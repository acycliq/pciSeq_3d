
import numpy as np
import pytest
from pciSeq.src.core.utils.ops_utils import (
    b_upd_naive,
    b_upd_optimized,
    b_upd_gpu,
    b_upd_fixed_point,
    _HAS_CUPY,
)


def _make_test_inputs(seed=42, nC=5, nG=50, nK=3):
    """Build a small synthetic problem matching test_b_upd_mathematical_consistency."""
    np.random.seed(seed)
    mu     = np.random.rand(nG, nK).astype(np.float32)
    Ac     = np.random.rand(nC).astype(np.float32)
    eta    = np.random.rand(nG).astype(np.float32)
    theta  = np.random.rand(nC, nK).astype(np.float32)
    gamma  = np.random.rand(nC, nG, nK).astype(np.float32)
    b_init = np.random.randn(nC, nG, nK).astype(np.float32) * 0.1
    counts = np.random.poisson(lam=5, size=(nC, nG)).astype(np.float32)
    class_prob = np.ones((nC, nK), dtype=np.float32)
    precision = np.zeros((nK, nG, nG), dtype=np.float32)
    for k in range(nK):
        A = np.random.rand(nG, nG).astype(np.float32)
        precision[k] = A @ A.T + np.eye(nG, dtype=np.float32) * 5
    return b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob


def test_b_upd_mathematical_consistency():
    """
    Verifies that the optimized PCG implementation matches
    the naive Newton-Raphson implementation derived from the paper.
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = _make_test_inputs()

    # Run Naive
    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)

    # Run Optimized; high max_iter so we hit the float32 floor against naive
    b_opt = b_upd_optimized(b_init.copy(), mu, Ac, eta, theta, gamma,
                            precision, counts, class_prob, max_iter=50)

    max_diff = np.abs(b_naive - b_opt).max()
    print(f"Max difference between Naive and Optimized: {max_diff:.2e}")
    assert max_diff < 1e-3, f"Optimized b_upd diverges from naive. Max diff: {max_diff}"


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy/GPU not available")
def test_b_upd_gpu_consistency():
    """
    Verifies that the GPU PCG implementation matches the naive Newton-Raphson
    gold standard. Same problem and tolerance as the CPU optimized test.
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = _make_test_inputs()

    # Naive (CPU, float64 LU)
    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)

    # GPU PCG with numpy inputs (drop-in mode: function uploads/downloads internally)
    b_gpu = b_upd_gpu(b_init.copy(), mu, Ac, eta, theta, gamma,
                      precision, counts, class_prob, max_iter=50)

    max_diff = np.abs(b_naive - b_gpu).max()
    print(f"Max difference between Naive and GPU: {max_diff:.2e}")
    assert max_diff < 1e-3, f"GPU b_upd diverges from naive. Max diff: {max_diff}"


@pytest.mark.skipif(not _HAS_CUPY, reason="CuPy/GPU not available")
def test_b_upd_gpu_matches_cpu_optimized():
    """
    Verifies that GPU PCG produces the same b as the CPU optimized PCG
    when run with identical inputs. Both run the same Jacobi-PCG inner loop;
    only the per-element sum order differs, so we expect float32-rounding
    agreement (well below the 1e-3 production tolerance).
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = _make_test_inputs()

    b_cpu = b_upd_optimized(b_init.copy(), mu, Ac, eta, theta, gamma,
                            precision, counts, class_prob, max_iter=20)
    b_gpu = b_upd_gpu(b_init.copy(), mu, Ac, eta, theta, gamma,
                      precision, counts, class_prob, max_iter=20)

    max_diff = np.abs(b_cpu - b_gpu).max()
    print(f"Max difference between CPU optimized and GPU: {max_diff:.2e}")
    assert max_diff < 1e-3, f"GPU b_upd disagrees with CPU optimized. Max diff: {max_diff}"


def _make_realistic_inputs(seed=42, nC=50, nG=50, nK=3, mu_max=65.0,
                           prior_floor=0.05):
    """Build a problem matching pciSeq's actual operating regime: heavy-tailed
    mu with realistic max, gamma-distributed multipliers, and a weak prior so
    that ||Pk^(-1) D||_2 > 1 for some (c, k). This is the regime where
    fixed-point will diverge.
    """
    rng = np.random.default_rng(seed)
    mu     = (rng.random((nG, nK)) ** 4).astype(np.float32) * mu_max
    Ac     = np.ones(nC, dtype=np.float32)
    eta    = rng.gamma(shape=2.0, scale=0.5, size=nG).astype(np.float32)
    theta  = rng.gamma(shape=2.0, scale=0.5, size=(nC, nK)).astype(np.float32)
    gamma  = rng.gamma(shape=2.0, scale=0.5, size=(nC, nG, nK)).astype(np.float32)
    b_init = np.zeros((nC, nG, nK), dtype=np.float32)
    counts = rng.poisson(2, (nC, nG)).astype(np.float32)
    class_prob = np.ones((nC, nK), dtype=np.float32)

    precision = np.zeros((nK, nG, nG), dtype=np.float32)
    for k in range(nK):
        A = rng.standard_normal((nG, nG)).astype(np.float32) * 0.3
        cov = A @ A.T + np.eye(nG, dtype=np.float32) * prior_floor
        precision[k] = np.linalg.inv(cov).astype(np.float32)
    return b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob


def test_b_upd_fixed_point_safe_regime():
    """
    In the friendly synthetic regime (small mu, strong prior `Pk = AA^T + 5I`)
    the fixed-point solver converges in ~3 iterations to within 1e-3 of naive.
    Same problem and tolerance as test_b_upd_mathematical_consistency.
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = _make_test_inputs()

    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)
    b_fp    = b_upd_fixed_point(b_init.copy(), mu, Ac, eta, theta, gamma,
                                precision, counts, class_prob, max_iter=5)

    max_diff = np.abs(b_naive - b_fp).max()
    print(f"Max difference between Naive and fixed-point (safe regime): {max_diff:.2e}")
    assert max_diff < 1e-3, (
        f"Fixed-point diverges from naive in safe regime. Max diff: {max_diff}. "
        "This should not happen; check the implementation."
    )


def test_b_upd_fixed_point_diverges_in_realistic_regime():
    """
    Documents the failure mode: in pciSeq's actual operating regime
    (heavy-tailed mu, weak prior), `||Pk^(-1) D||_2 > 1` for some (c, k)
    and the fixed-point iteration diverges. This test asserts the divergence
    so a future change that "accidentally" makes fixed-point converge here
    will trip the assertion and force a re-evaluation of the assumptions.

    If you want fixed-point to actually solve a real-pciSeq-shape problem,
    you need either a stronger prior or a robust solver (b_upd_optimized).
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = (
        _make_realistic_inputs(prior_floor=0.05)
    )

    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)
    b_fp    = b_upd_fixed_point(b_init.copy(), mu, Ac, eta, theta, gamma,
                                precision, counts, class_prob, max_iter=5)

    max_diff = np.abs(b_naive - b_fp).max()
    print(f"Max difference between Naive and fixed-point (realistic regime): {max_diff:.2e}")
    # Must be >> 1: explicit assertion that fixed-point diverged here.
    assert max_diff > 1.0, (
        f"Fixed-point unexpectedly converged in the realistic regime "
        f"(max diff {max_diff}). The conditioning analysis assumed "
        "||Pk^(-1) D||_2 > 1 for some cells; this no longer holds and the "
        "convergence guarantees in the docstring need to be revisited."
    )


def test_b_upd_optimized_robust_in_realistic_regime():
    """
    Counterpart to the previous test: PCG/Jacobi remains stable in the same
    realistic regime where fixed-point diverges. With max_iter=50 the worst
    error stays bounded (no divergence), even if not as tight as the friendly
    regime.
    """
    b_init, mu, Ac, eta, theta, gamma, precision, counts, class_prob = (
        _make_realistic_inputs(prior_floor=0.05)
    )

    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)
    b_pcg   = b_upd_optimized(b_init.copy(), mu, Ac, eta, theta, gamma,
                              precision, counts, class_prob, max_iter=50)

    max_diff = np.abs(b_naive - b_pcg).max()
    print(f"Max difference between Naive and PCG (realistic regime): {max_diff:.2e}")
    # PCG is stable: error is bounded (no divergence), even if loose.
    assert max_diff < 100.0, (
        f"PCG unexpectedly diverged in the realistic regime "
        f"(max diff {max_diff}). This contradicts the unconditional-stability "
        "guarantee of CG on SPD systems; investigate."
    )


if __name__ == "__main__":
    test_b_upd_mathematical_consistency()
    test_b_upd_fixed_point_safe_regime()
    test_b_upd_fixed_point_diverges_in_realistic_regime()
    test_b_upd_optimized_robust_in_realistic_regime()
    if _HAS_CUPY:
        test_b_upd_gpu_consistency()
        test_b_upd_gpu_matches_cpu_optimized()
    else:
        print("CuPy not available; skipping GPU tests.")
