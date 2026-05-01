
import numpy as np
import pytest
from pciSeq.src.core.utils.ops_utils import b_upd_naive, b_upd_optimized

def test_b_upd_mathematical_consistency():
    """
    Verifies that the optimized PCG implementation matches 
    the naive Newton-Raphson implementation derived from the paper.
    """
    # Test Case: 5 Cells, 50 Genes, 3 Classes
    nC = 5
    nG = 50
    nK = 3

    np.random.seed(42)
    mu = np.random.rand(nG, nK).astype(np.float32)
    Ac = np.random.rand(nC).astype(np.float32)
    eta = np.random.rand(nG).astype(np.float32)
    theta = np.random.rand(nC, nK).astype(np.float32)
    gamma = np.random.rand(nC, nG, nK).astype(np.float32)
    b_init = np.random.randn(nC, nG, nK).astype(np.float32) * 0.1
    counts = np.random.poisson(lam=5, size=(nC, nG)).astype(np.float32)
    
    # Force all updates to run for full verification by setting class_prob to 1.0
    class_prob = np.ones((nC, nK), dtype=np.float32)

    # Precision Matrix (Σ^-1)
    precision = np.zeros((nK, nG, nG), dtype=np.float32)
    for k in range(nK):
        A = np.random.rand(nG, nG).astype(np.float32)
        precision[k] = A @ A.T + np.eye(nG, dtype=np.float32) * 5

    # Run Naive
    b_naive = b_upd_naive(b_init, mu, Ac, eta, theta, gamma, precision, counts)
    
    # Run Optimized
    # Use higher max_iter for exact comparison with Newton-Raphson
    b_opt = b_upd_optimized(b_init.copy(), mu, Ac, eta, theta, gamma, precision, counts, class_prob, max_iter=50)

    # Check consistency
    max_diff = np.abs(b_naive - b_opt).max()
    print(f"Max difference between Naive and Optimized: {max_diff:.2e}")
    
    # We expect high precision (Newton-Raphson is the gold standard)
    assert max_diff < 1e-3, f"Optimized b_upd results diverge from naive math. Max diff: {max_diff}"

if __name__ == "__main__":
    test_b_upd_mathematical_consistency()
