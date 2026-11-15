"""
The Unclassified class, config['unclassified_class'].

One more class with no reference. Its mean expression per gene is an unknown
with a Gamma prior, shape r and rate r / M, and VarBayes.unclassified_upd
re-estimates it on every pass from the cells assigned to it. These tests check the bookkeeping (where the
column sits, that everything else is left alone), the update against plain
loops written straight from the formula, and that the Gamma it lands on is the
best one for the piece of the objective that depends on it.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln, psi

from pciSeq.src.core.main import VarBayes
from pciSeq.src.core.datatypes.singleCell import UNCLASSIFIED

GENES = ["Gene_%d" % i for i in range(10)]


def _build(base_opts, on=True, r=1.0, seed=7, classes=("Type_A", "Type_B", "Type_C")):
    rng = np.random.default_rng(seed)
    scref = pd.DataFrame(rng.random((10, len(classes))) * 100, index=GENES, columns=list(classes))
    scref.index.name = "gene_name"
    scref.columns.name = "class_name"

    xs = [10, 20, 30, 40, 10, 20, 30, 40]
    ys = [10, 20, 30, 40, 50, 50, 50, 50]
    cells_df = pd.DataFrame({
        "label": list(range(1, 9)),
        "area": [100, 120, 90, 110, 95, 105, 115, 100],
        "x0": xs, "y0": ys, "z0": [5] * 8,
        "values": [[[x, y, 5]] for x, y in zip(xs, ys)],
    })
    n = 400
    spots_df = pd.DataFrame({
        "x": rng.uniform(0, 50, n), "y": rng.uniform(0, 60, n), "z": rng.uniform(0, 10, n),
        "plane_id": rng.integers(0, 3, n),
        "label": rng.choice(np.arange(1, 9), n),
        "gene_name": rng.choice(GENES, n),
        "score": rng.uniform(0.5, 1.0, n), "intensity": rng.uniform(100, 1000, n),
    })
    opts = {**base_opts, "is3D": True, "img_dim": {"w": 100, "h": 100, "n_planes": 10},
            "label_map": {}, "rRho": 1000.0, "rTheta": 25.0, "mrf_beta": 1.0,
            "unclassified_class": on, "rUnclassified": r}
    return VarBayes(spots_df=spots_df, cells_df=cells_df, scRNAseq=scref, config=opts), scref


def _one_pass(vb, numba=True):
    """One pass of the loop up to and including the new step."""
    if numba:
        vb.geneCount_upd_numba(); vb.rho_upd(); vb.eta_upd_numba(); vb.theta_upd_numba()
        vb.gamma_upd_numba(); vb.cell_to_cellType_numba()
    else:
        vb.geneCount_upd(); vb.rho_upd(); vb.eta_upd(); vb.theta_upd()
        vb.gamma_upd(); vb.cell_to_cellType()


def test_off_by_default(base_opts):
    vb, _ = _build(base_opts, on=False)
    assert vb.single_cell.unclassified_idx is None
    assert UNCLASSIFIED not in vb.single_cell.classes
    assert vb.nK == 4


def test_column_is_there_and_zero_stays_last(base_opts):
    vb, scref = _build(base_opts)
    names = list(vb.single_cell.classes)
    assert names == ["Type_A", "Type_B", "Type_C", UNCLASSIFIED, "Zero"]
    assert list(vb.cellTypes.names) == names
    assert vb.nK == 5
    u = vb.single_cell.unclassified_idx
    assert u == 3

    # it starts as the top k mean that is the least like any class of the
    # reference, worked out here with plain loops
    start = vb.single_cell.mean_expression_adj.values[:, u]
    ref = scref.sort_index().values
    best, best_cos = None, np.inf
    for k in range(1, ref.shape[1] + 1):
        cand = np.array([np.sort(row)[-k:].mean() for row in ref])
        cos = max(cand @ ref[:, j] / np.linalg.norm(cand) / np.linalg.norm(ref[:, j]) for j in range(ref.shape[1]))
        if cos < best_cos:
            best, best_cos = cand, cos
    np.testing.assert_allclose(start, best * base_opts["Inefficiency"], rtol=1e-6)

    # the reference kept for the export is the one that was passed in
    assert UNCLASSIFIED not in vb.single_cell.raw_data.columns


def test_start_picks_the_k_furthest_from_any_class():
    """Two classes with no gene in common: the max (k=1) and the mean (k=2)
    are the same up to a scale, so they tie and the first one is kept. Then a
    random table, with the pick checked against plain loops."""
    from pciSeq.src.core.datatypes.singleCell import unclassified_start
    two = np.array([[4.0, 0.0], [0.0, 4.0]])
    col, k = unclassified_start(two)
    assert k == 1
    np.testing.assert_allclose(col, [4.0, 4.0])

    rng = np.random.default_rng(3)
    ref = rng.gamma(0.3, 10.0, size=(40, 6))
    col, k = unclassified_start(ref)
    cos = []
    for kk in range(1, 7):
        cand = np.array([np.sort(row)[-kk:].mean() for row in ref])
        cos.append(max(cand @ ref[:, j] / np.linalg.norm(cand) / np.linalg.norm(ref[:, j]) for j in range(6)))
    assert k == int(np.argmin(cos)) + 1
    np.testing.assert_allclose(col, [np.sort(row)[-k:].mean() for row in ref])


def test_a_reference_class_with_the_same_name_is_refused(base_opts):
    with pytest.raises(ValueError, match=UNCLASSIFIED):
        _build(base_opts, classes=("Type_A", "Type_B", UNCLASSIFIED))


def test_update_matches_plain_loops(base_opts):
    """shape and rate, element by element, from the formula in the docstring."""
    vb, _ = _build(base_opts, r=1.5)
    vb.initialise_state()
    _one_pass(vb)
    # give the class some cells, whatever the class step said
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(vb.nK), size=vb.nC)
    vb.cells.classProb = P
    # row 0 is the background. Give it reads and a class probability, so that
    # an update that lets it in comes out different from the loops below
    N0 = np.array(vb.cells.geneCount, dtype=np.float64)
    N0[0, :] = 7.0
    vb.cells.geneCount = N0
    assert P[0, vb.single_cell.unclassified_idx] > 0

    u = vb.single_cell.unclassified_idx
    eps, r = vb.config["SpotReg"], vb.config["rUnclassified"]
    m0 = vb._unclassified_prior_mean.copy()
    af = np.asarray(vb.cells.ini_cell_props["area_factor"], dtype=np.float64)
    theta, gamma, eta = vb.cells.theta_bar, vb.spots.gamma_bar, vb.genes.eta_bar
    N = vb.cells.geneCount
    others_before = vb.single_cell.mean_expression_adj.drop(columns=UNCLASSIFIED).values.copy()
    log_others_before = vb.single_cell.log_mean_expression.drop(columns=UNCLASSIFIED).values.copy()

    shape = np.full(vb.nG, r, dtype=np.float64)
    rate = r / m0
    for g in range(vb.nG):
        for c in range(1, vb.nC):          # row 0 is the background
            shape[g] += P[c, u] * float(N[c, g])
            rate[g] += P[c, u] * af[c] * float(eta[g]) * float(theta[c, u]) * float(gamma[c, g, u])

    vb.unclassified_upd()

    np.testing.assert_allclose(vb._unclassified_shape, shape, rtol=1e-6)
    np.testing.assert_allclose(vb._unclassified_rate, rate, rtol=1e-6)
    mu = np.maximum(shape / rate - eps, 0.0)
    np.testing.assert_allclose(vb.single_cell.mean_expression_adj.values[:, u], mu, rtol=1e-5, atol=1e-7)
    # the log table holds the expected log, not the log of the mean
    np.testing.assert_allclose(vb.single_cell.log_mean_expression.values[:, u], psi(shape) - np.log(rate),
                               rtol=1e-5, atol=1e-6)
    assert (vb.single_cell.log_mean_expression.values[:, u] < np.log(shape / rate)).all()

    # no other column moves, and Zero is still all zeros
    np.testing.assert_array_equal(vb.single_cell.mean_expression_adj.drop(columns=UNCLASSIFIED).values, others_before)
    np.testing.assert_array_equal(vb.single_cell.log_mean_expression.drop(columns=UNCLASSIFIED).values, log_others_before)
    assert (vb.single_cell.mean_expression_adj.values[:, -1] == 0).all()
    assert vb.single_cell.mean_expression_adj.values.dtype == np.float32


def _objective(a, b, shape, rate):
    """The piece of the objective that depends on q(m) = Gamma(a, b), for one
    gene: E_q[(shape - 1) log m - rate m] plus the entropy of q. shape - 1 and
    rate are what multiplies log m and m in the expected log joint."""
    e_log, e_m = psi(a) - np.log(b), a / b
    entropy = a - np.log(b) + gammaln(a) + (1.0 - a) * psi(a)
    return (shape - 1.0) * e_log - rate * e_m + entropy


def test_the_gamma_posterior_is_the_best_one(base_opts):
    """Moving the shape, the rate, or both, away from the update can only lower
    the objective. The coefficients are built here with loops, not read off the
    code, so a wrong shape or rate in unclassified_upd shows as a better
    neighbour."""
    vb, _ = _build(base_opts, r=0.7)
    vb.initialise_state()
    _one_pass(vb)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(vb.nK), size=vb.nC)
    vb.cells.classProb = P

    u = vb.single_cell.unclassified_idx
    r, m0 = vb.config["rUnclassified"], vb._unclassified_prior_mean.copy()
    af = np.asarray(vb.cells.ini_cell_props["area_factor"], dtype=np.float64)
    coef_log = np.full(vb.nG, r - 1.0)      # multiplies log m[g]
    coef_lin = r / m0                       # multiplies -m[g]
    for g in range(vb.nG):
        for c in range(1, vb.nC):
            coef_log[g] += P[c, u] * float(vb.cells.geneCount[c, g])
            coef_lin[g] += (P[c, u] * af[c] * float(vb.genes.eta_bar[g]) * float(vb.cells.theta_bar[c, u])
                            * float(vb.spots.gamma_bar[c, g, u]))

    vb.unclassified_upd()
    a, b = vb._unclassified_shape, vb._unclassified_rate
    best = _objective(a, b, coef_log + 1.0, coef_lin)
    for fa, fb in [(1.05, 1.0), (0.95, 1.0), (1.0, 1.05), (1.0, 0.95), (1.05, 1.05), (0.95, 0.95),
                   (1.05, 0.95), (0.95, 1.05)]:
        moved = _objective(a * fa, b * fb, coef_log + 1.0, coef_lin)
        assert (moved < best).all(), (fa, fb)


def test_mean_and_expected_log_of_the_gamma_by_integration(base_opts):
    """The two numbers the update writes into the tables, checked against the
    integrals they stand for: E[m] and E[log m] under Gamma(shape, rate)."""
    from scipy.integrate import quad
    from scipy.stats import gamma as gamma_dist
    vb, _ = _build(base_opts, r=1.3)
    vb.initialise_state()
    _one_pass(vb)
    vb.cells.classProb = np.random.default_rng(2).dirichlet(np.ones(vb.nK), size=vb.nC)
    vb.unclassified_upd()
    u = vb.single_cell.unclassified_idx
    eps = vb.config["SpotReg"]
    for g in range(vb.nG):
        a, b = vb._unclassified_shape[g], vb._unclassified_rate[g]
        q = gamma_dist(a, scale=1.0 / b)
        mean = quad(lambda x: x * q.pdf(x), 0, np.inf)[0]
        mean_log = quad(lambda x: np.log(x) * q.pdf(x), 0, np.inf)[0]
        np.testing.assert_allclose(vb.single_cell.mean_expression_adj.values[g, u] + eps, mean, rtol=1e-4)
        np.testing.assert_allclose(vb.single_cell.log_mean_expression.values[g, u], mean_log, rtol=1e-4, atol=1e-5)


def test_elbo_term_is_the_kl_between_the_two_gammas(base_opts):
    """The piece of the elbo for the mean expression of the class, checked
    against the integral it stands for: sum over genes of
    integral q(m) * (log q(m) - log p(m)) dm."""
    from scipy.integrate import quad
    from scipy.stats import gamma as gamma_dist
    from pciSeq.src.core.utils.elbo import unclassified_kl
    vb, _ = _build(base_opts, r=1.3)
    vb.initialise_state()
    # still on the prior, so nothing to pay
    assert unclassified_kl(vb) == pytest.approx(0.0, abs=1e-9)
    _one_pass(vb)
    vb.cells.classProb = np.random.default_rng(2).dirichlet(np.ones(vb.nK), size=vb.nC)
    vb.unclassified_upd()
    r = vb.config["rUnclassified"]
    total = 0.0
    for g in range(vb.nG):
        q = gamma_dist(vb._unclassified_shape[g], scale=1.0 / vb._unclassified_rate[g])
        p = gamma_dist(r, scale=vb._unclassified_prior_mean[g] / r)
        total += quad(lambda x: q.pdf(x) * (q.logpdf(x) - p.logpdf(x)), 0, np.inf, limit=200)[0]
    assert total > 0
    np.testing.assert_allclose(unclassified_kl(vb), total, rtol=1e-5)


def test_elbo_has_no_such_term_with_the_class_off(base_opts):
    from pciSeq.src.core.utils.elbo import unclassified_kl
    vb, _ = _build(base_opts, on=False)
    vb.initialise_state()
    assert unclassified_kl(vb) == 0.0


def test_the_update_is_the_top_of_the_elbo(base_opts):
    """unclassified_upd should leave the elbo at its highest point over the
    shape and the rate: the step itself must not lower it, and moving either
    of the two away from what the step found must lower it."""
    from pciSeq.src.core.utils.elbo import calc_elbo
    vb, _ = _build(base_opts, r=1.3)
    vb.initialise_state()
    for _ in range(3):
        _one_pass(vb)
        before = calc_elbo(vb)
        vb.unclassified_upd()
        assert calc_elbo(vb) >= before - 1e-6
        vb.spots_to_cell_numba()
    _one_pass(vb)
    vb.unclassified_upd()

    eps = vb.config["SpotReg"]
    a, b = vb._unclassified_shape.copy(), vb._unclassified_rate.copy()

    def put(shape, rate):
        vb._unclassified_shape, vb._unclassified_rate = shape, rate
        vb.single_cell.set_unclassified(np.maximum(shape / rate - eps, 0.0), psi(shape) - np.log(rate))
        return calc_elbo(vb)

    best = put(a, b)
    for fa, fb in [(1.02, 1.0), (0.98, 1.0), (1.0, 1.02), (1.0, 0.98), (1.05, 1.05), (0.95, 0.95)]:
        assert put(a * fa, b * fb) < best, (fa, fb)


def test_with_no_cells_it_stays_at_the_prior(base_opts):
    vb, _ = _build(base_opts, r=1.7)       # not 1, so a missing log(r) shows
    vb.initialise_state()
    _one_pass(vb)
    u = vb.single_cell.unclassified_idx
    r, M = vb.config["rUnclassified"], vb._unclassified_prior_mean
    start = vb.single_cell.mean_expression_adj.values[:, u].copy()
    log_start = vb.single_cell.log_mean_expression.values[:, u].copy()
    # what initialise_state left there is the prior: mean M, expected log psi(r) - log(r / M)
    np.testing.assert_allclose(start + vb.config["SpotReg"], M, rtol=1e-5)
    np.testing.assert_allclose(log_start, psi(r) - np.log(r) + np.log(M), rtol=1e-5, atol=1e-6)
    P = vb.cells.classProb.copy()
    P[:, u] = 0.0
    vb.cells.classProb = P
    vb.unclassified_upd()
    np.testing.assert_allclose(vb.single_cell.mean_expression_adj.values[:, u], start, rtol=1e-5)
    np.testing.assert_allclose(vb.single_cell.log_mean_expression.values[:, u], log_start, rtol=1e-5, atol=1e-6)


def test_cells_of_a_class_left_out_of_the_reference_teach_it_their_profile(base_opts):
    """Hand the class the reads of cells that follow one profile and its
    estimate times the scale factors gives those reads back."""
    vb, _ = _build(base_opts, r=1e-3)
    vb.initialise_state()
    _one_pass(vb)
    u = vb.single_cell.unclassified_idx
    P = np.zeros((vb.nC, vb.nK)); P[:, u] = 1.0
    vb.cells.classProb = P
    vb.unclassified_upd()

    af = np.asarray(vb.cells.ini_cell_props["area_factor"], dtype=np.float64)
    m = vb.single_cell.mean_expression_adj.values[:, u] + vb.config["SpotReg"]
    predicted = np.einsum('g, c, g, c, cg -> g', m, af[1:], vb.genes.eta_bar, vb.cells.theta_bar[1:, u],
                          vb.spots.gamma_bar[1:, :, u])
    observed = vb.cells.geneCount[1:].sum(axis=0)
    seen = observed > 0
    np.testing.assert_allclose(predicted[seen], observed[seen], rtol=2e-2)


def test_numpy_and_numba_paths_agree_with_the_class_on(base_opts):
    a, _ = _build(base_opts)
    b, _ = _build(base_opts)
    a.initialise_state(); b.initialise_state()
    for _ in range(3):
        _one_pass(a, numba=True); a.unclassified_upd(); a.spots_to_cell_numba(); a.gaussian_upd_numba()
        _one_pass(b, numba=False); b.unclassified_upd(); b.spots_to_cell(); b.gaussian_upd()
    u = a.single_cell.unclassified_idx
    np.testing.assert_allclose(a.single_cell.mean_expression_adj.values, b.single_cell.mean_expression_adj.values,
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(a.cells.classProb, b.cells.classProb, atol=1e-4)
    np.testing.assert_allclose(a.spots.parent_cell_prob, b.spots.parent_cell_prob, atol=1e-4)
    # the cached scaled expression follows the column
    af = np.asarray(a.cells.ini_cell_props["area_factor"], dtype=np.float32)
    np.testing.assert_allclose(a.scaled_exp[:, :, u],
                               np.outer(af, a.single_cell.mean_expression_adj.values[:, u]), rtol=1e-6)
    assert not np.allclose(a.single_cell.mean_expression_adj.values[:, u],
                           a._unclassified_prior_mean - a.config["SpotReg"])


def test_the_whole_loop_runs_with_the_class_on(base_opts):
    vb, _ = _build({**base_opts, "max_iter": 15})
    cellData, geneData = vb.run()
    assert np.isfinite(vb.single_cell.mean_expression_adj.values).all()
    assert np.isfinite(vb.cells.classProb).all()
    np.testing.assert_allclose(vb.cells.classProb.sum(axis=1), 1.0, atol=1e-5)
