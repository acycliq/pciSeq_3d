"""
Checks that the numba versions of the gamma, eta, theta and class steps give
the same result as the numpy ones.

Same idea as test_spots_to_cell_ab.py: the numpy versions are the readable
reference, the numba ones are what the loop runs. Each pair writes the same
arrays, so run one after the other on the same state and compare. Where a
step reads what it writes (classProb in the class step), the starting value
is put back before the second run.

The big arrays are float32 on both sides, so the tolerance is float32
rounding, not the 1e-12 of the spot test.
"""
import numpy as np

RTOL = 1e-5
ATOL = 1e-5


def _ready(vb):
    vb.initialise_state()
    vb.geneCount_upd()
    vb.gamma_upd()


def test_gamma_upd_numba_matches_numpy(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd()

    vb.gamma_upd()
    beta_np = vb.spots._post_rate.copy()
    lg_np = vb.spots.log_gamma_bar.copy()
    g_np = vb.spots.gamma_bar.copy()

    vb.gamma_upd_numba()
    beta_nb = vb.spots._post_rate
    lg_nb = vb.spots.log_gamma_bar
    g_nb = vb.spots.gamma_bar

    assert beta_nb.dtype == beta_np.dtype == np.float32
    np.testing.assert_allclose(beta_nb, beta_np, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(lg_nb, lg_np, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(g_nb, g_np, rtol=RTOL, atol=ATOL)


def test_cell_to_cellType_numba_matches_numpy(minimal_varbayes):
    vb = minimal_varbayes
    _ready(vb)
    prob_before = vb.cells.classProb.copy()

    vb.cell_to_cellType()
    contr_np = vb.cells.nb_contr.copy()
    prob_np = vb.cells.classProb.copy()

    vb.cells.classProb = prob_before.copy()
    vb.cell_to_cellType_numba()
    contr_nb = vb.cells.nb_contr
    prob_nb = vb.cells.classProb

    assert contr_nb.dtype == contr_np.dtype == np.float32
    np.testing.assert_allclose(contr_nb, contr_np, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(prob_nb, prob_np, rtol=RTOL, atol=ATOL)


def test_eta_upd_numba_matches_numpy(minimal_varbayes):
    vb = minimal_varbayes
    _ready(vb)

    vb.eta_upd()
    eta_np = vb.genes.eta_bar.copy()
    logeta_np = vb.genes.logeta_bar.copy()

    vb.eta_upd_numba()
    np.testing.assert_allclose(vb.genes.eta_bar, eta_np, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(vb.genes.logeta_bar, logeta_np, rtol=RTOL, atol=ATOL)


def test_theta_upd_numba_matches_numpy(minimal_varbayes):
    vb = minimal_varbayes
    _ready(vb)

    vb.theta_upd()
    theta_np = vb.cells.theta_bar.copy()

    vb.theta_upd_numba()
    np.testing.assert_allclose(vb.cells.theta_bar, theta_np, rtol=RTOL, atol=ATOL)
