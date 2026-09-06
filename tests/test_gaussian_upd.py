"""
gaussian_upd is switched off in the loop but has to keep working.

The call in main_loop is commented out, so nothing else in the suite touches
centroid_upd or cov_upd. Without this test they could break silently and nobody
would find out until the step is switched back on.
"""
import numpy as np


def _one_pass(vb):
    vb.initialise_state()
    for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd,
                 vb.gamma_upd, vb.cell_to_cellType, vb.spots_to_cell_numba):
        step()


def test_gaussian_upd_runs_and_moves_the_cells(minimal_varbayes):
    vb = minimal_varbayes
    _one_pass(vb)
    cov_before = vb.cells.cov.copy()
    centroid_before = vb.cells.centroid.values.copy()

    vb.gaussian_upd()          # centroid_upd then cov_upd

    assert np.isfinite(vb.cells.cov).all(), 'cov came back non-finite'
    assert np.isfinite(vb.cells.centroid.values).all(), 'centroid came back non-finite'
    assert vb.cells.cov.shape == cov_before.shape
    assert vb.cells.centroid.values.shape == centroid_before.shape
    assert not np.allclose(vb.cells.cov, cov_before), 'cov did not move at all'


def test_the_background_cell_keeps_the_prior_covariance(minimal_varbayes):
    """cov_upd maps index 0 back to the prior on purpose, it is not a real cell."""
    vb = minimal_varbayes
    _one_pass(vb)
    prior = vb.cells.ini_cov()[0].copy()
    vb.gaussian_upd()
    assert np.allclose(vb.cells.cov[0], prior)
