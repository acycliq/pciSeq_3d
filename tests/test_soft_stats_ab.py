"""
Checks that soft_stats_numba gives the same soft count, mean and scatter as
soft_stats, the numpy reference the cell geometry update is written against.
Both read the spot assignments and write nothing, so run both on the same
state and compare. Sums are float64 on both sides, only their order differs.
"""
import numpy as np

from pciSeq.src.core.utils import gaussian_model


def test_soft_stats_numba_matches_numpy(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd()
    vb.gamma_upd()
    vb.cell_to_cellType()
    vb.spots_to_cell_numba()

    d = vb.cells.dim
    n_np, x_np, S_np = gaussian_model.soft_stats(vb.spots, vb.cells, d)
    n_nb, x_nb, S_nb = gaussian_model.soft_stats_numba(vb.spots, vb.cells, d, nchunk=3)

    np.testing.assert_allclose(n_nb, n_np, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(x_nb, x_np, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(S_nb, S_np, rtol=1e-10, atol=1e-6)
