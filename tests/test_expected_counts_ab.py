"""
Checks the spot step now that it reads the single cell reference by gene id.

The spot kernel used to be handed a spots x classes array, expected_counts:
the genes x classes table of the reference with the row of each spot's gene
copied out, once per spot, by gene NAME. Now it is handed the table itself and
reads row gene_id[s] for spot s, so by POSITION. That is the same number only
as long as row g of the table is the gene with id g.

So there are three things to check here:
  1. the new kernel gives the very same arrays as the old one. The old kernel
     is kept below, in _spots_to_cell_kernel_before.
  2. VarBayes._check_gene_order stops the run when the reference rows and the
     gene ids do not line up.
  3. the trap is real: with the rows out of order and the check out of the
     way, the result does come out wrong. If this ever stops failing the way
     it should, the tests above are not looking at the right thing.
"""
import numpy as np
import pandas as pd
import pytest
from numba import njit, prange

from pciSeq.src.core.utils.numba_kernels import spots_to_cell_numba_kernel


@njit(parallel=True)
def _spots_to_cell_kernel_before(parent, gene_id, classProb, log_gamma_bar, log_theta,
                                 expected_counts, logeta_bar, nNb,
                                 wSpotCell, attention, expr_fluct, cell_ineff, gene_ineff):
    """NOTE: this is spots_to_cell_numba_kernel before it read the reference
    by gene id. expected_counts is the spots x classes array, one row per spot.
    It is not used anywhere in pciSeq anymore, it lives here only to check the
    new code against."""
    nS = parent.shape[0]
    nK = classProb.shape[1]
    for s in prange(nS):
        g = gene_id[s]
        le = logeta_bar[s]
        for n in range(nNb):
            cell = parent[s, n]
            t1 = 0.0
            t2 = 0.0
            t3 = 0.0
            for k in range(nK):
                cpk = classProb[cell, k]
                t1 += expected_counts[s, k] * cpk
                t2 += cpk * log_gamma_bar[cell, g, k]
                t3 += cpk * log_theta[cell, k]
            wSpotCell[s, n] = t1 + t2 + t3 + le
            attention[s, n] = t1
            expr_fluct[s, n] = t2
            cell_ineff[s, n] = t3
            gene_ineff[s, n] = le


def _ready(vb):
    """The steps the loop runs before the spot step."""
    vb.initialise_state()
    for name in ('geneCount_upd', 'rho_upd', 'eta_upd', 'theta_upd', 'gamma_upd', 'cell_to_cellType'):
        getattr(vb, name)()
    return vb


def _kernel_inputs(vb):
    nN = vb.nN
    nS = vb.spots.data.gene_name.shape[0]
    common = (np.ascontiguousarray(vb.spots.parent_cell_id), np.ascontiguousarray(vb.spots.gene_id),
              np.ascontiguousarray(vb.cells.classProb), np.ascontiguousarray(vb.spots.log_gamma_bar),
              np.ascontiguousarray(np.log(vb.cells.theta_bar)))
    logeta_bar = np.ascontiguousarray(vb.genes.logeta_bar[vb.spots.gene_id])
    outs = lambda: [np.zeros([nS, nN])] + [np.zeros([nS, nN], dtype=np.float32) for _ in range(4)]
    return common, logeta_bar, nN - 1, outs


def test_reading_by_gene_id_gives_the_same_arrays(minimal_varbayes):
    vb = _ready(minimal_varbayes)
    common, logeta_bar, nNb, outs = _kernel_inputs(vb)

    # before: the row of each spot's gene, picked by gene NAME
    gn = vb.spots.data.gene_name.values
    by_name = np.ascontiguousarray(vb.single_cell.log_mean_expression.loc[gn].values)
    before = outs()
    _spots_to_cell_kernel_before(*common, by_name, logeta_bar, nNb, *before)

    # now: the table itself, read by gene id
    table = np.ascontiguousarray(vb.single_cell.log_mean_expression.values)
    now = outs()
    spots_to_cell_numba_kernel(*common, table, logeta_bar, nNb, *now)

    assert table.shape == (vb.nG, vb.nK) and by_name.shape == (vb.nS, vb.nK)
    for a, b in zip(now, before):
        assert a.dtype == b.dtype
        assert np.array_equal(a, b)


def test_the_reference_rows_are_the_gene_panel(minimal_varbayes):
    vb = minimal_varbayes
    assert np.array_equal(vb.single_cell.log_mean_expression.index.values, vb.genes.gene_panel)
    assert np.array_equal(vb.single_cell.mean_expression_adj.index.values, vb.genes.gene_panel)
    # and every spot's id points back at its own gene name
    assert np.array_equal(vb.genes.gene_panel[vb.spots.gene_id], vb.spots.data.gene_name.values)
    vb._check_gene_order()   # says nothing


def _swap_two_rows(table):
    order = list(table.index)
    order[0], order[1] = order[1], order[0]
    return table.loc[order]


@pytest.mark.parametrize('which', ['_log_mean_expression_adj', '_mean_expression_adj'])
def test_rows_out_of_order_stop_the_run(minimal_varbayes, which):
    vb = minimal_varbayes
    setattr(vb.single_cell, which, _swap_two_rows(getattr(vb.single_cell, which)))
    with pytest.raises(ValueError, match='not in the same order'):
        vb._check_gene_order()


def test_a_missing_row_stops_the_run(minimal_varbayes):
    vb = minimal_varbayes
    vb.single_cell._log_mean_expression_adj = vb.single_cell._log_mean_expression_adj.iloc[:-1]
    with pytest.raises(ValueError, match='not in the same order'):
        vb._check_gene_order()


def test_the_check_runs_when_varbayes_is_built(minimal_varbayes, monkeypatch):
    # a check that is never called protects nothing. Building the fixture has
    # already gone through _setup_components, so swap the check for one that
    # shouts and go through it again with the same pieces
    import inspect
    from pciSeq.src.core.main import VarBayes
    assert '_check_gene_order()' in inspect.getsource(VarBayes._setup_components)

    def shout(self):
        raise RuntimeError('the check was called')
    monkeypatch.setattr(VarBayes, '_check_gene_order', shout)
    monkeypatch.setattr('pciSeq.src.core.main.Cells', lambda *a, **k: minimal_varbayes.cells)
    monkeypatch.setattr('pciSeq.src.core.main.Spots', lambda *a, **k: minimal_varbayes.spots)
    monkeypatch.setattr('pciSeq.src.core.main.Genes', lambda *a, **k: minimal_varbayes.genes)
    monkeypatch.setattr('pciSeq.src.core.main.SingleCell', lambda *a, **k: minimal_varbayes.single_cell)
    monkeypatch.setattr('pciSeq.src.core.main.CellClass', lambda *a, **k: minimal_varbayes.cellTypes)
    with pytest.raises(RuntimeError, match='the check was called'):
        minimal_varbayes._setup_components(None, None, None)


def test_the_trap_is_real(minimal_varbayes):
    """With the reference rows swapped and nobody checking, the numba spot
    step (by position) and the plain numpy one (by name) must disagree. This
    is what a silent mix up would look like, and it shows that comparing the
    two, as tests/test_spots_to_cell_ab.py does, would catch it."""
    vb = _ready(minimal_varbayes)

    vb.spots_to_cell()
    p_by_name = vb.spots.parent_cell_prob.copy()
    vb.spots_to_cell_numba()
    p_by_position = vb.spots.parent_cell_prob.copy()
    np.testing.assert_allclose(p_by_position, p_by_name, rtol=1e-12, atol=1e-12)

    # now break the order behind the back of the check
    sc = vb.single_cell
    sc._log_mean_expression_adj = _swap_two_rows(sc._log_mean_expression_adj)
    vb.spots_to_cell()
    p_by_name_2 = vb.spots.parent_cell_prob.copy()
    vb.spots_to_cell_numba()
    p_by_position_2 = vb.spots.parent_cell_prob.copy()

    # by name does not care about the order of the rows
    np.testing.assert_allclose(p_by_name_2, p_by_name, rtol=1e-12, atol=1e-12)
    # by position does, and gets it wrong
    assert not np.allclose(p_by_position_2, p_by_name_2, rtol=1e-6, atol=1e-6)
