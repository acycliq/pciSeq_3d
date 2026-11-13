"""
Checks that the spot_id column of cellData is the same as it was before the
contributing spots were rewritten.

cells_summary now builds the column with summary.contributing_spot_lists, which
lays the spots out in flat arrays and makes lists only for the genes that are
shown. Before that it took the cells by genes array of lists from
get_contributing_spots_2, reordered it and turned every entry into a list. That
older way is kept below, in _spot_id_lists_before, and the two must give the
very same nested lists.
"""
import numpy as np

from pciSeq.src.core import summary

TOL = 0.001
SPOT_TOL = TOL / 10


def _spot_id_lists_before(cells, spots, genes, iCounts, shown, tol):
    """NOTE: this is the code cells_summary used for the spot_id column before
    contributing_spot_lists replaced it. It is not used anywhere in pciSeq
    anymore, it lives here only to check the new code against."""
    spot_ids = summary.get_contributing_spots_2(cells, spots, genes, tol)

    # reorder each row so spots for genes with higher read counts come first
    spot_ids = np.take_along_axis(spot_ids, iCounts, axis=1)

    # keep only the genes with counts above the threshold, and convert each to a plain
    # list. np.asarray first: a gene with no spot above spot_tol comes back as the empty
    # python list that aggregate fills in, and that has no tolist of its own.
    return [
        [np.asarray(d).tolist() for d in row[mask]]
        for row, mask in zip(spot_ids, shown)
    ]


def _fitted(vb, passes=1):
    vb.initialise_state()
    for _ in range(passes):
        for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd, vb.gamma_upd,
                     vb.cell_to_cellType, vb.spots_to_cell_numba):
            step()
    vb.geneCount_upd()
    return vb


def _both(vb, tol):
    cells, spots, genes = vb.cells, vb.spots, vb.genes
    iCounts = np.argsort(-1 * cells.geneCount, axis=1)
    gene_count = np.take_along_axis(cells.geneCount, iCounts, axis=1)
    shown = [d > TOL for d in gene_count]

    before = _spot_id_lists_before(cells, spots, genes, iCounts, shown, tol)
    now = summary.contributing_spot_lists(cells, spots, genes, iCounts, shown, tol)

    assert len(now) == len(before) == cells.nC
    # row 0 is the background. It is dropped from cellData and the new code does
    # not bother to fill it, so compare from row 1 on
    assert now[1:] == before[1:]
    # and the same python types all the way down, the tsv is printed from these
    for row in now[1:]:
        assert all(type(lst) is list for lst in row)
        assert all(type(s) is int for lst in row for s in lst)
    return now


def test_same_lists_at_the_cut_off_cells_summary_uses(minimal_varbayes):
    vb = _fitted(minimal_varbayes)
    now = _both(vb, SPOT_TOL)
    assert any(lst for row in now[1:] for lst in row), 'no spot was listed at all'


def test_same_lists_after_more_passes(minimal_varbayes):
    _both(_fitted(minimal_varbayes, passes=4), SPOT_TOL)


def test_same_lists_with_no_cut_off(minimal_varbayes):
    # tol 0 lists every candidate spot with a probability above zero
    _both(_fitted(minimal_varbayes), 0.0)


def test_same_lists_with_a_high_cut_off(minimal_varbayes):
    # 0.5 leaves plenty of shown genes without a single spot, so with an empty list
    now = _both(_fitted(minimal_varbayes), 0.5)
    assert any(lst == [] for row in now[1:] for lst in row)


def test_same_lists_on_the_float32_start(minimal_varbayes):
    # before any spot assignment the probs are float32, all of them 1 / nN
    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd()
    assert vb.spots.parent_cell_prob.dtype == np.float32
    _both(vb, SPOT_TOL)


def test_spot_ids_are_the_table_index_not_the_row_number(minimal_varbayes):
    # the lists hold spots.data.index values. Shift the index so a mix-up with the
    # row number would show
    vb = _fitted(minimal_varbayes)
    vb.spots.data.index = vb.spots.data.index + 1000
    now = _both(vb, SPOT_TOL)
    assert min(s for row in now[1:] for lst in row for s in lst) >= 1000
