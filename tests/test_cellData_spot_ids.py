"""
cellData.spot_id lists the spots that make up each gene's count, not the neighbourhood.

It used to list, for every gene of a cell, every spot that had the cell among its
candidates, with no look at the probability. Nine out of ten of those had a probability
of zero for that cell: on silver180 a cell holding 31 counts listed 1867 spots, and
cellData.tsv was 205 MB, most of it these lists.

Now a spot is listed only if its probability for the cell is above tol / 10. Why a tenth
and not tol itself: a gene is shown when its count is above tol, and a small count can be
made of several spots that are each below tol. With tol for both, 3868 of 420120 shown
genes on silver180 would have had a count and an empty list. With tol / 10 there were
none, and the spots that are kept still add up to 99.995% of the counts.
"""
import numpy as np

from pciSeq.src.core import summary

TOL = 0.001
SPOT_TOL = TOL / 10


def _fitted(vb):
    vb.initialise_state()
    for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd, vb.gamma_upd,
                 vb.cell_to_cellType, vb.spots_to_cell_numba):
        step()
    vb.geneCount_upd()      # so the counts are the ones that go with the final assignments
    return vb


def _prob_of(vb, spot, cell):
    """The probability that this spot belongs to this cell."""
    row = int(np.flatnonzero(vb.spots.data.index.values == spot)[0])
    return float(vb.spots.parent_cell_prob[row][vb.spots.parent_cell_id[row] == cell].sum())


def test_every_listed_spot_is_above_the_cut_off(minimal_varbayes):
    vb = _fitted(minimal_varbayes)
    cells, _ = summary.collect_data(vb.cells, vb.spots, vb.genes, vb.config['is3D'])
    listed = [(r.Cell_Num, s) for _, r in cells.iterrows() for lst in r.spot_id for s in lst]
    assert listed, 'nothing was listed at all, the test model has no assigned spots'
    assert all(_prob_of(vb, s, c) > SPOT_TOL for c, s in listed)


def test_the_listed_spots_add_up_to_the_count(minimal_varbayes):
    vb = _fitted(minimal_varbayes)
    cells, _ = summary.collect_data(vb.cells, vb.spots, vb.genes, vb.config['is3D'])
    for _, r in cells.iterrows():
        for count, lst in zip(r.CellGeneCount, r.spot_id):
            total = sum(_prob_of(vb, s, r.Cell_Num) for s in lst)
            # the count is cut to 3 decimals on the way out, and what is left off the
            # list is at most nNeighbors + 1 spots of SPOT_TOL each
            assert abs(total - count) < 0.001 + SPOT_TOL * vb.spots.parent_cell_id.shape[1]


def test_no_cut_off_lists_every_candidate_like_it_used_to(minimal_varbayes):
    vb = _fitted(minimal_varbayes)
    everything = summary.get_contributing_spots_2(vb.cells, vb.spots, vb.genes)
    trimmed = summary.get_contributing_spots_2(vb.cells, vb.spots, vb.genes, SPOT_TOL)
    n_all = sum(len(lst) for row in everything for lst in row)
    n_cut = sum(len(lst) for row in trimmed for lst in row)
    assert n_all == vb.spots.parent_cell_id.size
    assert n_cut < n_all


def test_a_shown_gene_with_no_spot_above_the_cut_off_does_not_crash(minimal_varbayes, monkeypatch):
    # it did not come up once on silver180 but it can: then that gene's list is empty.
    # The entry comes back from aggregate as a plain python list, which has no tolist,
    # and the old conversion would have fallen over on it. A cut-off of 0.5 forces the
    # case here: plenty of shown genes have no single spot that is more than half theirs.
    vb = _fitted(minimal_varbayes)
    real = summary.get_contributing_spots_2
    monkeypatch.setattr(summary, 'get_contributing_spots_2',
                        lambda cells, spots, genes, tol=0.0: real(cells, spots, genes, 0.5))
    cells, _ = summary.collect_data(vb.cells, vb.spots, vb.genes, vb.config['is3D'])

    lists = [lst for row in cells.spot_id for lst in row]
    assert any(lst == [] for lst in lists), 'the test did not produce an empty list'
    assert any(lst for lst in lists), 'the test emptied every list, that is not the case being checked'
    # every gene that is shown still has a list lined up with it, empty or not
    assert (cells.spot_id.map(len) == cells.Genenames.map(len)).all()
