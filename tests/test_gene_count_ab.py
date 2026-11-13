"""
Checks that geneCount_upd_numba gives the very same arrays as geneCount_upd.

Same idea as the other A/B tests: geneCount_upd is the readable reference, the
numba one is what the loop runs. Here the two must agree exactly, not just to a
tolerance. Both add the same numbers in the same order and in the same float
type, so there is no rounding to allow for.

There is a third check too. geneCount_upd was written with numpy_groupies before
it moved to a coo matrix. That older code is kept below, in _gene_count_npg, and
it has to give the very same arrays as well.
"""
import pickle

import numpy as np
import numpy_groupies as npg


def _gene_count_npg(vb):
    """NOTE: this is the code of geneCount_upd as it was before the coo_matrix
    rewrite (commit 79105b64). It is not used anywhere in pciSeq anymore, it lives
    here only as one more thing to check the other two against. It builds a
    (cell, gene) index for every spot and neighbour and lets npg.aggregate sum
    the probs that share one.

    Returns the counts and the background, it does not write to vb.
    """
    # make an array nS-by-nN and fill it with the spots id
    gene_ids = np.tile(vb.spots.gene_id, (vb.nN, 1)).T

    # flatten it
    gene_ids = gene_ids.ravel()

    # make corresponding arrays for cell_id and probs
    cell_ids = vb.spots.parent_cell_id.ravel()
    probs = vb.spots.parent_cell_prob.ravel()

    # make the array to be used as index in the group-by operation
    group_idx = np.vstack((cell_ids, gene_ids))

    # For each cell aggregate the number of spots from the same gene.
    # It will produce an array of size nC-by-nG where the entry at (c,g)
    # is the gene counts of gene g within cell c
    N_cg = npg.aggregate(group_idx, probs, size=(vb.nC, vb.nG))

    out = np.zeros([vb.nC, vb.nG], dtype=np.float32)
    out[1:, :] = N_cg[1:, :]
    return out, N_cg[0, :]


def _both(vb):
    vb.geneCount_upd()
    counts_ref = vb.cells.geneCount.copy()
    background_ref = vb.cells.background_counts.copy()

    # third check: the code from before the coo_matrix rewrite
    counts_npg, background_npg = _gene_count_npg(vb)
    assert counts_npg.dtype == counts_ref.dtype
    assert background_npg.dtype == background_ref.dtype
    assert np.array_equal(counts_npg, counts_ref)
    assert np.array_equal(background_npg, background_ref)

    vb.geneCount_upd_numba()
    counts = vb.cells.geneCount
    background = vb.cells.background_counts

    assert counts.dtype == counts_ref.dtype == np.float32
    assert background.dtype == background_ref.dtype
    assert np.array_equal(counts, counts_ref)
    assert np.array_equal(background, background_ref)


def test_first_call_float32_probs(minimal_varbayes):
    # straight after initialise_state the probs are the flat 1/nN start, in float32
    vb = minimal_varbayes
    vb.initialise_state()
    assert vb.spots.parent_cell_prob.dtype == np.float32
    _both(vb)


def test_after_a_spot_assignment(minimal_varbayes):
    # one pass of the loop, so the probs are the real float64 ones from the softmax
    vb = minimal_varbayes
    vb.initialise_state()
    for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd, vb.gamma_upd,
                 vb.cell_to_cellType, vb.spots_to_cell):
        step()
    assert vb.spots.parent_cell_prob.dtype == np.float64
    _both(vb)


def test_random_probs_and_background_in_any_column(minimal_varbayes):
    # nothing in the kernel should lean on the background sitting in the last
    # column only. Shuffle the candidate cells of every spot and try again
    vb = minimal_varbayes
    vb.initialise_state()
    rng = np.random.default_rng(0)
    parent = vb.spots.parent_cell_id.copy()
    for row in parent:
        rng.shuffle(row)
    vb.spots.parent_cell_id = parent
    w = rng.random(parent.shape)
    vb.spots.parent_cell_prob = w / w.sum(axis=1, keepdims=True)
    _both(vb)


def test_index_follows_the_neighbours(minimal_varbayes):
    # the index is made from parent_cell_id. Give the spots new neighbours and it
    # has to be made again, otherwise the counts go to the wrong cells
    vb = minimal_varbayes
    vb.initialise_state()
    first = vb.spots.by_cell_index(vb.nC)
    assert vb.spots.by_cell_index(vb.nC) is first        # built once, then handed back

    vb.spots.parent_cell_id = vb.spots.parent_cell_id[:, ::-1].copy()
    vb.spots.parent_cell_prob = np.ascontiguousarray(vb.spots.parent_cell_prob[:, ::-1])
    assert vb.spots.by_cell_index(vb.nC) is not first
    _both(vb)


def test_index_stays_out_of_the_pickle(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd_numba()
    spots = pickle.loads(pickle.dumps(vb.spots))
    assert spots._by_cell_index is None
    # and it comes back when asked for
    indptr, order, gene_sorted = spots.by_cell_index(vb.nC)
    assert indptr[-1] == order.shape[0] == gene_sorted.shape[0] == spots.parent_cell_id.size
