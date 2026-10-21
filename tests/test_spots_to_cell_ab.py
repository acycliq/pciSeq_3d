"""
Checks that spots_to_cell_numba gives the same result as spots_to_cell.

They are two implementations of the same step: spots_to_cell is the plain numpy
loop (the reference), spots_to_cell_numba is the fast numba version used by default.
This test keeps them in sync, so a change to one that diverges from the other is
caught here. Both only write parent_cell_prob and the diagnostic arrays (not their
inputs), so we can run one after the other on the same state and compare.
"""
import numpy as np


def _run_both(vb):
    """Run the numpy loop then the numba one on the same state, return the arrays."""
    keys = ['parent_cell_prob', 'attention', 'expr_fluctuations', 'cell_inefficiency',
            'gene_inefficiency', 'mvn_loglik_arr']
    vb.spots_to_cell()
    loop = {k: getattr(vb.spots, k).copy() for k in keys}
    vb.spots_to_cell_numba()
    numba = {k: getattr(vb.spots, k).copy() for k in keys}
    return loop, numba


def _assert_same(loop, numba):
    for k in loop:
        np.testing.assert_allclose(numba[k], loop[k], rtol=1e-12, atol=1e-12, err_msg=k)


def test_spots_to_cell_numba_matches_loop(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    _assert_same(*_run_both(vb))


def test_spots_to_cell_numba_matches_loop_with_zero_mass(minimal_varbayes):
    """Same check, but on a state where the Zero class actually matters.

    Right after initialise_state every cell has the prior as its class probs, theta
    is a constant and gamma is its prior, so the Zero column barely moves the
    score. Here the class probs, theta and gamma are given random values, with a
    good chunk of probability on Zero, so the special treatment of Zero in the
    spot step (no gamma, theta or eta term for it) is exercised in both versions.
    """
    vb = minimal_varbayes
    vb.initialise_state()
    rng = np.random.default_rng(0)
    nC, nK = vb.cells.classProb.shape

    # float64 on purpose: the numba kernel adds up in float64 while einsum on
    # float32 arrays adds up in float32, and with random inputs that is a 1e-7
    # gap that has nothing to do with the logic being compared
    probs = rng.dirichlet(np.ones(nK), size=nC)
    probs[:, -1] = rng.uniform(0.2, 0.9, size=nC)     # plenty of Zero mass
    probs[:, :-1] *= (1 - probs[:, -1])[:, None] / probs[:, :-1].sum(axis=1, keepdims=True)
    vb.cells.classProb = probs.astype(np.float64)
    vb.cells._theta_bar = rng.uniform(0.3, 3.0, size=(nC, nK)).astype(np.float64)
    vb.cells.geneCount = rng.poisson(2.0, size=(vb.nC, vb.nG)).astype(np.float32)
    vb.gamma_upd()
    # the spot step reads logeta_bar, gamma_upd above read eta_bar; set both
    vb.genes._eta_bar = rng.uniform(0.5, 2.0, size=vb.nG).astype(np.float64)
    vb.genes._logeta_bar = np.log(vb.genes._eta_bar)

    loop, numba = _run_both(vb)
    _assert_same(loop, numba)

    # and the Zero treatment is really in there: the eta term is scaled down by
    # the Zero mass of the cell, so it is not the same on every slot of a spot
    assert np.ptp(loop['gene_inefficiency'][:, :-1], axis=1).max() > 0
