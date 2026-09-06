"""Tests for the zero_boost protection in cell_to_cellType.

The point of zero_boost is that a cell with no reads of its own cannot be
pulled off the Zero class by whatever its neighbours happen to be. These check
the boost is the right size, that it fades with reads, and that it does nothing
at all when the flag is off.
"""
import numpy as np
import pytest


@pytest.fixture
def vb(minimal_varbayes):
    """One pass of the loop, in the order main_loop runs it, so geneCount and
    everything cell_to_cellType reads is actually worked out."""
    v = minimal_varbayes
    v.initialise_state()
    v.geneCount_upd()
    v.rho_upd()
    v.eta_upd()
    v.theta_upd()
    v.gamma_upd()
    return v


def _mrf_after_call(vb):
    """Run one cell_to_cellType and hand back the mrf it actually used."""
    vb.cell_to_cellType()
    return vb.cells.mrf


def test_off_by_default(vb):
    """With the flag off the mrf is the plain neighbour term, untouched."""
    vb.config['zero_boost'] = False
    expected = vb.cells.calc_mrf()
    assert np.allclose(_mrf_after_call(vb), expected)


def test_missing_key_does_not_raise(vb):
    """An old opts dict with no zero_boost key at all still runs."""
    vb.config.pop('zero_boost', None)
    vb.cell_to_cellType()


def test_empty_cell_gets_the_full_boost(vb):
    """A cell holding no reads gets beta * nNeighbors on the Zero column."""
    vb.config['zero_boost'] = True
    vb.config['zero_boost_r0'] = 2.0
    vb.cells.geneCount[:] = 0.0

    mrf = _mrf_after_call(vb)
    full = vb.config['mrf_beta'] * vb.config['nNeighbors']
    assert np.allclose(mrf[:, -1], full)


def test_boost_fades_as_reads_arrive(vb):
    """More reads means less protection, strictly decreasing."""
    vb.config['zero_boost'] = True
    vb.config['zero_boost_r0'] = 2.0

    vb.cells.geneCount[:] = 0.0
    # cell i carries i reads, so the boost should fall off down the column
    for i in range(vb.cells.geneCount.shape[0]):
        vb.cells.geneCount[i, 0] = float(i)

    zero_col = _mrf_after_call(vb)[:, -1]
    assert np.all(np.diff(zero_col) < 0)


def test_boost_matches_the_formula(vb):
    """The Zero column is exactly beta * nNeighbors * exp(-reads / r0)."""
    vb.config['zero_boost'] = True
    vb.config['zero_boost_r0'] = 3.0

    reads = vb.cells.total_counts
    expected = (vb.config['mrf_beta'] * vb.config['nNeighbors']
                * np.exp(-reads / 3.0))
    assert np.allclose(_mrf_after_call(vb)[:, -1], expected)


def test_real_classes_are_untouched(vb):
    """Only the Zero column moves. The mrf can still clean up real classes."""
    vb.config['zero_boost'] = True
    baseline = vb.cells.calc_mrf()
    mrf = _mrf_after_call(vb)
    assert np.allclose(mrf[:, :-1], baseline[:, :-1])


def test_r0_controls_the_reach(vb):
    """A larger r0 keeps protecting cells further up the read range."""
    vb.config['zero_boost'] = True
    vb.cells.geneCount[:] = 0.0
    vb.cells.geneCount[:, 0] = 5.0

    vb.config['zero_boost_r0'] = 1.0
    small = _mrf_after_call(vb)[:, -1].copy()
    vb.config['zero_boost_r0'] = 10.0
    large = _mrf_after_call(vb)[:, -1].copy()
    assert np.all(large > small)


def test_boost_can_hold_a_cell_on_zero(vb):
    """The whole point: neighbours all voting one real class cannot take an
    empty cell off Zero once the boost is on."""
    vb.cells.geneCount[:] = 0.0

    vb.config['zero_boost'] = False
    vb.cell_to_cellType()
    without = vb.cells.classProb[:, -1].copy()

    vb.config['zero_boost'] = True
    vb.cell_to_cellType()
    with_boost = vb.cells.classProb[:, -1].copy()

    assert np.all(with_boost >= without - 1e-12)
    assert np.any(with_boost > without)
