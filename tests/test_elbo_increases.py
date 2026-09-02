"""
The ELBO is what the variational loop is supposed to be climbing, so it should
go up. If it goes down the objective and the update are not the same model.

Not a hard guarantee here: the mrf term is a Potts prior added without its
normalising constant, so the loop is not strictly coordinate ascent on one
function. What we can insist on is that it does not go DOWN most of the time,
which is what a mismatch looks like (dev_3d 4fe8a4f found the elbo falling on
more than half the steps because it scored a different mrf from the update).
"""
import numpy as np
import pytest

from pciSeq.src.core.utils.elbo import calc_elbo


def _iterate(vb, n):
    """Run n plain iterations and record the elbo after each."""
    vb.initialise_state()
    out = []
    for _ in range(n):
        vb.geneCount_upd()
        vb.rho_upd()
        vb.eta_upd()
        vb.theta_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()
        vb.spots_to_cell_numba()
        out.append(calc_elbo(vb))
    return np.array(out)


def test_elbo_is_a_finite_number(minimal_varbayes):
    e = _iterate(minimal_varbayes, 3)
    assert np.all(np.isfinite(e)), e


def test_elbo_never_drops_meaningfully(minimal_varbayes):
    """It should climb, then sit still.

    Counting the sign of each step is no good: once it has converged the steps
    are floating point noise and half of them come out negative. What matters is
    that no step is a real drop, i.e. large next to how far the elbo travelled
    in the first place.
    """
    e = _iterate(minimal_varbayes, 12)
    steps = np.diff(e)
    travelled = e.max() - e.min()
    worst_drop = -steps.min() if steps.min() < 0 else 0.0

    assert steps[0] > 0, 'elbo did not go up at all on the first step: %s' % e
    assert worst_drop < 0.01 * travelled, (
        'elbo dropped by %.3g, which is %.1f%% of the %.3g it travelled. '
        'trace: %s' % (worst_drop, 100 * worst_drop / travelled, travelled, e))


def test_elbo_settles(minimal_varbayes):
    """Once converged it should stop moving, not wander."""
    e = _iterate(minimal_varbayes, 12)
    late = np.abs(np.diff(e[-4:]))
    travelled = e.max() - e.min()
    assert late.max() < 1e-4 * travelled, (
        'elbo still moving by %.3g at the end of a converged run' % late.max())


def test_mrf_prior_uses_what_the_update_used(minimal_varbayes):
    """mrf_prior must read cells.mrf, not rebuild it from the new classProb."""
    from pciSeq.src.core.utils.elbo import mrf_prior
    vb = minimal_varbayes
    _iterate(vb, 1)
    stashed = vb.cells.mrf.copy()
    scored_with_stash = mrf_prior(vb)
    # rebuilding now uses the classProb cell_to_cellType already overwrote
    vb.cells.mrf = None
    scored_with_rebuild = mrf_prior(vb)
    vb.cells.mrf = stashed
    assert scored_with_stash != scored_with_rebuild, (
        'the two agree, so this test cannot tell them apart on this fixture')
    assert mrf_prior(vb) == scored_with_stash
