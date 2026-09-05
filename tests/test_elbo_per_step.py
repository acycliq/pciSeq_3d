"""
Checks the elbo_per_step flag, both ways.

Off is the default and must cost nothing, because the elbo is expensive and
every run pays for the check. On has to score the named steps and leave the
others alone.

The telescoping check at the bottom is the useful one: if you score every step,
the deltas have to add up to the change over the whole iteration, since each
step's "after" is the next step's "before". If that ever stops holding then the
instrumentation is measuring the wrong thing.
"""
import re

import pytest

STEPS = ['geneCount_upd', 'rho_upd', 'eta_upd', 'theta_upd', 'gamma_upd',
         'cell_to_cellType', 'spots_to_cell']


def _one_iteration(vb):
    vb._step_times = {}
    vb.iter_num = 0
    for name in STEPS:
        fn = vb.spots_to_cell_numba if name == 'spots_to_cell' else getattr(vb, name)
        vb._step(name, fn)


def _deltas(caplog):
    """Pull the step name and delta back out of the log lines."""
    out = {}
    for line in caplog.text.splitlines():
        m = re.search(r'elbo step (\S+)\s+([-+][\d.]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def test_off_by_default(minimal_varbayes):
    assert minimal_varbayes.config.get('elbo_per_step', []) == []


def test_off_scores_nothing(minimal_varbayes, caplog):
    vb = minimal_varbayes
    vb.config['elbo_per_step'] = []
    vb.initialise_state()
    _one_iteration(vb)
    with caplog.at_level('INFO'):
        _one_iteration(vb)
    assert _deltas(caplog) == {}, 'nothing should be scored when the list is empty'


def test_scores_only_the_named_step(minimal_varbayes, caplog):
    vb = minimal_varbayes
    vb.config['elbo_per_step'] = ['cell_to_cellType']
    vb.initialise_state()
    _one_iteration(vb)          # first pass, gets mvn_loglik_arr populated
    with caplog.at_level('INFO'):
        _one_iteration(vb)
    assert set(_deltas(caplog)) == {'cell_to_cellType'}


def test_all_scores_every_step(minimal_varbayes, caplog):
    vb = minimal_varbayes
    vb.config['elbo_per_step'] = 'all'
    vb.initialise_state()
    _one_iteration(vb)
    with caplog.at_level('INFO'):
        _one_iteration(vb)
    assert set(_deltas(caplog)) == set(STEPS)


def test_the_first_iteration_is_skipped(minimal_varbayes, caplog):
    """calc_elbo needs spots.mvn_loglik_arr, which is None until the first
    spots_to_cell, so there is nothing to score on the way in."""
    vb = minimal_varbayes
    vb.config['elbo_per_step'] = 'all'
    vb.initialise_state()
    assert vb.spots.mvn_loglik_arr is None
    with caplog.at_level('INFO'):
        _one_iteration(vb)
    assert _deltas(caplog) == {}, 'nothing is scoreable before the spatial term exists'


def test_the_deltas_telescope(minimal_varbayes, caplog):
    """Every step scored, so the deltas must sum to the whole-iteration change."""
    from pciSeq.src.core.utils.elbo import calc_elbo
    vb = minimal_varbayes
    vb.config['elbo_per_step'] = 'all'
    vb.initialise_state()
    _one_iteration(vb)

    before = calc_elbo(vb)
    with caplog.at_level('INFO'):
        _one_iteration(vb)
    after = calc_elbo(vb)

    scored = _deltas(caplog)
    assert set(scored) == set(STEPS)
    # the logged deltas are rounded to 2dp, so allow for that piling up
    assert sum(scored.values()) == pytest.approx(after - before, abs=0.05 * len(STEPS))
