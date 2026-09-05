"""
Checks the verbose flag actually works, both ways.

Off is the default and must cost nothing. On has to produce the step timings,
the ELBO and the convergence detail without falling over, since that path is
never exercised by a normal run.
"""
import logging

import pytest


def _one_iteration(vb):
    vb.initialise_state()
    vb._step_times = {}
    vb.iter_num = 0
    vb._step('geneCount_upd', vb.geneCount_upd)
    vb._step('rho_upd', vb.rho_upd)
    vb._step('eta_upd', vb.eta_upd)
    vb._step('theta_upd', vb.theta_upd)
    vb._step('gamma_upd', vb.gamma_upd)
    vb._step('cell_to_cellType', vb.cell_to_cellType)
    vb._step('spots_to_cell', vb.spots_to_cell_numba)


def test_off_by_default(minimal_varbayes):
    assert minimal_varbayes.config.get('verbose', False) is False


def test_off_records_no_timings(minimal_varbayes):
    vb = minimal_varbayes
    vb.config['verbose'] = False
    _one_iteration(vb)
    assert vb._step_times == {}, 'timing should cost nothing when verbose is off'


def test_on_records_a_timing_per_step(minimal_varbayes):
    vb = minimal_varbayes
    vb.config['verbose'] = True
    _one_iteration(vb)
    assert set(vb._step_times) == {
        'geneCount_upd', 'rho_upd', 'eta_upd', 'theta_upd', 'gamma_upd',
        'cell_to_cellType', 'spots_to_cell'}
    assert all(t >= 0 for t in vb._step_times.values())


def test_the_elbo_actually_runs(minimal_varbayes):
    """It is only ever called under verbose, so nothing else would catch a break."""
    from pciSeq.src.core.utils.elbo import calc_elbo
    vb = minimal_varbayes
    vb.config['verbose'] = True
    _one_iteration(vb)
    elbo = calc_elbo(vb)
    assert elbo == elbo, 'elbo came back NaN'


def test_convergence_detail_is_logged_only_when_verbose(minimal_varbayes, caplog):
    from pciSeq.src.core.utils import convergence
    vb = minimal_varbayes
    _one_iteration(vb)
    p0 = vb.spots.parent_cell_prob.copy()

    with caplog.at_level(logging.INFO):
        convergence.has_converged(vb.spots, p0, 0.02, False)
    assert 'convergence detail' not in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO):
        convergence.has_converged(vb.spots, p0, 0.02, True)
    assert 'convergence detail' in caplog.text
