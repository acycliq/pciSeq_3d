"""The live tools: asking about a run while it is still running.

Uses the 16 cell fixture from test_label_identifiers, labels 101..116, so the
label translation is exercised too. The tools are driven between iterations, the
way the live viewer's callback will drive them.
"""
import numpy as np
import pytest

from pciSeq.src.mcp.live import Live


@pytest.fixture
def fitted_halfway(minimal_varbayes):
    """A model stopped between iterations, with a couple of updates behind it."""
    vb = minimal_varbayes
    vb.initialise_state()
    live = Live(vb)
    # two iterations in the order main_loop runs them, remembering between them as
    # the live viewer's callback will
    for i in range(2):
        vb.iter_num = i
        vb.geneCount_upd()
        vb.rho_upd()
        vb.eta_upd()
        vb.theta_upd()
        vb.gamma_upd()
        vb.cell_to_cellType()
        vb.spots_to_cell_numba()
        vb.iter_delta.append(0.5 / (i + 1))
        live.remember()
    return vb, live


def test_progress_reports_where_the_loop_is(fitted_halfway):
    vb, live = fitted_halfway
    p = live.progress()
    assert p['iteration'] == vb.iter_num == 1
    assert p['max_iter'] == vb.config['max_iter'] and p['converged'] is False
    assert p['delta'] == pytest.approx(0.25)
    assert p['delta_history'] == [pytest.approx(0.5), pytest.approx(0.25)]
    assert 'tolerance' in p and 'delta_is' in p


def test_cell_reports_the_current_state_in_segmentation_labels(fitted_halfway):
    vb, live = fitted_halfway
    labels = [live.to_external(r) for r in range(1, len(vb.cells.classProb))]
    for lab in labels:
        c = live.cell(lab)
        assert c['cell'] == lab
        assert c['iteration'] == 1
        probs = [x['prob'] for x in c['classes']]
        assert probs == sorted(probs, reverse=True)
        assert set(x['class'] for x in c['classes']) <= set(vb.cells.class_names)
        assert c['total_counts'] >= 0
        assert 'soft' in c['counts_are']
        # the fixture remembered twice, so the movement fields are there
        assert 'changed_since_last_iteration' in c


def test_the_background_row_is_never_a_cell(fitted_halfway):
    _, live = fitted_halfway
    with pytest.raises(ValueError, match='background'):
        live.cell(0)
    with pytest.raises(KeyError):
        live.cell(9999)


def test_class_counts_add_up_to_the_cells(fitted_halfway):
    vb, live = fitted_halfway
    n = len(vb.cells.classProb) - 1          # row 0 is the background
    cc = live.class_counts()
    assert cc['n_cells'] == n
    assert sum(r['cells'] for r in cc['classes']) == n
    assert sum(r['soft'] for r in cc['classes']) == pytest.approx(n, rel=1e-4)
    assert cc['classes'][0]['class'] == 'Zero'          # Zero first, then by size


def test_changed_cells_compares_with_the_last_snapshot(fitted_halfway):
    vb, live = fitted_halfway
    before = np.argmax(vb.cells.classProb, axis=1).copy()
    # move one cell by hand, then ask what changed
    row = 3
    other = (int(before[row]) + 1) % vb.cells.classProb.shape[1]
    vb.cells.classProb[row] = 0
    vb.cells.classProb[row, other] = 1.0
    out = live.changed_cells()
    assert out['n_changed'] == 1
    moved = out['cells'][0]
    assert moved['cell'] == live.to_external(row)
    assert moved['to'] == vb.cells.class_names[other]
    assert out['compared_with_iteration'] == 1


def test_nothing_remembered_yet_says_so(fitted_halfway):
    vb, _ = fitted_halfway
    fresh = Live(vb)
    with pytest.raises(RuntimeError, match='nothing has been remembered'):
        fresh.changed_cells()


def test_scale_factors_report_eta_and_theta(fitted_halfway):
    _, live = fitted_halfway
    s = live.scale_factors()
    assert s['eta']['mean'] > 0
    assert len(s['eta']['highest']) == 5
    assert s['theta']['min'] <= s['theta']['mean'] <= s['theta']['max']


def test_before_the_first_class_update_there_is_nothing_to_report(minimal_varbayes):
    """A run asked about too early must say so, not fall over."""
    vb = minimal_varbayes
    live = Live(vb)
    p = live.progress()
    assert p['iteration'] is None and p['delta'] is None
    vb.cells.classProb = None
    with pytest.raises(RuntimeError, match='first class update'):
        live.cell(1)
