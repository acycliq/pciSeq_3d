"""
Checks the diagnostics database comes out with everything in it.

The dashboard and the viewer read this file, so a missing key or column is a
hard failure for them. This builds a tiny run, exports it, and looks at what
actually landed in the sqlite file rather than trusting the code reads right.
"""
import json
import os
import sqlite3

import numpy as np
import pytest

from pciSeq.src.core.io import export_diagnostics

EXPECTED_METADATA = {
    'nC', 'nG', 'nK', 'rSpot', 'SpotReg', 'class_names', 'eta_bar',
    'mean_gene_reads_per_class', 'sc_mean_expression', 'log_prior',
    'nS', 'nN', 'misread_density', 'rho_bar', 'log_rho_bar',
    'hard_misread_counts', 'hard_misread_by_plane',
    'gene_panel', 'label_map', 'gene_total_spots', 'pciSeq_provenance',
}

EXPECTED_CELL_COLUMNS = {
    'internal_label', 'scaled_means', 'theta_bar', 'gene_count', 'class_prob',
    'theta', 'assigned_class_idx', 'gamma_assigned', 'mrf',
    'gene_loglik', 'runner_up_idx', 'contr_assigned', 'contr_runner_up',
}

EXPECTED_SPOT_COLUMNS = {
    'spot_id', 'gene_idx', 'x', 'y', 'z', 'neighbor_cell_ids', 'mvn_loglik',
    'attention', 'expr_fluct', 'cell_inefficiency', 'gene_inefficiency', 'bonus',
}


@pytest.fixture
def exported_db(minimal_varbayes, tmp_path):
    vb = minimal_varbayes
    vb.initialise_state()
    # one pass of the loop, in the order main_loop runs them, so everything the
    # export reads has actually been worked out
    vb.geneCount_upd()
    vb.rho_upd()
    vb.eta_upd()
    vb.theta_upd()
    vb.gamma_upd()
    vb.cell_to_cellType()
    vb.spots_to_cell_numba()
    export_diagnostics(vb, str(tmp_path))
    db = os.path.join(str(tmp_path), 'diagnostics', 'diagnostics.db')
    assert os.path.exists(db), 'no database was written'
    return sqlite3.connect(db)


def _columns(con, table):
    return {r[1] for r in con.execute('PRAGMA table_info(%s)' % table)}


def test_metadata_has_every_key(exported_db):
    got = {k for (k,) in exported_db.execute('select key from metadata')}
    assert EXPECTED_METADATA <= got, 'missing: %s' % (EXPECTED_METADATA - got)


def test_cells_table_columns(exported_db):
    got = _columns(exported_db, 'cells')
    assert EXPECTED_CELL_COLUMNS <= got, 'missing: %s' % (EXPECTED_CELL_COLUMNS - got)


def test_spots_table_columns(exported_db):
    got = _columns(exported_db, 'spots')
    assert EXPECTED_SPOT_COLUMNS <= got, 'missing: %s' % (EXPECTED_SPOT_COLUMNS - got)


def test_provenance_says_which_code_made_it(exported_db):
    (val,) = exported_db.execute(
        "select value from metadata where key='pciSeq_provenance'").fetchone()
    prov = json.loads(val)
    for k in ('version', 'branch', 'commit', 'commit_date', 'created_at'):
        assert k in prov, k


def test_the_new_blobs_are_not_empty(exported_db):
    (mrf,) = exported_db.execute('select mrf from cells limit 1').fetchone()
    assert mrf and len(mrf) > 0
    row = exported_db.execute('select gene_inefficiency, bonus from spots limit 1').fetchone()
    assert row is not None and all(b is not None for b in row)


def test_the_saved_class_score_gives_back_the_stored_probabilities(exported_db):
    """gene_loglik is the run's own gene term, from its last class update, so with
    the prior and the spatial term it has to give the stored class_prob back through
    a softmax, for every cell. That is the point of saving it: an explanation read
    from it is the run's, whatever pciSeq reads the file years later."""
    from scipy.special import softmax
    meta = dict(exported_db.execute('select key, value from metadata'))
    nG, nK = int(meta['nG']), int(meta['nK'])
    log_prior = np.asarray(json.loads(meta['log_prior']), dtype=np.float32)
    rows = exported_db.execute(
        'select class_prob, mrf, gene_loglik, assigned_class_idx, runner_up_idx, '
        'contr_assigned, contr_runner_up from cells').fetchall()
    assert rows
    f32 = lambda b: np.frombuffer(b, dtype=np.float32)
    for cp, mrf, gl, a, r, ca, cr in rows:
        cp, mrf, gl, ca, cr = f32(cp), f32(mrf), f32(gl), f32(ca), f32(cr)
        assert gl.shape == (nK,) and ca.shape == (nG,) and cr.shape == (nG,)
        np.testing.assert_allclose(softmax(gl + log_prior + mrf), cp, atol=1e-5)
        # the per gene terms add up to the saved totals
        assert ca.sum() == pytest.approx(gl[a], rel=1e-4, abs=1e-3)
        if r >= 0:
            assert r != a and cp[r] > 0 and cp[r] == np.sort(cp)[-2]
            assert cr.sum() == pytest.approx(gl[r], rel=1e-4, abs=1e-3)
        else:
            assert (np.delete(cp, a) == 0).all()
