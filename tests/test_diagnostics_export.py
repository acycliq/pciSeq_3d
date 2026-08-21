import os
import sqlite3
import numpy as np
import pytest
from scipy.special import softmax
from pciSeq.src.core.utils.io_utils import export_diagnostics
from pciSeq.src.core.utils.tie_freeze import TIE_RETURNS, TIE_START


def _pin(vb, c):
    """Pin cell c on its real current row, by making the detector think it has
    been bouncing and then running an ordinary cell type update."""
    freezer = vb.tie_freezer
    current = vb.cells.classProb[c].argmax()
    freezer.labels[-1, c] = (current + 1) % vb.nK
    freezer.labels[0, c] = current
    freezer.returns[c] = TIE_RETURNS - 1
    vb.iter_num = TIE_START + 10
    vb.cell_to_cellType()
    assert freezer.frozen[c], "the cell should have been pinned"

def test_export_diagnostics_includes_mrf(minimal_varbayes, tmp_path):
    """Test that diagnostics export includes the mrf column in cells table."""
    vb = minimal_varbayes
    vb.initialise_state()
    
    # Run one iteration to populate mrf
    vb.geneCount_upd()
    vb.gamma_upd()
    vb.cell_to_cellType()
    
    # Setup output directory
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Export diagnostics
    export_diagnostics(vb, str(output_dir))
    
    # Verify database exists
    db_path = output_dir / "diagnostics" / "diagnostics.db"
    assert db_path.exists()
    
    # Connect and check schema
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if mrf column exists in cells table
    cursor.execute("PRAGMA table_info(cells)")
    columns = [row[1] for row in cursor.fetchall()]
    assert "mrf" in columns, "mrf column missing from cells table"
    
    # Check if data is populated
    cursor.execute("SELECT mrf FROM cells LIMIT 1")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] is not None
    
    # Verify the blob can be converted back to numpy and has correct size
    mrf_blob = row[0]
    mrf_array = np.frombuffer(mrf_blob, dtype=np.float32)
    assert mrf_array.shape == (vb.nK,), f"Expected shape ({vb.nK},), got {mrf_array.shape}"
    
    # Verify values match stashed mrf
    stashed_mrf = vb.cells.mrf[0].astype(np.float32)
    assert np.allclose(mrf_array, stashed_mrf)

    conn.close()


def test_pinned_cell_rebuilds_from_the_exported_row(minimal_varbayes, tmp_path):
    """A pinned cell has to be explainable from what the database holds.

    The viewer does not read class_prob to draw its component chart, it
    rebuilds the posterior out of scaled_means, eta_bar, theta_bar, gene_count,
    log_prior and mrf. For a pinned cell the live values would not add up to
    the class being shown, so the export writes the ones the cell was pinned
    on. This does the viewer's arithmetic by hand on the exported row and
    checks it lands back on the stored class_prob.
    """
    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd()
    vb.gamma_upd()
    vb.cell_to_cellType()

    c = 3
    _pin(vb, c)
    pinned = vb.cells.classProb[c].copy()

    # the rest of the run: everything the pinned cell depended on moves on
    vb.genes._eta_bar = vb.genes.eta_bar * 1.4
    vb.cellTypes._initial_weights = vb.cellTypes._initial_weights * np.linspace(0.5, 1.5, vb.nK)
    vb.cells.theta_bar[c] = vb.cells.theta_bar[c] * 0.7
    vb.cells.geneCount[c] = vb.cells.geneCount[c] + 3.0
    vb.cells.mrf[c] = vb.cells.mrf[c] + 2.0

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    export_diagnostics(vb, str(output_dir))

    conn = sqlite3.connect(str(output_dir / "diagnostics" / "diagnostics.db"))
    cursor = conn.cursor()
    cursor.execute('SELECT scaled_means, theta_bar, gene_count, class_prob, mrf, '
                   'frozen_eta_bar, frozen_log_prior, frozen_iter FROM cells WHERE cell_id = ?', (c,))
    row = cursor.fetchone()

    f32 = lambda b: np.frombuffer(b, dtype=np.float32)
    scaled_means = f32(row[0]).reshape(vb.nG, vb.nK)
    theta_bar = f32(row[1])
    gene_count = f32(row[2])
    class_prob = f32(row[3])
    mrf = f32(row[4])

    assert row[5] is not None, "a pinned cell needs its own eta_bar in the row"
    assert row[6] is not None, "a pinned cell needs its own log_prior in the row"
    assert row[7] == TIE_START + 10, "the iteration it was pinned on should be recorded"
    eta_bar = f32(row[5])
    log_prior = f32(row[6])

    # what the viewer does: NB log-likelihood per gene and class, summed over
    # genes, plus the prior and the mrf, through a softmax
    rSpot = vb.config['rSpot']
    scaled_exp = scaled_means * eta_bar[:, None] * theta_bar[None, :] + vb.config['SpotReg']
    q = scaled_exp / (rSpot + scaled_exp)
    contr = gene_count[:, None] * np.log(q) + rSpot * np.log(1 - q)
    rebuilt = softmax(contr.sum(axis=0) + log_prior + mrf)

    assert np.allclose(class_prob, pinned, atol=1e-5), \
        "the exported class_prob should be the pinned row"
    assert np.allclose(rebuilt, pinned, atol=1e-4), \
        "the exported row must rebuild the class the viewer is showing"

    # an ordinary cell keeps the live values and leaves the frozen ones empty
    other = 1 if c != 1 else 2
    cursor.execute('SELECT frozen_eta_bar, frozen_log_prior, frozen_iter '
                   'FROM cells WHERE cell_id = ?', (other,))
    assert cursor.fetchone() == (None, None, None), \
        "cells that were never pinned should not carry a copy"

    conn.close()
