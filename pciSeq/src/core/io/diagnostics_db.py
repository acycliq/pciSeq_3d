"""The sqlite database the viewer reads for its per cell and per spot panels."""

import os
import json
import sqlite3
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)



def export_diagnostics(varBayes: Any, output_dir: str) -> None:
    """Export diagnostics data (check_cell and check_spot) to a single SQLite database.

    Writes to {output_dir}/diagnostics/diagnostics.db

    Tables:
      - metadata: key-value pairs (including JSON arrays)
      - cells: per-cell diagnostic data
      - spots: per-spot diagnostic data

    Args:
        varBayes: Fitted VarBayes object
        output_dir: Base data directory
    """
    diagnostics_dir = os.path.join(output_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)

    db_path = os.path.join(diagnostics_dir, 'diagnostics.db')
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Create Tables ---
    cursor.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)')

    cursor.execute('''
        CREATE TABLE cells (
            cell_id INTEGER PRIMARY KEY,
            scaled_means BLOB,
            theta_bar BLOB,
            gene_count BLOB,
            class_prob BLOB,
            theta REAL,
            assigned_class_idx INTEGER,
            gamma_assigned BLOB,
            mrf BLOB
        )
    ''')

    cursor.execute('''
        CREATE TABLE spots (
            spot_id INTEGER PRIMARY KEY,
            gene_idx INTEGER,
            x INTEGER,
            y INTEGER,
            z INTEGER,
            neighbor_cell_ids TEXT,
            mvn_loglik BLOB,
            attention BLOB,
            expr_fluct BLOB,
            cell_inefficiency BLOB,
            gene_inefficiency BLOB,
            bonus BLOB
        )
    ''')

    # --- Gather Data ---
    cells = varBayes.cells
    genes = varBayes.genes
    spots = varBayes.spots

    # Label Map
    label_map = {}
    if varBayes.config.get('label_map'):
        label_map = {str(k): int(v) for k, v in varBayes.config['label_map'].items()}

    # Gene Panel
    gene_panel = genes.gene_panel.tolist()

    # Misread Density
    misread_series = genes.misread_density
    if hasattr(misread_series, 'to_dict'):
        misread_dict = {str(k): float(v) for k, v in misread_series.to_dict().items()}
    else:
        logger.error("Diagnostics export skipped: 'misread_density' is missing or invalid.")
        return

    # The misread density the model actually learned, as opposed to the prior above.
    rho_bar = genes.rho_bar
    rho_bar_dict = ({str(g): float(v) for g, v in zip(genes.gene_panel, rho_bar)}
                    if rho_bar is not None else {})

    # E[log rho] per gene. This is the term the background column in spots_to_cell
    # uses, not log(rho_bar), because E[log rho] is not log(E[rho]).
    log_rho_bar = genes.log_rho_bar
    log_rho_bar_dict = ({str(g): float(v) for g, v in zip(genes.gene_panel, log_rho_bar)}
                        if log_rho_bar is not None else {})

    # Spots whose best guess is the background column, counted per gene and per
    # gene per plane.
    hard_misread_counts = []
    hard_misread_by_plane = {}
    try:
        prob = spots.parent_cell_prob
        if prob is not None and len(prob):
            is_misread = np.argmax(prob, axis=1) == (prob.shape[1] - 1)
            hard_misread_counts = np.bincount(
                spots.gene_id, weights=is_misread.astype(np.float32),
                minlength=len(genes.gene_panel)).astype(int).tolist()
            if 'plane_id' in spots.data.columns:
                plane_ids = spots.data['plane_id'].to_numpy()
                for idx in np.where(is_misread)[0]:
                    gene = genes.gene_panel[spots.gene_id[idx]]
                    plane = int(plane_ids[idx])
                    hard_misread_by_plane.setdefault(gene, {})
                    hard_misread_by_plane[gene][plane] = hard_misread_by_plane[gene].get(plane, 0) + 1
    except Exception as e:
        logger.warning('Could not compute hard misread counts: %s', e)

    # --- Populate Metadata ---
    # Compute scaled_means for metadata nC (and for cells table)
    # logger.info('Computing scaled_exp for diagnostics export...')
    scaled_means = varBayes.scaled_exp
    nC, nG, nK = scaled_means.shape

    nS = spots.nS
    # Check if we can get nN (needs neighbor_ids)
    neighbor_ids = spots.parent_cell_id
    nN = 0
    if neighbor_ids is not None:
        nN = neighbor_ids.shape[1]

    meta_items = [
        # Cell-related
        ('nC', str(nC)),
        ('nG', str(nG)),
        ('nK', str(nK)),
        ('rSpot', str(float(varBayes.config['rSpot']))),
        ('SpotReg', str(float(varBayes.config['SpotReg']))),
        ('class_names', json.dumps(cells.class_names.tolist())),
        ('eta_bar', json.dumps(genes.eta_bar.astype(np.float32).tolist())),
        ('mean_gene_reads_per_class', json.dumps(cells.mean_gene_reads_per_class().astype(np.float32).tolist())),
        ('sc_mean_expression', json.dumps(varBayes.single_cell.mean_expression.values.astype(np.float32).tolist())),
        ('log_prior', json.dumps(varBayes.cellTypes.log_prior.astype(np.float32).tolist())),

        # Spot-related
        ('nS', str(int(nS))),
        ('nN', str(int(nN))),
        ('misread_density', json.dumps(misread_dict)),
        ('rho_bar', json.dumps(rho_bar_dict)),
        ('log_rho_bar', json.dumps(log_rho_bar_dict)),
        ('hard_misread_counts', json.dumps(hard_misread_counts)),
        ('hard_misread_by_plane', json.dumps(hard_misread_by_plane)),

        # Shared
        ('gene_panel', json.dumps(gene_panel)),
        ('label_map', json.dumps(label_map)),
    ]

    # Per-gene observed spot counts (for η scatter in dashboard)
    try:
        if hasattr(spots, 'counts_per_gene') and spots.counts_per_gene is not None:
            gene_total_spots = spots.counts_per_gene.astype(int).tolist()
        else:
            vc = spots.data['gene_name'].value_counts()
            gene_total_spots = [int(vc.get(g, 0)) for g in gene_panel]
        meta_items.append(('gene_total_spots', json.dumps(gene_total_spots)))
    except Exception:
        pass

    # which pciSeq made this. version, branch, commit, build_date, created_at,
    # serialised_at, hostname, os, python version, package versions.
    meta_items.append(('pciSeq_provenance', json.dumps(getattr(varBayes, 'metadata', {}))))

    cursor.executemany('INSERT INTO metadata VALUES (?, ?)', meta_items)
    # logger.info('Inserted %d metadata entries', len(meta_items))

    # --- Populate Cells Table ---
    scaled_means_f32 = scaled_means.astype(np.float32)
    theta_bar_f32 = cells.theta_bar.astype(np.float32)
    gene_count_f32 = cells.geneCount.astype(np.float32)
    class_prob_f32 = cells.classProb.astype(np.float32)

    # Dashboard columns: theta, assigned_class_idx, gamma_assigned
    # theta = sum_k zeta[c,k] * theta_bar[c,k]
    theta_scalar = np.einsum('ck,ck->c', class_prob_f32, theta_bar_f32).astype(np.float32)
    assigned_class_idx = np.argmax(class_prob_f32, axis=1).astype(np.int32)

    # gamma_assigned[c, :] = gamma_bar[c, :, assigned_class[c]]
    gamma_bar = varBayes.spots.gamma_bar.astype(np.float32)  # (nC, nG, nK)
    gamma_assigned = gamma_bar[np.arange(nC), :, assigned_class_idx]   # (nC, nG)

    # the mrf term the last class update used. None if the run never got that
    # far, write zeros then so the column is still the right shape.
    mrf_f32 = (cells.mrf.astype(np.float32) if cells.mrf is not None
               else np.zeros((nC, nK), dtype=np.float32))

    batch_size = 10000
    for batch_start in range(0, nC, batch_size):
        batch_end = min(batch_start + batch_size, nC)
        batch_data = []
        for c in range(batch_start, batch_end):
            batch_data.append((
                c,
                scaled_means_f32[c].tobytes(),
                theta_bar_f32[c].tobytes(),
                gene_count_f32[c].tobytes(),
                class_prob_f32[c].tobytes(),
                float(theta_scalar[c]),
                int(assigned_class_idx[c]),
                gamma_assigned[c].tobytes(),
                mrf_f32[c].tobytes(),
            ))
        cursor.executemany('INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', batch_data)
        # if (batch_end % 10000 == 0) or (batch_end == nC):
        #     logger.info('Inserted %d/%d cells', batch_end, nC)

    # --- Populate Spots Table ---
    if spots.mvn_loglik_arr is None or spots.attention is None or spots.expr_fluctuations is None or spots.cell_inefficiency is None or neighbor_ids is None:
        logger.warning('check_spot data missing; spots table will be empty.')
    else:
        mvn_f32 = spots.mvn_loglik_arr.astype(np.float32)
        attn_f32 = spots.attention.astype(np.float32)
        expr_f32 = spots.expr_fluctuations.astype(np.float32)
        cineff_f32 = spots.cell_inefficiency.astype(np.float32)
        gineff_f32 = spots.gene_inefficiency.astype(np.float32)
        # the inside cell bonus the model adds before the softmax in spots_to_cell
        bonus_f32 = (spots.bonus_mask * varBayes.config['InsideCellBonus']).astype(np.float32)
        gene_idx = spots.gene_id.astype(np.int32)
        xs = spots.data['x'].astype(np.int32).to_numpy()
        ys = spots.data['y'].astype(np.int32).to_numpy()
        zs = spots.data['z'].astype(np.int32).to_numpy()

        batch_size = 10000
        for start in range(0, nS, batch_size):
            end = min(start + batch_size, nS)
            batch = []
            for i in range(start, end):
                neigh_json = json.dumps(list(map(int, neighbor_ids[i].tolist())))
                batch.append((
                    int(spots.data.index[i]),
                    int(gene_idx[i]),
                    int(xs[i]), int(ys[i]), int(zs[i]),
                    neigh_json,
                    mvn_f32[i].tobytes(),
                    attn_f32[i].tobytes(),
                    expr_f32[i].tobytes(),
                    cineff_f32[i].tobytes(),
                    gineff_f32[i].tobytes(),
                    bonus_f32[i].tobytes(),
                ))
            cursor.executemany('''
                INSERT INTO spots (spot_id, gene_idx, x, y, z, neighbor_cell_ids, mvn_loglik, attention, expr_fluct, cell_inefficiency, gene_inefficiency, bonus)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', batch)
            # if (end % 50000 == 0) or (end == nS):
            #     logger.info('Inserted %d/%d spots', end, nS)

    conn.commit()
    conn.close()

    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    logger.info('Saved at: %s (%.1f MB)', db_path, db_size_mb)



def export_db_tables(out_dir: str, con: Any) -> None:
    """Export all database tables to CSV files.

    Args:
        out_dir: Output directory for CSV files
        con: Database connection object
    """
    tables = con.get_db_tables()
    for table in tables:
        export_db_table(table, out_dir, con)



def export_db_table(table_name: str, out_dir: str, con: Any) -> None:
    """Export single database table to CSV.

    Args:
        table_name: Name of table to export
        out_dir: Output directory
        con: Database connection object
    """
    df = con.from_redis(table_name)
    fname = os.path.join(out_dir, table_name + '.csv')
    df.to_csv(fname, index=False)
    logger.info('Saved at %s', fname)
