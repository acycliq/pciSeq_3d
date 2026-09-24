"""Answer questions about a finished run.

These are the tools an agent calls. They are plain functions on purpose: the logic
should not know whether it is being reached over MCP, from a viewer, or from a
notebook. The MCP server is a thin wrapper on top.

Everything comes out of diagnostics.db, which is self contained: its metadata table
carries the gene panel, the class names, eta, the class means, the prior and the
label map, so no pickle and none of the big tsv files are needed.

Three rules hold for every tool here:

1. cell labels in and out are the labels of your segmentation, never the internal
   ones. See `to_internal` in cell_utils for why the two differ.
2. counts are soft, weighted by the assignment probability, and every answer that
   reports one says so. Containment is hard and says that too.
3. row 0 of the model arrays is the background pseudocell and is never reported as
   a cell.

One thing to know about precision. The saved files keep 3 decimals, so a probability
under 0.0005 reads as 0.0 in them. cell_row and spot_row hand back exactly what the
file says, which is right for "what does the file report", but it means neither can
show you a probability finer than that. explain_spot recomputes from diagnostics.db
and gives the full float32, so use that when the small numbers matter. Note also that
cellData truncates and geneData rounds, summary.py:31 against summary.py:102, so the
last digit of the two files is not arrived at the same way.
"""
import json
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MISSING = object()

# counts below this are dropped from the per gene listings. Same threshold cellData
# uses, so the two agree about which genes a cell holds.
_COUNT_TOL = 0.001

# what a cell's blob columns unpack to, and the shape to give them
_CELL_BLOBS = {
    'scaled_means': ('nG', 'nK'),   # area_factor * mu, before eta and theta
    'theta_bar': ('nK',),
    'gene_count': ('nG',),
    'class_prob': ('nK',),
    'gamma_assigned': ('nG',),      # the winning class only, see the class docstring
    'mrf': ('nK',),
}


class Run:
    """A finished pciSeq run, opened for questions.

    Parameters
    ----------
    path : str or Path
        The run folder, or the diagnostics.db itself. Anything above the database
        works, the file is searched for underneath.

    Notes
    -----
    diagnostics.db keeps `gamma_assigned`, the gamma of the winning class only, not
    the whole (nC, nG, nK) array. Questions about gamma under a different class need
    the pickle.
    """

    def __init__(self, path):
        self.path = _find_db(path)
        # check_same_thread=False because an MCP server runs each tool call on a
        # worker thread, and the pool does not promise the same thread twice: open_run
        # would build this connection on one thread and cell_counts hit it from
        # another, which sqlite refuses by default. It only started showing up as a
        # one-in-four failure in the full test suite. sqlite3.threadsafety is 1 in
        # the usual build, meaning the module is safe but a shared connection is not,
        # so every query also goes through _query, which holds a lock.
        self._con = sqlite3.connect('file:%s?mode=ro' % self.path, uri=True,
                                    check_same_thread=False)
        self._lock = threading.Lock()
        self._meta_cache = {}

        self.nC = int(self._meta('nC'))
        self.nG = int(self._meta('nG'))
        self.nK = int(self._meta('nK'))
        self.nS = int(self._meta('nS'))
        self.gene_panel = np.array(self._meta('gene_panel', parse=True))
        self.class_names = np.array(self._meta('class_names', parse=True))

        # label_map comes back from JSON with string keys
        raw = self._meta('label_map', parse=True, default=None)
        self.label_map = {int(k): int(v) for k, v in raw.items()} if raw else None
        self._reverse_map = ({v: k for k, v in self.label_map.items()}
                             if self.label_map else None)

    # ------------------------------------------------------------ plumbing

    def _query(self, sql, params=()):
        """One row from the db, under the lock. See __init__ for why the lock."""
        with self._lock:
            return self._con.execute(sql, params).fetchone()

    def _meta(self, key, parse=False, default=_MISSING):
        """One row of the metadata table.

        `parse` decodes the value as JSON. `default` is returned when the key is
        absent, otherwise a missing key raises.
        """
        if key not in self._meta_cache:
            row = self._query(
                "SELECT value FROM metadata WHERE key = ?", (key,))
            self._meta_cache[key] = row[0] if row else None
        val = self._meta_cache[key]
        if val is None:
            if default is _MISSING:
                raise KeyError('diagnostics.db has no metadata key %r' % key)
            return default
        return json.loads(val) if parse else val

    def to_internal(self, label):
        """Segmentation label to the row it occupies in the arrays."""
        try:
            label = int(label)
        except (TypeError, ValueError):
            raise ValueError('%r is not a cell label. Spots assigned to the background '
                             'belong to no cell, so there is nothing to look up.' % (label,))
        if self.label_map is None:
            return label
        try:
            return self.label_map[label]
        except KeyError:
            raise KeyError('no cell %s in this segmentation' % label)

    def to_external(self, row):
        """The row of an array back to the segmentation label."""
        if self._reverse_map is None:
            return int(row)
        try:
            return self._reverse_map[int(row)]
        except KeyError:
            raise KeyError('no cell at row %s, this run has %d' % (row, self.nC))

    def _cell(self, label):
        """Every array of one cell, keyed by column name, taking a segmentation label."""
        row = self.to_internal(label)
        if row == 0:
            raise ValueError('cell 0 is the background pseudocell, not a cell')
        cols = list(_CELL_BLOBS) + ['theta', 'assigned_class_idx']
        got = self._query(
            "SELECT %s FROM cells WHERE cell_id = ?" % ', '.join(cols), (row,))
        if got is None:
            raise KeyError('cell %s is not in diagnostics.db' % label)

        out = {}
        sizes = {'nG': self.nG, 'nK': self.nK}
        for name, shape in _CELL_BLOBS.items():
            arr = np.frombuffer(got[cols.index(name)], dtype=np.float32)
            out[name] = arr.reshape([sizes[s] for s in shape])
        out['theta'] = float(got[cols.index('theta')])
        out['assigned_class_idx'] = int(got[cols.index('assigned_class_idx')])
        out['row'] = row
        return out

    # --------------------------------------------------------------- tools

    def summary(self):
        """What this run is: its size, the code that made it, and what it can answer."""
        prov = self._meta('pciSeq_provenance', parse=True, default={})
        return {
            'path': str(self.path),
            'cells': self.nC - 1,          # row 0 is the background
            'spots': self.nS,
            'genes': self.nG,
            'classes': self.nK,
            'pciSeq_version': prov.get('version'),
            'commit': prov.get('commit'),
            'labels_were_renumbered': self.label_map is not None,
            'has_containment': self._has_containment(),
        }

    def cell(self, label):
        """The headline facts about one cell."""
        c = self._cell(label)
        order = np.argsort(-c['class_prob'])
        counts = c['gene_count']
        top_genes = np.argsort(-counts)[:10]
        return {
            'cell': int(label),
            'total_counts': float(counts.sum()),
            'counts_are': 'soft, weighted by the spot assignment probabilities',
            'classes': [{'class': str(self.class_names[k]), 'prob': float(c['class_prob'][k])}
                        for k in order[:5]],
            'top_genes': [{'gene': str(self.gene_panel[g]), 'counts': float(counts[g])}
                          for g in top_genes if counts[g] > _COUNT_TOL],
            'theta': c['theta'],
            'theta_is': 'the soft scalar, averaged over the class probabilities, '
                        'not theta_bar of the assigned class',
        }

    def explain_cell(self, label, vs_class=None, top_n=10):
        """Why this cell got its class, gene by gene.

        The class score is the sum of the per gene negative binomial contributions,
        plus the class prior, plus the spatial term. This recomputes those from the
        stored arrays and compares the assigned class against another one.

        The recomputation uses the final eta, and the loop updates eta after the last
        class update, so it reproduces the stored probabilities exactly only once the
        run has converged. On an unconverged run the argmax still agrees but the
        probabilities can be off in the second decimal.

        Parameters
        ----------
        label : int
            Cell label, as in your segmentation.
        vs_class : str, optional
            The class to compare against. Defaults to the runner up.
        top_n : int, default 10
            How many genes to report on each side.
        """
        c = self._cell(label)
        contr = self._nb_contributions(c)          # (nG, nK)

        assigned = c['assigned_class_idx']
        if vs_class is None:
            order = np.argsort(-c['class_prob'])
            other = int(order[1] if order[0] == assigned else order[0])
        else:
            hit = np.where(self.class_names == vs_class)[0]
            if not len(hit):
                raise ValueError('no class %r in this run' % vs_class)
            other = int(hit[0])
            if other == assigned:
                raise ValueError('cell %s is already %s, pick another class to compare'
                                 % (label, vs_class))

        diff = contr[:, assigned] - contr[:, other]
        favours_assigned = np.argsort(-diff)[:top_n]
        favours_other = np.argsort(diff)[:top_n]
        log_prior = np.asarray(self._meta('log_prior', parse=True), dtype=np.float32)

        def side(idx, keep):
            return [{'gene': str(self.gene_panel[g]),
                     'counts': float(c['gene_count'][g]),
                     'diff': float(diff[g])}
                    for g in idx if keep(diff[g])]

        out = {
            'cell': int(label),
            'assigned': str(self.class_names[assigned]),
            'compared_with': str(self.class_names[other]),
            'prob_assigned': float(c['class_prob'][assigned]),
            'prob_compared': float(c['class_prob'][other]),
            'score': {
                'gene_loglik': {'assigned': float(contr[:, assigned].sum()),
                                'compared': float(contr[:, other].sum())},
                'log_prior': {'assigned': float(log_prior[assigned]),
                              'compared': float(log_prior[other])},
                'spatial': {'assigned': float(c['mrf'][assigned]),
                            'compared': float(c['mrf'][other])},
            },
            'genes_favouring_assigned': side(favours_assigned, lambda d: d > 0),
            'genes_favouring_compared': side(favours_other, lambda d: d < 0),
            'counts_are': 'soft, weighted by the spot assignment probabilities',
        }
        out['narrative'] = narrate_cell(out)
        return out

    def explain_spot(self, spot_id):
        """Why this spot went to the cell it did, term by term.

        One row per candidate cell plus the background. The terms are the ones the
        model used in its last spot update, so the probabilities here are the ones
        in geneData, at full float32 precision rather than the 3 decimals the file
        keeps. Use this rather than spot_row when a probability below 0.0005 matters.
        """
        # sqlite3 quietly matches nothing for a numpy integer, so a pandas index
        # value comes back as 'no such spot'. Plain int first.
        try:
            spot_id = int(spot_id)
        except (TypeError, ValueError):
            raise ValueError('%r is not a spot id' % (spot_id,))
        got = self._query(
            "SELECT gene_idx, x, y, z, neighbor_cell_ids, mvn_loglik, attention, "
            "expr_fluct, cell_inefficiency, gene_inefficiency, bonus "
            "FROM spots WHERE spot_id = ?", (spot_id,))
        if got is None:
            raise KeyError('no spot %s in this run' % spot_id)

        gene = str(self.gene_panel[got[0]])
        neighbours, terms, misread, score, prob = self._spot_scores(got)

        rows = []
        for i, internal in enumerate(neighbours[:-1]):
            cls_idx = self._query("SELECT assigned_class_idx FROM cells WHERE cell_id = ?",
                                  (internal,))
            rows.append({
                'cell': self.to_external(internal),
                'class': str(self.class_names[cls_idx[0]]) if cls_idx else None,
                'Gaussian fit': float(terms['mvn_loglik'][i]),
                'class expression': float(terms['attention'][i]),
                'cell scale': float(terms['cell_inefficiency'][i]),
                'cell-gene scale': float(terms['expr_fluct'][i]),
                'gene efficiency': float(terms['gene_inefficiency'][i]),
                'bonus': float(terms['bonus'][i]),
                'sum': float(score[i]),
                'prob': float(prob[i]),
            })
        rows.append({'cell': 'background', 'misread': misread,
                     'sum': misread, 'prob': float(prob[-1])})

        out = {
            'spot': int(spot_id),
            'gene': gene,
            'position': {'x': got[1], 'y': got[2], 'z': got[3],
                         'z_is': 'the anisotropy scaled z the model works in, '
                                 'not the plane index'},
            'candidates': rows,
            'assigned_to': rows[int(np.argmax(prob))]['cell'],
        }
        out['narrative'] = narrate_spot(out)
        return out

    def cell_counts(self, label, gene=None):
        """How many reads a cell holds. Soft, and the answer says so."""
        c = self._cell(label)
        counts = c['gene_count']
        note = ('these are soft counts: each spot contributes its probability of '
                'belonging to this cell, so they are estimates and not whole numbers')
        if gene is not None:
            hit = np.where(self.gene_panel == gene)[0]
            if not len(hit):
                raise ValueError('no gene %r in the panel' % gene)
            return {'cell': int(label), 'gene': gene,
                    'counts': float(counts[hit[0]]), 'counts_are': note}
        order = np.argsort(-counts)
        return {
            'cell': int(label),
            'total_counts': float(counts.sum()),
            'per_gene': [{'gene': str(self.gene_panel[g]), 'counts': float(counts[g])}
                         for g in order if counts[g] > _COUNT_TOL],
            'counts_are': note,
        }

    def spots_in_cell(self, label, gene=None):
        """How many spots physically sit inside a cell's segmentation mask.

        Hard containment, no probabilities. A spot can sit outside every cell and
        still be assigned to one, so this is a different number from `cell_counts`.
        """
        if not self._has_containment():
            raise NotImplementedError(
                'this run has no containment data. geneData gained the inside_cell '
                'column after it was produced, so the run has to be repeated to '
                'answer this. cell_counts works on any run.')
        label = int(label)
        self.to_internal(label)                  # just to raise if it is not a cell

        import pyarrow.compute as pc

        per_gene = {}
        for tb, gene_of in self._arrow_spots(['inside_cell', 'gene_id']):
            hits = tb.filter(pc.equal(tb['inside_cell'], label))
            for gid, n in zip(*np.unique(hits['gene_id'].to_numpy(), return_counts=True)):
                per_gene[gene_of[int(gid)]] = per_gene.get(gene_of[int(gid)], 0) + int(n)

        note = ('hard count: spots whose pixel falls inside this cell\'s segmentation '
                'mask, no probabilities involved. The model may have assigned some of '
                'them elsewhere, and cell_counts may include spots from outside the mask')
        if gene is not None:
            if gene not in set(gene_of.values()):
                raise ValueError('no gene %r in the panel' % gene)
            return {'cell': label, 'gene': gene, 'spots': per_gene.get(gene, 0),
                    'spots_are': note}
        return {
            'cell': label,
            'total_spots': sum(per_gene.values()),
            'per_gene': [{'gene': g, 'spots': n}
                         for g, n in sorted(per_gene.items(), key=lambda kv: -kv[1])],
            'spots_are': note,
        }

    def spots_of_cell(self, label, min_prob=None):
        """Which spots belong to a cell, under one of two definitions.

        With `min_prob` unset: the spots whose most likely parent is this cell, the
        argmax. With `min_prob` set: every spot with a probability on this cell above
        it, which is what cellData.spot_id holds at 0.0001. The two are different
        lists, and the answer says which one it is. Either way every spot comes with
        its probability, sorted highest first.
        """
        label = int(label)
        self.to_internal(label)                  # raises if it is not a cell

        rows = []
        for tb, gene_of in self._arrow_spots(['spot_id', 'gene_id', 'neighbour_array',
                                              'neighbour_prob']):
            for sid, gid, cands, probs in zip(tb['spot_id'].to_pylist(),
                                              tb['gene_id'].to_pylist(),
                                              tb['neighbour_array'].to_pylist(),
                                              tb['neighbour_prob'].to_pylist()):
                if label not in cands:
                    continue
                p = probs[cands.index(label)]
                if min_prob is None:
                    keep = cands[0] == label
                else:
                    keep = p > min_prob
                if keep:
                    rows.append({'spot': int(sid), 'gene': gene_of[int(gid)], 'prob': float(p)})
        rows.sort(key=lambda r: -r['prob'])

        if min_prob is None:
            note = ('spots whose most likely parent is this cell. That is the argmax, '
                    'not a hard assignment: the lowest probability here can be well '
                    'under 0.5, and the cell also draws counts from spots whose most '
                    'likely parent is another cell')
        else:
            note = ('every spot with probability above %g on this cell. Their '
                    'probabilities add up to the cell\'s soft counts, so this is the '
                    'decomposition of cell_counts. Probabilities are rounded to 3 '
                    'decimals in the viewer files, so anything under 0.0005 is absent'
                    % min_prob)
        return {
            'cell': label,
            'definition': 'most likely parent' if min_prob is None else 'prob > %g' % min_prob,
            'n_spots': len(rows),
            'sum_of_probs': float(sum(r['prob'] for r in rows)),
            'spots': rows,
            'spots_are': note,
        }

    def cell_row(self, label):
        """The cellData row of one cell, value for value.

        Everything the tsv has except the three drawing columns. Read straight out of
        cellData.tsv when it is there, which is both exact and fast, a targeted grep
        on a 250 MB file takes about 30 ms. Without the tsv the row is rebuilt from
        the viewer files, which matches on every column except the boundary cases of
        spot_id; see `_exact_spot_ids`.
        """
        label = int(label)
        self.to_internal(label)

        row = self._tsv_row('cellData.tsv', r'^%d\t' % label)
        if row is not None:
            out = {k: row[k] for k in ('Cell_Num', 'X', 'Y', 'Z', 'Genenames',
                                       'CellGeneCount', 'spot_id', 'ClassName', 'Prob')}
            out['source'] = 'cellData.tsv'
        else:
            import pyarrow.compute as pc
            tb = self._arrow_cells()
            hit = tb.filter(pc.equal(tb['cell_id'], label))
            if hit.num_rows == 0:
                raise KeyError('no cell %s in this run' % label)
            r = hit.to_pylist()[0]
            genes = r['gene_names']
            per_gene = self._exact_spot_ids(label)
            out = {
                'Cell_Num': label,
                'X': _tsv(r['X']), 'Y': _tsv(r['Y']), 'Z': _tsv(r['Z']),
                'Genenames': genes,
                'CellGeneCount': [_tsv(v) for v in r['gene_counts']],
                'spot_id': [per_gene.get(g, []) for g in genes],
                'ClassName': r['class_name'],
                'Prob': [_tsv(v) for v in r['prob']],
                'source': 'rebuilt from the viewer files, cellData.tsv is not in this '
                          'run. spot_id can miss a few spots sitting within 0.0005 of '
                          'the 0.0001 cut-off',
            }
        out['counts_are'] = ('soft, weighted by the spot assignment probabilities. '
                             'spot_id lists every spot with probability above 0.0001 on '
                             'this cell, lined up with Genenames, and their probabilities '
                             'add up to CellGeneCount')
        return out

    def spot_row(self, spot_id):
        """The geneData row of one spot, value for value.

        Read out of geneData.tsv when it is there, otherwise rebuilt from the viewer
        files; `source` says which. Both give the same numbers, the viewer files are
        written from the same frame.

        neighbour_prob is rounded to 3 decimals here because that is how geneData
        stores it (summary.py:102), so a candidate holding 0.0004 shows up as 0.0 and
        you cannot tell it apart from one holding nothing at all. That is fine for
        reporting what the file says, and it is why cell_row's spot_id list can
        contain spots whose probability reads 0.0: the cut-off there is 0.0001,
        applied before the rounding. When the small probabilities matter use
        explain_spot, which recomputes them from diagnostics.db at full precision.
        """
        try:
            spot_id = int(spot_id)
        except (TypeError, ValueError):
            raise ValueError('%r is not a spot id' % (spot_id,))
        # spot_id is the third column of geneData
        row = self._tsv_row('geneData.tsv', r'^[^\t]*\t[^\t]*\t%d\t' % spot_id)
        if row is not None:
            row['source'] = 'geneData.tsv'
            return row

        import pyarrow.compute as pc
        for tb, gene_of in self._arrow_spots(None):
            hit = tb.filter(pc.equal(tb['spot_id'], spot_id))
            if hit.num_rows:
                r = hit.to_pylist()[0]
                out = {
                    'gene_name': gene_of[int(r['gene_id'])],
                    'gene_id': int(r['gene_id']),
                    'spot_id': spot_id,
                    'x': _tsv(r['x']), 'y': _tsv(r['y']), 'z': _tsv(r['z']),
                    'plane_id': int(r['plane_id']),
                    'neighbour': int(r['neighbour_array'][0]),
                    'neighbour_array': [int(c) for c in r['neighbour_array']],
                    'neighbour_prob': [_tsv(v) for v in r['neighbour_prob']],
                    'omp_score': _tsv(r['omp_score']),
                    'omp_intensity': _tsv(r['omp_intensity']),
                    'is_hard_misread': int(r['is_hard_misread']),
                    'source': 'the viewer files, geneData.tsv is not in this run',
                }
                if 'inside_cell' in r:
                    out['inside_cell'] = int(r['inside_cell'])
                return out
        raise KeyError('no spot %s in this run' % spot_id)

    # ------------------------------------------------------------ internals

    def _spot_scores(self, got):
        """The candidate ids, the six terms, the misread density, the scores and the
        softmax for one spots-table row (columns from gene_idx onward)."""
        gene = str(self.gene_panel[got[0]])
        neighbours = json.loads(got[4])
        terms = {name: np.frombuffer(got[i], dtype=np.float32)
                 for i, name in enumerate(
                     ['mvn_loglik', 'attention', 'expr_fluct',
                      'cell_inefficiency', 'gene_inefficiency', 'bonus'], start=5)}
        misread = float(self._meta('log_rho_bar', parse=True)[gene])
        # the last slot of every array is the background, which scores the misread
        # density instead of the six terms
        score = sum(v[:-1] for v in terms.values())
        score = np.append(score, misread)
        prob = np.exp(score - score.max())
        prob = prob / prob.sum()
        return neighbours, terms, misread, score, prob

    def _exact_spot_ids(self, label, tol=0.0001):
        """cellData.spot_id for one cell: per gene, the spots with an unrounded
        probability above tol on the cell.

        The arrow files round probabilities to 3 decimals, so they cannot answer this
        on their own, but they do list every candidate of every spot whatever its
        probability. So: candidates from the arrow files, then the exact probability
        of each from the db, which is indexed by spot id.

        This is the fallback, only used when cellData.tsv is missing, and it is not
        quite exact. The recomputation sits within about 5e-4 of what the model held,
        which is nothing anywhere except at the 0.0001 cut-off, where it flips a spot
        in or out. Measured on espio cell 18223: 506 spots against the 542 cellData
        lists, the 36 missing ones all recomputing to just under 1e-4. Read the tsv
        when it is there.
        """
        internal = self.to_internal(label)
        candidates = []
        for tb, _ in self._arrow_spots(['spot_id', 'neighbour_array']):
            for sid, cands in zip(tb['spot_id'].to_pylist(), tb['neighbour_array'].to_pylist()):
                if label in cands:
                    candidates.append(int(sid))

        per_gene = {}
        for sid in candidates:
            got = self._query(
                "SELECT gene_idx, x, y, z, neighbor_cell_ids, mvn_loglik, attention, "
                "expr_fluct, cell_inefficiency, gene_inefficiency, bonus "
                "FROM spots WHERE spot_id = ?", (sid,))
            neighbours, _, _, _, prob = self._spot_scores(got)
            p = float(prob[neighbours.index(internal)])
            if p > tol:
                per_gene.setdefault(str(self.gene_panel[got[0]]), []).append(sid)
        return per_gene

    def _tsv_row(self, name, pattern):
        """One row of a tsv, found by grep, parsed with the header.

        None when the file is not there. grep rather than pandas on purpose: these
        files run to hundreds of MB and reading one row out of cellData.tsv this way
        takes about 30 ms, against seconds to load the lot.
        """
        import ast
        import subprocess
        path = self.path.parent.parent.parent / 'tsv' / name
        if not path.is_file():
            return None
        hit = subprocess.run(['grep', '-m1', '-P', pattern, str(path)],
                             capture_output=True, text=True)
        if hit.returncode or not hit.stdout.strip():
            return None
        with open(path) as fid:
            header = fid.readline().rstrip('\n').split('\t')

        out = {}
        for col, raw in zip(header, hit.stdout.rstrip('\n').split('\t')):
            if raw.startswith('[') or raw.startswith('{'):
                out[col] = ast.literal_eval(raw)
            else:
                try:
                    out[col] = int(raw)
                except ValueError:
                    try:
                        out[col] = float(raw)
                    except ValueError:
                        out[col] = raw
        return out

    def _arrow_cells(self):
        """The arrow cells table, read once. It is cellData minus spot_id and the
        drawing columns, at the tsv's precision."""
        if not hasattr(self, '_cells_tb'):
            import pyarrow as pa
            import pyarrow.feather as feather
            d = self.path.parent.parent / 'arrow_cells'
            self._cells_tb = pa.concat_tables(
                [feather.read_table(f) for f in sorted(d.glob('*.feather'))])
        return self._cells_tb

    def _arrow_spots(self, columns):
        """Yield each arrow spots shard as a table, plus the gene id to name map.

        The shards carry segmentation labels, they went through the same translation
        as geneData, so a label compares directly against neighbour_array.
        """
        import pyarrow.feather as feather
        shard_dir = self.path.parent.parent / 'arrow_spots'
        gene_of = {int(k): v for k, v in
                   json.loads((shard_dir / 'gene_dict.json').read_text()).items()}
        for f in sorted(shard_dir.glob('*.feather')):
            yield feather.read_table(f, columns=columns), gene_of

    def _has_containment(self):
        """Whether this run carries the inside_cell column, added Sept 2026."""
        shard = self.path.parent.parent / 'arrow_spots'
        files = sorted(shard.glob('*.feather')) if shard.is_dir() else []
        if not files:
            return False
        import pyarrow.feather as feather
        return 'inside_cell' in feather.read_table(files[0]).schema.names

    def _nb_contributions(self, c):
        """Per gene, per class negative binomial log-likelihood for one cell.

        The same arithmetic as compute_gene_loglikelihood_matrix, for one row:
        the expected count is the class mean scaled by the cell's area, by the gene
        efficiency and by the cell's theta, with SpotReg added on.
        """
        eta = np.asarray(self._meta('eta_bar', parse=True), dtype=np.float32)
        spot_reg = float(self._meta('SpotReg'))
        r_spot = float(self._meta('rSpot'))

        expected = np.einsum('gk,g,k->gk', c['scaled_means'], eta, c['theta_bar']) + spot_reg
        p = expected / (r_spot + expected)
        return c['gene_count'][:, None] * np.log(p) + r_spot * np.log(1 - p)


def narrate_cell(e):
    """The story behind an explain_cell result, in plain words.

    Built from the numbers rather than left to the agent, so the story is the same
    whoever asks and cannot get the sign of a log-likelihood the wrong way round.
    Written for a reader who has never heard of a nat: the score differences are
    turned into odds ('about sixty to one') or into words ('overwhelmingly'), and the
    genes are ranked rather than numbered. The numbers themselves stay in the dict
    for anyone who wants them.
    """
    a, o = e['assigned'], e['compared_with']
    s = e['score']
    d = {
        'genes': s['gene_loglik']['assigned'] - s['gene_loglik']['compared'],
        'prior': s['log_prior']['assigned'] - s['log_prior']['compared'],
        'neighbours': s['spatial']['assigned'] - s['spatial']['compared'],
    }
    margin = sum(d.values())
    decider = max(d, key=lambda k: abs(d[k]))
    for_a = [g['gene'] for g in e['genes_favouring_assigned']]
    for_o = [g['gene'] for g in e['genes_favouring_compared']]

    def p(x):
        return 'less than 0.01' if x < 0.005 else '%.2f' % x

    def names(gs, n=3):
        gs = gs[:n]
        return gs[0] if len(gs) == 1 else ', '.join(gs[:-1]) + ' and ' + gs[-1]

    out = []
    out.append('Cell %d was called %s, with probability %s. The closest alternative was %s, '
               'at %s.' % (e['cell'], a, p(e['prob_assigned']), o, p(e['prob_compared'])))
    out.append('pciSeq decides a cell\'s class from three things: how well its gene counts '
               'match what each class typically expresses (the gene log-likelihood), how '
               'common each class is to begin with (the prior), and what the neighbouring '
               'cells were called (the spatial term). The class that comes out best '
               'overall wins.')

    # the genes, ranked, no numbers
    if d['genes'] > 0:
        out.append('The genes point to %s, %s. The strongest evidence comes from %s: the '
                   'cell holds these in the amounts a %s cell typically does and a %s cell '
                   'does not.' % (a, _strength(d['genes']), names(for_a), a, o))
        if for_o:
            out.append('A few genes, %s, look more like %s, but they are outweighed.'
                       % (names(for_o), o))
    else:
        out.append('On its genes alone the cell looks more like %s, %s, mostly because of '
                   '%s.' % (o, _strength(-d['genes']), names(for_o)))
        if for_a:
            out.append('The genes arguing for %s are %s.' % (a, names(for_a)))

    # the prior
    if abs(d['prior']) < 0.05:
        out.append('The prior treats the two classes alike.')
    else:
        who = a if d['prior'] > 0 else o
        out.append('The prior favours %s, because that class is more common to begin with.'
                   % who)

    # the neighbourhood
    if abs(d['neighbours']) < 0.4:
        out.append('The neighbouring cells make no real difference either way.')
    elif d['neighbours'] > 0:
        out.append('The neighbouring cells are %s %s, which %s the call.'
                   % (_mostly(d['neighbours']), a,
                      'strengthens' if d['genes'] > 0 else 'is what carries'))
    else:
        out.append('The neighbouring cells lean towards %s, which counts against the call.'
                   % o)

    # what settled it
    if decider == 'genes':
        out.append('So the genes settled it%s.'
                   % (', and the neighbourhood agreed' if d['neighbours'] > 0.4 else ''))
    elif decider == 'neighbours' and d['genes'] <= 0:
        out.append('So this call is the neighbourhood overruling the genes: on its genes '
                   'alone the cell would have been called %s, but surrounded by %s cells '
                   'the balance comes out for %s, %s.' % (o, a, a, _strength(margin)))
    elif decider == 'neighbours':
        out.append('So the neighbourhood settled it. The genes agreed, but only mildly; the '
                   'surrounding cells made the difference.')
    else:
        out.append('So the prior settled it.')
    return ' '.join(out)


def narrate_spot(e):
    """The story behind an explain_spot result, in plain words.

    Same idea as narrate_cell: computed from the numbers so it is the same whoever
    asks, no units in the output, differences said as odds or as a word. The shape
    follows the spot page of the docs: where the spot is against how well its gene
    fits each cell, and which of the two carried the call.
    """
    gene, spot = e['gene'], e['spot']
    cells = [c for c in e['candidates'] if c['cell'] != 'background']
    bg = next(c for c in e['candidates'] if c['cell'] == 'background')
    by_prob = sorted(cells, key=lambda c: -c['prob'])
    winner = by_prob[0]
    to_bg = e['assigned_to'] == 'background'

    def expr(c):
        return c['class expression'] + c['cell scale'] + c['cell-gene scale']

    def p(x):
        return 'less than 0.01' if x < 0.005 else '%.2f' % x

    def sure(x):
        return ('confidently' if x > 0.9 else 'fairly confidently' if x > 0.6
                else 'narrowly' if x > 0.4 else 'with no clear winner')

    out = []
    # the call
    if to_bg:
        out.append('Spot %d is a %s spot. It was assigned to the background, with probability '
                   '%s, meaning the model takes it for a misread rather than a read from any '
                   'cell. The nearest cell, %d, gets %s.'
                   % (spot, gene, p(bg['prob']), winner['cell'], p(winner['prob'])))
    else:
        others = ', '.join('cell %d (%s)' % (c['cell'], p(c['prob'])) for c in by_prob[1:3])
        out.append('Spot %d is a %s spot. It was assigned to cell %d, %s, with probability %s. '
                   'The next candidates are %s, and the chance it is a misread is %s.'
                   % (spot, gene, winner['cell'], sure(winner['prob']), p(winner['prob']),
                      others, p(bg['prob'])))

    # how a spot is scored
    out.append('pciSeq weighs each nearby cell on two things: how close the spot is to the '
               'cell\'s centre (the Gaussian fit), and how well a %s spot fits that cell, '
               'which combines whether the cell\'s class expresses %s (class expression), '
               'whether the cell holds more reads overall than its class predicts (cell '
               'scale), and whether it already holds more %s than its class predicts '
               '(cell-gene scale). The background is scored on how often %s spots turn out '
               'to be misreads. The best total wins.' % (gene, gene, gene, gene))

    if to_bg:
        out.append('Here no cell scores well enough: cell %d is the nearest, and its class '
                   'is %s, but %s does not fit it well, and %s misreads are common enough in '
                   'this run for the background to win.'
                   % (winner['cell'], winner['class'], gene, gene))
        return ' '.join(out)

    # position
    by_dist = sorted(cells, key=lambda c: -c['Gaussian fit'])
    nearest = by_dist[0]
    if nearest['cell'] == winner['cell']:
        rivals = [c for c in by_dist[1:3]]
        out.append('Cell %d is the nearest candidate: %s on position alone.'
                   % (winner['cell'], ' and '.join(
                       '%s over cell %d' % (_strength(winner['Gaussian fit'] - c['Gaussian fit']), c['cell'])
                       for c in rivals)))
    else:
        out.append('Cell %d is not the nearest: cell %d is closer, %s on position alone.'
                   % (winner['cell'], nearest['cell'],
                      _strength(nearest['Gaussian fit'] - winner['Gaussian fit'])))

    # expression
    runner = by_prob[1] if len(by_prob) > 1 else None
    classes = {c['class'] for c in cells}
    if len(classes) == 1:
        out.append('Every candidate is a %s cell, so on class alone %s fits them all equally; '
                   'what separates them is how much each already holds.' % (winner['class'], gene))
    else:
        out.append('Cell %d is a %s cell, and that class %s %s.'
                   % (winner['cell'], winner['class'],
                      'expresses' if winner['class expression'] > 0 else 'barely expresses', gene))
        if runner and runner['class'] != winner['class']:
            out.append('Cell %d is a %s cell, which %s.'
                       % (runner['cell'], runner['class'],
                          'does too' if runner['class expression'] > 0 else 'does not'))
        elif runner:
            out.append('Cell %d is a %s cell too.' % (runner['cell'], runner['class']))
    if runner:
        pieces = []
        d_sc = winner['cell scale'] - runner['cell scale']
        d_cg = winner['cell-gene scale'] - runner['cell-gene scale']
        if abs(d_sc) > 0.2:
            pieces.append('cell %d holds more reads overall than its class predicts'
                          % (winner['cell'] if d_sc > 0 else runner['cell']))
        if abs(d_cg) > 0.2:
            pieces.append('cell %d already holds more %s than its class predicts'
                          % (winner['cell'] if d_cg > 0 else runner['cell'], gene))
        if pieces:
            out.append('Between the top two, %s.' % ' and '.join(pieces))

    # the inside cell bonus, when it is switched on
    bonus = [c for c in cells if c.get('bonus', 0)]
    if bonus:
        out.append('The spot\'s pixel lies inside the mask of cell %d, which adds a bonus for '
                   'it.' % bonus[0]['cell'])

    # verdict, winner against the runner up
    if runner:
        d_pos = winner['Gaussian fit'] - runner['Gaussian fit']
        d_exp = expr(winner) - expr(runner)
        if d_pos > 0.4 and d_exp > 0.4:
            out.append('So cell %d is both the nearer and the better fit for the gene, and '
                       'the call is clear.' % winner['cell'])
        elif d_pos > 0.4:
            out.append('So cell %d is the better fit for the gene, %s, but cell %d is closer, '
                       '%s, and distance carries the call.'
                       % (runner['cell'], _strength(-d_exp) if d_exp < -0.4 else 'slightly',
                          winner['cell'], _strength(d_pos)))
        elif d_exp > 0.4:
            out.append('So cell %d is closer, %s, but cell %d fits the gene better, %s, and '
                       'expression carries the call.'
                       % (runner['cell'], _strength(-d_pos) if d_pos < -0.4 else 'slightly',
                          winner['cell'], _strength(d_exp)))
        else:
            out.append('So the two are close on both counts, and the call is a narrow one.')
    if bg['prob'] > 0.1:
        out.append('There is a real chance, %s, that the spot is a misread.' % p(bg['prob']))
    return ' '.join(out)


def _strength(d):
    """A log-likelihood difference as something a reader can picture: the odds, e**d
    to one, in round numbers while the number still means anything, and a word once
    it does not. Meant to close a sentence: 'the genes point to X, about 60 to one'."""
    import math
    if d < 0.4:
        return 'only just'
    odds = math.exp(d)
    if odds < 3:
        return 'slightly'
    if odds < 20:
        return 'about %d to one' % round(odds)
    if odds < 100:
        return 'about %d to one' % (round(odds / 5) * 5)
    if odds < 20000:
        return 'about %s to one' % format(int(round(odds, -2)), ',')
    return 'overwhelmingly, beyond any doubt'


def _mostly(d):
    """How much of the neighbourhood agrees, from the size of the spatial term."""
    return 'overwhelmingly' if d > 8 else 'mostly' if d > 2 else 'somewhat more'


def _tsv(v):
    """A float the way the tsv files print it: three decimals.

    Only used on the rebuild path. The viewer files were already written from the
    rounded frame, so this is a no-op on them in practice, it is here so the rebuilt
    row cannot come out with more digits than the real file would have had.
    """
    return round(float(v), 3)


def _find_db(path):
    """Accept the run folder, anything above it, or the database itself."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    hits = sorted(path.rglob('diagnostics.db'))
    if not hits:
        raise FileNotFoundError('no diagnostics.db under %s' % path)
    if len(hits) > 1:
        raise ValueError('%d diagnostics.db files under %s, point at one of them:\n  %s'
                         % (len(hits), path, '\n  '.join(str(h) for h in hits[:10])))
    return hits[0]


def open_run(path):
    """Open a finished run. The entry point every other tool goes through."""
    return Run(path)
