"""The agent tools, checked against the run they read.

Every tool here is cross-checked against something the run wrote independently:
explain_cell against the stored class probabilities, explain_spot against
geneData.neighbour, cell_counts against cellData.CellGeneCount, spots_in_cell
against geneData.inside_cell. If a tool drifts from the model it fails here.

Uses the 16 cell fixture from test_label_identifiers, labels 101..116, so the
label translation is exercised too.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from pciSeq.src.mcp.tools import open_run, Run
from tests.test_label_identifiers import _run

LABELS = list(range(101, 117))


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    """One run, saved, plus the frames it returned.

    Run long enough to converge: explain_cell recomputes the class score with the
    final eta, and eta is updated after the last class update, so the recomputation
    only matches the stored probabilities exactly once the run has settled. At two
    iterations the error is 1e-2, at ten it is 1e-7."""
    tmp = tmp_path_factory.mktemp('mcp')
    rng = np.random.default_rng(7)
    # tol 0.5 would call it converged after two iterations whatever max_iter says
    cellData, geneData = _run(rng, tmp, save=True, max_iter=40, tol=0.001)
    return open_run(tmp), cellData.set_index('Cell_Num'), geneData.set_index('spot_id')


# ------------------------------------------------------------ opening it

def test_open_run_finds_the_db_from_the_run_folder(fitted):
    run, _, _ = fitted
    assert run.path.name == 'diagnostics.db'


def test_summary_does_not_count_the_background(fitted):
    run, cellData, _ = fitted
    s = run.summary()
    assert s['cells'] == 16 == len(cellData)
    assert s['labels_were_renumbered'] is True
    assert s['has_containment'] is True


def test_labels_round_trip(fitted):
    run, _, _ = fitted
    for lab in LABELS:
        assert run.to_external(run.to_internal(lab)) == lab


# ---------------------------------------------------------- explain_cell

def test_explain_cell_reproduces_the_stored_class_probabilities(fitted):
    """The whole point. The recomputed score must give back what the model stored,
    otherwise the gene by gene story is not the model's story."""
    run, _, _ = fitted
    log_prior = np.asarray(run._meta('log_prior', parse=True), dtype=np.float32)
    for lab in LABELS:
        c = run._cell(lab)
        mine = softmax(run._nb_contributions(c).sum(axis=0) + log_prior + c['mrf'])
        assert np.allclose(mine, c['class_prob'], atol=1e-4), lab


def test_explain_cell_agrees_with_cellData(fitted):
    run, cellData, _ = fitted
    for lab in LABELS:
        e = run.explain_cell(lab)
        assert e['assigned'] == cellData.loc[lab, 'ClassName'][0]
        assert e['prob_assigned'] == pytest.approx(cellData.loc[lab, 'Prob'][0], abs=1e-3)
        assert e['assigned'] != e['compared_with']


def test_explain_cell_against_a_named_class(fitted):
    run, _, _ = fitted
    e = run.explain_cell(101)
    other = 'TypeA' if e['assigned'] != 'TypeA' else 'TypeB'
    assert run.explain_cell(101, vs_class=other)['compared_with'] == other
    with pytest.raises(ValueError):
        run.explain_cell(101, vs_class=e['assigned'])
    with pytest.raises(ValueError):
        run.explain_cell(101, vs_class='NoSuchClass')


# ---------------------------------------------------------- explain_spot

def test_explain_spot_agrees_with_geneData(fitted):
    """The softmax of the stored terms has to land on the same cell geneData did,
    and the candidates have to come out as segmentation labels."""
    run, _, geneData = fitted
    for sid in geneData.index[:200]:
        e = run.explain_spot(sid)
        probs = [c['prob'] for c in e['candidates']]
        assert sum(probs) == pytest.approx(1.0, abs=1e-5)

        expected = geneData.loc[sid, 'neighbour']
        assert e['assigned_to'] == ('background' if expected == 0 else expected)

        cells = [c['cell'] for c in e['candidates'] if c['cell'] != 'background']
        assert set(cells) <= set(LABELS)


def test_explain_spot_gene_matches(fitted):
    run, _, geneData = fitted
    sid = geneData.index[0]
    assert run.explain_spot(sid)['gene'] == geneData.loc[sid, 'gene_name']


# ----------------------------------------------------------- cell_counts

def test_cell_counts_matches_cellData_and_says_it_is_soft(fitted):
    run, cellData, _ = fitted
    for lab in LABELS:
        cc = run.cell_counts(lab)
        assert cc['total_counts'] == pytest.approx(sum(cellData.loc[lab, 'CellGeneCount']), abs=0.05)
        assert 'soft' in cc['counts_are']
        got = {g['gene']: g['counts'] for g in cc['per_gene']}
        want = dict(zip(cellData.loc[lab, 'Genenames'], cellData.loc[lab, 'CellGeneCount']))
        for g, n in want.items():
            assert got[g] == pytest.approx(n, abs=0.002), (lab, g)


def test_cell_counts_for_one_gene(fitted):
    run, _, _ = fitted
    all_of_them = {g['gene']: g['counts'] for g in run.cell_counts(105)['per_gene']}
    for g, n in all_of_them.items():
        assert run.cell_counts(105, gene=g)['counts'] == pytest.approx(n)
    with pytest.raises(ValueError):
        run.cell_counts(105, gene='NoSuchGene')


# --------------------------------------------------------- spots_in_cell

def test_spots_in_cell_matches_geneData_and_says_it_is_hard(fitted):
    run, _, geneData = fitted
    for lab in LABELS:
        sc = run.spots_in_cell(lab)
        assert sc['total_spots'] == int((geneData.inside_cell == lab).sum())
        assert 'no probabilities' in sc['spots_are']
        assert all(isinstance(g['spots'], int) for g in sc['per_gene'])


def test_spots_in_cell_for_one_gene(fitted):
    run, _, geneData = fitted
    inside = geneData[geneData.inside_cell == 103]
    for g, n in inside.gene_name.value_counts().items():
        assert run.spots_in_cell(103, gene=g)['spots'] == n


def test_soft_and_hard_are_different_numbers(fitted):
    """If these ever agreed for every cell the two tools would be redundant."""
    run, _, _ = fitted
    soft = [run.cell_counts(l)['total_counts'] for l in LABELS]
    hard = [run.spots_in_cell(l)['total_spots'] for l in LABELS]
    assert any(abs(s - h) > 0.5 for s, h in zip(soft, hard))


# --------------------------------------------------------- the guard rails

def test_background_is_refused_with_a_clear_message(fitted):
    """The agent reads assigned_to: 'background' and may hand it straight back."""
    run, _, _ = fitted
    with pytest.raises(ValueError, match='background'):
        run.cell_counts('background')
    with pytest.raises(ValueError, match='background'):
        run.cell(0)


def test_unknown_cell_and_spot(fitted):
    run, _, _ = fitted
    with pytest.raises(KeyError):
        run.cell(999999)
    with pytest.raises(KeyError):
        run.explain_spot(999999999)


def test_open_run_on_an_empty_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_run(tmp_path)


# --------------------------------------------------------- spots_of_cell

def test_argmax_spots_match_geneData_neighbour(fitted):
    run, _, geneData = fitted
    for lab in LABELS:
        got = run.spots_of_cell(lab)
        assert got['definition'] == 'most likely parent'
        assert {r['spot'] for r in got['spots']} == set(geneData.index[geneData.neighbour == lab])
        assert 'not a hard assignment' in got['spots_are']


def test_thresholded_spots_match_geneData_and_sum_to_the_counts(fitted):
    run, cellData, geneData = fitted
    for lab in LABELS:
        got = run.spots_of_cell(lab, min_prob=0.0001)
        assert got['definition'] == 'prob > 0.0001'

        want = set()
        for sid, cands, probs in zip(geneData.index, geneData.neighbour_array, geneData.neighbour_prob):
            if lab in cands and probs[list(cands).index(lab)] > 0.0001:
                want.add(sid)
        assert {r['spot'] for r in got['spots']} == want
        assert got['sum_of_probs'] == pytest.approx(sum(cellData.loc[lab, 'CellGeneCount']), abs=0.05)
        assert 'decomposition of cell_counts' in got['spots_are']


def test_the_two_definitions_nest(fitted):
    """Argmax is a subset of any threshold below 0.5, and normally much smaller."""
    run, _, _ = fitted
    for lab in LABELS:
        argmax = {r['spot'] for r in run.spots_of_cell(lab)['spots']}
        wide = {r['spot'] for r in run.spots_of_cell(lab, min_prob=0.0001)['spots']}
        assert argmax <= wide


def test_spots_come_sorted_by_probability(fitted):
    run, _, _ = fitted
    probs = [r['prob'] for r in run.spots_of_cell(105, min_prob=0.0)['spots']]
    assert probs == sorted(probs, reverse=True)


def test_spots_of_cell_refuses_a_bad_label(fitted):
    run, _, _ = fitted
    with pytest.raises(KeyError):
        run.spots_of_cell(999999)


# ------------------------------------------ the tsv rows, value for value

@pytest.fixture(scope='module')
def tsv(fitted):
    """The two tsv files as the run wrote them, lists parsed back."""
    from pciSeq.src.core.io import read_tsv
    run, _, _ = fitted
    d = run.path.parent.parent.parent / 'tsv'
    return (read_tsv(str(d / 'cellData.tsv')).set_index('Cell_Num'),
            read_tsv(str(d / 'geneData.tsv')).set_index('spot_id'))


def test_cell_row_is_the_cellData_row(fitted, tsv):
    run, _, _ = fitted
    cells, _ = tsv
    for lab in LABELS:
        got = run.cell_row(lab)
        want = cells.loc[lab]
        assert got['Cell_Num'] == lab
        for col in ('X', 'Y', 'Z'):
            assert got[col] == pytest.approx(want[col], abs=1e-3), (lab, col)
        assert got['Genenames'] == list(want['Genenames'])
        assert got['CellGeneCount'] == pytest.approx(list(want['CellGeneCount']), abs=1e-3)
        assert got['ClassName'] == list(want['ClassName'])
        assert got['Prob'] == pytest.approx(list(want['Prob']), abs=1e-3)
        # spot_id: same spots under each gene, order inside a gene is not promised
        assert [sorted(s) for s in got['spot_id']] == [sorted(s) for s in want['spot_id']], lab


def test_spot_row_is_the_geneData_row(fitted, tsv):
    run, _, _ = fitted
    _, spots = tsv
    for sid in list(spots.index[:150]) + list(spots.index[-50:]):
        got = run.spot_row(sid)
        want = spots.loc[sid]
        assert got['gene_name'] == want['gene_name']
        assert got['gene_id'] == want['gene_id']
        for col in ('x', 'y', 'z', 'omp_score', 'omp_intensity'):
            assert got[col] == pytest.approx(want[col], abs=1e-3), (sid, col)
        assert got['plane_id'] == want['plane_id']
        assert got['neighbour'] == want['neighbour']
        assert got['neighbour_array'] == list(want['neighbour_array'])
        assert got['neighbour_prob'] == pytest.approx(list(want['neighbour_prob']), abs=1e-3)
        assert got['is_hard_misread'] == want['is_hard_misread']
        assert got['inside_cell'] == want['inside_cell']


def test_cell_row_spot_ids_are_the_soft_count_decomposition(fitted):
    """The probabilities of the listed spots add up to the counts, per gene."""
    run, _, _ = fitted
    row = run.cell_row(105)
    for gene, count, sids in zip(row['Genenames'], row['CellGeneCount'], row['spot_id']):
        total = sum(next(c['prob'] for c in run.explain_spot(s)['candidates'] if c['cell'] == 105)
                    for s in sids)
        assert total == pytest.approx(count, abs=0.002), gene


# --------------------------------------------------------- threads

def test_a_run_opened_on_one_thread_answers_on_another(fitted):
    """An MCP server runs each tool call on a worker thread and does not promise the
    same one twice, so open_run and the next call can land on different threads.
    sqlite refuses a connection used across threads unless told otherwise, and that
    showed up as a one-in-four failure of the full suite before it was caught.
    This forces the case every time."""
    import concurrent.futures as cf
    import threading

    run, cellData, _ = fitted
    here = threading.get_ident()
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        there = pool.submit(threading.get_ident).result()
        assert there != here
        got = pool.submit(run.cell_counts, 105).result()
        spot = pool.submit(run.explain_spot, 0).result()
    assert got['total_counts'] == pytest.approx(sum(cellData.loc[105, 'CellGeneCount']), abs=0.05)
    assert sum(c['prob'] for c in spot['candidates']) == pytest.approx(1.0, abs=1e-5)


def test_queries_from_many_threads_at_once(fitted):
    """The lock in Run._query. Hammer one connection from several threads and every
    answer has to be the right one, not a crash and not somebody else's row."""
    import concurrent.futures as cf

    run, _, _ = fitted
    want = {lab: run.cell(lab)['classes'][0]['class'] for lab in LABELS}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run.cell, lab) for lab in LABELS * 5]
        for lab, fut in zip(LABELS * 5, futures):
            assert fut.result()['classes'][0]['class'] == want[lab]


# --------------------------------------------------------- the narrative

def test_narrative_tells_the_story_without_units(fitted):
    """Every explain_cell answer carries a plain-words story built from its own
    numbers. It names both classes, the top gene and what settled it, and it never
    mentions a nat: the evidence comes out as odds or as a word, because the reader
    it is written for has never heard of a log-likelihood."""
    import re
    run, _, _ = fitted
    for lab in LABELS:
        e = run.explain_cell(lab)
        story = e['narrative']
        assert e['assigned'] in story and e['compared_with'] in story
        top = (e['genes_favouring_assigned'] or e['genes_favouring_compared'])[0]['gene']
        assert top in story
        assert re.search(r'So (the genes settled it|this call is the neighbourhood|'
                         r'the neighbourhood settled it|the prior settled it)', story), lab
        assert not re.search(r'\bnats?\b', story), lab
        # probabilities like 1.00 and 0.01 are allowed; a negative number or anything
        # 10 or more with a decimal point is a log-likelihood that leaked through
        assert not re.search(r'-\d|\b\d{2,}\.\d', story), 'raw score leaked: %d' % lab
        assert '\u2014' not in story and '\u2013' not in story


def test_narrative_when_the_neighbourhood_overrules_the_genes():
    """The fixture may not contain such a cell, so build the case by hand: genes
    favour the other class by 4 (about 55 to one), the neighbourhood favours the
    assigned one by 12. The story has to say the neighbourhood overruled the genes,
    name the class the genes alone would have picked, and give the odds in words."""
    from pciSeq.src.mcp.tools import narrate_cell
    e = {
        'cell': 4308, 'assigned': 'L5 ET', 'compared_with': 'L4/5 IT',
        'prob_assigned': 0.999, 'prob_compared': 0.001,
        'score': {'gene_loglik': {'assigned': -300.0, 'compared': -296.0},
                  'log_prior': {'assigned': -4.3, 'compared': -4.3},
                  'spatial': {'assigned': 12.5, 'compared': 0.5}},
        'genes_favouring_assigned': [{'gene': 'Cpne7', 'counts': 9.0, 'diff': 8.2}],
        'genes_favouring_compared': [{'gene': 'Car4', 'counts': 6.0, 'diff': -7.2}],
    }
    story = narrate_cell(e)
    assert 'looks more like L4/5 IT, about 55 to one, mostly because of Car4' in story
    assert 'neighbouring cells are overwhelmingly L5 ET, which is what carries the call' in story
    assert 'neighbourhood overruling the genes' in story
    assert 'would have been called L4/5 IT' in story
    assert 'comes out for L5 ET, about 3,000 to one' in story


def test_narrative_when_the_genes_decide():
    from pciSeq.src.mcp.tools import narrate_cell
    e = {
        'cell': 1, 'assigned': 'A', 'compared_with': 'B',
        'prob_assigned': 1.0, 'prob_compared': 0.0,
        'score': {'gene_loglik': {'assigned': -100.0, 'compared': -133.6},
                  'log_prior': {'assigned': -4.3, 'compared': -4.3},
                  'spatial': {'assigned': 4.8, 'compared': 0.0}},
        'genes_favouring_assigned': [{'gene': 'Ndnf', 'counts': 8.0, 'diff': 15.3},
                                     {'gene': 'Rgs5', 'counts': 7.0, 'diff': 15.3}],
        'genes_favouring_compared': [{'gene': 'Npy', 'counts': 3.0, 'diff': -7.6}],
    }
    story = narrate_cell(e)
    assert 'The genes point to A, overwhelmingly, beyond any doubt.' in story
    assert 'The strongest evidence comes from Ndnf and Rgs5' in story
    assert 'at less than 0.01' in story
    assert 'So the genes settled it, and the neighbourhood agreed.' in story


def test_strength_wording():
    """The odds bands, so a change in rounding shows up here and not in a doc."""
    from pciSeq.src.mcp.tools import _strength
    assert _strength(0.2) == 'only just'
    assert _strength(0.8) == 'slightly'
    assert _strength(2.0) == 'about 7 to one'
    assert _strength(4.1) == 'about 60 to one'
    assert _strength(8.4) == 'about 4,400 to one'
    assert _strength(12.5) == 'overwhelmingly, beyond any doubt'


# ---------------------------------------------------- the spot narrative

def test_spot_narrative_tells_the_story_without_units(fitted):
    import re
    run, _, geneData = fitted
    for sid in list(geneData.index[:40]) + list(geneData.index[-10:]):
        e = run.explain_spot(sid)
        story = e['narrative']
        assert e['gene'] in story
        if e['assigned_to'] == 'background':
            assert 'assigned to the background' in story
        else:
            assert 'assigned to cell %d' % e['assigned_to'] in story
            assert re.search(r'carries the call|the call is clear|a narrow one', story), sid
        assert not re.search(r'\bnats?\b', story)
        assert not re.search(r'-\d|\b\d{2,}\.\d', story), 'raw score leaked: %d' % sid
        assert '\u2014' not in story and '\u2013' not in story


def test_spot_candidates_carry_their_class(fitted):
    run, cellData, _ = fitted
    e = run.explain_spot(0)
    for c in e['candidates']:
        if c['cell'] != 'background':
            assert c['class'] == cellData.loc[c['cell'], 'ClassName'][0]


def _spot(assigned, cands, bg_prob, bg_misread=-14.0):
    rows = []
    for cell, cls, gauss, cx, sc, cg, prob in cands:
        rows.append({'cell': cell, 'class': cls, 'Gaussian fit': gauss, 'class expression': cx,
                     'cell scale': sc, 'cell-gene scale': cg, 'gene efficiency': -0.35,
                     'bonus': 0.0, 'sum': gauss + cx + sc + cg - 0.35, 'prob': prob})
    rows.append({'cell': 'background', 'misread': bg_misread, 'sum': bg_misread, 'prob': bg_prob})
    return {'spot': 7, 'gene': 'Synpr', 'position': {}, 'candidates': rows, 'assigned_to': assigned}


def test_spot_narrative_when_distance_carries_it():
    """The docs spot: nearest by a clear margin, every candidate the same class, the
    runner up holds more reads. Distance wins."""
    from pciSeq.src.mcp.tools import narrate_spot
    e = _spot(18223, [(18223, 'DG', -10.4, 0.94, -0.66, 0.18, 0.74),
                      (21574, 'DG', -12.8, 0.94, 0.10, 0.07, 0.13),
                      (17371, 'DG', -12.1, 0.94, -0.30, -0.54, 0.10)], 0.01)
    s = narrate_spot(e)
    assert 'assigned to cell 18223, fairly confidently, with probability 0.74' in s
    assert 'Cell 18223 is the nearest candidate: about 5 to one over cell 17371 and about 11 to one over cell 21574' in s
    assert 'Every candidate is a DG cell' in s
    assert 'cell 21574 holds more reads overall than its class predicts' in s
    assert 'distance carries the call' in s


def test_spot_narrative_when_expression_carries_it():
    from pciSeq.src.mcp.tools import narrate_spot
    e = _spot(2, [(2, 'CA2', -11.0, 0.9, 0.5, 0.6, 0.66),
                  (1, 'CA1', -10.6, -1.2, 0.0, 0.0, 0.32)], 0.01)
    s = narrate_spot(e)
    assert 'Cell 2 is not the nearest: cell 1 is closer' in s
    assert 'Cell 2 is a CA2 cell, and that class expresses Synpr' in s
    assert 'Cell 1 is a CA1 cell, which does not' in s
    assert 'expression carries the call' in s


def test_spot_narrative_when_the_background_wins():
    from pciSeq.src.mcp.tools import narrate_spot
    e = _spot('background', [(6482, 'VLMC', -15.6, -1.3, 0.6, -0.36, 0.02)], 0.95)
    s = narrate_spot(e)
    assert 'assigned to the background, with probability 0.95' in s
    assert 'takes it for a misread' in s
    assert 'cell 6482 is the nearest, and its class is VLMC' in s
    assert 'carries the call' not in s


# ----------------------------------------- settings, convergence and planes

def test_run_info_reports_the_settings_and_how_it_ended(fitted):
    """The fixture is run with max_iter 40 and tolerance 0.001, so those two have to
    come straight back, along with a convergence verdict in words."""
    run, _, _ = fitted
    info = run.run_info()
    assert info['settings']['max_iter'] == 40
    assert info['settings']['CellCallTolerance'] == 0.001
    assert info['voxel_size'] == [1, 1, 1] and info['is3D'] is False
    assert isinstance(info['converged'], bool) and info['iterations'] >= 1
    assert ('converged after' in info['ended']) == info['converged']
    assert 'label_map' not in info['settings']
    s = run.summary()
    assert s['has_settings'] and s['iterations'] == info['iterations']


def test_run_info_on_a_run_without_settings(fitted):
    """Runs written before the export carry no config row. The answer must say so
    rather than invent settings."""
    run, _, _ = fitted
    saved = run.config, run.run_record
    run.config, run.run_record = None, None
    try:
        info = run.run_info()
        assert info['settings'] is None and 'before pciSeq exported' in info['note']
        assert info['commit'] is not None
        pos = run.explain_spot(0)['position']
        assert 'plane' not in pos and 'no voxel_size' in pos['z_is']
    finally:
        run.config, run.run_record = saved


def test_explain_spot_gives_the_plane_on_an_anisotropic_3d_run(rng, tmp_path):
    """Three planes with voxel_size [1, 1, 4]: the model works in a z four times the
    plane index, and the tool has to hand back the plane, not the scaled z."""
    import pciSeq
    from scipy.sparse import coo_matrix
    from pciSeq.src.mcp.tools import open_run
    from tests.test_label_identifiers import _label_image, GENES, CLASSES

    spots = pd.DataFrame({
        'gene_name': rng.choice(GENES, 900),
        'x': rng.uniform(0, 79, 900).astype(np.float32),
        'y': rng.uniform(0, 79, 900).astype(np.float32),
        'z_plane': rng.integers(0, 3, 900).astype(np.float32),
    })
    scref = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50, index=GENES, columns=CLASSES)
    scref.index.name = 'gene_name'
    lab = _label_image()
    # one matrix per plane: process_labels remaps each plane in place, so the same
    # object three times would be renumbered on the first pass and fail on the second
    _, geneData = pciSeq.fit(spots=spots, coo=[coo_matrix(lab) for _ in range(3)], scRNAseq=scref,
                             opts={'max_iter': 3, 'save_data': True, 'output_path': str(tmp_path),
                                   'voxel_size': [1, 1, 4], 'CellCallTolerance': 0.5})
    run = open_run(tmp_path)
    assert run.run_info()['voxel_size'] == [1, 1, 4] and run.run_info()['is3D'] is True
    gd = geneData.set_index('spot_id')
    for sid in gd.index[:150]:
        pos = run.explain_spot(sid)['position']
        assert pos['plane'] == gd.loc[sid, 'plane_id'], sid
        assert pos['z'] == int(gd.loc[sid, 'z'])
