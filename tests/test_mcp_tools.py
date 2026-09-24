"""The agent tools, checked against the run they read.

Every tool here is cross-checked against something the run wrote independently:
explain_cell against the stored class probabilities, explain_spot against
geneData.neighbour, cell_counts against cellData.CellGeneCount, spots_in_cell
against geneData.inside_cell. If a tool drifts from the model it fails here.

Uses the 16 cell fixture from test_label_identifiers, labels 101..116, so the
label translation is exercised too.
"""
import numpy as np
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
