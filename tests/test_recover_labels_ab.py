"""
Checks that recover_original_labels gives the very same geneData as it did
before its three spot columns were mapped with a lookup table.

The function puts the segmentation labels back on the results at the very end of
fit. For geneData it used to call fetch_label once per spot and column. That
older code is kept below, in _gene_data_before, and the two must give the same
table: values, dtypes and python lists of python ints in neighbour_array.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

import pciSeq
import pciSeq.app as app
from pciSeq.src.core.utils.cell_utils import (
    recover_original_labels, reversed_label_map, fetch_label,
    _label_lookup_table, _lookup_labels,
)

GENES = ['g%02d' % i for i in range(12)]
CLASSES = ['A', 'B', 'C']


def _gene_data_before(geneData, label_map):
    """NOTE: this is what recover_original_labels did to geneData before the
    lookup table. It is not used anywhere in pciSeq anymore, it lives here only
    to check the new code against. One python call per spot and column."""
    reverse_map = reversed_label_map(label_map)
    return geneData.assign(
        neighbour=geneData.neighbour.map(lambda x: fetch_label(x, reverse_map)),
        neighbour_array=geneData.neighbour_array.map(lambda x: fetch_label(x, reverse_map)),
        inside_cell=geneData.inside_cell.map(lambda x: fetch_label(x, reverse_map))
    )


def _label_image():
    """16 square cells on an 80x80 grid, labelled 101 up. Gappy on purpose."""
    lab = np.zeros((80, 80), dtype=np.int32)
    i = 101
    for r in range(4, 76, 18):
        for c in range(4, 76, 18):
            lab[r:r + 9, c:c + 9] = i
            i += 3
    return lab


def _inputs_of_a_real_run(rng, tmp_path, monkeypatch):
    """Run a small fit on labels that have to be renumbered and catch what it
    hands to recover_original_labels, so the test works on the real thing: the
    real tables and the real label_map with whatever types it holds."""
    caught = {}
    real = app.recover_original_labels

    def spy(cellData, geneData, cellBoundaries, cellBoundaries_list, label_map):
        caught['args'] = (cellData, geneData, cellBoundaries, cellBoundaries_list, label_map)
        return real(cellData, geneData, cellBoundaries, cellBoundaries_list, label_map)

    monkeypatch.setattr(app, 'recover_original_labels', spy)
    spots = pd.DataFrame({
        'gene_name': rng.choice(GENES, 900),
        'x': rng.uniform(0, 79, 900).astype(np.float32),
        'y': rng.uniform(0, 79, 900).astype(np.float32),
    })
    scref = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50, index=GENES, columns=CLASSES)
    scref.index.name = 'gene_name'
    opts = {'max_iter': 2, 'save_data': False, 'CellCallTolerance': 0.5, 'output_path': str(tmp_path)}
    pciSeq.fit(spots=spots, coo=coo_matrix(_label_image()), scRNAseq=scref, opts=opts)
    return caught['args']


@pytest.mark.slow
def test_same_geneData_on_a_real_run(rng, tmp_path, monkeypatch):
    cellData, geneData, cellBoundaries, cellBoundaries_list, label_map = _inputs_of_a_real_run(rng, tmp_path, monkeypatch)
    assert label_map, 'the labels were not renumbered, the test checks nothing'

    before = _gene_data_before(geneData, label_map)
    _, now, _, _ = recover_original_labels(cellData, geneData, cellBoundaries, cellBoundaries_list, label_map)

    assert (now.dtypes == before.dtypes).all()
    pd.testing.assert_frame_equal(now, before, check_exact=True)
    for a, b in zip(now.neighbour_array, before.neighbour_array):
        assert type(a) is list and a == b
        assert [type(v) for v in a] == [type(v) for v in b]
    assert now.to_csv(sep='\t', index=False) == before.to_csv(sep='\t', index=False)
    # and the labels really are the segmentation ones now
    assert set(now.neighbour) <= {0} | set(range(101, 101 + 3 * 16, 3))


def test_lookup_gives_what_fetch_label_gives():
    d = {0: 0, 1: 205, 2: 101, 3: 9000, 7: 44}       # gaps on both sides
    lut, known = _label_lookup_table(d)
    labels = np.array([[0, 1, 2], [3, 7, 0]])
    assert _lookup_labels(labels, lut, known).tolist() == [fetch_label(row, d) for row in labels.tolist()]


@pytest.mark.parametrize('bad', [4, 8, 500, -1])
def test_an_unknown_label_still_raises(bad):
    # 4 is a gap inside the table, 8 and 500 are past its end, -1 is before it.
    # fetch_label raises KeyError on all of them and so must the lookup
    d = {0: 0, 1: 205, 2: 101, 3: 9000, 7: 44}
    lut, known = _label_lookup_table(d)
    with pytest.raises(KeyError):
        fetch_label(bad, d)
    with pytest.raises(KeyError):
        _lookup_labels(np.array([0, 1, bad]), lut, known)


def test_an_empty_geneData_goes_through():
    empty = pd.DataFrame({'neighbour': [], 'neighbour_array': [], 'inside_cell': []})
    cells = pd.DataFrame({'Cell_Num': [1]})
    bounds = pd.DataFrame({'cell_id': [1]})
    out = recover_original_labels(cells, empty, bounds, [bounds], {101: 1, 0: 0})
    assert len(out[1]) == 0
    assert list(out[0].Cell_Num) == [101]
