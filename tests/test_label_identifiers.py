"""The two cell identifiers, and the translation between them.

pciSeq has two numbers for the same cell. The segmentation label is whatever came
in on the label image, which is what the user recognises and what every flat file
reports. The internal label is the row index into the arrays in the pickle file, 1..nC-1,
with row 0 the background.

They are only different when the input labels were not already sequential, so a
missing translation is correct on every tidy test dataset and wrong on a real
segmentation with gaps in it. That is why these tests use labels 101..116: they
do not overlap the internal ones at all, so a skipped translation cannot pass by luck.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

import pciSeq
from pciSeq.src.core.utils.cell_utils import (
    recover_original_labels, to_external, to_internal,
)

GENES = ['g%d' % i for i in range(6)]
CLASSES = ['TypeA', 'TypeB', 'TypeC']

# 0 stays 0 on both sides, it is the background
LABEL_MAP = {0: 0, 101: 1, 102: 2, 103: 3}


# ------------------------------------------------------------- the helpers

def test_nothing_happens_when_there_was_no_renumbering():
    assert to_internal(12, None) == 12
    assert to_external(12, None) == 12
    assert to_internal([5, 12], None) == [5, 12]


def test_the_two_directions_are_inverses():
    assert to_internal(102, LABEL_MAP) == 2
    assert to_external(2, LABEL_MAP) == 102
    assert to_external(to_internal(103, LABEL_MAP), LABEL_MAP) == 103


def test_a_list_goes_in_and_a_list_comes_out():
    assert to_internal([101, 102, 103], LABEL_MAP) == [1, 2, 3]
    assert to_external([1, 2, 3], LABEL_MAP) == [101, 102, 103]
    # a single element list used to collapse to a scalar, which would quietly
    # change the shape of geneData.neighbour_array on a one neighbour run
    assert to_external([1], LABEL_MAP) == [101]


def test_background_is_zero_on_both_sides():
    assert to_internal(0, LABEL_MAP) == 0
    assert to_external(0, LABEL_MAP) == 0


def test_numpy_numbers_work_too():
    assert to_internal(np.int64(102), LABEL_MAP) == 2
    assert to_external(np.array([1, 2]), LABEL_MAP) == [101, 102]


def test_an_unknown_label_raises_instead_of_going_quiet():
    """It used to come back as None and end up in the output file as a null."""
    with pytest.raises(KeyError):
        to_internal(999, LABEL_MAP)


# ------------------------------------------------ recover_original_labels

def test_the_no_op_path_still_returns_four_things():
    """It used to return three, so unpacking it blew up."""
    empty = pd.DataFrame()
    out = recover_original_labels(empty, empty, empty, [empty], None)
    assert len(out) == 4


# --------------------------------------------------------- the whole run

def _label_image():
    """16 square cells on an 80x80 grid, labelled 101 up. Gappy on purpose."""
    lab = np.zeros((80, 80), dtype=np.int32)
    i = 101
    for r in range(4, 76, 18):
        for c in range(4, 76, 18):
            lab[r:r + 9, c:c + 9] = i
            i += 1
    return lab


def _run(rng, tmp_path, save=False, max_iter=2, tol=0.5):
    spots = pd.DataFrame({
        'gene_name': rng.choice(GENES, 900),
        'x': rng.uniform(0, 79, 900).astype(np.float32),
        'y': rng.uniform(0, 79, 900).astype(np.float32),
    })
    scref = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50,
                         index=GENES, columns=CLASSES)
    scref.index.name = 'gene_name'
    opts = {'max_iter': max_iter, 'save_data': save, 'CellCallTolerance': tol,
            'output_path': str(tmp_path)}
    return pciSeq.fit(spots=spots, coo=coo_matrix(_label_image()),
                      scRNAseq=scref, opts=opts)


@pytest.mark.slow
def test_cellData_reports_the_segmentation_labels(rng, tmp_path):
    cellData, _ = _run(rng, tmp_path)
    assert set(cellData.Cell_Num) == set(range(101, 117))


@pytest.mark.slow
def test_geneData_reports_the_segmentation_labels(rng, tmp_path):
    """Both neighbour and neighbour_array, they are easy to translate one and forget
    the other, and then the same row carries two different numbering schemes."""
    _, geneData = _run(rng, tmp_path)
    allowed = {0} | set(range(101, 117))

    assert set(geneData.neighbour) <= allowed
    flat = {int(v) for row in geneData.neighbour_array for v in row}
    assert flat <= allowed


@pytest.mark.slow
def test_the_two_geneData_columns_agree(rng, tmp_path):
    """neighbour is the most likely parent, so it has to be one of the candidates
    in neighbour_array. It would not be if only one of them got translated."""
    _, geneData = _run(rng, tmp_path)
    for nbr, arr in zip(geneData.neighbour, geneData.neighbour_array):
        assert nbr in list(arr)


# ------------------------------------------- the physical containment column

@pytest.mark.slow
def test_geneData_carries_the_cell_the_spot_sits_in(rng, tmp_path):
    """geneData.inside_cell is the cell whose mask the spot falls in, 0 for none. It
    goes through the same translation as the other two, so it is segmentation labels."""
    _, geneData = _run(rng, tmp_path)
    assert 'inside_cell' in geneData.columns
    assert set(geneData.inside_cell) <= {0} | set(range(101, 117))


@pytest.mark.slow
def test_inside_cell_matches_the_label_image(rng, tmp_path):
    """Exact, not approximate. Read the label image back at each spot's pixel and it
    has to agree, otherwise the column is not worth having."""
    _, geneData = _run(rng, tmp_path)
    lab = _label_image()
    from_image = lab[geneData.y.astype(int), geneData.x.astype(int)]
    assert (geneData.inside_cell.values == from_image).all()


@pytest.mark.slow
def test_inside_cell_is_not_the_same_thing_as_neighbour(rng, tmp_path):
    """The point of the column. neighbour is what the model decided, inside_cell is
    where the segmentation put it, and on a real run they disagree for some spots. If
    these two ever matched everywhere the column would be telling us nothing new."""
    _, geneData = _run(rng, tmp_path)
    assert (geneData.inside_cell.values != geneData.neighbour.values).any()


@pytest.mark.slow
def test_inside_cell_reaches_the_arrow_shards(rng, tmp_path):
    """_spots_arrow_table lists its columns one by one, so a new one that is not in
    that list is dropped without a word. The tsv would look right and the viewer
    would see nothing."""
    import pyarrow.feather as feather

    _run(rng, tmp_path, save=True)
    shards = list((tmp_path / 'pciSeq' / 'data' / 'viewer_data' / 'arrow_spots').glob('*.feather'))
    assert shards

    table = feather.read_table(shards[0])
    assert 'inside_cell' in table.schema.names, 'missing, add it to _spots_arrow_table'
    assert set(table.column('inside_cell').to_pylist()) <= {0} | set(range(101, 117))


# --------------------------------------------- the methods on the model

def test_the_model_converts_both_ways(minimal_varbayes):
    """obj.to_internal / obj.to_external, so a user holding the pickle needs no
    import and does not have to dig label_map out of config themselves."""
    obj = minimal_varbayes
    obj.config['label_map'] = LABEL_MAP

    assert obj.to_internal(102) == 2
    assert obj.to_external(2) == 102
    assert obj.to_external(obj.to_internal(103)) == 103
    assert obj.to_internal([101, 102]) == [1, 2]


def test_the_model_methods_are_a_no_op_without_a_map(minimal_varbayes):
    obj = minimal_varbayes
    obj.config['label_map'] = None
    assert obj.to_internal(102) == 102
    assert obj.to_external(102) == 102


@pytest.mark.slow
def test_nC_counts_the_background_row(rng, tmp_path):
    """The docs quote the internal labels as 1..nC-1, so pin the off-by-one they are
    quoting. nC counts the background at row 0, so 16 segmented cells give nC 17 and
    the last real cell is 16, not 17."""
    from pciSeq.app import cell_type
    from pciSeq.src.validation import validate_inputs
    from pciSeq.src.preprocess.main import stage_data

    spots = pd.DataFrame({
        'gene_name': rng.choice(GENES, 900),
        'x': rng.uniform(0, 79, 900).astype(np.float32),
        'y': rng.uniform(0, 79, 900).astype(np.float32),
    })
    scref = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50,
                         index=GENES, columns=CLASSES)
    scref.index.name = 'gene_name'
    s, coo, scd, cfg = validate_inputs(
        spots, coo_matrix(_label_image()), scref,
        {'max_iter': 2, 'save_data': False, 'output_path': str(tmp_path)})
    cells, _borders, sp, label_map = stage_data(s, coo, cfg)
    _, _, obj = cell_type(cells, sp, scd, cfg)

    n_real = len(set(label_map)) - 1          # the map carries the background 0 too
    assert n_real == 16
    assert obj.nC == n_real + 1
    assert obj.cells.classProb.shape[0] == obj.nC
    assert max(label_map.values()) == obj.nC - 1
