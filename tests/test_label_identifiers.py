"""The two cell identifiers, and the translation between them.

pciSeq has two numbers for the same cell. The segmentation label is whatever came
in on the label image, which is what the user recognises and what every flat file
reports. The internal label is 1..nC, which is what indexes the model arrays.

They are only different when the input labels were not already sequential, so a
missing translation is correct on every tidy test dataset and wrong on a real
segmentation with gaps in it. That is why these tests use labels 101..116: they
do not overlap 1..nC at all, so a skipped translation cannot pass by luck.
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


def _run(rng, tmp_path):
    spots = pd.DataFrame({
        'gene_name': rng.choice(GENES, 900),
        'x': rng.uniform(0, 79, 900).astype(np.float32),
        'y': rng.uniform(0, 79, 900).astype(np.float32),
    })
    scref = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50,
                         index=GENES, columns=CLASSES)
    scref.index.name = 'gene_name'
    opts = {'max_iter': 2, 'save_data': False, 'CellCallTolerance': 0.5,
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
