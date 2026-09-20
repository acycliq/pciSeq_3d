"""
coo can be handed to fit as a plain numpy label image, not only as sparse matrices.

A 3D array (planes, height, width) always worked. A 2D one (height, width) did not:
parse_args looped over its first axis, so a 4 by 5 image came back as four "planes"
of shape (1, 5), one per pixel row, and nothing raised. The run would then carry on
as if it was a tall 3D stack.

These only call parse_args, never fit, so they are quick and leave the log alone.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from pciSeq.app import parse_args


def _spots():
    """parse_args only passes the spots through, so anything will do."""
    return pd.DataFrame({'x': [1.0], 'y': [1.0], 'gene_name': ['a']})


def _label_img():
    """A 4 by 5 label image with two cells in it."""
    img = np.zeros((4, 5), dtype=np.uint32)
    img[0:2, 0:2] = 1
    img[2:4, 3:5] = 2
    return img


def test_a_2d_array_is_one_plane():
    img = _label_img()
    _, coo, _, _ = parse_args(spots=_spots(), coo=img)

    assert len(coo) == 1
    assert isinstance(coo[0], coo_matrix)
    assert coo[0].shape == (4, 5)
    # and the labels come through untouched
    assert np.array_equal(coo[0].toarray(), img)


def test_a_3d_array_is_one_matrix_per_plane():
    stack = np.stack([_label_img(), _label_img() * 0, _label_img()])
    _, coo, _, _ = parse_args(spots=_spots(), coo=stack)

    assert len(coo) == 3
    assert all(m.shape == (4, 5) for m in coo)
    for plane, m in zip(stack, coo):
        assert np.array_equal(m.toarray(), plane)


def test_the_array_can_be_positional_too():
    _, coo, _, _ = parse_args(_spots(), _label_img())
    assert len(coo) == 1
    assert coo[0].shape == (4, 5)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4, 5)])
def test_any_other_number_of_dimensions_is_refused(shape):
    with pytest.raises(ValueError, match="2D .* or 3D"):
        parse_args(spots=_spots(), coo=np.zeros(shape, dtype=np.uint32))


def test_a_list_of_sparse_matrices_is_left_alone():
    planes = [coo_matrix(_label_img()), coo_matrix(_label_img())]
    _, coo, _, _ = parse_args(spots=_spots(), coo=planes)
    assert coo is planes
