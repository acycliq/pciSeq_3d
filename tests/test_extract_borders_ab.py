"""
Checks that extract_borders gives the same result as extract_borders_dip.

They are two implementations of the same thing: extract_borders_dip is the plain
single threaded version (the reference), extract_borders is the threaded one that
everything actually calls. This test keeps them in sync, so a change to one that
diverges from the other is caught here rather than showing up as wrong cell
outlines in the viewer.
"""
import numpy as np
import pytest

from pciSeq.src.preprocess.cell_processing import extract_borders, extract_borders_dip


def _label_plane(size, n_cells, seed):
    """A plane of round blobs, the shape a segmentation actually produces."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.uint32)
    yy, xx = np.ogrid[:size, :size]
    for lab in range(1, n_cells + 1):
        cy, cx = rng.integers(15, size - 15, 2)
        r = rng.integers(4, 11)
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = lab
    return img


@pytest.mark.parametrize('size,n_cells,seed', [(200, 25, 0), (400, 90, 1), (600, 200, 2)])
def test_the_two_agree(size, n_cells, seed):
    img = _label_plane(size, n_cells, seed)
    a = extract_borders_dip(img).sort_values('label').reset_index(drop=True)
    b = extract_borders(img).sort_values('label').reset_index(drop=True)

    assert list(a.label) == list(b.label), 'different set of labels'
    for lab, ca, cb in zip(a.label, a.coords, b.coords):
        assert list(ca) == list(cb), 'coords differ for label %s' % lab


def test_an_empty_plane_is_handled():
    img = np.zeros((50, 50), dtype=np.uint32)
    assert len(extract_borders_dip(img)) == len(extract_borders(img))


def test_the_polygons_come_back_closed():
    """Both append the first point at the end so the ring closes."""
    img = _label_plane(200, 10, 3)
    for fn in (extract_borders_dip, extract_borders):
        for coords in fn(img).coords:
            assert coords[0] == coords[-1], '%s left a polygon open' % fn.__name__
