"""
Checks that the nearest cells found with scipy's KDTree are the ones scikit-learn's
ball tree found.

Cells.nearest_cells is used twice at the start of a run: for every spot, its
nNeighbors + 1 nearest cells (they become the spot's candidate parents and are
never changed after that), and for every cell, its nearest cells (the
neighbourhood of the spatial prior). It used to be scikit-learn's
NearestNeighbors. That older code is kept below, in _nearest_cells_before, and
the two must give the same cells in the same order at the same distances.

The one exception is ties. When two cells are at exactly the same distance from
a point neither library promises an order. The background, cell 0, is the case
that always happens: it is parked at -sys.maxsize, every real cell is equally
far from it, so its own row of neighbours is arbitrary. That row is left out of
the comparison. It is never used for a real cell, see the last test.
"""
import sys

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from pciSeq.src.core.datatypes.cells import Cells


def _nearest_cells_before(zyx_coords, zyx, n):
    """NOTE: this is how the nearest cells were found before KDTree (the body of
    Cells.nn followed by kneighbors). It is not used anywhere in pciSeq anymore,
    it lives here only to check the new code against."""
    nbrs = NearestNeighbors(n_neighbors=n, algorithm='ball_tree').fit(zyx_coords)
    return nbrs.kneighbors(zyx)


class _FakeCells:
    """Just enough of Cells to call nearest_cells on a set of coordinates."""
    nn = Cells.nn
    nearest_cells = Cells.nearest_cells
    nearest_neighbours = Cells.nearest_neighbours

    def __init__(self, zyx_coords, nNeighbors):
        self.zyx_coords = zyx_coords
        self.config = {'nNeighbors': nNeighbors}


def _cells(nC, d, seed, spread=3000.0):
    """Cell centroids like read_image_objects makes them: float32, with the
    background in row 0 at -sys.maxsize. In 2D the z column is all zeros."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0, spread, (nC - 1, 3))
    if d == 2:
        xyz[:, 0] = 0
    return np.vstack(([[-sys.maxsize] * 3], xyz)).astype(np.float32)


@pytest.mark.parametrize('d', [2, 3])
@pytest.mark.parametrize('nNeighbors', [6, 9])
def test_same_cells_for_the_spots(d, nNeighbors):
    zyx_coords = _cells(4000, d, 0)
    rng = np.random.default_rng(1)
    spots = rng.uniform(0, 3000, (60_000, 3)).astype(np.float32)
    if d == 2:
        spots[:, 0] = 0
    dist_0, idx_0 = _nearest_cells_before(zyx_coords, spots, nNeighbors + 1)
    dist, idx = _FakeCells(zyx_coords, nNeighbors).nearest_cells(spots)

    assert idx.shape == idx_0.shape == (60_000, nNeighbors + 1)
    assert np.array_equal(idx, idx_0)
    assert np.array_equal(dist, dist_0)
    assert idx.dtype == idx_0.dtype and dist.dtype == dist_0.dtype


@pytest.mark.parametrize('d', [2, 3])
def test_same_cells_for_the_cells(d):
    zyx_coords = _cells(4000, d, 2)
    dist_0, idx_0 = _nearest_cells_before(zyx_coords, zyx_coords, 10)
    dist, idx = _FakeCells(zyx_coords, 9).nearest_cells(zyx_coords)

    # every real cell: same neighbours, same order, same distances
    assert np.array_equal(idx[1:], idx_0[1:])
    assert np.array_equal(dist[1:], dist_0[1:])
    # the background's own row is the tie case. Same distances, any cells
    assert np.array_equal(dist[0], dist_0[0])
    assert idx[0, 0] == 0


def test_the_background_is_never_a_neighbour_of_a_real_cell():
    # this is why its arbitrary row does not matter to any real cell
    zyx_coords = _cells(4000, 3, 3)
    nbrs = _FakeCells(zyx_coords, 9).nearest_neighbours()
    assert nbrs['indices'].shape == (4000, 9)
    assert not (nbrs['indices'][1:] == 0).any()
    # and no cell is its own neighbour
    assert not (nbrs['indices'][1:] == np.arange(1, 4000)[:, None]).any()


def test_too_few_cells_is_an_error():
    # scikit-learn raised here. KDTree would hand back an index one past the last
    # cell and carry on, so the check is done by hand
    zyx_coords = _cells(5, 3, 4)
    with pytest.raises(ValueError):
        _nearest_cells_before(zyx_coords, zyx_coords, 10)
    with pytest.raises(ValueError):
        _FakeCells(zyx_coords, 9).nearest_cells(zyx_coords)


def test_same_cells_on_a_model(minimal_varbayes):
    vb = minimal_varbayes
    n = vb.config['nNeighbors'] + 1
    spotZYX = vb.spots.data[['z', 'y', 'x']].values
    dist_0, idx_0 = _nearest_cells_before(vb.cells.zyx_coords, spotZYX, n)
    dist, idx = vb.cells.nearest_cells(spotZYX)
    assert np.array_equal(idx, idx_0) and np.array_equal(dist, dist_0)

    vb.initialise_state()
    expected = idx_0.copy()
    expected[:, -1] = 0                     # the last column is the background
    assert np.array_equal(vb.spots.parent_cell_id, expected)
