"""
Checks that the batch geometry functions give exactly what the one cell at a
time functions give.

cells_summary used to loop over the cells and call gaussian_contour and
gaussian_ellipsoid_props on each. It now calls gaussian_contours and
gaussian_ellipsoids_props once for all the cells. The one cell functions are
still in geometry.py as the reference, and the loops cells_summary had around
them are kept below. The numbers end up in cellData (gaussian_contour,
sphere_scale, sphere_rotation), so they must agree to the last bit, not to a
tolerance.
"""
import numpy as np
import pytest

from pciSeq.src.core import summary
from pciSeq.src.core.utils.geometry import (
    gaussian_contour, gaussian_contours, gaussian_ellipsoid_props, gaussian_ellipsoids_props,
)


def _contours_before(centroid_df, covs):
    """NOTE: this is the loop cells_summary had for the gaussian_contour column
    before gaussian_contours replaced it. It is not used anywhere in pciSeq
    anymore, it lives here only to check the new code against."""
    contour = []
    for i in range(len(covs)):
        mu = centroid_df.iloc[i].tolist()
        cov = covs[i]
        ellipsis = gaussian_contour(mu[:2], cov[:2, :2], 3).astype(np.int64)
        contour.append(ellipsis.tolist())
    return contour


def _sphere_props_before(covs):
    """NOTE: this is summary.sphere_props as it was before
    gaussian_ellipsoids_props replaced the loop. Not used anywhere in pciSeq
    anymore, here only as the check."""
    sphere_scale = []
    sphere_rotation = []
    for i in range(len(covs)):
        cov = covs[i]
        scale, rotation = gaussian_ellipsoid_props(cov)
        sphere_scale.append(scale)
        sphere_rotation.append(rotation)
    return sphere_scale, sphere_rotation


def _random_covs(n, d, seed, dtype=np.float64):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d, d)) * rng.uniform(0.5, 30, (n, 1, 1))
    covs = a @ np.transpose(a, (0, 2, 1)) + 1e-3 * np.eye(d)
    return covs.astype(dtype)


def _odd_covs(d):
    """The awkward ones: a perfect sphere (all eigenvalues tie), axis aligned,
    two equal axes, a strong negative correlation, nearly flat, and noise below
    the 1e-10 cut that gaussian_ellipsoid_props zeroes."""
    out = [np.eye(d) * 25.0, np.diag([100.0, 4.0, 1.0][:d]), np.diag([9.0, 9.0, 2.0][:d])]
    neg = np.eye(d) * 10.0
    neg[0, 1] = neg[1, 0] = -8.0
    out.append(neg)
    flat = np.eye(d) * 50.0
    flat[d - 1, d - 1] = 1e-6
    out.append(flat)
    noisy = np.eye(d) * 16.0
    noisy[0, d - 1] = noisy[d - 1, 0] = 3e-11
    out.append(noisy)
    return np.array(out)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_same_ellipsoid_props_on_random_covariances(dtype):
    covs = _random_covs(3000, 3, 0, dtype)
    assert gaussian_ellipsoids_props(covs) == _sphere_props_before(covs)


def test_same_ellipsoid_props_on_the_awkward_ones():
    covs = _odd_covs(3)
    now = gaussian_ellipsoids_props(covs)
    assert now == _sphere_props_before(covs)
    # plain python floats all the way down, cellData.tsv is printed from these
    assert all(type(v) is float for part in now for row in part for v in row)


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def test_same_contours_on_random_covariances(dtype):
    import pandas as pd
    rng = np.random.default_rng(1)
    covs = _random_covs(3000, 3, 2, dtype)
    cent = pd.DataFrame(rng.uniform(0, 6000, (3000, 3)).astype(np.float32), columns=['x', 'y', 'z'])
    now = gaussian_contours(cent.values[:, :2], covs[:, :2, :2], 3).astype(np.int64).tolist()
    assert now == _contours_before(cent, covs)


def test_same_contours_on_the_awkward_ones_and_in_2d():
    import pandas as pd
    for d in (2, 3):
        covs = _odd_covs(d)
        cent = pd.DataFrame(np.arange(len(covs) * d, dtype=np.float32).reshape(len(covs), d) * 37.5,
                            columns=['x', 'y', 'z'][:d])
        now = gaussian_contours(cent.values[:, :2], covs[:, :2, :2], 3)
        assert now.dtype == np.float32 and now.shape == (len(covs), 40, 2)
        assert now.astype(np.int64).tolist() == _contours_before(cent, covs)


def test_no_cells_is_fine():
    assert gaussian_ellipsoids_props(np.zeros((0, 3, 3))) == ([], [])
    assert gaussian_contours(np.zeros((0, 2)), np.zeros((0, 2, 2))).shape == (0, 40, 2)


def test_same_columns_on_a_model(minimal_varbayes):
    vb = minimal_varbayes
    vb.initialise_state()
    for _ in range(3):
        for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd, vb.gamma_upd,
                     vb.cell_to_cellType, vb.spots_to_cell_numba, vb.gaussian_upd):
            step()
    cells = vb.cells
    now = gaussian_contours(cells.centroid.values[:, :2], cells.cov[:, :2, :2], 3).astype(np.int64).tolist()
    assert now == _contours_before(cells.centroid, cells.cov)
    if cells.cov.shape[1] == 3:
        assert summary.sphere_props(cells) == _sphere_props_before(cells.cov)
