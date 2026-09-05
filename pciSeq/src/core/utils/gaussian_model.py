"""The gaussian each cell's spots are taken to be scattered around."""

import logging
import numpy as np
import pandas as pd
import numpy_groupies as npg

logger = logging.getLogger(__name__)


def empirical_mean(spots, cells):

    # get the total gene counts per cell
    N_c = cells.total_counts

    xyz_spots = spots.xyz_coords
    prob = spots.parent_cell_prob
    n = cells.config['nNeighbors'] + 1

    # multiply the x coord of the spots by the cell prob
    a = np.tile(xyz_spots[:, 0], (n, 1)).T * prob

    # multiply the y coord of the spots by the cell prob
    b = np.tile(xyz_spots[:, 1], (n, 1)).T * prob

    # multiply the z coord of the spots by the cell prob
    c = np.tile(xyz_spots[:, 2], (n, 1)).T * prob

    # aggregated x and y coordinate
    idx = spots.parent_cell_id
    x_agg = npg.aggregate(idx.ravel(), a.ravel(), size=len(N_c))
    y_agg = npg.aggregate(idx.ravel(), b.ravel(), size=len(N_c))
    z_agg = npg.aggregate(idx.ravel(), c.ravel(), size=len(N_c))

    # get the estimated cell centers
    x_bar = np.nan * np.ones(N_c.shape)
    y_bar = np.nan * np.ones(N_c.shape)
    z_bar = np.nan * np.ones(N_c.shape)

    x_bar[N_c > 0] = x_agg[N_c > 0] / N_c[N_c > 0]
    y_bar[N_c > 0] = y_agg[N_c > 0] / N_c[N_c > 0]
    z_bar[N_c > 0] = z_agg[N_c > 0] / N_c[N_c > 0]

    # cells with N_c = 0 will end up with x_bar = y_bar = np.nan
    xyz_bar_fitted = np.array(list(zip(x_bar.T, y_bar.T, z_bar.T)))

    # if you have a value for the estimated centroid use that, otherwise
    # use the initial (starting values) centroids
    ini_cent = cells.ini_centroids()
    xyz_bar = np.array(tuple(zip(*[ini_cent['x'], ini_cent['y'], ini_cent['z']])))

    # # sanity check. NaNs or Infs should appear together
    # assert np.all(np.isfinite(x_bar) == np.isfinite(y_bar))
    # use the fitted centroids where possible otherwise use the initial ones
    xyz_bar[np.isfinite(x_bar)] = xyz_bar_fitted[np.isfinite(x_bar)]
    return pd.DataFrame(xyz_bar, columns=['x', 'y', 'z'], dtype=np.float32)


def expected_covariance(scale_matrix, dof):
    """
        Calculate the expected covariance matrix from a scale matrix and degrees of freedom.

        Parameters
        ----------
        scale_matrix : np.ndarray
            Scale matrix of shape (C, d, d) where d must be 2 or 3
        dof : np.ndarray
            Degrees of freedom,shape (C,).
            Values will be automatically adjusted if below d + 2

        Returns
        -------
        np.ndarray
            Expected covariance matrix of same shape as input scale_matrix

        Raises
        ------
        ValueError
            If matrix dimensions are invalid or don't match
    """
    # Get the last two dimensions
    *_, d1, d2 = scale_matrix.shape

    # Check square
    if d1 != d2:
        raise ValueError(f"scale_matrix must be square, got shape {scale_matrix}")

    # Check dimension is 2 or 3
    if d1 not in (2, 3):
        raise ValueError(f"scale_matrix dimension must be 2 or 3, got {d1}")

    # Adjust degrees of freedom if needed, maybe I should drop a warning?
    min_dof = d1 + 1
    dof[dof <= min_dof] = min_dof + 1

    return scale_matrix / (dof[:, None, None] - d1 - 1)
