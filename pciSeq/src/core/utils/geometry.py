"""Geometric calculation utilities for cell and spot analysis."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
import logging

# Configure logging
logger = logging.getLogger(__name__)


def gaussian_contour(mu: Union[List, np.ndarray],
                     cov: np.ndarray,
                     sdwidth: float = 3) -> np.ndarray:
    """Draw an ellipsoid for a given covariance matrix and mean vector.

    Args:
        mu: Mean vector [x, y]
        cov: 2x2 covariance matrix
        sdwidth: Number of standard deviations to include (default: 3)

    Returns:
        Array of contour points (N x 2)

    Example:
        cov_1 = [[1, 0.5], [0.5, 1]]
        means_1 = [1, 1]
        ellipsis_1 = gaussian_contour(means_1, cov_1)
        plt.plot(ellipsis_1[0], ellipsis_1[1])
    """
    mu = np.array(mu)

    # Generate points around unit circle
    npts = 40
    tt = np.linspace(0, 2 * np.pi, npts)
    ap = np.zeros((2, npts))
    ap[0, :] = np.cos(tt)
    ap[1, :] = np.sin(tt)

    # Transform by eigenvalues/vectors of covariance
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = sdwidth * np.sqrt(eigvals)
    eigvals = eigvals * np.eye(2)

    # Generate ellipse points
    vd = eigvecs.dot(eigvals)
    out = vd.dot(ap) + mu.reshape(2, -1)

    return np.array(list(zip(*out)), dtype=np.float32)


def euler_angles_OLD(r: np.ndarray) -> List[float]:
    """Calculate Euler angles from rotation matrix.

    Args:
        r: 3x3 rotation matrix

    Returns:
        List of angles [theta_x, theta_y, theta_z]
    """
    theta_x = np.arctan2(r[2, 1], r[2, 2])
    theta_y = np.arcsin(r[2, 0])
    theta_z = np.arctan2(r[1, 0], r[0, 0])
    return [theta_x, theta_y, theta_z]


def gaussian_ellipsoid_props(cov: np.ndarray,
                             sdwidth: float = 3) -> Tuple[List[float], List[float]]:
    # Perform Eigen decomposition (PCA)
    # zero the numerical noise. abs, not cov < tol: that was zeroing every negative
    # off-diagonal too, so a fitted covariance with a negative correlation came out
    # as a different matrix
    tol = 1.0e-10
    cov = np.where(np.abs(cov) < tol, 0, cov)
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]  # Sort the eigenvalues
    eigvecs = eigvecs[:, idx]  # Sort the eigenvectors

    scaling = sdwidth * np.sqrt(eigvals)

    # Ensure eigenvectors are orthonormal (should already be, but this fixes small numerical issues)
    eigvecs, _ = np.linalg.qr(eigvecs)  # QR decomposition to make sure they are orthogonal

    # Ensure a right-handed coordinate system
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1  # Flip the last eigenvector

    # Extract Euler angles from the rotation matrix
    r = eigvecs.T
    rotation = euler_angles(r)
    return scaling.tolist(), rotation


def euler_angles(r: NDArray[np.float32]) -> List[float]:
    """Extract Euler angles from a 3D rotation matrix using ZYX convention.

    Decomposes a rotation matrix into three sequential rotations around fixed
    axes (intrinsic rotations) following the ZYX (yaw-pitch-roll) convention:
    1. Rotation around Z axis (yaw/heading) - ψ (psi)
    2. Rotation around Y axis (pitch/attitude) - θ (theta)
    3. Rotation around X axis (roll/bank) - φ (phi)

    Parameters
    ----------
    r : NDArray[float32], shape (3, 3)
        Rotation matrix R where:
            R = Rz(ψ) @ Ry(θ) @ Rx(φ)
        Must be orthogonal with determinant 1

    Returns
    -------
    tuple of float
        Euler angles in radians (φ, θ, ψ) where:
            - φ (phi): rotation around X axis [-π/2, π/2]
            - θ (theta): rotation around Y axis [-π/2, π/2]
            - ψ (psi): rotation around Z axis [-π, π]
    """
    # Input validation
    if r.shape != (3, 3):
        raise ValueError("Input matrix must be 3x3")
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6):
        raise ValueError("Input matrix must be orthogonal")
    if not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError("Input matrix must have determinant 1")

    # Extract Euler angles
    # Note: np.arctan2 returns angles in [-π, π]
    phi = np.arctan2(r[2, 1], r[2, 2])  # X rotation (roll)
    # clip: rounding can push r[2, 0] a hair past 1 and arcsin gives nan
    theta = np.arcsin(-np.clip(r[2, 0], -1.0, 1.0))  # Y rotation (pitch)
    psi = np.arctan2(r[1, 0], r[0, 0])  # Z rotation (yaw)

    # Check for gimbal lock.
    # Gimbal lock occurs when θ = ±π/2, making φ and ψ indistinguishable.
    # if np.isclose(abs(theta), np.pi / 2, atol=1e-6):
    #     logger.warning("Warning: Gimbal lock detected (pitch = ±90°)")

    # plain python floats, not np.float32. On numpy 2 a float32 prints as
    # np.float32(...) and that text ends up in cellData.tsv, which read_tsv cant eval.
    # rounded to 7 digits, that is all the precision a float32 has anyway.
    return [round(float(phi), 7), round(float(theta), 7), round(float(psi), 7)]


def gaussian_contours(mus: np.ndarray, covs: np.ndarray, sdwidth: float = 3) -> np.ndarray:
    """gaussian_contour for many cells in one go.

    mus is (n, 2) and covs is (n, 2, 2). Returns (n, 40, 2) float32, row i being
    what gaussian_contour(mus[i], covs[i], sdwidth) returns. That function is the
    reference, tests/test_geometry_batch_ab.py checks the two agree to the last bit.

    cells_summary used to call gaussian_contour once per cell. Most of that time
    went on things that are the same for every cell (the points on the unit
    circle) or that numpy does for a whole stack at once (the eigen
    decomposition), and on building the result through python tuples.
    """
    mus = np.asarray(mus, dtype=np.float64)
    n = mus.shape[0]

    # Points around the unit circle, same for every cell. npts is for drawing
    # only, the number of points on a cell's outline. Must match gaussian_contour.
    npts = 40
    tt = np.linspace(0, 2 * np.pi, npts)
    ap = np.zeros((2, npts))
    ap[0, :] = np.cos(tt)
    ap[1, :] = np.sin(tt)

    out = np.empty((n, npts, 2), dtype=np.float64)
    if n == 0:
        return out.astype(np.float32)

    # all the eigen decompositions in one call
    eigvals, eigvecs = np.linalg.eig(covs)
    eigvals = sdwidth * np.sqrt(eigvals)
    eye = np.eye(2)
    for i in range(n):
        # these two products are left exactly as gaussian_contour has them, so the
        # rounding is the same and the numbers come out the same
        vd = eigvecs[i].dot(eigvals[i] * eye)
        out[i] = (vd.dot(ap) + mus[i].reshape(2, -1)).T
    return out.astype(np.float32)


def gaussian_ellipsoids_props(covs: np.ndarray, sdwidth: float = 3) -> Tuple[List[List[float]], List[List[float]]]:
    """gaussian_ellipsoid_props for many cells in one go.

    covs is (n, 3, 3). Returns two lists of n entries, the scaling and the
    rotation of every cell, entry i being what gaussian_ellipsoid_props(covs[i],
    sdwidth) returns. That function is the reference and has the comments on why
    each step is there, tests/test_geometry_batch_ab.py checks the two agree to
    the last bit. Every numpy call below takes the whole stack of matrices where
    the reference takes one.
    """
    covs = np.asarray(covs)
    if covs.shape[0] == 0:
        return [], []

    tol = 1.0e-10
    covs = np.where(np.abs(covs) < tol, 0, covs)
    eigvals, eigvecs = np.linalg.eig(covs)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals, axis=1)[:, ::-1]
    eigvals = np.take_along_axis(eigvals, idx, axis=1)
    eigvecs = np.take_along_axis(eigvecs, idx[:, None, :], axis=2)

    scaling = sdwidth * np.sqrt(eigvals)

    eigvecs, _ = np.linalg.qr(eigvecs)

    # Ensure a right-handed coordinate system
    flip = np.linalg.det(eigvecs) < 0
    eigvecs[flip, :, -1] *= -1

    # the checks of euler_angles, on all the matrices at once
    r = np.transpose(eigvecs, (0, 2, 1))
    if not np.allclose(r @ np.transpose(r, (0, 2, 1)), np.eye(3), atol=1e-6):
        raise ValueError("Input matrix must be orthogonal")
    if not np.isclose(np.linalg.det(r), 1, atol=1e-6).all():
        raise ValueError("Input matrix must have determinant 1")

    phi = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    theta = np.arcsin(-np.clip(r[:, 2, 0], -1.0, 1.0))
    psi = np.arctan2(r[:, 1, 0], r[:, 0, 0])

    # python's round on python floats, like euler_angles. np.round does not always
    # give the same last digit
    rotation = [[round(a, 7), round(b, 7), round(c, 7)]
                for a, b, c in zip(phi.tolist(), theta.tolist(), psi.tolist())]
    return scaling.tolist(), rotation


def ellipsoid_to_cov(scale: List[float],
                     rotation: List[float],
                     sdwidth: float = 3) -> np.ndarray:
    """The covariance matrix back from sphere_scale and sphere_rotation.

    The inverse of gaussian_ellipsoid_props, so the two cellData columns are
    enough to get the fitted covariance of a cell without the pickle.

    Parameters
    ----------
    scale : list of float, length 3
        The semi-axes of the sdwidth-sigma ellipsoid, cellData's sphere_scale.
    rotation : list of float, length 3
        Euler angles [phi, theta, psi] in radians, cellData's sphere_rotation,
        with R = Rz(psi) @ Ry(theta) @ Rx(phi) as in euler_angles.
    sdwidth : float
        How many standard deviations the scale covers. 3 in cellData.

    Returns
    -------
    ndarray, shape (3, 3)
        The covariance. Its diagonal is the variance along x, y, z.

    Example
    -------
        row = cellData.iloc[0]
        cov = ellipsoid_to_cov(row['sphere_scale'], row['sphere_rotation'])
    """
    phi, theta, psi = rotation
    c, s = np.cos(phi), np.sin(phi)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    c, s = np.cos(theta), np.sin(theta)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = np.cos(psi), np.sin(psi)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    r = rz @ ry @ rx
    # gaussian_ellipsoid_props took r = eigvecs.T, so the eigenvectors are the columns of r.T
    v = r.T
    var = (np.asarray(scale, dtype=np.float64) / sdwidth) ** 2
    return v @ np.diag(var) @ v.T


def adjust_for_anisotropy(
        spots: pd.DataFrame,
        voxel_size: Tuple[float, float, float]
) -> pd.DataFrame:
    """Adjust spot coordinates for non-cubic voxel dimensions.

    Transforms spatial coordinates to account for anisotropic voxels while
    preserving gene names and z-plane information. Applies scaling based on
    physical voxel dimensions.

    Parameters
    ----------
    spots : pd.DataFrame
        Spot data with columns:
            - gene_name : str, gene identifier
            - x : float, x coordinate
            - y : float, y coordinate
            - z_plane : float, z-plane index
            - score : float, spot caller score, carried through untouched
            - intensity : float, spot caller intensity, carried through untouched

    voxel_size : tuple of float, length 3
        Physical voxel dimensions (dx, dy, dz) in consistent units.
        Example: (0.5, 0.5, 2.0) for 0.5μm x 0.5μm x 2.0μm voxels

    Returns
    -------
    pd.DataFrame
        Adjusted spot data with columns:
            - gene_name : str
            - x : float32, scaled x coordinate
            - y : float32, scaled y coordinate
            - z : float32, scaled z coordinate
            - z_plane : float32, original z-plane index
    """

    # Scale the coordinates
    coords = spots[['x', 'y', 'z_plane']].values
    scaled_coords = anisotropy_calc(coords, voxel_size)

    # Build the frame one column at a time. The gene names must not be stacked
    # next to the numbers in one array: numpy can only do that by turning every
    # number into a python object, and they all have to be cast back afterwards.
    #
    # Note for future: consider preserving the original index from the input 'spots' DataFrame.
    # Keeping the index can be useful for filtering spots before calling pciSeq.fit().
    # The line below creates a new index, which can make it harder to exclude spots
    # if they were removed or cleaned earlier in the codebase.
    # Should be safe, but worth testing.
    return pd.DataFrame({
        'gene_name': spots.gene_name.astype(str).values,
        'x': scaled_coords[:, 0].astype(np.float32),
        'y': scaled_coords[:, 1].astype(np.float32),
        'z': scaled_coords[:, 2].astype(np.float32),
        'z_plane': spots.z_plane.values.astype(np.float32),
        'score': spots.score.values.astype(np.float32),
        'intensity': spots.intensity.values.astype(np.float32),
    })


def anisotropy_calc(data: np.ndarray,
                    voxel_size: Tuple[float, float, float],
                    inverse = False) -> np.ndarray:
    """Calculate anisotropic scaling for spot coordinates.

    Adjusts coordinates to account for different voxel dimensions in x, y, and z.
    All dimensions are scaled relative to the x dimension.

    Args:
        data: Array of shape (N, 3) containing spot coordinates [x, y, z]
        voxel_size: Physical dimensions of voxels as (x, y, z) in same units
        inverse: If True, return the original coordinates. If False, return the scaled coordinates. Default is False.

    Returns:
        np.ndarray: Scaled coordinates of shape (N, 3)

    Notes:
        - X dimension is used as reference (Sx = 1)
        - Y and Z are scaled relative to X
        - Output maintains same shape as input
        - Uses float32 for memory efficiency
    """
    # Unpack voxel dimensions
    x, y, z = voxel_size

    # Calculate scaling factors relative to x dimension
    Sx = x / x  # Always 1
    Sy = y / x  # Scale y relative to x
    Sz = z / x  # Scale z relative to x

    # Create anisotropic scaling matrix
    scaling_matrix = np.array([
        [Sx, 0, 0],
        [0, Sy, 0],
        [0, 0, Sz]
    ], dtype=np.float32)

    if not inverse:
        # Apply scaling: matrix multiplication then transpose back
        # data.T converts from (N,3) to (3,N) for matrix multiplication
        # final .T converts back to (N,3)
        return scaling_matrix.dot(data.T).T
    elif inverse:
        # does the inverse of the above.
        # Coordinates are already scaled relative to x dimension, and
        # we want to return the original coordinates.
        # In most cases where the voxel size differs only on z (ie Sx=Sy but Sz!=Sx)
        # that will be equivalent as dividing z by Sz/Sx
        S = scaling_matrix
        W = np.linalg.inv(S.T.dot(S)).dot(S.T)
        return W.dot(data.T).T
    else:
        raise ValueError('inverse must be True or False')


# The scaled z and the plane. pciSeq works in coordinates where z is stretched to
# the units of x, z = z_plane * voxel_size[2] / voxel_size[0] (anisotropy_calc
# above). That z is what the spot files, geneData and cellData carry, so it is
# NOT a plane number. These two go between the two.
#
# Plane p is the stretch from p up to p + 1, so a z goes to its plane by floor,
# never round. A spot with z_plane 21.56 has plane_id 21 (spot_processing.py),
# and a cell centroid at plane 21.56 is on plane 21 too. With round the cell tools
# said 22 while the viewer's Centroid Plane said 21, fixed in e427077.
#
# The 1e-4 keeps a z sitting exactly on a plane from landing one below when
# z / ratio comes out as 56.99999 in floating point.
#
# The viewer has twins of these: planeOf in electron/run.js and src/utils/planes.js
# in pciSeq_viewer. Keep all three the same; tests/test_geometry.py and the
# viewer's tests/planes.test.js check the same numbers.
def plane_of(z, voxel_size):
    """The plane a scaled z is on: floor(z * voxel_size[0] / voxel_size[2]).

    Takes a number or an array, gives an int or an int array.
    """
    vx, _, vz = voxel_size
    p = np.floor(np.asarray(z, dtype=np.float64) * vx / vz + 1e-4).astype(np.int64)
    return int(p) if p.ndim == 0 else p


def plane_to_z(plane, voxel_size):
    """The scaled z where a plane starts, plane * voxel_size[2] / voxel_size[0]."""
    vx, _, vz = voxel_size
    return np.asarray(plane, dtype=np.float64) * vz / vx





def get_img_shape(coo: List[coo_matrix]) -> List[int]:
    """Get image dimensions from list of sparse matrices.

    Args:
        coo: List of sparse coordinate matrices

    Returns:
        List of dimensions [n_planes, height, width]

    Raises:
        AssertionError: If matrices have different shapes
    """
    n = len(coo)
    img_shape = set([d.shape for d in coo])
    assert len(img_shape) == 1, 'pages do not have the same shape'
    img_shape = img_shape.pop()
    return [n, img_shape[0], img_shape[1]]
