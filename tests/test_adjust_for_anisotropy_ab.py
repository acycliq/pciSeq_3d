"""
Checks that adjust_for_anisotropy gives the same spots frame as before.

It now builds the frame one column at a time. Before, it stacked the gene
names and the numbers into one numpy array, which turns every number into a
python object, and then cast them all back. That older code is kept below, in
_adjust_for_anisotropy_before, and the two must give the very same frame:
same columns, same dtypes, same index, same values.
"""
import numpy as np
import pandas as pd
import pytest

from pciSeq.src.core.utils.geometry import adjust_for_anisotropy, anisotropy_calc


def _adjust_for_anisotropy_before(spots, voxel_size):
    """NOTE: this is adjust_for_anisotropy before the frame was built column by
    column. It is not used anywhere in pciSeq anymore, it lives here only to
    check the new code against."""
    gene_col = spots.gene_name.values[:, None]
    z_plane = spots.z_plane.values[:, None].astype(np.float32)
    score = spots.score.values[:, None].astype(np.float32)
    intensity = spots.intensity.values[:, None].astype(np.float32)
    coords = spots[['x', 'y', 'z_plane']].values
    scaled_coords = anisotropy_calc(coords, voxel_size)
    data_adj = np.hstack([gene_col, scaled_coords, z_plane, score, intensity])
    return pd.DataFrame(
        data=data_adj,
        columns=['gene_name', 'x', 'y', 'z', 'z_plane', 'score', 'intensity']
    ).astype({
        'gene_name': str,
        'x': np.float32,
        'y': np.float32,
        'z': np.float32,
        'z_plane': np.float32,
        'score': np.float32,
        'intensity': np.float32,
    })


def _spots(n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    genes = np.array(['g%03d' % i for i in range(40)], dtype=object)
    return pd.DataFrame({
        'gene_name': genes[rng.integers(0, 40, n)],
        'x': (rng.random(n) * 6000).astype(dtype),
        'y': (rng.random(n) * 8000).astype(dtype),
        'z_plane': rng.integers(0, 69, n).astype(dtype),
        'score': rng.random(n).astype(dtype),
        'intensity': rng.random(n).astype(dtype),
    })


def _check(spots, voxel_size):
    before = _adjust_for_anisotropy_before(spots, voxel_size)
    now = adjust_for_anisotropy(spots, voxel_size)
    pd.testing.assert_frame_equal(now, before, check_exact=True)
    assert list(now.dtypes) == list(before.dtypes)
    # the gene names are plain python strings in both
    assert type(now.gene_name.iloc[0]) is type(before.gene_name.iloc[0])


@pytest.mark.parametrize('voxel_size', [[1, 1, 1], [0.28, 0.28, 0.7], [0.5, 0.4, 2.0]])
def test_same_frame_from_float32_spots(voxel_size):
    _check(_spots(50_000, np.float32), voxel_size)


@pytest.mark.parametrize('voxel_size', [[1, 1, 1], [0.28, 0.28, 0.7]])
def test_same_frame_from_float64_spots(voxel_size):
    # a spots frame that never went through the validator, so still float64
    _check(_spots(50_000, np.float64), voxel_size)


def test_same_frame_when_the_index_has_gaps():
    # remove_oob drops rows before this runs, so the index coming in has holes.
    # Both versions hand back a fresh 0..N-1 index
    spots = _spots(1000, np.float32).iloc[::3]
    _check(spots, [0.28, 0.28, 0.7])
    assert adjust_for_anisotropy(spots, [0.28, 0.28, 0.7]).index.equals(pd.RangeIndex(len(spots)))

