"""Tests for VarBayes._roi_volume, the denominator of the background density.

The background misread density is N_0,g / A_total and it gets compared against
the cell gaussians in the same softmax. Those gaussians live on anisotropy
scaled coordinates, so A_total has to be the scaled volume too. Counting raw
voxels leaves the two sides in different units, the background comes out too
strong, and too many spots end up as background instead of going to a cell.
"""

import types

import pytest

from pciSeq.src.core.main import VarBayes


def _roi(w, h, n_planes, voxel_size):
    """Call _roi_volume with a stub holding only the config it reads."""
    stub = types.SimpleNamespace(config={
        'img_dim': {'w': w, 'h': h, 'n_planes': n_planes},
        'voxel_size': voxel_size,
    })
    return VarBayes._roi_volume(stub)


def test_isotropic_voxels_leave_the_volume_alone():
    # the default, and every 2d run. Nothing should change for these.
    assert _roi(100, 50, 1, [1, 1, 1]) == pytest.approx(5000)
    assert _roi(100, 50, 10, [1, 1, 1]) == pytest.approx(50000)


def test_thick_planes_stretch_the_volume():
    # a plane twice as thick as a pixel is wide counts double
    assert _roi(100, 50, 1, [1, 1, 2]) == pytest.approx(10000)


def test_the_yao_voxel():
    # 0.28 x 0.28 x 0.7 gives Sz = 2.5, the config this was found on
    assert _roi(100, 50, 1, [0.28, 0.28, 0.7]) == pytest.approx(12500)


def test_only_z_is_corrected():
    """Pins down the assumption rather than the ideal.

    We correct z only, taking the pixels to be square in xy. anisotropy_calc
    does also scale y by voxel_size[1]/voxel_size[0], so a dataset with non
    square pixels would need that factor too and this volume would be out by
    it. No such dataset has come up, and this test is here so the assumption is
    visible if one ever does.
    """
    assert _roi(100, 50, 1, [1, 2, 2]) == pytest.approx(10000)


def test_scaling_is_relative_to_x():
    """Only the ratios matter, not the units. The same physical voxel declared
    in different units has to give the same answer, because the coordinates are
    scaled relative to x either way."""
    microns = _roi(100, 50, 4, [0.28, 0.28, 0.7])
    nanometres = _roi(100, 50, 4, [280, 280, 700])
    assert microns == pytest.approx(nanometres)
