"""
Checks stage_image on a wide, a tall and a square image.

When no zoom_levels is given, the deepest level should be the first one that is big
enough to hold the image, whichever side is the longer one. Each image carries a white
block in a corner that is not symmetric, so a pyramid that came out flipped, swapped
or shifted would put the block in the wrong place.

The tiles are jpg, so pixel values are compared loosely.
"""
import sqlite3

import numpy as np
import pytest

from pciSeq.src.tiling.read_tiles import read_tiles

try:
    from pciSeq.src.tiling.stage_image import stage_image, tile_maker, _fit_zoom_level
except (ImportError, OSError):          # no pyvips, or no libvips behind it
    pytest.skip("libvips is not available", allow_module_level=True)

# (width, height, the level expected). Level z is 256 * 2**z on the longer side.
SHAPES = {
    "wide": (700, 300, 2),
    "tall": (300, 700, 2),
    "square": (600, 600, 2),
}


def _image(w, h):
    """Black, with a white block near the top right corner."""
    a = np.zeros((h, w), dtype=np.uint8)
    a[_block(w, h)] = 255
    return a


def _block(w, h):
    """rows 10% to 30% from the top, columns 70% to 90% from the left"""
    return slice(h // 10, 3 * h // 10), slice(7 * w // 10, 9 * w // 10)


def _levels(path):
    con = sqlite3.connect(path)
    try:
        meta = dict(con.execute("select name, value from metadata"))
        levels = [r[0] for r in con.execute("select distinct zoom_level from tiles order by 1")]
    finally:
        con.close()
    return meta, levels


@pytest.mark.parametrize("w, h, level", [
    (256, 256, 0), (257, 100, 1), (512, 512, 1), (100, 513, 2),
    (6408, 4382, 5), (4382, 6408, 5), (8192, 8192, 5),
    (10000, 10000, 6),
])
def test_fit_zoom_level(w, h, level):
    assert _fit_zoom_level(w, h) == level


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("use_buffer", [True, False])
def test_level_is_picked_from_the_longer_side(tmp_path, shape, use_buffer):
    w, h, level = SHAPES[shape]
    path = stage_image(_image(w, h), out_dir=str(tmp_path), name=shape,
                       use_buffer=use_buffer, progress=False)

    meta, levels = _levels(path)
    assert levels == list(range(level + 1))
    assert int(meta["maxzoom"]) == level
    assert (int(meta["width"]), int(meta["height"])) == (w, h)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_picture_lands_where_it_should(tmp_path, shape):
    w, h, _ = SHAPES[shape]
    path = stage_image(_image(w, h), out_dir=str(tmp_path), name=shape, progress=False)

    im, scale = read_tiles(path, plane=0)          # one pixel per image pixel
    got = np.asarray(im.convert("L"), dtype=float)
    assert got.shape == (h, w)
    assert scale == pytest.approx(1.0)

    rows, cols = _block(w, h)
    inside = got[rows.start + 5:rows.stop - 5, cols.start + 5:cols.stop - 5]
    outside = got.copy()
    outside[rows.start - 5:rows.stop + 5, cols.start - 5:cols.stop + 5] = 0
    assert inside.mean() > 230
    assert outside.max() < 40


def test_zoom_levels_given_by_the_user_is_kept(tmp_path):
    path = stage_image(_image(700, 300), out_dir=str(tmp_path), name="deep",
                       zoom_levels=4, progress=False)
    meta, levels = _levels(path)
    assert levels == [0, 1, 2, 3, 4]
    assert int(meta["maxzoom"]) == 4


def test_tile_maker_says_which_level_it_wrote(tmp_path):
    info = tile_maker(_image(300, 700), out_dir=str(tmp_path / "tiles"))
    assert info["zoom_levels"] == 2
    assert info["original_dims"] == [300, 700]
