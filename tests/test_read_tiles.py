"""
Checks read_tiles against a pyramid we build by hand.

The tiles here are made with pillow rather than stage_image, so the test runs without
libvips and without a real dataset. It puts a known pattern in the image, so a crop can
be checked against the pixels it is supposed to contain.
"""
import io
import sqlite3

import numpy as np
import pytest
from PIL import Image

from pciSeq.src.tiling.read_tiles import read_tiles

TILE = 256
W, H = 1000, 600          # a small "original image"
ZOOM_LEVELS = 3           # so the top level is 256 * 2**3 = 2048 on the long side


def _pattern(w, h):
    """A picture where every pixel says where it is: x in red, y in green."""
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[..., 0] = (xx * 255 // max(w - 1, 1)).astype(np.uint8)
    a[..., 1] = (yy * 255 // max(h - 1, 1)).astype(np.uint8)
    return Image.fromarray(a)


@pytest.fixture(scope="module")
def mbtiles(tmp_path_factory):
    """A pyramid built the same way stage_image builds one, in png so it stays exact."""
    path = tmp_path_factory.mktemp("tiles") / "pattern.mbtiles"
    con = sqlite3.connect(path)
    con.execute("create table tiles (plane_id integer, zoom_level integer, "
                "tile_column integer, tile_row integer, tile_data blob)")
    con.execute("create table metadata (name text, value text)")
    for k, v in {"format": "png", "minzoom": "0", "maxzoom": str(ZOOM_LEVELS),
                 "width": str(W), "height": str(H), "plane_count": "2"}.items():
        con.execute("insert into metadata values (?, ?)", (k, v))

    base = _pattern(W, H)
    for plane in (0, 1):
        for z in range(ZOOM_LEVELS + 1):
            side = TILE * 2 ** z
            factor = side / max(W, H)
            im = base.resize((round(W * factor), round(H * factor)), Image.LANCZOS)
            if plane == 1:                       # make the second plane tell itself apart
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            for ty in range((im.height + TILE - 1) // TILE):
                for tx in range((im.width + TILE - 1) // TILE):
                    tile = im.crop((tx * TILE, ty * TILE, (tx + 1) * TILE, (ty + 1) * TILE))
                    b = io.BytesIO()
                    tile.save(b, format="PNG")
                    con.execute("insert into tiles values (?, ?, ?, ?, ?)",
                                (plane, z, tx, ty, sqlite3.Binary(b.getvalue())))
    con.commit()
    con.close()
    return str(path)


def test_default_is_one_pixel_per_image_pixel(mbtiles):
    im, scale = read_tiles(mbtiles, bbox=(100, 50, 300, 150))
    assert im.size == (200, 100)
    assert scale == pytest.approx(1.0)


def test_whole_plane_when_no_bbox(mbtiles):
    im, scale = read_tiles(mbtiles)
    assert im.size == (W, H)
    assert scale == pytest.approx(1.0)


def test_max_width_sets_the_output_size(mbtiles):
    im, scale = read_tiles(mbtiles, bbox=(100, 50, 300, 150), width=400)
    assert im.size == (400, 200)
    assert scale == pytest.approx(2.0)


def test_crop_lands_where_it_should(mbtiles):
    """The pattern encodes position, so the corner pixels say which region came back."""
    x0, y0, x1, y1 = 400, 200, 500, 300
    im, _ = read_tiles(mbtiles, bbox=(x0, y0, x1, y1))
    a = np.asarray(im)
    assert abs(int(a[0, 0, 0]) - x0 * 255 // (W - 1)) <= 4        # red is x
    assert abs(int(a[0, 0, 1]) - y0 * 255 // (H - 1)) <= 4        # green is y
    assert abs(int(a[-1, -1, 0]) - (x1 - 1) * 255 // (W - 1)) <= 4
    assert abs(int(a[-1, -1, 1]) - (y1 - 1) * 255 // (H - 1)) <= 4


def test_bbox_is_clamped_to_the_image(mbtiles):
    im, _ = read_tiles(mbtiles, bbox=(-50, -50, W + 500, H + 500))
    assert im.size == (W, H)


def test_planes_are_separate(mbtiles):
    a, _ = read_tiles(mbtiles, plane=0, bbox=(0, 0, 100, 100))
    b, _ = read_tiles(mbtiles, plane=1, bbox=(0, 0, 100, 100))
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_zoom_argument_reads_that_level(mbtiles):
    # level 1 is 512 on the long side, so half of the image scale
    im, scale = read_tiles(mbtiles, bbox=(0, 0, W, H), zoom=1)
    assert scale == pytest.approx(1.0)      # still resampled back to 1:1
    im, _ = read_tiles(mbtiles, bbox=(0, 0, W, H), zoom=1, width=512)
    assert im.size[0] == 512


def test_bad_bbox_is_rejected(mbtiles):
    with pytest.raises(ValueError):
        read_tiles(mbtiles, bbox=(300, 50, 100, 150))


def test_missing_plane_is_reported(mbtiles):
    with pytest.raises(ValueError, match="no tiles"):
        read_tiles(mbtiles, plane=7, bbox=(0, 0, 100, 100))
