"""cell_image and plane_image, the pictures drawn from the background tiles.

Uses the 16 cell fixture from test_label_identifiers: 9 x 9 squares on an 80 x 80
image, labels 101 up, so we know exactly where every cell is. The tiles are a flat
grey built here with pillow, so anything red in the picture is the outline we drew,
and we can check it lands on the cell.
"""
import io
import sqlite3

import numpy as np
import pytest
from PIL import Image

from pciSeq.src.mcp.tools import open_run
from tests.test_label_identifiers import _run, _label_image

TILE = 256


def _grey_mbtiles(path, w=80, h=80, maxzoom=2):
    """A one plane pyramid of flat grey, laid out the way stage_image does it."""
    con = sqlite3.connect(path)
    con.execute("create table tiles (plane_id integer, zoom_level integer, "
                "tile_column integer, tile_row integer, tile_data blob)")
    con.execute("create table metadata (name text, value text)")
    for k, v in {"format": "png", "minzoom": "0", "maxzoom": str(maxzoom),
                 "width": str(w), "height": str(h)}.items():
        con.execute("insert into metadata values (?, ?)", (k, v))
    for z in range(maxzoom + 1):
        f = TILE * 2 ** z / max(w, h)
        im = Image.new("RGB", (round(w * f), round(h * f)), (90, 90, 90))
        for ty in range((im.height + TILE - 1) // TILE):
            for tx in range((im.width + TILE - 1) // TILE):
                b = io.BytesIO()
                im.crop((tx * TILE, ty * TILE, (tx + 1) * TILE, (ty + 1) * TILE)).save(b, "PNG")
                con.execute("insert into tiles values (0, ?, ?, ?, ?)", (z, tx, ty, b.getvalue()))
    con.commit()
    con.close()


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('cell_image')
    _run(np.random.default_rng(7), tmp, save=True)
    r = open_run(tmp)
    _grey_mbtiles(r.path.parent.parent / 'dapi_test.mbtiles')
    return r


def _red(im):
    """Where the picture is clearly red, as (rows, cols)."""
    a = np.asarray(im).astype(int)
    return np.nonzero((a[..., 0] > 180) & (a[..., 1] < 150))


def test_close_up_draws_the_cell_where_it_is(run):
    import pyarrow.compute as pc
    im, info = run.cell_image(101, width=600)
    assert im.size == (600, 400)          # 3:2
    # the outline the viewer stores is simplified, a corner can sit a pixel off the
    # mask, so check against that and not against the square in the label image
    b = run._outlines(0)
    o = b.filter(pc.equal(b['label'], 101)).to_pylist()[0]
    xs, ys = np.array(o['x_list']), np.array(o['y_list'])
    x0, y0 = info['bbox'][:2]
    s = info['scale']
    rows, cols = _red(im)
    assert len(rows) > 0
    # the red sits on that outline, give or take the line width
    assert cols.min() >= (xs.min() - x0) * s - 4 and cols.max() <= (xs.max() - x0) * s + 4
    assert rows.min() >= (ys.min() - y0) * s - 4 and rows.max() <= (ys.max() - y0) * s + 4
    # and it is the square of cell 101, not some other cell
    lab = _label_image()
    assert lab[int(round(np.mean(rows) / s + y0)), int(round(np.mean(cols) / s + x0))] == 101
    assert info['plane'] == 0 and info['cell_has_outline_on_this_plane'] is True


def test_close_up_outlines_the_other_cells_in_the_window(run):
    # 106 is in the middle of the grid, so the window catches cells on every side
    _, info = run.cell_image(106)
    assert info['other_cells_outlined'] > 0


def test_box_stays_inside_the_image_at_the_edge(run):
    # 101 is in the top left corner, the window must not hang off the image
    _, info = run.cell_image(101)
    x0, y0, x1, y1 = info['bbox']
    assert x0 >= 0 and y0 >= 0 and x1 <= 80 and y1 <= 80


def test_context_is_the_whole_plane_with_a_ring(run):
    im, info = run.cell_image(116, context=True, width=300)
    # 80 x 80 is taller than 3:2, so rows are trimmed top and bottom
    assert im.size == (300, 200)
    assert info['bbox'] == [0.0, 13.3, 80.0, 66.7]
    rows, cols = _red(im)
    # 116 is the bottom right cell, so the ring is in the bottom right
    assert cols.mean() > 150 and rows.mean() > 100


def test_background_and_unknown_cells_are_refused(run):
    with pytest.raises(ValueError, match='background'):
        run.cell_image(0)
    with pytest.raises(KeyError):
        run.cell_image(999)


def test_no_mbtiles_says_what_to_do(tmp_path):
    _run(np.random.default_rng(7), tmp_path, save=True)
    r = open_run(tmp_path)
    with pytest.raises(FileNotFoundError, match='mbtiles'):
        r.cell_image(101)


# ------------------------------------------------------------ plane_image

def test_plane_image_is_the_whole_plane_untrimmed(run):
    im, info = run.plane_image(width=200)
    assert im.size == (200, 200)          # 80 x 80, nothing cut to 3:2
    assert info['bbox'] == [0.0, 0.0, 80.0, 80.0] and info['plane'] == 0
    # nothing drawn on it, so it is the flat grey the tiles hold
    a = np.asarray(im).astype(int)
    assert np.abs(a - 90).max() <= 2


def test_plane_image_crops_to_the_bbox(run):
    im, info = run.plane_image(bbox=[10, 20, 50, 40], width=400)
    assert im.size == (400, 200)
    assert info['scale'] == 10.0 and info['bbox'] == [10.0, 20.0, 50.0, 40.0]


def test_plane_image_clamps_the_bbox_and_says_so(run):
    _, info = run.plane_image(bbox=[-10, -10, 200, 30])
    assert info['bbox'] == [0.0, 0.0, 80.0, 30.0]


def test_plane_image_refuses_a_plane_that_is_not_there(run):
    with pytest.raises(ValueError, match='planes 0 to 0'):
        run.plane_image(plane=3)
