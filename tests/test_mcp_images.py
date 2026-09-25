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


def _grey_mbtiles(path, w=80, h=80, maxzoom=2, grey=90, name=None):
    """A one plane pyramid of flat grey, laid out the way stage_image does it. name
    goes into the metadata the way stage_image writes it, grey lets a test tell two
    backgrounds apart."""
    con = sqlite3.connect(path)
    con.execute("create table tiles (plane_id integer, zoom_level integer, "
                "tile_column integer, tile_row integer, tile_data blob)")
    con.execute("create table metadata (name text, value text)")
    for k, v in {"format": "png", "minzoom": "0", "maxzoom": str(maxzoom),
                 "width": str(w), "height": str(h), **({"name": name} if name else {})}.items():
        con.execute("insert into metadata values (?, ?)", (k, v))
    for z in range(maxzoom + 1):
        f = TILE * 2 ** z / max(w, h)
        im = Image.new("RGB", (round(w * f), round(h * f)), (grey,) * 3)
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


# ------------------------------------------------- several background images

@pytest.fixture(scope='module')
def two_backgrounds(tmp_path_factory):
    """A run with a DAPI and a GCaMP background, told apart by their grey level."""
    tmp = tmp_path_factory.mktemp('two_backgrounds')
    _run(np.random.default_rng(7), tmp, save=True)
    r = open_run(tmp)
    vd = r.path.parent.parent
    _grey_mbtiles(vd / 'dapi.mbtiles', grey=60, name='DAPI')
    _grey_mbtiles(vd / 'gcamp.mbtiles', grey=200, name='GCaMP')
    return r


def _grey_of(im):
    return int(np.median(np.asarray(im)[..., 1]))


def test_several_backgrounds_and_no_channel_is_refused_with_the_names(two_backgrounds):
    for call in (lambda: two_backgrounds.plane_image(),
                 lambda: two_backgrounds.cell_image(106)):
        with pytest.raises(ValueError) as e:
            call()
        msg = str(e.value)
        assert 'DAPI' in msg and 'GCaMP' in msg and 'Ask the user' in msg


def test_channel_picks_the_background(two_backgrounds):
    im, info = two_backgrounds.plane_image(channel='GCaMP', width=100)
    assert abs(_grey_of(im) - 200) <= 2 and info['background'] == 'GCaMP'
    assert info['backgrounds_in_this_run'] == ['DAPI', 'GCaMP']
    im, info = two_backgrounds.plane_image(channel='dapi', width=100)
    assert abs(_grey_of(im) - 60) <= 2 and info['background'] == 'DAPI'
    im, info = two_backgrounds.cell_image(106, channel='gcamp', width=300)
    assert abs(_grey_of(im) - 200) <= 2 and info['background'] == 'GCaMP'


def test_part_of_the_name_is_enough(two_backgrounds):
    _, info = two_backgrounds.plane_image(channel='cam', width=100)
    assert info['background'] == 'GCaMP'


def test_an_unknown_channel_lists_what_there_is(two_backgrounds):
    with pytest.raises(ValueError, match='no background image matching .*DAPI.*GCaMP'):
        two_backgrounds.plane_image(channel='tdTomato')


def test_one_background_needs_no_channel(run):
    _, info = run.plane_image(width=100)
    assert info['backgrounds_in_this_run'] == ['dapi_test']
    assert 'background_note' not in info


def test_one_background_is_used_whatever_channel_says(run):
    """The name of a lone image proves nothing about the stain, so it is not
    checked: 'GCaMP' on a file called dapi_test still draws, and says why."""
    for ch in ('GCaMP', 'dapi', 'anything'):
        _, info = run.plane_image(width=100, channel=ch)
        assert info['background'] == 'dapi_test'
        assert 'only one background image' in info['background_note']
    _, info = run.cell_image(106, channel='GCaMP', width=300)
    assert 'only one background image' in info['background_note']


# ------------------------------------------------------- the mrf neighbours

def test_neighbours_are_saved_nearest_first(run):
    """On the 4 x 4 grid, cells 18 px apart, 106 has 102 above, 105 and 107 beside
    it and 110 below, all at 18 px, before any diagonal at 25 px. The labels have to
    come out as segmentation labels, not internal rows."""
    nb = run.cell(106)['neighbours']
    assert set(nb[:4]) == {102, 105, 107, 110}
    assert 106 not in nb and all(101 <= n <= 116 for n in nb)
    assert len(nb) == run.run_info()['settings']['nNeighbors']


def test_close_up_can_outline_only_the_neighbours(run):
    _, info = run.cell_image(106, neighbours=True)
    assert info['neighbours'] == run.cell(106)['neighbours']
    # a 2D run, so every neighbour has an outline on the one plane
    assert info['neighbours_not_on_this_plane'] == []
    assert info['other_cells_outlined'] == len(info['neighbours'])
    _, every = run.cell_image(106)
    assert 'neighbours' not in every


def test_old_run_without_neighbours_says_so(tmp_path):
    _run(np.random.default_rng(7), tmp_path, save=True)
    r = open_run(tmp_path)
    _grey_mbtiles(r.path.parent.parent / 'dapi_test.mbtiles')
    r._has_nbrs = False           # what a run from before the column looks like
    assert r.cell(106)['neighbours'] is None
    _, info = r.cell_image(106, neighbours=True)
    assert 'before diagnostics.db kept' in info['neighbours_note']
    assert info['other_cells_outlined'] > 0
