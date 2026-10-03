"""
pciSeq.allen_regions fits the Allen atlas to a section from hand drawn landmarks.

Nothing here downloads the atlas. The tests build a tiny fake one (a few slices with
two made up structures) and check the parts on it: the transform and its inverse, the
landmark text, the smoothing, the cut at the image border, and then the whole thing
end to end, where the fit has to find a rotation and scale we chose ourselves.
"""
import json

import numpy as np
import pytest
from contourpy import contour_generator
from matplotlib.path import Path as Polygon

import pciSeq.allen_regions as ar

BLOB, BAND = 7, 9          # ids of the two fake structures
STRUCTURES = {
    1: {'id': 1, 'acronym': 'root', 'name': 'root'},
    BLOB: {'id': BLOB, 'acronym': 'BLOB', 'name': 'a lopsided blob'},
    BAND: {'id': BAND, 'acronym': 'BAND', 'name': 'a band next to it'},
}


def fake_slice():
    """60 x 80 voxels of root with a lopsided blob (no symmetry, so only one rotation
    fits) and a band beside it"""
    sl = np.ones((60, 80), dtype=np.uint32)
    sl[15:40, 20:35] = BLOB
    sl[30:40, 35:55] = BLOB
    sl[15:22, 35:42] = BLOB
    sl[44:50, 15:60] = BAND
    return sl


def fake_volume():
    """three slices, the structures only in the middle one, left half. The right half
    stays root so the fit has one place to find them"""
    vol = np.ones((3, 60, 160), dtype=np.uint32)
    vol[1, :, :80] = fake_slice()
    return vol


def outline_um(sl, structure_id):
    """a structure's outline in atlas um, traced the way the module does it"""
    mask = np.pad(sl == structure_id, 1).astype(float)
    ring = max(contour_generator(z=mask).lines(0.5), key=len)
    return (ring - 1) * ar.VOX + ar.VOX / 2


def area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


# the section we pretend to have: turned 40 degrees, 80% of the atlas size, shifted
TRUE = np.array([np.radians(40.0), np.log(1.25), 300.0, 900.0])


@pytest.mark.parametrize('mirror', [False, True])
def test_to_image_undoes_to_atlas(mirror):
    pts = np.random.default_rng(0).uniform(-500, 500, (50, 2))
    back = ar.to_image(ar.to_atlas(pts, TRUE, mirror), TRUE, mirror)
    assert np.allclose(back, pts)


def test_landmark_text():
    assert ar.parse_landmark('Dentate Gyrus=BLOB', STRUCTURES) == ('Dentate Gyrus', BLOB, 'same')
    assert ar.parse_landmark('my CA1=BAND:inside', STRUCTURES) == ('my CA1', BAND, 'inside')
    # a region name may have an = in it, the last one splits
    assert ar.parse_landmark('a=b=BLOB', STRUCTURES)[0] == 'a=b'
    with pytest.raises(ValueError, match='not an Allen structure'):
        ar.parse_landmark('x=NOPE', STRUCTURES)
    with pytest.raises(ValueError, match='write it as'):
        ar.parse_landmark('x=BLOB:around', STRUCTURES)


def test_a_landmark_too_small_is_refused():
    # a 4 um square holds no point of the 10 um grid. It used to give a nan cost and
    # the fit quietly returned the first slice it tried
    tiny = np.array([[0, 0], [4, 0], [4, 4], [0, 4.0]])
    with pytest.raises(ValueError, match='too small'):
        ar.Landmark(tiny, BAND, 'inside', 'speck')


def test_smoothing_does_not_shrink():
    ring = outline_um(fake_slice(), BLOB)
    assert len(ring) >= ar.SMOOTH_MIN
    smoothed = ar.smooth(ring)
    assert abs(area(smoothed) / area(ring) - 1) < 0.02
    # and a small outline is left alone
    small = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    assert np.array_equal(ar.smooth(small), small)


def test_clip_to_box_cuts_at_the_border():
    # a 10 x 10 square hanging 4 over the right side of a 6 wide box
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]])
    cut = ar.clip_to_box(square, 6, 20)
    assert area(cut) == pytest.approx(60)
    assert cut[:, 0].max() == 6
    # all of it off the box: nothing left
    assert len(ar.clip_to_box(square + 100, 6, 20)) == 0


def test_the_fit_finds_the_rotation_and_scale():
    sl = fake_slice()
    # draw the landmarks where the section would show them
    blob = ar.Landmark(ar.to_image(outline_um(sl, BLOB), TRUE, False), BLOB, 'same', 'blob')
    band = ar.Landmark(ar.to_image(outline_um(sl, BAND), TRUE, False), BAND, 'inside', 'band')
    cost, params, mirror = ar.fit_slice(sl, [blob, band])
    assert cost < 10                                  # um, well under one voxel
    assert np.exp(params[1]) == pytest.approx(1.25, rel=0.03)
    # the drawn blob lands back on the atlas blob, whichever of mirror/rotation it used
    landed = ar.to_atlas(blob.points, params, mirror)
    truth = ar.to_atlas(blob.points, TRUE, False)
    assert np.abs(landed - truth).max() < ar.VOX


def test_a_slice_without_the_structure_is_skipped():
    blob = ar.Landmark(outline_um(fake_slice(), BLOB), BLOB, 'same', 'blob')
    assert ar.fit_slice(np.ones((60, 80), dtype=np.uint32), [blob]) is None


def test_points_in_the_outlines_match_the_atlas():
    """a point is in an exported outline when the atlas has that region at the point,
    to within 1%. The first version smoothed with a moving average, which shrinks, and
    cut at the image border by whole voxels; on real data the regions at the border
    lost a fifth of their cells, and here the band was 2.7% short"""
    sl = fake_slice()
    best = {'slice': sl, 'params': TRUE, 'mirror': False}
    # an image that cuts through both structures
    lo = ar.to_image(np.array([[25, 10]]) * ar.VOX, TRUE, False)[0]
    px = 0.5
    shift = np.array([TRUE[0], TRUE[1], 0, 0])
    best['params'] = TRUE.copy()
    best['params'][2:] = TRUE[2:] + ar.to_atlas(lo[None], shift, False)[0]   # image origin at lo
    size_um = [700.0, 600.0]
    feats = ar.regions(best, STRUCTURES, size_um, px)
    assert {f['properties']['name'] for f in feats} == {'Allen BLOB', 'Allen BAND'}
    assert all(f['properties']['by'] == 'allen' for f in feats)

    pts = np.random.default_rng(1).uniform([0, 0], np.array(size_um) / px, (40000, 2))
    vox = np.round(ar.to_atlas(pts * px, best['params'], False) / ar.VOX - 0.5).astype(int)
    ok = (vox[:, 0] >= 0) & (vox[:, 0] < sl.shape[1]) & (vox[:, 1] >= 0) & (vox[:, 1] < sl.shape[0])
    truth = np.zeros(len(pts), int)
    truth[ok] = sl[vox[ok, 1], vox[ok, 0]]
    for f in feats:
        ring = np.array(f['geometry']['coordinates'][0])
        assert ring.min() >= 0 and (ring <= np.array(size_um) / px).all()   # nothing off the image
        inside = Polygon(ring).contains_points(pts).sum()
        expected = (truth == f['properties']['allen_id']).sum()
        assert expected > 500
        assert inside == pytest.approx(expected, rel=0.01)


def test_the_command_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ar, 'load_atlas', lambda: (fake_volume(), STRUCTURES))
    px = 0.5
    sl = fake_slice()
    rings = {name: ar.to_image(outline_um(sl, sid), TRUE, False) / px
             for name, sid in (('my blob', BLOB), ('my band', BAND))}
    # slide the drawings so they sit on the image, 100 px in from the corner
    corner = np.vstack(list(rings.values())).min(0) - 100
    drawn = {'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'properties': {'name': name},
         'geometry': {'type': 'Polygon', 'coordinates': [(ring - corner).tolist()]}}
        for name, ring in rings.items()]}
    # a cell annotation has no geometry, the command must not trip on it
    drawn['features'].append({'type': 'Feature', 'properties': {'name': 'cells', 'kind': 'cells'}, 'geometry': None})
    outlines, out = tmp_path / 'outlines.geojson', tmp_path / 'allen.geojson'
    outlines.write_text(json.dumps(drawn))

    ar.main([str(outlines), str(out), '--landmark', 'my blob=BLOB', '--landmark', 'my band=BAND:inside',
             '--pixel-size', str(px), '--image-size', '4000', '4000'])
    summary = json.loads(capsys.readouterr().out)

    assert summary['atlas_slice'] == 1
    assert summary['scale'] == pytest.approx(1.25, rel=0.03)
    assert summary['section_vs_atlas_pct'] == round(100 / summary['scale'])
    # the two halves of the real atlas are near mirror images, so the hemisphere is
    # never reported as found
    assert 'hemisphere' not in summary and 'mirrored' not in summary
    assert 'not determined' in summary['hemisphere_is']
    names = {f['properties']['name'] for f in json.loads(out.read_text())['features']}
    assert {'Allen BLOB', 'Allen BAND'} <= names

    with pytest.raises(ValueError, match='no drawn region'):
        ar.main([str(outlines), str(out), '--landmark', 'nothing=BLOB', '--pixel-size', '0.5', '--image-size', '10', '10'])
