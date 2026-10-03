"""Allen brain regions on a coronal mouse section, fitted from one or two regions
the user drew by hand (the landmarks).

EXPERIMENTAL. Good for a rough map of what is where; not good enough for thin
layers. On the one section tried so far (espio, hippocampus) the outlines were
right near the landmarks and about 50 um off further away, which is as much as a
cell layer is wide. See "What is missing" at the end.

    python -m pciSeq.allen_regions outlines.geojson allen.geojson \\
        --landmark "Dentate Gyrus=DG-sg" --landmark "CA1=CA1:inside" \\
        --pixel-size 0.28 --image-size 6408 4382

outlines.geojson is the viewer's annotations file. Each --landmark says which of
its regions is which Allen structure (by acronym), and how the drawing relates to
it: 'same' (the default) when the drawing is the structure's own outline, for
example the DG granule layer, 'inside' when it only lies inside the structure,
for example the CA1 pyramidal band, since the Allen 3D atlas has CA1 as one field
with no layers.

How it works: every coronal slice of the Allen mouse brain atlas (CCFv3, 25 um)
is tried, both hemispheres, and a rotation, scale, shift and mirror is fitted that
lays the Allen landmarks onto the drawn ones. The best slice wins and all its
regions are mapped onto the image and written as allen.geojson, in image pixels,
each tagged by: allen, so the viewer can open them next to the user's own.

The fit is a straight one (no bending), so it is best near the landmarks and
gets rougher further away. Only numpy, scipy and matplotlib are needed; the
atlas (4 MB) is downloaded once into ~/.cache/pciSeq/allen_ccf.

What is missing (bead pciSeq_3d-7wp has the details):
  - the cut is taken as exactly coronal; a real section is a little tilted
  - the fit cannot bend, so uneven shrinkage is not followed
  - the newer Allen annotation (Allen-CCF-2020) has the CA1, CA2 and CA3 layers,
    this one has them as whole fields
  - only the outer edge of a region is kept, holes are ignored

Bead pciSeq_3d-7wp. Uses the Allen Mouse Brain Common Coordinate Framework
(Wang et al. 2020, Cell), under the Allen Institute Terms of Use.
"""
import argparse
import gzip
import json
import socket
import sys
import urllib.request
from pathlib import Path

import numpy as np
from contourpy import contour_generator
from matplotlib.path import Path as Polygon
from scipy import ndimage, optimize

ATLAS_URL = ('http://download.alleninstitute.org/informatics-archive/current-release/'
             'mouse_ccf/annotation/ccf_2017/annotation_25.nrrd')
STRUCTURES_URL = 'http://api.brain-map.org/api/v2/structure_graph_download/1.json'
CACHE = Path.home() / '.cache' / 'pciSeq' / 'allen_ccf'
VOX = 25.0           # um per atlas voxel
STEP = 10.0          # um between the points sampled along the drawings
W_INSIDE = 1.0       # weight of an 'inside' landmark against a 'same' one
MIN_VOXELS = 4       # region pieces smaller than this are left out
MIN_POINTS = 10      # a drawn landmark must give at least this many sample points
SMOOTH_ROUNDS = 10   # smoothing passes, to round off the atlas's 25 um staircase
SMOOTH_MIN = 40      # outlines with fewer points (a few voxels) are left as they are


# ---------------------------------------------------------------- the atlas

def _ipv4_first(*args, **kwargs):
    """getaddrinfo with the IPv4 addresses first. Allen's download server lists two
    dozen IPv6 addresses first, and on a network where IPv6 does not get through,
    urllib waits out its timeout on each in turn (curl races both, so it never
    notices)"""
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda a: a[0] != socket.AF_INET)


_getaddrinfo = socket.getaddrinfo


def _cached(url, name, timeout=30):
    """download once; a half download is never left in the cache"""
    path = CACHE / name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        socket.getaddrinfo = _ipv4_first
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
        except OSError as e:
            raise RuntimeError(f'could not download the Allen atlas file {url}: {e}. Check the '
                               f'internet connection and try again') from e
        finally:
            socket.getaddrinfo = _getaddrinfo
        tmp = path.with_suffix('.part')
        tmp.write_bytes(data)
        tmp.replace(path)
    return path


def load_atlas():
    """the annotation volume as [AP, DV, LR], and the structures by id"""
    raw = _cached(ATLAS_URL, 'annotation_25.nrrd').read_bytes()
    head, body = raw.split(b'\n\n', 1)
    sizes = [int(v) for line in head.decode().splitlines() if line.startswith('sizes:')
             for v in line.split()[1:]]
    volume = np.frombuffer(gzip.decompress(body), dtype='<u4').reshape(sizes[::-1]).transpose(2, 1, 0)
    structures = {}

    def walk(node):
        structures[node['id']] = node
        for child in node.get('children', []):
            walk(child)
    for root in json.loads(_cached(STRUCTURES_URL, 'structures.json').read_text())['msg']:
        walk(root)
    return volume, structures


# ---------------------------------------------------------------- the transform

# params (angle, log scale, tx, ty) and a mirror flag map the drawing (um, image
# axes) onto the atlas slice (um, x = left-right, y = dorsal-ventral)

def to_atlas(p, params, mirror):
    th, ls, tx, ty = params
    q = p * np.array([-1.0 if mirror else 1.0, 1.0])
    c, s = np.cos(th), np.sin(th)
    return np.exp(ls) * q @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])


def to_image(p, params, mirror):
    th, ls, tx, ty = params
    c, s = np.cos(th), np.sin(th)
    q = (p - np.array([tx, ty])) / np.exp(ls) @ np.array([[c, -s], [s, c]])
    return q * np.array([-1.0 if mirror else 1.0, 1.0])


# ---------------------------------------------------------------- the drawings

def along(ring, step=STEP):
    """points every `step` um along a closed outline"""
    ring = np.vstack([ring, ring[:1]])
    pts = []
    for a, b in zip(ring[:-1], ring[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        pts.append(a + (b - a) * np.arange(n)[:, None] / n)
    return np.vstack(pts)


def within(ring, step=STEP):
    """points on a grid inside an outline"""
    lo, hi = ring.min(0), ring.max(0)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    p = np.c_[xs.ravel(), ys.ravel()]
    return p[Polygon(ring).contains_points(p)]


def distance_map(points, step=5.0, margin=400.0):
    """distance (um) to the nearest of `points`, on a grid; and how to sample it"""
    lo = points.min(0) - margin
    shape = np.ceil((points.max(0) + margin - lo) / step).astype(int)
    img = np.ones(shape[::-1], bool)
    idx = ((points - lo) / step).astype(int)
    img[idx[:, 1], idx[:, 0]] = False
    return ndimage.distance_transform_edt(img) * step, lo, step


def sample(dist, xy, far=2000.0):
    """bilinear value of a distance map at (x, y) grid coordinates"""
    return ndimage.map_coordinates(dist, [xy[:, 1], xy[:, 0]], order=1, mode='constant', cval=far)


class Landmark:
    """one drawn region and the Allen structure it stands for"""

    def __init__(self, ring_um, structure_id, mode, name='landmark'):
        self.structure_id, self.mode = structure_id, mode
        self.points = along(ring_um) if mode == 'same' else within(ring_um)
        if len(self.points) < MIN_POINTS:
            raise ValueError(f'the drawn region {name!r} is too small to use as a landmark '
                             f'(it must be a few tens of um across)')
        if mode == 'same':
            self.user_dist, self.user_lo, self.user_step = distance_map(self.points)

    def slice_maps(self, sl):
        """what the cost needs from one atlas slice, or None if the structure is not in it"""
        mask = sl == self.structure_id
        if mask.sum() < MIN_VOXELS:
            return None
        if self.mode == 'inside':
            return {'outside': ndimage.distance_transform_edt(~mask) * VOX}
        edge = mask & ~ndimage.binary_erosion(mask)
        ey, ex = np.nonzero(edge)
        return {'edge': ndimage.distance_transform_edt(~edge) * VOX,
                'edge_um': np.c_[ex, ey] * VOX + VOX / 2}

    def cost(self, params, mirror, maps):
        """mean um off: both ways along the outline for 'same', outside the structure for 'inside'"""
        at = to_atlas(self.points, params, mirror) / VOX - 0.5
        if self.mode == 'inside':
            return W_INSIDE * sample(maps['outside'], at).mean()
        back = (to_image(maps['edge_um'], params, mirror) - self.user_lo) / self.user_step
        return (sample(maps['edge'], at).mean() + sample(self.user_dist, back).mean()) / 2


# ---------------------------------------------------------------- the fit

def fit_slice(sl, landmarks):
    """best (cost, params, mirror) for one slice, or None if a landmark is missing there"""
    maps = [lm.slice_maps(sl) for lm in landmarks]
    if any(m is None for m in maps):
        return None

    def cost(params, mirror):
        return sum(lm.cost(params, mirror, m) for lm, m in zip(landmarks, maps))

    # start from every rotation (15 degree steps), mirror and a few scales, centres
    # lined up, then polish the best start
    anchors = [lm for lm in landmarks if lm.mode == 'same'] or landmarks
    user_c = np.vstack([lm.points for lm in anchors]).mean(0)
    atlas_c = np.vstack([np.argwhere(sl == lm.structure_id)[:, ::-1] * VOX + VOX / 2 for lm in anchors]).mean(0)
    best = None
    for mirror in (False, True):
        for th in np.radians(np.arange(0, 360, 15)):
            for s in (0.8, 1.0, 1.2):
                p0 = np.array([th, np.log(s), 0.0, 0.0])
                p0[2:] = atlas_c - to_atlas(user_c[None], p0, mirror)[0]
                c0 = cost(p0, mirror)
                if best is None or c0 < best[0]:
                    best = (c0, p0, mirror)
    r = optimize.minimize(cost, best[1], args=(best[2],), method='Nelder-Mead',
                          options={'xatol': 1e-3, 'fatol': 0.05, 'maxiter': 2000})
    return r.fun, r.x, best[2]


def fit(volume, landmarks):
    """the best slice over all coronal slices and both hemispheres"""
    half = volume.shape[2] // 2
    best = None
    for ap in range(volume.shape[0]):
        for side, sl in (('left', volume[ap, :, :half]), ('right', volume[ap, :, half:])):
            res = fit_slice(sl, landmarks)
            if res and (best is None or res[0] < best['cost']):
                best = {'cost': res[0], 'params': res[1], 'mirror': res[2], 'ap': ap, 'side': side, 'slice': sl}
    if best is None:
        raise ValueError('no coronal slice holds all the landmark structures')
    best['per_landmark'] = []
    for lm in landmarks:
        best['per_landmark'].append(float(lm.cost(best['params'], best['mirror'], lm.slice_maps(best['slice']))))
    return best


# ---------------------------------------------------------------- the regions

def smooth(ring, rounds=SMOOTH_ROUNDS, shrink=0.5, grow=-0.53):
    """round off a closed outline without shrinking it (Taubin smoothing): each round
    pulls every point towards the middle of its two neighbours, then pushes it back
    out by a little more. A plain moving average made every region smaller, and the
    thin ones lost a tenth of their cells"""
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < SMOOTH_MIN:
        return ring
    p = ring.astype(float)
    for _ in range(rounds):
        for step in (shrink, grow):
            p = p + step * ((np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0)) / 2 - p)
    return p


def clip_to_box(ring, w, h):
    """cut a closed outline at the box [0, w] x [0, h] (Sutherland-Hodgman, one side at a time)"""
    def cut(pts, axis, limit, keep_below):
        inside = lambda p: p[axis] <= limit if keep_below else p[axis] >= limit
        out = []
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            if inside(a) != inside(b):        # the edge crosses the side: add the crossing point
                t = (limit - a[axis]) / (b[axis] - a[axis])
                cross = a + t * (b - a)
                out += [a, cross] if inside(a) else [cross]
            elif inside(a):
                out.append(a)
        return np.array(out).reshape(-1, 2)
    for axis, limit, keep_below in ((0, 0, False), (0, w, True), (1, 0, False), (1, h, True)):
        if len(ring) == 0:
            break
        ring = cut(ring, axis, limit, keep_below)
    return ring


def regions(best, structures, image_size_um, pixel_size):
    """every Allen region of the slice that falls on the image, as features in pixels"""
    sl, params, mirror = best['slice'], best['params'], best['mirror']
    w, h = image_size_um
    rect = to_atlas(np.array([[0, 0], [w, 0], [w, h], [0, h]]), params, mirror) / VOX - 0.5
    yy, xx = np.mgrid[0:sl.shape[0], 0:sl.shape[1]]
    on_image = Polygon(rect).contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(sl.shape)
    # one voxel more all round, so voxels only partly on the image are traced whole;
    # the outline is cut at the image border afterwards, exactly
    on_image = ndimage.binary_dilation(on_image, structure=np.ones((3, 3), bool))
    size_px = np.array(image_size_um) / pixel_size

    features = []
    for rid in np.unique(sl[on_image]):
        node = structures.get(int(rid))
        if rid == 0 or node is None or node['acronym'] == 'root':   # root: brain, no named region
            continue
        pieces, n = ndimage.label((sl == rid) & on_image)
        sizes = ndimage.sum(np.ones_like(pieces), pieces, range(1, n + 1))
        kept = [i + 1 for i in np.argsort(-sizes) if sizes[i] >= MIN_VOXELS]
        for k, label in enumerate(kept):
            mask = np.pad(pieces == label, 1).astype(float)
            outer = max(contour_generator(z=mask).lines(0.5), key=len)    # the outer edge, not holes
            um = (smooth(outer) - 1) * VOX + VOX / 2
            px = clip_to_box(to_image(um, params, mirror) / pixel_size, *size_px)
            if len(px) < 3:           # the piece was only in the extra voxel, off the image
                continue
            name = f"Allen {node['acronym']}" + (f' {k + 1}' if len(kept) > 1 else '')
            features.append({
                'type': 'Feature',
                'properties': {'kind': 'region', 'name': name, 'visible': True, 'by': 'allen',
                               'allen_name': node['name'], 'allen_id': int(rid)},
                'geometry': {'type': 'Polygon', 'coordinates': [np.round(px, 1).tolist()]},
            })
    return features


# ---------------------------------------------------------------- command line

def parse_landmark(text, structures):
    """'Dentate Gyrus=DG-sg' or 'CA1=CA1:inside' -> (region name, structure id, mode)"""
    name, _, rest = text.rpartition('=')
    acronym, _, mode = rest.partition(':')
    mode = mode or 'same'
    if not name or mode not in ('same', 'inside'):
        raise ValueError(f'landmark {text!r}: write it as "region name=ACRONYM" or "region name=ACRONYM:inside"')
    ids = [i for i, s in structures.items() if s['acronym'] == acronym]
    if not ids:
        raise ValueError(f'{acronym!r} is not an Allen structure acronym')
    return name, ids[0], mode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('outlines', help="the viewer's annotations file (GeoJSON) holding the drawn regions")
    ap.add_argument('out', help='where to write the Allen regions (GeoJSON)')
    ap.add_argument('--landmark', action='append', required=True,
                    help='"region name=ACRONYM", add ":inside" when the drawing only lies inside it')
    ap.add_argument('--pixel-size', type=float, required=True, help='um per image pixel')
    ap.add_argument('--image-size', type=float, nargs=2, required=True, metavar=('WIDTH', 'HEIGHT'),
                    help='image size in pixels')
    args = ap.parse_args(argv)

    volume, structures = load_atlas()
    drawn = {f['properties'].get('name'): f for f in json.loads(Path(args.outlines).read_text())['features']}
    landmarks = []
    for text in args.landmark:
        name, sid, mode = parse_landmark(text, structures)
        f = drawn.get(name)
        if not f or not f.get('geometry') or f['geometry']['type'] != 'Polygon':
            raise ValueError(f'no drawn region called {name!r} in {args.outlines}')
        landmarks.append(Landmark(np.array(f['geometry']['coordinates'][0], float) * args.pixel_size, sid, mode, name))

    best = fit(volume, landmarks)
    size_um = [v * args.pixel_size for v in args.image_size]
    features = regions(best, structures, size_um, args.pixel_size)
    Path(args.out).write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))

    # what the fit found, for whoever ran it (the viewer passes it on to the chat)
    th, ls = best['params'][:2]
    summary = {
        'regions': len(features),
        'atlas_slice': int(best['ap']),
        'rotation_deg': round(float(np.degrees(th)) % 360, 1),
        'scale': round(float(np.exp(ls)), 3),
        'section_vs_atlas_pct': round(100 / float(np.exp(ls))),
        'hemisphere_is': 'not determined: the two halves of the atlas are near mirror images, so the '
                         'fit cannot tell which hemisphere the section is from',
        'landmark_fit_um': {t: round(c, 1) for t, c in zip(args.landmark, best['per_landmark'])},
        'fit_um_is': "per landmark: for 'same' the mean distance between the drawn and the Allen "
                     "outline, for 'inside' the mean distance of the drawing outside the structure",
        'scale_is': 'atlas um per section um',
        'section_vs_atlas_pct_is': 'the section\'s size as a percent of the atlas (100 / scale); under 100 '
                                   'means the section is smaller, as fixed and cut tissue usually is. '
                                   'Quote this, do not work it out from the scale',
    }
    print(json.dumps(summary))


if __name__ == '__main__':
    sys.exit(main())
