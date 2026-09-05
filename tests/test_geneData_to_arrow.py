"""Tests for geneData_to_arrow, the per-plane spots Arrow export.

One feather file per plane, dense: every plane in [0, num_planes) gets a file and
a manifest entry, empty planes included. The manifest has to carry the plane key
or the viewer cannot skip planes, total_rows has to equal the input rows or spots
go missing silently, and shards have to come out in plane order or the dense
index the viewer relies on is a lie.
"""

import json

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pytest

from pciSeq.src.core.io import geneData_to_arrow


def make_spots_df(n, planes):
    """Minimal spots frame with the columns spots_summary() produces."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'gene_name': [f'gene{i % 7}' for i in range(n)],
        'gene_id': rng.integers(1, 8, n),
        'spot_id': np.arange(n),
        'x': rng.random(n) * 1000,
        'y': rng.random(n) * 1000,
        'z': rng.random(n) * 20,
        'plane_id': planes,
        'neighbour_array': [list(rng.integers(0, 5, 3)) for _ in range(n)],
        'neighbour_prob': [list(rng.random(3)) for _ in range(n)],
        'omp_score': rng.random(n),
        'omp_intensity': rng.random(n),
        'is_hard_misread': rng.integers(0, 2, n),
    })


def run(tmp_path, df, num_planes=None):
    geneData_to_arrow(df, str(tmp_path), num_planes=num_planes)
    out = tmp_path / 'viewer_data' / 'arrow_spots'
    manifest = json.loads((out / 'manifest.json').read_text())
    return out, manifest


def test_dense_planes_with_gaps_write_empty_files_with_same_schema(tmp_path):
    # plane 2 has no spots, plane 4 only exists because num_planes says so
    df = make_spots_df(1000, np.random.default_rng(0).choice([0, 1, 3], 1000))
    out, manifest = run(tmp_path, df, num_planes=5)

    assert [s['plane'] for s in manifest['shards']] == [0, 1, 2, 3, 4]
    rows = {s['plane']: s['rows'] for s in manifest['shards']}
    assert rows == {0: 315, 1: 330, 2: 0, 3: 355, 4: 0}
    assert manifest['total_rows'] == 1000

    # every manifest entry has a url and a plane key, and the file exists
    for s in manifest['shards']:
        assert 'plane' in s and 'rows' in s
        assert (out / s['url']).exists()

    # empty shard carries the same schema as a populated one
    empty = feather.read_table(out / 'spots_plane_002.feather')
    full = feather.read_table(out / 'spots_plane_001.feather')
    assert empty.schema.equals(full.schema)
    assert empty.num_rows == 0

    # no spots lost: the sum of shard rows on disk equals the input
    on_disk = sum(feather.read_table(out / s['url']).num_rows for s in manifest['shards'])
    assert on_disk == 1000


@pytest.mark.parametrize('bad_planes,label', [
    ([0, np.nan, 1], 'nan'),
    ([-1, 0, 1], 'negative'),
    ([0.5, 1, 2], 'fractional'),
    (['not_a_number', '1', '2'], 'unparseable_string'),
])
def test_bad_plane_id_raises(tmp_path, bad_planes, label):
    df = make_spots_df(3, bad_planes)
    with pytest.raises(ValueError, match='plane_id'):
        geneData_to_arrow(df, str(tmp_path), num_planes=3)


def test_failed_validation_leaves_previous_output_intact(tmp_path):
    # a run that fails on bad input must not delete shards the old manifest points at
    out, manifest = run(tmp_path, make_spots_df(4, [0, 1, 1, 2]), num_planes=3)
    before = sorted(p.name for p in out.glob('*.feather'))
    assert len(before) == 3

    with pytest.raises(ValueError, match='plane_id'):
        geneData_to_arrow(make_spots_df(3, [0, np.nan, 1]), str(tmp_path), num_planes=3)

    # directory and manifest unchanged, every url still resolves
    assert sorted(p.name for p in out.glob('*.feather')) == before
    manifest_after = json.loads((out / 'manifest.json').read_text())
    assert manifest_after == manifest
    for s in manifest['shards']:
        assert (out / s['url']).exists()


def test_stale_shards_from_previous_runs_are_pruned(tmp_path):
    out, _ = run(tmp_path, make_spots_df(4, [0, 1, 1, 2]), num_planes=3)

    # leftovers from the old row-chunked layout and from a longer previous run
    (out / 'spots_shard_000.feather').write_bytes(b'junk')
    (out / 'spots_plane_099.feather').write_bytes(b'junk')

    run(tmp_path, make_spots_df(4, [0, 1, 1, 2]), num_planes=3)
    names = sorted(p.name for p in out.glob('spots_*.feather'))
    assert names == ['spots_plane_000.feather', 'spots_plane_001.feather', 'spots_plane_002.feather']


def test_string_plane_ids_are_normalised_and_sorted_numerically(tmp_path):
    # '10' must not sort before '2' the way it would as a string
    df = make_spots_df(6, ['10', '2', '10', '2', '2', '0'])
    out, manifest = run(tmp_path, df)

    assert manifest['total_rows'] == 6
    assert [s['plane'] for s in manifest['shards']] == list(range(11))
    rows = {s['plane']: s['rows'] for s in manifest['shards']}
    assert rows[0] == 1 and rows[2] == 3 and rows[10] == 2


def test_float_plane_ids_are_normalised(tmp_path):
    df = make_spots_df(4, [0.0, 1.0, 1.0, 2.0])
    _, manifest = run(tmp_path, df, num_planes=3)

    assert manifest['total_rows'] == 4
    assert [s['plane'] for s in manifest['shards']] == [0, 1, 2]


def test_num_planes_widens_rather_than_dropping_spots(tmp_path):
    # the caller underestimates the plane count: planes that have spots still get written
    df = make_spots_df(4, [0, 7, 7, 7])
    out, manifest = run(tmp_path, df, num_planes=3)

    assert manifest['total_rows'] == 4
    assert [s['plane'] for s in manifest['shards']] == list(range(8))


def test_empty_input_writes_all_empty_shards(tmp_path):
    out, manifest = run(tmp_path, make_spots_df(0, []), num_planes=4)

    assert manifest['total_rows'] == 0
    assert len(manifest['shards']) == 4
    assert all(s['rows'] == 0 for s in manifest['shards'])
    for s in manifest['shards']:
        assert feather.read_table(out / s['url']).num_rows == 0


def test_gene_dict_covers_all_genes(tmp_path):
    df = make_spots_df(50, np.random.default_rng(1).integers(0, 3, 50))
    out, manifest = run(tmp_path, df, num_planes=3)

    gene_dict = json.loads((out / 'gene_dict.json').read_text())
    assert set(gene_dict) == {str(g) for g in df['gene_id'].unique()}
