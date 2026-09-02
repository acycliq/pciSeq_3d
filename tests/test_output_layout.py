"""
Checks the files land where the viewer expects and the spots are split by plane.

The viewer reads everything under viewer_data, and it fetches spots one plane at
a time, so it wants one shard per plane, including planes with no spots in them.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from pciSeq.src.core.utils.io_utils import geneData_to_arrow, _dense_planes


@pytest.fixture
def spots_df():
    # plane 1 is deliberately empty, it must still get a shard
    return pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0],
        'y': [1.0, 2.0, 3.0, 4.0],
        'z': [0.0, 0.0, 2.0, 2.0],
        'plane_id': [0, 0, 2, 2],
        'spot_id': [0, 1, 2, 3],
        'gene_id': [0, 1, 0, 1],
        'gene_name': ['a', 'b', 'a', 'b'],
        'neighbour_array': [[1, 2], [1, 2], [1, 2], [1, 2]],
        'neighbour_prob': [[0.5, 0.5]] * 4,
        'is_hard_misread': np.array([0, 1, 0, 0], dtype=np.uint8),
    })


def test_one_shard_per_plane_including_empty_ones(spots_df, tmp_path):
    geneData_to_arrow(spots_df, str(tmp_path), num_planes=4)
    d = tmp_path / 'viewer_data' / 'arrow_spots'
    names = sorted(p.name for p in d.glob('spots_plane_*.feather'))
    assert names == ['spots_plane_000.feather', 'spots_plane_001.feather',
                     'spots_plane_002.feather', 'spots_plane_003.feather']


def test_manifest_matches_the_shards(spots_df, tmp_path):
    geneData_to_arrow(spots_df, str(tmp_path), num_planes=4)
    d = tmp_path / 'viewer_data' / 'arrow_spots'
    manifest = json.loads((d / 'manifest.json').read_text())
    assert manifest['total_rows'] == 4
    assert [s['plane'] for s in manifest['shards']] == [0, 1, 2, 3]
    assert [s['rows'] for s in manifest['shards']] == [2, 0, 2, 0]


def test_is_hard_misread_survives_into_the_shard(spots_df, tmp_path):
    from pyarrow import feather
    geneData_to_arrow(spots_df, str(tmp_path), num_planes=4)
    t = feather.read_table(
        (tmp_path / 'viewer_data' / 'arrow_spots' / 'spots_plane_000.feather').as_posix())
    assert 'is_hard_misread' in t.column_names
    assert t.column('is_hard_misread').to_pylist() == [0, 1]


def test_a_plane_id_that_makes_no_sense_is_rejected(spots_df, tmp_path):
    bad = spots_df.copy()
    bad.loc[0, 'plane_id'] = -1
    with pytest.raises(ValueError):
        geneData_to_arrow(bad, str(tmp_path), num_planes=4)


def test_dense_planes_fills_the_gaps(spots_df):
    out = [(p, len(d)) for p, d in _dense_planes(spots_df, 4)]
    assert out == [(0, 2), (1, 0), (2, 2), (3, 0)]


def test_write_data_produces_the_whole_lot(minimal_varbayes, tmp_path):
    """End to end through write_data: tsv, arrow, the zarr store and the pickle.

    The other tests call each writer on its own. This one checks they are
    actually wired up, since a missing call site would not show up above.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from pciSeq.src.core.summary import collect_data
    from pciSeq.src.core.utils.io_utils import write_data

    vb = minimal_varbayes
    vb.initialise_state()
    vb.geneCount_upd()
    vb.rho_upd()
    vb.eta_upd()
    vb.theta_upd()
    vb.gamma_upd()
    vb.cell_to_cellType()
    vb.spots_to_cell_numba()

    cellData, geneData = collect_data(vb.cells, vb.spots, vb.genes, vb.config['is3D'])
    dim = vb.config['img_dim']
    coo = [coo_matrix((dim['h'], dim['w']), dtype=np.int32) for _ in range(dim['n_planes'])]
    boundaries = pd.DataFrame({'label': [], 'coords': []})

    cfg = dict(vb.config)
    cfg['output_path'] = str(tmp_path)
    write_data(cellData, geneData, boundaries, [boundaries] * dim['n_planes'],
               coo, vb, cfg)

    data = tmp_path / 'pciSeq' / 'data'
    assert (data / 'tsv' / 'cellData.tsv').exists()
    assert (data / 'tsv' / 'geneData.tsv').exists()
    assert (data / 'viewer_data' / 'arrow_spots' / 'manifest.json').exists()
    assert (data / 'viewer_data' / 'diagnostics' / 'diagnostics.db').exists()
    assert (data / 'debug' / 'pciSeq.pickle').exists()
    assert list(data.glob('*.zarr')) or (data / 'pciSeq.zarr').exists(), \
        'no spatialdata zarr store was written'

    # the misread flag has to reach the tsv, the viewer reads it
    head = (data / 'tsv' / 'geneData.tsv').read_text().splitlines()[0]
    assert 'is_hard_misread' in head
