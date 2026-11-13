"""
Checks that spots_summary gives the very same table as it did before its plain
columns stopped going through python lists.

The table is what fit returns as geneData and what geneData.tsv is printed from,
so same here means same in every way that can be seen from outside: the columns
and their order, the dtypes, the index, every value, and for the two ragged
columns (neighbour_array, neighbour_prob) python lists of python numbers.
"""
import numpy as np
import pandas as pd

from pciSeq.src.core import summary


def _spots_summary_before(spots, is3D):
    """NOTE: this is the code of summary.spots_summary as it was before the plain
    columns were passed to pandas as numpy arrays (commit d428fc4f still has it).
    It is not used anywhere in pciSeq anymore, it lives here only to check the
    new code against. Every column goes through tolist."""
    # check for duplicates (ie spots with the same coordinates with or without the same gene name).
    # is_duplicate = spots.data.duplicated(subset=['x', 'y'])

    idx = np.argsort(-1 * spots.parent_cell_prob, axis=1)
    p = np.take_along_axis(spots.parent_cell_prob, idx, axis=1).round(3)
    nbrs = np.take_along_axis(spots.parent_cell_id, idx, axis=1)
    max_nbrs = nbrs[:, 0]

    out = pd.DataFrame({'gene_name': spots.data.gene_name.tolist(),
                        'gene_id': spots.gene_id.tolist(),
                        'spot_id': spots.data.index.tolist(),
                        'x': np.round(spots.data.x.astype('float64'), 3).tolist(),
                        'y': np.round(spots.data.y.astype('float64'), 3).tolist(),
                        'plane_id': spots.data.plane_id.tolist(),
                        'neighbour': max_nbrs.tolist(),
                        'neighbour_array': nbrs.tolist(),
                        'neighbour_prob': p.tolist(),
                        # 'omp_score': ((spots.data.score * 1000).astype(np.int32)/1000).tolist()
                        })
    # These three used to be written only for 3D, but the viewer reads all of
    # them whatever the data is, so a 2D run left it with nothing to draw. In 2D
    # z is zero for every spot anyway, and score and intensity are filled in by
    # the validator when the caller has none.
    out['z'] = np.round(spots.data.z.astype('float64'), 3).tolist()
    out['omp_score'] = np.round(spots.data.score.astype('float64'), 3).tolist()
    out['omp_intensity'] = np.round(spots.data.intensity.astype('float64'), 3).tolist()
    # which cell the spot physically sits in, straight off the label image, 0 if none.
    # Not the same as 'neighbour': that is the cell the model decided the spot came
    # from, this is just where the segmentation put it. The model only ever looks at
    # it through InsideCellBonus, which is normally 0. Named after inside_cell in
    # label_processing, which is what works it out.
    out['inside_cell'] = spots.data.label.tolist()
    # move column z after x, y
    z_pos = out.columns.get_loc('y') + 1
    out.insert(z_pos, 'z', out.pop('z'))

    return out


def _fitted(vb, passes=1):
    vb.initialise_state()
    for _ in range(passes):
        for step in (vb.geneCount_upd, vb.rho_upd, vb.eta_upd, vb.theta_upd, vb.gamma_upd,
                     vb.cell_to_cellType, vb.spots_to_cell_numba):
            step()
    return vb


def _both(vb):
    before = _spots_summary_before(vb.spots, vb.config['is3D'])
    now = summary.spots_summary(vb.spots, vb.config['is3D'])

    assert list(now.columns) == list(before.columns)
    assert (now.dtypes == before.dtypes).all()
    pd.testing.assert_frame_equal(now, before, check_exact=True)

    # the ragged columns: python lists of python numbers, like before
    for col, kind in (('neighbour_array', int), ('neighbour_prob', float)):
        for a, b in zip(now[col], before[col]):
            assert type(a) is list and a == b
            assert all(type(v) is kind for v in a)
    # and the text of the tsv is the same, character for character
    assert now.to_csv(sep='\t', index=False) == before.to_csv(sep='\t', index=False)
    return now


def test_same_table_after_a_pass(minimal_varbayes):
    now = _both(_fitted(minimal_varbayes))
    assert len(now) == minimal_varbayes.spots.data.shape[0]


def test_same_table_after_more_passes(minimal_varbayes):
    _both(_fitted(minimal_varbayes, passes=4))


def test_same_table_on_the_float32_start(minimal_varbayes):
    # before any spot assignment the probs are float32, all equal, so the order of
    # the candidate cells is decided by how argsort breaks the ties
    vb = minimal_varbayes
    vb.initialise_state()
    _both(vb)


def test_same_table_when_the_spot_index_has_gaps(minimal_varbayes):
    # spot_id is the index of the spot table and it has gaps when spots were
    # dropped on the way in. A column built from the index must not pick up the
    # index as its own row labels
    vb = _fitted(minimal_varbayes)
    vb.spots.data.index = vb.spots.data.index * 3 + 7
    now = _both(vb)
    assert list(now.index) == list(range(len(now)))
    assert list(now.spot_id) == list(vb.spots.data.index)


def test_same_table_with_narrow_number_types(minimal_varbayes):
    # whatever the numpy type of a column in the spot table, the output is int64
    # or float64, as it was when everything went through a list
    vb = _fitted(minimal_varbayes)
    d = vb.spots.data
    d['plane_id'] = d['plane_id'].astype(np.int16)
    d['label'] = d['label'].astype(np.uint32)
    d['x'] = d['x'].astype(np.float32)
    d['score'] = d['score'].astype(np.float32)
    now = _both(vb)
    assert now.plane_id.dtype == np.int64 and now.inside_cell.dtype == np.int64
    assert now.x.dtype == np.float64 and now.omp_score.dtype == np.float64
