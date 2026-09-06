"""A full 2D run, all the way out to the files the viewer reads.

These exist because three separate 2D bugs got through the rest of the suite.
Every one of them lived in the output path, which nothing else exercises end to
end, so the unit tests stayed green while a 2D run produced files the viewer
could not open:

  - score and intensity were required by adjust_for_anisotropy but listed in no
    validator, so spots without them died with an AttributeError
  - cells_summary only built Z for 3D, so cellData_to_arrow refused to write and
    the viewer had no cells
  - spots_summary only built z, omp_score and omp_intensity for 3D, so the
    viewer crashed reading them in dataLoaders.js

The point of these tests is the column contract between what pciSeq writes and
what the viewer reads. They are slow because they run the whole thing.
"""
import numpy as np
import pandas as pd
import pytest
import pyarrow.feather as feather
from scipy.sparse import coo_matrix

import pciSeq

GENES = ['g%d' % i for i in range(6)]
CLASSES = ['TypeA', 'TypeB', 'TypeC']


def _label_image():
    """16 square cells on an 80x80 grid. Needs to be more than nNeighbors."""
    lab = np.zeros((80, 80), dtype=np.int32)
    i = 1
    for r in range(4, 76, 18):
        for c in range(4, 76, 18):
            lab[r:r + 9, c:c + 9] = i
            i += 1
    return lab


def _spots(rng, n=900, **extra):
    df = pd.DataFrame({
        'gene_name': rng.choice(GENES, n),
        'x': rng.uniform(0, 79, n).astype(np.float32),
        'y': rng.uniform(0, 79, n).astype(np.float32),
    })
    for k, v in extra.items():
        df[k] = np.float32(v)
    return df


def _scrnaseq(rng):
    df = pd.DataFrame(rng.random((len(GENES), len(CLASSES))) * 50,
                      index=GENES, columns=CLASSES)
    df.index.name = 'gene_name'
    return df


def _run(rng, tmp_path, nplanes=1, save=False, **spot_cols):
    spots = _spots(rng, **spot_cols)
    if nplanes > 1:
        spots['z_plane'] = rng.integers(0, nplanes, len(spots)).astype(np.float32)
    lab = _label_image()
    coo = coo_matrix(lab) if nplanes == 1 else [coo_matrix(lab) for _ in range(nplanes)]
    opts = {
        'max_iter': 2,
        'save_data': save,
        'CellCallTolerance': 0.5,
        'output_path': str(tmp_path),
    }
    return pciSeq.fit(spots=spots, coo=coo, scRNAseq=_scrnaseq(rng), opts=opts)


# ---------------------------------------------------------------- the frames

@pytest.mark.slow
def test_2d_runs_without_score_or_intensity(rng, tmp_path):
    """The plain 2D contract: gene_name, x, y and nothing else."""
    cellData, geneData = _run(rng, tmp_path)
    assert len(cellData) > 0
    assert len(geneData) > 0


@pytest.mark.slow
def test_2d_cellData_carries_Z(rng, tmp_path):
    """cellData_to_arrow demands Z, so cells_summary has to produce it."""
    cellData, _ = _run(rng, tmp_path)
    assert 'Z' in cellData.columns
    assert (cellData['Z'] == 0).all()
    # Z sits right after X and Y, the viewer reads them positionally
    assert list(cellData.columns)[:4] == ['Cell_Num', 'X', 'Y', 'Z']


@pytest.mark.slow
def test_2d_geneData_carries_the_viewer_columns(rng, tmp_path):
    """dataLoaders.js reads z, omp_score and omp_intensity whatever the data is."""
    _, geneData = _run(rng, tmp_path)
    for col in ('z', 'omp_score', 'omp_intensity'):
        assert col in geneData.columns, '%s missing, the viewer reads it' % col
    assert (geneData['z'] == 0).all()


@pytest.mark.slow
def test_missing_score_and_intensity_default_to_one(rng, tmp_path):
    """A caller with no spot quality metrics gets 1.0, not a crash."""
    _, geneData = _run(rng, tmp_path)
    assert (geneData['omp_score'] == 1.0).all()
    assert (geneData['omp_intensity'] == 1.0).all()


@pytest.mark.slow
def test_supplied_score_and_intensity_survive(rng, tmp_path):
    """Real values are carried through untouched, not overwritten."""
    _, geneData = _run(rng, tmp_path, score=0.42, intensity=0.17)
    assert np.allclose(geneData['omp_score'].astype(float), 0.42)
    assert np.allclose(geneData['omp_intensity'].astype(float), 0.17)


# ------------------------------------------------------------ the arrow files

@pytest.mark.slow
def test_2d_writes_the_arrow_files_the_viewer_needs(rng, tmp_path):
    """The whole reason these tests exist: a 2D run used to write no cells."""
    _run(rng, tmp_path, save=True)
    vd = tmp_path / 'pciSeq' / 'data' / 'viewer_data'

    cells = list((vd / 'arrow_cells').glob('*.feather'))
    spots = list((vd / 'arrow_spots').glob('*.feather'))
    bounds = list((vd / 'arrow_boundaries').glob('*.feather'))
    assert cells, 'no arrow_cells shard, the viewer has nothing to draw'
    assert spots, 'no arrow_spots shard'
    assert bounds, 'no arrow_boundaries shard'

    cell_cols = feather.read_table(cells[0]).schema.names
    for col in ('cell_id', 'X', 'Y', 'Z', 'class_name', 'prob'):
        assert col in cell_cols, '%s missing from arrow_cells' % col

    spot_cols = feather.read_table(spots[0]).schema.names
    for col in ('x', 'y', 'z', 'gene_id', 'spot_id', 'omp_score', 'omp_intensity'):
        assert col in spot_cols, '%s missing from arrow_spots' % col


@pytest.mark.slow
def test_2d_writes_one_plane(rng, tmp_path):
    """2D is a single plane, so one spot shard and one boundary shard."""
    _run(rng, tmp_path, save=True)
    vd = tmp_path / 'pciSeq' / 'data' / 'viewer_data'
    assert len(list((vd / 'arrow_spots').glob('*.feather'))) == 1
    assert len(list((vd / 'arrow_boundaries').glob('*.feather'))) == 1


# ------------------------------------------------------------------- vs 3D

@pytest.mark.slow
def test_3d_still_has_the_same_columns(rng, tmp_path):
    """The 2D fixes moved these out of an is3D branch, so check 3D kept them."""
    cellData, geneData = _run(rng, tmp_path, nplanes=5)
    assert 'Z' in cellData.columns
    for col in ('z', 'omp_score', 'omp_intensity'):
        assert col in geneData.columns
    # unlike 2D, z is not all zeros here
    assert geneData['z'].nunique() > 1
