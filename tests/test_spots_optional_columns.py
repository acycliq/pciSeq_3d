"""score and intensity are optional in the spots frame.

Neither is read by the model, they are just carried through to geneData as
omp_score and omp_intensity. Before this they were required in practice but
listed nowhere, so leaving them out blew up inside adjust_for_anisotropy with
an AttributeError rather than a useful message.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix

from pciSeq.src.validation.config import Config
from pciSeq.src.validation.validator import Validator


def _spots(**extra):
    df = pd.DataFrame({
        'gene_name': ['a', 'b', 'c'],
        'x': np.float32([1.0, 2.0, 3.0]),
        'y': np.float32([4.0, 5.0, 6.0]),
    })
    for k, v in extra.items():
        df[k] = np.float32(v)
    return df


def _validated(spots, nplanes):
    lab = np.zeros((8, 8), dtype=np.int32)
    lab[2:4, 2:4] = 1
    coo = coo_matrix(lab) if nplanes == 1 else [coo_matrix(lab) for _ in range(nplanes)]
    cfg = Config()
    cfg.set_runtime_attrs(coo)
    v = Validator.__new__(Validator)
    v.spots = spots
    v.config = cfg
    v._validate_spots_schema()
    return v.spots


@pytest.mark.parametrize('nplanes', [1, 5])
def test_both_missing_get_filled(nplanes):
    """Neither column supplied, both come back as 1.0."""
    out = _validated(_spots(z_plane=0.0), nplanes)
    assert (out['score'] == 1.0).all()
    assert (out['intensity'] == 1.0).all()


@pytest.mark.parametrize('nplanes', [1, 5])
def test_supplied_values_survive(nplanes):
    """A caller who has real values keeps them."""
    out = _validated(_spots(z_plane=0.0, score=0.42, intensity=0.17), nplanes)
    assert (out['score'] == np.float32(0.42)).all()
    assert (out['intensity'] == np.float32(0.17)).all()


def test_score_present_intensity_missing():
    """The coppafish case: score is always there, intensity may not be."""
    out = _validated(_spots(z_plane=0.0, score=0.42), nplanes=5)
    assert (out['score'] == np.float32(0.42)).all()
    assert (out['intensity'] == 1.0).all()


def test_z_plane_still_only_filled_in_2d():
    """2D manufactures z_plane, 3D still demands it. Unchanged behaviour."""
    out = _validated(_spots(), nplanes=1)
    assert (out['z_plane'] == 0).all()

    with pytest.raises(ValueError, match='z_plane'):
        _validated(_spots(), nplanes=5)
