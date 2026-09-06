"""
The three entries in the config that pciSeq works out for itself.

is3D, img_dim and label_map live in the same dictionary as the real settings but
are not settings: the code decides them from your data. config.RUNTIME_KEYS names
them so they are not a surprise when you print the config.
"""
import logging

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from pciSeq import config
from pciSeq.src.validation.config import Config


def _stack(n):
    return [coo_matrix(np.eye(4, dtype=np.uint32)) for _ in range(n)]


def test_the_three_are_named():
    assert set(config.RUNTIME_KEYS) == {'is3D', 'img_dim', 'label_map'}


def test_none_of_them_is_a_setting():
    """They are worked out, so none of them belongs in DEFAULT."""
    for k in config.RUNTIME_KEYS:
        assert k not in config.DEFAULT, '%s should not be a user option' % k


@pytest.mark.parametrize('n_planes,expected', [(1, False), (2, True), (5, True)])
def test_is3D_comes_from_the_segmentation(n_planes, expected):
    cfg = Config({})
    cfg.set_runtime_attrs(_stack(n_planes))
    assert cfg['is3D'] is expected


@pytest.mark.parametrize('key,value', [('is3D', True),
                                       ('img_dim', {'w': 1}),
                                       ('label_map', {1: 2})])
def test_passing_any_of_them_is_rejected(key, value):
    """They are not options, so the config refuses them like any unknown key."""
    with pytest.raises(KeyError, match='Unrecognized configuration option'):
        Config({key: value})
