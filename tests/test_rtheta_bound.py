"""
rTheta has to stay above 1.

theta is the mode of its Gamma posterior, (N_c + rTheta - 1) / (rTheta + expected).
Once rTheta drops to 1 or below, a cell holding no assigned reads comes out with a
theta of zero or negative. That is meaningless: theta multiplies the expected counts
and log(theta) is taken further downstream. Cells with no reads are not a corner
case, silver 180 has a few hundred of them.
"""
import pytest

from pciSeq.src.validation.validator import Validator


def _theta(reads, rTheta, expected=50.0):
    """What theta_upd and cells.calc_theta work out between them."""
    return (reads + rTheta - 1) / (rTheta + expected)


def test_an_empty_cell_goes_negative_below_one():
    assert _theta(reads=0, rTheta=0.5) < 0
    assert _theta(reads=0, rTheta=1.0) == 0
    assert _theta(reads=0, rTheta=1.01) > 0


def test_an_empty_cell_is_fine_above_one():
    for rTheta in (1.01, 2, 25, 1000):
        assert _theta(reads=0, rTheta=rTheta) > 0


def _validator(rTheta):
    """A validator carrying just enough config for _validate_config to run."""
    cfg = {
        'rTheta': rTheta,
        'cell_type_prior': 'uniform',
        'InsideCellBonus': 0,
        'MisreadDensity': 1e-5,
        'cell_centroid_prior': 10,
        'cell_cov_prior': 10,
    }
    return Validator(spots=None, coo=None, scdata=None, config=cfg)


@pytest.mark.parametrize('bad', [1, 1.0, 0.5, 0, -3])
def test_validator_rejects_rTheta_at_or_below_one(bad):
    with pytest.raises(ValueError, match='rTheta'):
        _validator(bad)._validate_config()


@pytest.mark.parametrize('ok', [1.01, 2, 25.0])
def test_validator_accepts_rTheta_above_one(ok):
    _validator(ok)._validate_config()          # must not raise
