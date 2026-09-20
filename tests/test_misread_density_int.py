"""
MisreadDensity can be given as a whole number.

The comment in config.py says "a number or a dictionary", and the validator that
turns it into {'default': value} takes an int or a float. The type table was the odd
one out: it listed (float, dict), so MisreadDensity=1 raised a TypeError while every
similar option (SpotReg, Inefficiency, rTheta) took an int without complaining.
"""
import pytest

from pciSeq.src.validation.config import _TYPE_SPECS


def _passes_the_type_check(value):
    allowed = _TYPE_SPECS["MisreadDensity"]
    return isinstance(value, allowed if isinstance(allowed, tuple) else (allowed,))


@pytest.mark.parametrize("value", [1, 0, 1e-5, {"default": 1e-5}])
def test_these_are_all_fine(value):
    assert _passes_the_type_check(value)


def test_a_string_is_still_refused():
    assert not _passes_the_type_check("1e-5")
