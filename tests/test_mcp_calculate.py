"""calculate, the arithmetic tool.

It exists so the agent does not add numbers up in its head. It reads the expression
itself instead of calling eval, so these check both sides: the arithmetic comes out
right, and anything that is not arithmetic is refused.
"""
import math

import pytest

from pciSeq.src.mcp.tools import calculate


def test_sums_and_brackets():
    assert calculate('15.29 + 9.8 + 3.1')['result'] == pytest.approx(28.19)
    assert calculate('(56.6 - 30.9) * 2')['result'] == pytest.approx(51.4)
    assert calculate('-4.33 - -4.33')['result'] == 0


def test_odds_from_a_log_likelihood_difference():
    # the use the instructions name: exp(d) turns a difference into odds
    assert calculate('exp(3)')['result'] == pytest.approx(math.e ** 3)
    assert calculate('log(exp(2.5))')['result'] == pytest.approx(2.5)
    assert calculate('round(exp(4.1))')['result'] == 60


def test_the_answer_says_what_was_asked():
    out = calculate('1 + 1')
    assert out == {'expression': '1 + 1', 'result': 2}


@pytest.mark.parametrize('bad', [
    '__import__("os").system("ls")',   # a name that is not one of ours
    'open("x")',
    '(1).__class__',                     # an attribute
    'round(2.5, ndigits=1)',             # a keyword argument
    'x + 1',                             # a variable
    '"a" * 3',                           # not a number
    'True + 1',                          # bool is an int in python, still refused
])
def test_anything_but_arithmetic_is_refused(bad):
    with pytest.raises(ValueError, match='only numbers'):
        calculate(bad)


def test_runaway_and_impossible_sums_are_refused_with_a_reason():
    with pytest.raises(ValueError, match='exponent'):
        calculate('10 ** 10 ** 10')
    with pytest.raises(ValueError, match='division by zero'):
        calculate('1 / 0')
    with pytest.raises(ValueError, match='math domain'):
        calculate('log(0)')
    with pytest.raises(ValueError, match='longer than'):
        calculate('1+' * 300 + '1')
    with pytest.raises(ValueError, match='cannot read'):
        calculate('2 +')
