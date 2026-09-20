"""
cell_type has to let an error out with the type it was raised with.

It used to catch everything and raise RuntimeError("Cell typing failed: ...") in its
place. So a ValueError about bad input and a crash deep in the model both reached the
caller as the same RuntimeError, and the ValueError its own docstring promised never
came out. fit, a few lines above it, always did the right thing: log and re-raise.

A config with nothing in it is enough. VarBayes checks for its required keys before it
builds anything, so this fails straight away, no fit runs and the log is left alone.
"""
import pandas as pd
import pytest

from pciSeq.app import cell_type


def test_a_value_error_comes_out_as_a_value_error():
    empty = pd.DataFrame()
    with pytest.raises(ValueError, match="Missing required config parameters"):
        cell_type(empty, empty, empty, {})


def test_it_is_not_turned_into_a_runtime_error():
    empty = pd.DataFrame()
    with pytest.raises(Exception) as err:
        cell_type(empty, empty, empty, {})
    assert not isinstance(err.value, RuntimeError)
