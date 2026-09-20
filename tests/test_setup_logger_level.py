"""
setup_logger has to use the level it is given.

It took a level argument, filled in INFO when there was none, and then handed a
hard coded logging.INFO to basicConfig anyway. So setup_logger(logging.DEBUG) still
gave INFO and the debug lines (the per plane ones in the tiling, for example) never
showed up.

setup_logger wipes the handlers on the root logger, so every test here puts them
back the way they were. Otherwise it would mess with whatever test runs next.
"""
import logging

import pytest

from pciSeq.src.core.logger import setup_logger


@pytest.fixture
def keep_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield root
    for h in root.handlers[:]:
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


def test_the_level_asked_for_is_the_level_set(keep_root_logger):
    setup_logger(logging.DEBUG)
    assert keep_root_logger.level == logging.DEBUG

    setup_logger(logging.WARNING)
    assert keep_root_logger.level == logging.WARNING


def test_no_level_still_means_info(keep_root_logger):
    setup_logger()
    assert keep_root_logger.level == logging.INFO
