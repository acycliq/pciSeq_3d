"""
The run log has to land in the folder the results go to.

Config opened the log file before it merged the user's options, so it only ever
saw the default output_path and the log went to /tmp/pciSeq/pciSeq.log no matter
what output_path was set to. The results went to the right place, the log did not.

log_file adds a handler to the root logger, so the test puts the handlers back
the way they were. Otherwise the next test would keep writing to this log.
"""
import logging
import os

import pytest

from pciSeq.src.validation.config import Config


@pytest.fixture
def root_with_console_only():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    for h in handlers:
        root.removeHandler(h)
    # log_file only adds the file when some handler is already there.
    root.addHandler(logging.StreamHandler())
    yield root
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)


def _log_files(root):
    return [h.baseFilename for h in root.handlers if isinstance(h, logging.FileHandler)]


def test_log_goes_under_output_path(root_with_console_only, tmp_path):
    Config({"output_path": str(tmp_path)})
    assert _log_files(root_with_console_only) == [str(tmp_path / "pciSeq" / "pciSeq.log")]
    assert os.path.exists(tmp_path / "pciSeq" / "pciSeq.log")


def test_override_lines_still_reach_the_log(root_with_console_only, tmp_path):
    root_with_console_only.setLevel(logging.INFO)
    Config({"output_path": str(tmp_path), "max_iter": 7})
    for h in root_with_console_only.handlers:
        h.flush()
    text = (tmp_path / "pciSeq" / "pciSeq.log").read_text()
    assert "Config override: max_iter = 7" in text
