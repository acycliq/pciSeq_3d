"""
The class colours in the realtime viewer.

The viewer is javascript and has no tests of its own, so this one runs a small node
script, tests/viewer_colors_check.js, against the viewer's real colors.js. It is
skipped when node is not installed.

What it guards: the palette used to stop at 65 colours, so in a bigger taxonomy every
class from the 66th on was drawn grey. And a loaded colour scheme only overrode the
classes it named, the rest kept a random auto colour that could pass for one of the
chosen ones. The rules now: one colour per real class, a class the scheme does not name
is grey, and Zero is always black.
"""
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
JS_DIR = os.path.join(REPO, 'pciSeq', 'src', 'realtime_viewer', 'static', 'js')


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_the_colour_rules():
    done = subprocess.run(['node', os.path.join(HERE, 'viewer_colors_check.js'), JS_DIR],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'FAIL' not in done.stdout
