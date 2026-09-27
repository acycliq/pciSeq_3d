"""The generated API pages must match what the generators produce right now.

docs/api/reference.md and docs/api/configuration.md are built from the
docstrings and from the comments in config.py by gen_api.py, and
docs/api/code-map.md from the source tree by gen_code_map.py. The deploy
workflow regenerates them before it builds, so the published site is never
stale, but the copies committed here can drift the moment someone edits a
docstring, or moves a function, and forgets to re-run the script. That leaves a
confusing diff sitting in the repo and makes the next unrelated change look
bigger than it is.

Re-run from the website folder to fix a failure here:

    cd website && python gen_api.py && python gen_code_map.py
"""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN = REPO / 'website' / 'gen_api.py'
GEN_MAP = REPO / 'website' / 'gen_code_map.py'
OUT = REPO / 'website' / 'docs' / 'api'


def _load(path):
    """Load a generator script by path, they are scripts rather than a package."""
    if not path.exists():
        pytest.skip('no %s in this checkout' % path.relative_to(REPO))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def gen_api():
    return _load(GEN)


@pytest.fixture(scope='module')
def gen_code_map():
    return _load(GEN_MAP)


def _check(page, produced):
    on_disk = (OUT / page).read_text()
    assert on_disk == produced, (
        '%s is out of date with the source it is generated from. '
        'Run: cd website && python gen_api.py && python gen_code_map.py' % page
    )


def test_reference_page_is_current(gen_api):
    """Catches a docstring edit that never made it into the page."""
    _check('reference.md', gen_api.gen_reference())


def test_configuration_page_is_current(gen_api):
    """Catches a config.py comment or default that never made it into the page."""
    _check('configuration.md', gen_api.gen_configuration())


def test_code_map_names_exist(gen_code_map):
    """Every entry of the hand-kept map has to point at a function that is still
    there under that name, or the page sends a reader to the wrong place."""
    assert gen_code_map.missing_map_entries() == []


def test_code_map_page_is_current(gen_code_map):
    """Catches a moved or renamed function that never made it into the page."""
    _check('code-map.md', gen_code_map.gen_code_map())
