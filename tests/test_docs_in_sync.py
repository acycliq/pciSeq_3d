"""The generated API pages must match what gen_api.py produces right now.

docs/api/reference.md and docs/api/configuration.md are built from the
docstrings and from the comments in config.py. The deploy workflow regenerates
them before it builds, so the published site is never stale, but the copies
committed here can drift the moment someone edits a docstring and forgets to
re-run gen_api.py. That leaves a confusing diff sitting in the repo and makes
the next unrelated change look bigger than it is.

Re-run it from the website folder to fix a failure here:

    cd website && python gen_api.py
"""
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GEN = REPO / 'website' / 'gen_api.py'
OUT = REPO / 'website' / 'docs' / 'api'


@pytest.fixture(scope='module')
def gen_api():
    """Load gen_api.py by path, it is a script rather than a package."""
    if not GEN.exists():
        pytest.skip('no website/gen_api.py in this checkout')
    spec = importlib.util.spec_from_file_location('gen_api', GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check(page, produced):
    on_disk = (OUT / page).read_text()
    assert on_disk == produced, (
        '%s is out of date with the source it is generated from. '
        'Run: cd website && python gen_api.py' % page
    )


def test_reference_page_is_current(gen_api):
    """Catches a docstring edit that never made it into the page."""
    _check('reference.md', gen_api.gen_reference())


def test_configuration_page_is_current(gen_api):
    """Catches a config.py comment or default that never made it into the page."""
    _check('configuration.md', gen_api.gen_configuration())
