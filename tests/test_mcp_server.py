"""The MCP wrapper round the tools.

tools.py is tested on its own in test_mcp_tools.py. This checks the thin layer on
top: every tool is registered, calls go through, and the errors tools.py raises on
purpose reach the client with their message intact. That last one matters most,
mcp hides the message of any exception it does not recognise, so without the
wrapping the agent would see 'Error executing tool' and nothing else.

Skipped when mcp is not installed, it is an optional extra.
"""
import asyncio
import json

import numpy as np
import pytest

mcp = pytest.importorskip('mcp')

from pciSeq.src.mcp import server as srv
from tests.test_label_identifiers import _run

TOOLS = {'open_run', 'cell', 'explain_cell', 'explain_spot', 'cell_counts', 'spots_in_cell',
         'spots_of_cell', 'cell_row', 'spot_row', 'docs', 'run_info'}


def call(name, **args):
    """Call a tool the way a client does, and hand back the parsed JSON."""
    res = asyncio.run(srv.server.call_tool(name, args))
    assert not res.is_error
    return json.loads(res.content[0].text)


def error_of(name, **args):
    """The message an agent would see when a tool refuses."""
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError) as exc:
        asyncio.run(srv.server.call_tool(name, args))
    return str(exc.value)


@pytest.fixture(scope='module')
def run_folder(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('mcp_server')
    _run(np.random.default_rng(3), tmp, save=True, max_iter=40, tol=0.001)
    return tmp


def test_every_tool_is_registered():
    names = {t.name for t in asyncio.run(srv.server.list_tools())}
    assert names == TOOLS


def test_every_tool_has_a_description_the_agent_can_use():
    for t in asyncio.run(srv.server.list_tools()):
        assert t.description and len(t.description) > 40, t.name


def test_calls_go_through(run_folder):
    s = call('open_run', path=str(run_folder))
    assert s['cells'] == 16

    e = call('explain_cell', label=105)
    assert e['assigned'] != e['compared_with']

    c = call('cell_counts', label=105)
    assert 'soft' in c['counts_are']

    h = call('spots_in_cell', label=105)
    assert 'no probabilities' in h['spots_are']

    a = call('spots_of_cell', label=105)
    b = call('spots_of_cell', label=105, min_prob=0.0001)
    assert a['n_spots'] <= b['n_spots']


def test_the_message_survives_when_a_tool_refuses(run_folder):
    """The whole reason server.py wraps the tools."""
    call('open_run', path=str(run_folder))
    assert 'no cell 999999' in error_of('cell', label=999999)
    assert 'background' in error_of('cell_counts', label=0)
    assert 'NoSuchGene' in error_of('cell_counts', label=105, gene='NoSuchGene')


def test_nothing_works_before_open_run():
    srv._run = None
    assert 'open_run' in error_of('cell', label=105)


def test_open_run_on_nothing_says_so(tmp_path):
    assert 'no diagnostics.db' in error_of('open_run', path=str(tmp_path))


# ------------------------------------------------------- packaging

def test_the_module_is_shipped_and_launchable():
    """Two ways a plain pip install can leave the server unreachable: the package
    missing from find_packages, or no console script so the only way in is the
    internal module path."""
    import ast
    import pathlib
    from setuptools import find_packages

    repo = pathlib.Path(__file__).resolve().parents[1]
    assert 'pciSeq.src.mcp' in find_packages(str(repo))

    setup_py = (repo / 'setup.py').read_text()
    assert 'pciseq-mcp = pciSeq.src.mcp.server:main' in setup_py
    assert callable(srv.main)


def test_mcp_is_an_extra_not_a_core_dependency():
    """A plain install must not drag in a web server, a jwt stack and a pydantic
    floor for the sake of a feature most users never touch. Same call spikelab
    makes. [all] rolls the extras up so nobody has to know their names."""
    import pathlib
    import runpy
    import unittest.mock as mock

    repo = pathlib.Path(__file__).resolve().parents[1]
    with mock.patch('setuptools.setup') as setup:
        runpy.run_path(str(repo / 'setup.py'), run_name='not_main')
    kwargs = setup.call_args.kwargs

    core = ' '.join(kwargs['install_requires'])
    assert 'mcp' not in core, 'mcp belongs in extras_require, not install_requires'

    extras = kwargs['extras_require']
    assert extras['mcp'] == ['mcp>=2']
    assert 'mcp>=2' in extras['all']


def test_missing_mcp_still_says_something_useful():
    """mcp ships with pciSeq now, so this only fires on an old install, but the
    message should still say what to do."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / 'pciSeq/src/mcp/server.py').read_text()
    assert 'pciSeq_3d[mcp]' in src
