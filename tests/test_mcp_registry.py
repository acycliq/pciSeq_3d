"""The registry the pciSeq viewer reads to find the MCP server.

Every test works on a file in a temp folder; the real one in the user's config
folder is never touched.
"""
import json
import sys

import pytest

mcp = pytest.importorskip('mcp')

from pciSeq.src.mcp import registry
from pciSeq.src.mcp import server as srv


@pytest.fixture
def reg(tmp_path, monkeypatch):
    path = tmp_path / 'pciseq' / 'mcp_servers.json'
    monkeypatch.setattr(registry, 'registry_path', lambda: path)
    return path


def test_register_writes_how_to_start_the_server(reg):
    e = registry.register('my env')
    data = json.loads(reg.read_text())
    assert data['servers'] == [e]
    assert e['name'] == 'my env' and e['python'] == sys.executable
    assert e['args'] == ['-m', 'pciSeq.src.mcp.server']
    import pciSeq
    assert (e['pciseq_version'], e['commit']) == (pciSeq.__version__, pciSeq.__commit__)


def test_registering_again_refreshes_rather_than_duplicates(reg):
    registry.register('first')
    registry.register('second')
    entries = registry.list_entries()
    assert [e['name'] for e in entries] == ['second']


def test_a_checkout_is_found_through_pythonpath(reg):
    # the tests import pciSeq from the repo, not from site-packages, so the viewer
    # has to be told where the code is
    import pciSeq
    from pathlib import Path
    e = registry.register()
    assert e['env']['PYTHONPATH'] == str(Path(pciSeq.__file__).resolve().parent.parent)


def test_unregister_and_a_python_that_is_gone(reg):
    registry.register()
    data = json.loads(reg.read_text())
    data['servers'].append(dict(data['servers'][0], python='/no/such/python', name='gone'))
    reg.write_text(json.dumps(data))
    assert {e['name']: e['exists'] for e in registry.list_entries()} == \
        {data['servers'][0]['name']: True, 'gone': False}
    assert registry.unregister() is True
    assert registry.unregister() is False
    assert [e['name'] for e in registry.list_entries()] == ['gone']


def test_the_command_line(reg, capsys):
    srv.main(['--register', '--name', 'cli'])
    assert 'registered "cli"' in capsys.readouterr().out
    srv.main(['--list'])
    assert 'cli' in capsys.readouterr().out
    srv.main(['--unregister'])
    assert 'removed' in capsys.readouterr().out


def test_the_server_reports_its_pciseq_version():
    import pciSeq
    assert srv.server.version == '%s+%s' % (pciSeq.__version__, pciSeq.__commit__)
