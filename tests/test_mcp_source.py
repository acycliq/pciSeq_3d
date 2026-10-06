"""The live viewer's chat can read the pciSeq source: the files on this machine,
which are the ones running the fit. No model and no fit needed for any of this."""
import urllib.request

import pytest

from pciSeq.src.mcp import chat, source


def test_a_file_comes_back_with_line_numbers():
    out = source.read_source('pciSeq/src/mcp/source.py')
    assert out['path'] == 'pciSeq/src/mcp/source.py'
    assert out['start_line'] == 1
    first = out['text'].split('\n')[0]
    assert first.lstrip().startswith('1  ') and 'Reading the pciSeq source code' in first
    assert 'MAX_LINES = 400' in out['text']
    assert 'running this fit' in out['source_is']


def test_the_three_ways_of_writing_a_path_reach_the_same_file():
    full = source.read_source('pciSeq/src/core/main.py')
    assert source.read_source('src/core/main.py')['path'] == full['path']
    # the code map page of the docs writes paths from pciSeq/src
    assert source.read_source('core/main.py')['path'] == full['path']
    assert full['total_lines'] > 400


def test_a_long_file_comes_in_slices():
    one = source.read_source('pciSeq/src/core/main.py')
    assert one['end_line'] == source.MAX_LINES
    assert 'start_line=%d' % (source.MAX_LINES + 1) in one['more']
    two = source.read_source('pciSeq/src/core/main.py', start_line=source.MAX_LINES + 1)
    assert two['start_line'] == source.MAX_LINES + 1
    assert two['text'].split('\n')[0].lstrip().startswith('%d  ' % (source.MAX_LINES + 1))
    with pytest.raises(ValueError, match='past the end'):
        source.read_source('pciSeq/src/core/main.py', start_line=10 ** 6)


@pytest.mark.parametrize('bad', ['../setup.py', '/etc/passwd', 'pciSeq/../../README.md',
                                 'pciSeq/src/mcp/nope.py'])
def test_nothing_outside_the_package_is_read(bad):
    with pytest.raises(ValueError, match='in the pciSeq source'):
        source.read_source(bad)


def test_only_python_files_are_read():
    # a real file inside the package that is not python
    with pytest.raises(ValueError, match='only the python source'):
        source.read_source('pciSeq/src/tiling/README.md')
    with pytest.raises(ValueError, match='is a folder'):
        source.read_source('pciSeq/src/core')


def test_a_folder_lists_its_folders_and_python_files():
    out = source.list_source('pciSeq/src')
    names = {e['name']: e['type'] for e in out['entries']}
    assert names['core'] == 'dir' and names['mcp'] == 'dir'
    assert '__pycache__' not in names
    top = source.list_source('')
    assert top['dir'] == 'pciSeq' and {'name': 'src', 'type': 'dir'} in top['entries']
    with pytest.raises(ValueError, match='is a file'):
        source.list_source('pciSeq/src/core/main.py')


def test_the_chat_offers_both_and_runs_them():
    names = [t['name'] for t in chat.TOOLS]
    assert 'list_source' in names and 'read_source' in names
    # neither needs the model being fitted, so no live object here
    out = chat.call_tool(None, 'read_source', {'path': 'core/main.py', 'start_line': 5})
    assert out['start_line'] == 5 and out['path'] == 'pciSeq/src/core/main.py'
    assert chat.call_tool(None, 'list_source', {})['dir'] == 'pciSeq'
    # a refusal reaches the model as an error it can read, not as an exception
    assert 'in the pciSeq source' in chat.call_tool(None, 'read_source', {'path': '../x.py'})['error']


# ---- a finished run: the code at the commit that made it

@pytest.fixture
def github(monkeypatch):
    """A fake GitHub holding one commit, and a record of what was asked for."""
    files = {'pciSeq/src/core/main.py': 'import numpy\n' * 500,
             'pciSeq/src/core/utils/geometry.py': 'def area():\n    return 1\n'}
    asked = []

    def fetch(url):
        asked.append(url)
        if url.startswith(source.RAW + 'abc1234/'):
            path = url[len(source.RAW + 'abc1234/'):]
            if path in files:
                return files[path]
        if url.startswith(source.CONTENTS) and '?ref=abc1234' in url:
            d = url[len(source.CONTENTS):url.index('?')]
            if d == 'pciSeq/src/core':
                import json
                return json.dumps([{'name': 'utils', 'type': 'dir'}, {'name': 'main.py', 'type': 'file'},
                                   {'name': '__pycache__', 'type': 'dir'}, {'name': 'notes.txt', 'type': 'file'}])
        raise FileNotFoundError(url)

    monkeypatch.setattr(source, 'fetch_text', fetch)
    return asked


def test_a_run_reads_github_at_the_commit_that_made_it(github):
    out = source.read_source_for_run('abc1234', 'core/main.py', start_line=401)
    assert out['path'] == 'pciSeq/src/core/main.py' and out['commit'] == 'abc1234'
    assert out['start_line'] == 401 and out['end_line'] == 500 and 'more' not in out
    assert out['text'].split('\n')[0].startswith('401  import numpy')
    assert 'GitHub at commit abc1234' in out['source_is']
    assert github == [source.RAW + 'abc1234/pciSeq/src/core/main.py']


def test_the_three_path_spellings_reach_github_too(github):
    for spelling in ('pciSeq/src/core/utils/geometry.py', 'src/core/utils/geometry.py', 'core/utils/geometry.py'):
        assert source.read_source_for_run('abc1234', spelling)['path'] == 'pciSeq/src/core/utils/geometry.py'
    with pytest.raises(ValueError, match='not a path inside the repo'):
        source.read_source_for_run('abc1234', 'pciSeq/../setup.py')
    with pytest.raises(ValueError, match='only the python source'):
        source.read_source_for_run('abc1234', 'README.md')


def test_a_folder_on_github_lists_folders_and_python_files(github):
    out = source.list_source_for_run('abc1234', 'core')
    assert out['dir'] == 'pciSeq/src/core' and out['commit'] == 'abc1234'
    assert out['entries'] == [{'name': 'utils', 'type': 'dir'}, {'name': 'main.py', 'type': 'file'}]


def test_a_commit_that_is_not_on_github_is_refused_not_replaced(github):
    with pytest.raises(FileNotFoundError, match='never pushed'):
        source.read_source_for_run('0000000', 'core/main.py')
    with pytest.raises(FileNotFoundError, match='never pushed'):
        source.list_source_for_run('0000000', 'core')


def test_a_run_without_a_commit_is_refused(github):
    with pytest.raises(ValueError, match='does not record the commit'):
        source.read_source_for_run(None, 'core/main.py')
    with pytest.raises(ValueError, match='does not record the commit'):
        source.list_source_for_run('', '')
    assert github == []


def test_no_network_is_a_plain_message_not_a_crash(monkeypatch):
    import urllib.error
    import urllib.request

    def down(req, timeout=0):
        raise urllib.error.URLError('Name or service not known')
    monkeypatch.setattr(urllib.request, 'urlopen', down)
    source.fetch_text.cache_clear()
    with pytest.raises(RuntimeError, match='could not reach GitHub'):
        source.read_source_for_run('abc1234', 'core/main.py')


def test_a_file_is_fetched_once_however_many_slices(monkeypatch):
    calls = []

    def one_file(req, timeout=0):
        calls.append(req.full_url)
        class R:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return ('x\n' * 900).encode()
        return R()
    monkeypatch.setattr(urllib.request, 'urlopen', one_file)
    source.fetch_text.cache_clear()
    for start in (1, 401, 801):
        assert source.read_source_for_run('abc1234', 'core/main.py', start)['start_line'] == start
    assert len(calls) == 1
    source.fetch_text.cache_clear()
