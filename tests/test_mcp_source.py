"""The live viewer's chat can read the pciSeq source: the files on this machine,
which are the ones running the fit. No model and no fit needed for any of this."""
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
