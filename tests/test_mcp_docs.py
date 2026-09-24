"""The documentation resources and the docs search tool.

Plain module tests first, then the server layer. Skipped when mcp is missing, as the
server tests are, but the module tests need nothing.
"""
import asyncio
import json
import pathlib

import pytest

from pciSeq.src.mcp import docs

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_docs_root_is_the_repo_checkout_here():
    """In a checkout there is no packed copy, setup.py only packs when run as
    __main__ by a real build, so the root is website/docs."""
    root = docs.docs_root()
    assert root is not None
    assert root.name == 'docs' and (root / 'index.md').is_file()


def test_every_page_is_listed_once_and_includes_are_not():
    pages = docs.list_pages()
    assert len(pages) == len(set(pages))
    assert 'index.md' in pages and 'api/mcp-server.md' in pages
    assert not any(p.startswith('_tables') or '_tables/' in p for p in pages)
    on_disk = {p.relative_to(REPO / 'website/docs').as_posix()
               for p in (REPO / 'website/docs').rglob('*.md')
               if not any(x in p.parts for x in ('node_modules', '.vitepress', '_tables'))}
    assert set(pages) == on_disk


def test_read_page_and_title():
    text = docs.read_page('api/mcp-server.md')
    assert text.startswith('---')
    assert docs.page_title(text) == 'MCP server'
    with pytest.raises(KeyError):
        docs.read_page('nope.md')
    with pytest.raises(KeyError):
        docs.read_page('../setup.py')


def test_search_finds_the_page_that_defines_a_term():
    hits = docs.search_docs('inside_cell')
    assert hits and hits[0]['page'] in ('api/working-with-results.md', 'api/mcp-server.md',
                                        'api/spatialdata-store.md')
    assert all('inside_cell' in h['text'] for h in hits)

    hits = docs.search_docs('segmentation label internal label')
    assert hits[0]['page'] == 'api/working-with-results.md'
    assert 'Cell identifiers' in hits[0]['heading'] or 'label' in hits[0]['text'].lower()


def test_search_ranks_all_words_above_some():
    hits = docs.search_docs('remove_flat_cells default', n=3)
    assert hits
    low = hits[0]['text'].lower()
    assert 'remove_flat_cells' in low and 'default' in low


def test_search_with_nothing_to_search():
    assert docs.search_docs('') == []
    assert docs.search_docs('zzzznotaword') == []


# ------------------------------------------------------------ the server

mcp = pytest.importorskip('mcp')
from pciSeq.src.mcp import server as srv  # noqa: E402


def test_docs_tool_is_registered_and_points_at_resources():
    names = {t.name for t in asyncio.run(srv.server.list_tools())}
    assert 'docs' in names
    res = asyncio.run(srv.server.call_tool('docs', {'query': 'misread density', 'n': 3}))
    body = json.loads(res.content[0].text)
    assert body['hits'] and body['hits'][0]['resource'].startswith('pciseq-docs://')


def test_resources_list_and_read():
    """One resource per page, plus the index. A template would not reach pages in
    subfolders, so every page is registered on its own with its title."""
    listed = asyncio.run(srv.server.list_resources())
    static = {str(r.uri) for r in listed}
    assert 'pciseq-docs://index' in static
    assert {'pciseq-docs://' + p for p in docs.list_pages()} <= static
    by_uri = {str(r.uri): r.name for r in listed}
    assert by_uri['pciseq-docs://api/mcp-server.md'] == 'MCP server'

    index = asyncio.run(srv.server.read_resource('pciseq-docs://index'))
    text = list(index)[0].content
    assert 'pciseq-docs://api/mcp-server.md' in text

    page = asyncio.run(srv.server.read_resource('pciseq-docs://api/mcp-server.md'))
    assert '# MCP server' in list(page)[0].content
