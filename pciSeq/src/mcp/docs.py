"""The documentation pages, for an agent that cannot open a browser.

Two things the server builds on: the list of pages and their text, exposed as MCP
resources, and a keyword search over them, exposed as a tool. Both are plain
functions here so they can be tested and used without mcp.

Where the pages come from, in order: a copy that setup.py packs into the wheel at
build time (pciSeq/src/mcp/_docs), so a pip install has them, and failing that the
website/docs folder of a repo checkout, found by walking up from this file. No
index, no embeddings: the whole corpus is 33 pages, it is searched on the spot.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SKIP = ('node_modules', '.vitepress', '_tables')


def docs_root():
    """The folder holding the markdown pages, or None when there is none."""
    packed = HERE / '_docs'
    if (packed / 'index.md').is_file():
        return packed
    for parent in HERE.parents:
        cand = parent / 'website' / 'docs'
        if (cand / 'index.md').is_file():
            return cand
    return None


def list_pages():
    """Every page, as its path relative to the docs root, sorted. Includes and
    build folders are left out."""
    root = docs_root()
    if root is None:
        return []
    pages = [p.relative_to(root).as_posix() for p in root.rglob('*.md')
             if not any(part in _SKIP for part in p.relative_to(root).parts)]
    return sorted(pages)


def read_page(path):
    """The markdown of one page. Raises KeyError for a path that is not a page."""
    root = docs_root()
    if root is None or path not in list_pages():
        raise KeyError('no docs page %r' % path)
    return (root / path).read_text()


def page_title(text):
    """The first heading of a page, or its description from the frontmatter."""
    m = re.search(r'^# (.+)$', text, re.M)
    if m:
        return m.group(1).strip()
    m = re.search(r'^description:\s*(.+)$', text, re.M)
    return m.group(1).strip() if m else ''


def _paragraphs(text):
    """(heading, paragraph) pairs, heading being the nearest one above. Frontmatter
    is dropped; code blocks are kept, config keys live in them."""
    text = re.sub(r'\A---.*?---\s*', '', text, count=1, flags=re.S)
    heading = ''
    out = []
    for block in re.split(r'\n\s*\n', text):
        block = block.strip()
        if not block:
            continue
        m = re.match(r'#+ (.+)', block)
        if m:
            heading = m.group(1).strip()
            rest = block[m.end():].strip()
            if rest:
                out.append((heading, rest))
            continue
        out.append((heading, block))
    return out


def search_docs(query, n=5):
    """Paragraphs matching a query, best first.

    Words are matched case-insensitively. A paragraph holding every word of the
    query outranks one holding some of them; ties go to the paragraph with more
    hits. Returns dicts with the page, its title, the nearest heading and the text.
    """
    words = [w for w in re.findall(r'[A-Za-z0-9_]+', query.lower()) if len(w) > 1]
    if not words:
        return []
    hits = []
    for page in list_pages():
        text = read_page(page)
        title = page_title(text)
        for heading, para in _paragraphs(text):
            low = para.lower()
            present = [w for w in words if w in low]
            if not present:
                continue
            count = sum(low.count(w) for w in present)
            hits.append((len(present), count, page, title, heading, para))
    hits.sort(key=lambda h: (-h[0], -h[1], h[2]))
    return [{'page': h[2], 'title': h[3], 'heading': h[4], 'text': h[5]} for h in hits[:n]]
