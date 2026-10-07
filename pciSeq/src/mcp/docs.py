"""The documentation pages, for an agent that cannot open a browser.

Two things the server builds on: the list of pages and their text, exposed as MCP
resources, and a keyword search over them, exposed as a tool. Both are plain
functions here so they can be tested and used without mcp.

Where the pages come from. With a finished run open: the copy saved inside it, so
a run is explained by the documentation of the pciSeq that produced it. A run with
no copy inside gets no documentation, the server refuses (server.py
_run_has_no_docs) rather than hand over pages of another version. With no finished
run open (pciSeq.fit saving the pages into a new run, the live viewer's chat):
the pages of the pciSeq installed here, which is the website/docs folder of a repo
checkout, found by walking up from this file, or the copy setup.py packs into the
wheel at build time (pciSeq/src/mcp/_docs), so a pip install has them. No index, no
embeddings: the whole corpus is a few dozen pages, it is searched on the spot.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SKIP = ('node_modules', '.vitepress', '_tables')
# Pages never handed to an agent. convergence.md is about the oscillation, which
# Dimitris keeps out of anything that leaves the repo, so it stays a local draft
# and is not searched, read or shipped. Same list in pack_docs, setup.py.
LEFT_OUT = ('the-model/convergence.md',)


# The pages saved inside the open run (page -> markdown), set by open_run. A run
# carries the documentation of the commit that fitted it, which is the one that
# describes its numbers; the copy on this machine may be newer. The viewer does the
# same (electron/run.js docsFromRun).
_run_pages = None


def use_run_pages(pages):
    """Read the documentation from these pages until told otherwise. Called with the
    docs saved inside a run when one is opened, and with None when it has none."""
    global _run_pages
    _run_pages = dict(pages) if pages else None


def source():
    """Where the pages being served come from, in words, for the agent."""
    if _run_pages:
        return ('the documentation saved inside this run when it was fitted, so it '
                'describes the pciSeq that produced these numbers')
    if docs_root() is None:
        return 'no documentation pages were found on this machine'
    return 'the documentation of the pciSeq installed on this machine'


def docs_root():
    """The folder holding the markdown pages, or None when there is none.

    The live website/docs of a checkout first, the copy setup.py packs into _docs
    only when there is no checkout above (a pip install). The other way round, an
    editable install (pip install -e, which also runs the packing) would read the
    docs as they were on the day of the install, and never see an edit after it.
    """
    for parent in HERE.parents:
        cand = parent / 'website' / 'docs'
        if (cand / 'index.md').is_file():
            return cand
    packed = HERE / '_docs'
    if (packed / 'index.md').is_file():
        return packed
    return None


def list_pages():
    """Every page, as its path relative to the docs root, sorted. Includes and
    build folders are left out."""
    if _run_pages:
        return sorted(p for p in _run_pages if p not in LEFT_OUT)
    root = docs_root()
    if root is None:
        return []
    pages = [p.relative_to(root).as_posix() for p in root.rglob('*.md')
             if not any(part in _SKIP for part in p.relative_to(root).parts)]
    pages = [p for p in pages if p not in LEFT_OUT]
    return sorted(pages)


def read_page(path):
    """The markdown of one page. Raises KeyError for a path that is not a page."""
    if _run_pages:
        if path not in list_pages():
            raise KeyError('no docs page %r' % path)
        return _run_pages[path]
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


def page_summary(text, max_length=160):
    """What a page is about, in one line. The description in the frontmatter is
    written for exactly this, so it wins; only a third of the pages carry one, and
    the copy saved inside a run drops the frontmatter, so the fallback is the page's
    first real paragraph, cut short. Tables, code, containers, html and a lone bold
    lead-in such as "**Simplifications.**" are passed over: they say nothing about
    the page."""
    described = re.search(r'^description:\s*(.+)$', text, re.M)
    if described:
        return described.group(1).strip()
    for _, para in _paragraphs(text):
        if re.match(r'^[|`:<]', para):
            continue
        line = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', re.sub(r'\s+', ' ', para)).strip()
        if not line or re.match(r'^\*\*[^*]+\*\*[.:]?$', line):
            continue
        return line if len(line) <= max_length else re.sub(r'\s+\S*$', '', line[:max_length - 1]) + '...'
    return ''


def _paragraphs(text):
    """(heading, paragraph) pairs, heading being the nearest one above. Frontmatter
    is dropped; code blocks are kept, config keys live in them. A table is split
    into its rows, one paragraph each: a row is the unit a reader wants back (a
    setting, a function and its line), and a long table would otherwise outrank
    the prose that explains the term."""
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
        if block.startswith('|'):
            out.extend((heading, row) for row in block.splitlines()
                       if not re.match(r'^\|[\s\-:|]*\|$', row))
            continue
        out.append((heading, block))
    return out


def search_docs(query, n=5):
    """Paragraphs matching a query, best first.

    Words are matched case-insensitively. A paragraph holding every word of the
    query outranks one holding some of them; among those, prose comes before a
    table row, since a paragraph explains and a row only points; then the one
    with more hits. Returns dicts with the page, its title, the nearest heading
    and the text.
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
            hits.append((len(present), count, para.startswith('|'), page, title, heading, para))
    hits.sort(key=lambda h: (-h[0], h[2], -h[1], h[3]))
    return [{'page': h[3], 'title': h[4], 'heading': h[5], 'text': h[6]} for h in hits[:n]]
