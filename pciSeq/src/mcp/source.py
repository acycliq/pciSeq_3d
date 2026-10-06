"""Reading the pciSeq source code, for the chat of the live viewer.

The chat runs inside the python that is doing the fit, so the files read here are the
very ones that are running. That is why there is no GitHub in this, unlike the desktop
viewer, which fetches the code at the commit a finished run was made with.

A path can be given three ways and all of them land on the same file:
    pciSeq/src/core/main.py     as it looks in the repo, and in the desktop viewer
    src/core/main.py            from the package folder
    core/main.py                as the code map page of the docs writes it
"""
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]       # the pciSeq package folder
MAX_LINES = 400                                 # a slice, same as the desktop viewer
SKIP = {'__pycache__', 'static', '_docs', 'node_modules'}


def where():
    """Which code this is, in words, for the agent to pass on."""
    from pciSeq import __version__, __commit__
    return ('the pciSeq installed on this machine, version %s, commit %s. It is the code '
            'that is running this fit' % (__version__, __commit__))


def _resolve(path):
    """The file or folder a path points at. Anything outside the package is refused,
    so '../' and absolute paths get nowhere."""
    p = (path or '').strip().lstrip('/')
    if p == 'pciSeq':
        p = ''
    elif p.startswith('pciSeq/'):
        p = p[len('pciSeq/'):]
    for base in (PKG, PKG / 'src'):
        cand = (base / p).resolve()
        inside = cand == PKG or PKG in cand.parents
        if inside and cand.exists() and not (SKIP & set(cand.relative_to(PKG).parts)):
            return cand
    raise ValueError('no %s in the pciSeq source. Call list_source to see what is there, '
                     'the model code is under pciSeq/src/core' % (path or 'such path'))


def _shown(p):
    """A path the way the repo writes it, pciSeq/src/core/main.py."""
    rel = p.relative_to(PKG).as_posix()
    return 'pciSeq' if rel == '.' else 'pciSeq/' + rel


def list_source(dir=''):
    """The folders and python files in one folder of the source."""
    d = _resolve(dir)
    if not d.is_dir():
        raise ValueError('%s is a file, read it with read_source' % _shown(d))
    entries = []
    for c in sorted(d.iterdir(), key=lambda c: (c.is_file(), c.name)):
        if c.name in SKIP or c.name.startswith('.'):
            continue
        if c.is_dir():
            entries.append({'name': c.name, 'type': 'dir'})
        elif c.suffix == '.py':
            entries.append({'name': c.name, 'type': 'file'})
    return {'dir': _shown(d), 'entries': entries, 'source_is': where()}


def read_source(path, start_line=1):
    """One python file, with line numbers, at most MAX_LINES lines from start_line."""
    f = _resolve(path)
    if f.is_dir():
        raise ValueError('%s is a folder, list it with list_source' % _shown(f))
    if f.suffix != '.py':
        raise ValueError('only the python source can be read, %s is not it' % f.name)
    lines = f.read_text(encoding='utf-8', errors='replace').split('\n')
    if lines and lines[-1] == '':
        lines.pop()                              # the newline at the end of the file
    start = max(1, int(start_line or 1))
    if start > len(lines):
        raise ValueError('%s has %d lines, start_line %d is past the end'
                         % (_shown(f), len(lines), start))
    end = min(len(lines), start + MAX_LINES - 1)
    width = len(str(end))
    text = '\n'.join('%*d  %s' % (width, n, lines[n - 1]) for n in range(start, end + 1))
    out = {'path': _shown(f), 'start_line': start, 'end_line': end,
           'total_lines': len(lines), 'text': text, 'source_is': where()}
    if end < len(lines):
        out['more'] = 'the file goes on, call again with start_line=%d' % (end + 1)
    return out
