"""Reading the pciSeq source code, for the agent.

The rule, in every way of asking: the code shown is the code that produced the
numbers. Two cases:

  * The chat of the live viewer runs inside the python that is doing the fit, so the
    installed files are the very ones running. list_source and read_source read them.
  * pciseq-mcp answers about a finished run, which records the commit of pciSeq that
    made it. list_source_for_run and read_source_for_run read the file at that commit
    on GitHub, as the desktop viewer does; a run that does not record its commit gets
    a refusal rather than some other version.

A path can be given three ways and all of them land on the same file:
    pciSeq/src/core/main.py     as it looks in the repo, and in the desktop viewer
    src/core/main.py            from the package folder
    core/main.py                as the code map page of the docs writes it
"""
import functools
import json
import urllib.error
import urllib.request
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]       # the pciSeq package folder
MAX_LINES = 400                                 # a slice, same as the desktop viewer
SKIP = {'__pycache__', 'static', '_docs', 'node_modules'}
REPO = 'acycliq/pciSeq_3d'
RAW = 'https://raw.githubusercontent.com/%s/' % REPO
CONTENTS = 'https://api.github.com/repos/%s/contents/' % REPO


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
    out = _slice(f.read_text(encoding='utf-8', errors='replace'), start_line, _shown(f))
    out['source_is'] = where()
    return out


def _slice(content, start_line, shown):
    """One slice of a file's text, with line numbers."""
    lines = content.split('\n')
    if lines and lines[-1] == '':
        lines.pop()                              # the newline at the end of the file
    start = max(1, int(start_line or 1))
    if start > len(lines):
        raise ValueError('%s has %d lines, start_line %d is past the end'
                         % (shown, len(lines), start))
    end = min(len(lines), start + MAX_LINES - 1)
    width = len(str(end))
    text = '\n'.join('%*d  %s' % (width, n, lines[n - 1]) for n in range(start, end + 1))
    out = {'path': shown, 'start_line': start, 'end_line': end,
           'total_lines': len(lines), 'text': text}
    if end < len(lines):
        out['more'] = 'the file goes on, call again with start_line=%d' % (end + 1)
    return out


# ---- a finished run: the code at the commit that made it

@functools.lru_cache(maxsize=64)
def fetch_text(url):
    """GET a url and return its text. The tests put a fake in its place.

    Cached, so reading a long file slice by slice fetches it once, and a folder
    listed twice costs one call against GitHub's rate limit."""
    req = urllib.request.Request(url, headers={'User-Agent': 'pciseq-mcp'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(url) from e
        raise RuntimeError('GitHub answered %d for %s' % (e.code, url)) from e
    except urllib.error.URLError as e:
        raise RuntimeError('could not reach GitHub (%s), so the code that made this run '
                           'cannot be shown from this machine' % e.reason) from e


def _repo_path(path):
    """The same three spellings as _resolve, as a path inside the repo on GitHub."""
    p = (path or '').strip().lstrip('/')
    if '..' in p.split('/'):
        raise ValueError('not a path inside the repo: %s' % path)
    if p in ('', 'pciSeq'):
        return 'pciSeq'
    if p.startswith('pciSeq/'):
        return p
    if p.startswith('src/'):
        return 'pciSeq/' + p
    return 'pciSeq/src/' + p


def _no_commit():
    raise ValueError('this run does not record the commit of pciSeq that made it, so the '
                     'code that produced it cannot be shown. The code map page names the '
                     'function and line for each quantity')


def _missing(what, commit):
    return FileNotFoundError('no %s on GitHub at commit %s. Either the path is wrong, call '
                             'list_source, or that commit was never pushed, in which case '
                             'the code that made this run is not available' % (what, commit))


def list_source_for_run(commit, dir=''):
    """A folder of the source at the commit that made the run."""
    if not commit:
        _no_commit()
    d = _repo_path(dir)
    try:
        items = json.loads(fetch_text(CONTENTS + d + '?ref=' + commit))
    except FileNotFoundError:
        raise _missing('folder ' + d, commit)
    if not isinstance(items, list):
        raise ValueError('%s is a file, read it with read_source' % d)
    entries = [{'name': i['name'], 'type': 'dir' if i['type'] == 'dir' else 'file'}
               for i in items
               if i['name'] not in SKIP and not i['name'].startswith('.')
               and (i['type'] == 'dir' or i['name'].endswith('.py'))]
    return {'dir': d, 'entries': entries, 'commit': commit, 'source_is': _for_run_where(commit)}


def read_source_for_run(commit, path, start_line=1):
    """One file of the source at the commit that made the run, in slices."""
    if not commit:
        _no_commit()
    f = _repo_path(path)
    if not f.endswith('.py'):
        raise ValueError('only the python source can be read, %s is not it' % f)
    try:
        content = fetch_text(RAW + commit + '/' + f)
    except FileNotFoundError:
        raise _missing(f, commit)
    out = _slice(content, start_line, f)
    out['commit'] = commit
    out['source_is'] = _for_run_where(commit)
    return out


def _for_run_where(commit):
    return 'the pciSeq source on GitHub at commit %s, the one that made this run' % commit
