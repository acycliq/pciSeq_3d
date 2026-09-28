"""Where the pciSeq viewer finds the MCP server: a small registry file.

The viewer is a desktop app. It cannot see the shell, so it does not know which
conda env or venv holds pciSeq_3d[mcp]. The same problem Jupyter has with kernels,
solved the same way: run `pciseq-mcp --register` once inside the env, and it writes
down how to start the server from there. The viewer reads the file and lists what is
registered. Nothing is guessed, and a conda env, a venv or a plain python all look
the same.

The file lives in the user's config folder, the one Electron calls appData, so the
viewer finds it with app.getPath('appData') and no guessing on its side either:

    Linux    $XDG_CONFIG_HOME/pciseq/mcp_servers.json, or ~/.config/pciseq/...
    macOS    ~/Library/Application Support/pciseq/mcp_servers.json
    Windows  %APPDATA%\\pciseq\\mcp_servers.json

One entry per python, so registering again from the same env just refreshes it.
"""
import datetime
import json
import os
import site
import sys
from pathlib import Path


def registry_path():
    """The registry file for this user, on this OS."""
    if sys.platform.startswith('win'):
        base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'pciseq' / 'mcp_servers.json'


def _read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {'servers': []}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)       # all at once, so the viewer never reads half a file


def this_entry(name=None):
    """How to start the server from the python running now."""
    import pciSeq
    entry = {
        'name': name or Path(sys.prefix).name,
        'python': sys.executable,
        'args': ['-m', 'pciSeq.src.mcp.server'],
        'env': {},
        'pciseq_version': pciSeq.__version__,
        'commit': pciSeq.__commit__,
        'registered_at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    # pciSeq imported from a checkout rather than installed (PYTHONPATH or the cwd):
    # the viewer starts python somewhere else, so it has to be told where the code is
    pkg_root = str(Path(pciSeq.__file__).resolve().parent.parent)
    installed = [str(Path(p).resolve()) for p in site.getsitepackages() + [site.getusersitepackages()]]
    if pkg_root not in installed:
        entry['env']['PYTHONPATH'] = pkg_root
    return entry


def register(name=None, path=None):
    """Add or refresh the entry for this python. Returns the entry."""
    path = Path(path) if path else registry_path()
    data = _read(path)
    entry = this_entry(name)
    data['servers'] = [s for s in data['servers'] if s.get('python') != entry['python']]
    data['servers'].append(entry)
    _write(path, data)
    return entry


def unregister(path=None, python=None):
    """Remove the entry for this python (or the one given). True if there was one."""
    path = Path(path) if path else registry_path()
    data = _read(path)
    python = python or sys.executable
    before = len(data['servers'])
    data['servers'] = [s for s in data['servers'] if s.get('python') != python]
    _write(path, data)
    return len(data['servers']) < before


def list_entries(path=None):
    """Every registered entry, each with 'exists': whether its python is still there."""
    path = Path(path) if path else registry_path()
    return [dict(s, exists=Path(s.get('python', '')).exists()) for s in _read(path)['servers']]
