"""Where the live viewer's chat keeps its API key and model.

The desktop viewer encrypts the key with Electron's safeStorage; python has no
keyring to lean on, so this follows what the mature command line tools do (gh, aws,
docker): a file in the user's config directory, readable only by the owner, with an
environment variable as the fallback.

    <config>/pciseq/chat.json          0600, {provider, model, api_key}
    ANTHROPIC_API_KEY / ZAI_API_KEY    used when the file has no key

The key never leaves this process: the settings route answers with whether a key is
set and where it came from, never with the key itself. Nothing here is logged.
"""
import json
import os
import stat
import sys
from pathlib import Path

# Anything that speaks Anthropic's Messages API. The same three the desktop
# viewer offers, so a user moving between them meets the same names.
PROVIDERS = {
    'anthropic': {'label': 'Anthropic', 'base_url': None,
                  'model': 'claude-sonnet-5', 'env': 'ANTHROPIC_API_KEY'},
    'zai': {'label': 'Z.ai (GLM)', 'base_url': 'https://api.z.ai/api/anthropic',
            'model': 'glm-5.2', 'env': 'ZAI_API_KEY'},
    'other': {'label': 'Other (Anthropic-compatible)', 'base_url': '',
              'model': '', 'env': None},
}


def config_path():
    """The settings file for this user, on this OS."""
    if sys.platform.startswith('win'):
        base = Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming')
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'pciseq' / 'chat.json'


def _read(path=None):
    path = Path(path) if path else config_path()
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _provider(data):
    p = data.get('provider', 'anthropic')
    return p if p in PROVIDERS else 'anthropic'


def load(path=None):
    """Everything the chat needs to call the model: provider, key, model, base url.

    The key is in here, so this is for the chat loop only; the page gets status().
    """
    data = _read(path)
    p = _provider(data)
    spec = PROVIDERS[p]
    key = data.get('api_key') or (os.environ.get(spec['env']) if spec['env'] else '')
    base = data.get('base_url') if p == 'other' else spec['base_url']
    return {
        'provider': p,
        'api_key': key or '',
        'key_from_env': bool(not data.get('api_key') and key),
        'model': data.get('model') or spec['model'],
        'base_url': base or None,
    }


def status(path=None):
    """What the page may know: everything except the key itself."""
    s = load(path)
    return {
        'provider': s['provider'],
        'providers': [{'id': k, 'label': v['label']} for k, v in PROVIDERS.items()],
        'has_key': bool(s['api_key']),
        'key_from_env': s['key_from_env'],
        'env_name': PROVIDERS[s['provider']]['env'],
        'model': s['model'],
        'base_url': s['base_url'] or '',
        'config_file': str(Path(path) if path else config_path()),
    }


def save(provider=None, api_key=None, model=None, base_url=None, path=None):
    """Write the settings. An empty api_key leaves the stored one alone, so the
    page can save a model change without handling the key."""
    path = Path(path) if path else config_path()
    data = _read(path)
    if provider:
        if provider not in PROVIDERS:
            raise ValueError('no provider %r' % provider)
        data['provider'] = provider
    if api_key:
        data['api_key'] = api_key.strip()
    if model and model.strip():
        data['model'] = model.strip()
    if base_url is not None:
        data['base_url'] = base_url.strip()
    if _provider(data) == 'other' and not data.get('base_url'):
        raise ValueError('give the address of the service for Other')

    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, stat.S_IRWXU)                  # 0700
    # write then move, so a reader never sees half a file, and never a moment
    # where the key sits in a world readable one
    tmp = path.with_suffix('.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)          # 0600
    return status(path)
