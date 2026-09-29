"""Where the live viewer's chat keeps its API key.

The key is the thing worth testing here: that the file is written the way a
credentials file should be (owner only), that the environment is the fallback, and
that whatever the page is told never contains the key itself.
"""
import json
import stat

import pytest

from pciSeq.src.mcp import chat_settings


@pytest.fixture
def cfg(tmp_path):
    return tmp_path / 'pciseq' / 'chat.json'


def test_the_settings_file_is_written_for_the_owner_only(cfg):
    chat_settings.save(provider='anthropic', api_key='sk-secret', model='claude-sonnet-5', path=cfg)
    assert json.loads(cfg.read_text())['api_key'] == 'sk-secret'
    assert stat.S_IMODE(cfg.stat().st_mode) == 0o600
    assert stat.S_IMODE(cfg.parent.stat().st_mode) == 0o700


def test_the_page_never_sees_the_key(cfg):
    chat_settings.save(provider='anthropic', api_key='sk-secret', path=cfg)
    st = chat_settings.status(cfg)
    assert st['has_key'] is True and st['key_from_env'] is False
    assert 'sk-secret' not in json.dumps(st)


def test_the_environment_is_the_fallback(cfg, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-from-env')
    st = chat_settings.status(cfg)
    assert st['has_key'] is True and st['key_from_env'] is True
    assert chat_settings.load(cfg)['api_key'] == 'sk-from-env'
    # a saved key wins over the environment
    chat_settings.save(api_key='sk-saved', path=cfg)
    assert chat_settings.load(cfg)['api_key'] == 'sk-saved'
    assert chat_settings.status(cfg)['key_from_env'] is False


def test_each_provider_brings_its_own_address_and_model(cfg):
    chat_settings.save(provider='zai', api_key='zai-key', path=cfg)
    s = chat_settings.load(cfg)
    assert s['base_url'] == 'https://api.z.ai/api/anthropic' and s['model'] == 'glm-5.2'
    with pytest.raises(ValueError, match='address'):
        chat_settings.save(provider='other', path=cfg)
    with pytest.raises(ValueError, match='no provider'):
        chat_settings.save(provider='nope', path=cfg)


def test_saving_a_model_does_not_need_the_key_again(cfg):
    chat_settings.save(provider='anthropic', api_key='sk-secret', path=cfg)
    chat_settings.save(model='claude-opus-5', path=cfg)
    s = chat_settings.load(cfg)
    assert s['api_key'] == 'sk-secret' and s['model'] == 'claude-opus-5'
