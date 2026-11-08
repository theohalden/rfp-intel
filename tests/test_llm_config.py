"""Select credentials by provider without contacting any API or logging secrets."""
import pytest
import config


def setup_env(monkeypatch):
    monkeypatch.setattr(config, 'load_dotenv', lambda *a, **k: None)
    for name in ('LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_MODEL', 'ANTHROPIC_API_KEY', 'ANTHROPIC_MODEL'):
        monkeypatch.delenv(name, raising=False)


def test_openai_config_does_not_require_anthropic(monkeypatch):
    setup_env(monkeypatch)
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-placeholder')
    monkeypatch.setenv('OPENAI_MODEL', 'configured-model')
    assert config.llm_environment() == ('test-placeholder', 'configured-model')


def test_missing_openai_config_explains_exact_fields(monkeypatch):
    setup_env(monkeypatch)
    with pytest.raises(ValueError, match='OPENAI_API_KEY and OPENAI_MODEL'):
        config.llm_environment()


def test_anthropic_selection_uses_its_own_credentials(monkeypatch):
    setup_env(monkeypatch)
    monkeypatch.setenv('LLM_PROVIDER', 'anthropic')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-placeholder')
    monkeypatch.setenv('ANTHROPIC_MODEL', 'configured-model')
    assert config.llm_environment() == ('test-placeholder', 'configured-model')


def test_tested_mini_model_has_reasoning_enabled_without_changing_other_models(monkeypatch):
    from agents import llm
    from config import Settings
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.delenv('OPENAI_REASONING_EFFORT', raising=False)
    monkeypatch.setattr(llm, 'llm_environment', lambda: ('test-placeholder', 'gpt-5.4-mini'))
    assert llm.create_llm(Settings()).reasoning == {'effort': 'medium'}
    monkeypatch.setattr(llm, 'llm_environment', lambda: ('test-placeholder', 'configured-model'))
    assert llm.create_llm(Settings()).reasoning is None


def test_reasoning_effort_is_explicitly_overridable_and_validated(monkeypatch):
    from agents import llm
    from config import Settings
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    monkeypatch.setattr(llm, 'llm_environment', lambda: ('test-placeholder', 'gpt-5.4-mini'))
    monkeypatch.setenv('OPENAI_REASONING_EFFORT', 'low')
    assert llm.create_llm(Settings()).reasoning == {'effort': 'low'}
    monkeypatch.setenv('OPENAI_REASONING_EFFORT', 'invalid')
    with pytest.raises(ValueError, match='OPENAI_REASONING_EFFORT'):
        llm.create_llm(Settings())


def test_anthropic_adapter_drops_openai_only_structured_output_options():
    from agents.llm import _SchemaAdapter
    class Client:
        def with_structured_output(self, schema, **kw):
            return kw
    out = _SchemaAdapter(Client(), drop=('method', 'strict')).with_structured_output(int, method='json_schema', strict=True, include_raw=True)
    assert out == {'include_raw': True}
