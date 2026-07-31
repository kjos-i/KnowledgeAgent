"""Tests for `llm_factory` — provider-agnostic LLM dispatch.

Covers the lazy key validation, cache key shape, ConfigError messages,
and the dispatcher's provider→kwargs mapping. No real LangChain
clients are constructed — `init_chat_model` is patched so the tests
run without any provider's API key set in the environment.
"""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from knowledge_agent.llm_factory import (
    ConfigError,
    _validate_provider_config,
    clear_cache,
    format_model_ref,
    get_llm,
    get_llm_ref,
    parse_model_ref,
    supports_temperature,
    to_model_ref,
)

# init_chat_model is imported lazily inside `_build_llm`, so we patch
# the source module — not knowledge_agent.llm_factory — to intercept.
_INIT_PATCH = "langchain.chat_models.init_chat_model"


def _secret(value: str | None) -> SecretStr | None:
    """Wrap a raw test key as `SecretStr` (matching real `Settings`), or None."""
    return SecretStr(value) if value is not None else None


class _FakeSettings:
    """Minimal stand-in for `Settings` — just the fields the factory reads.

    Rate-limit fields default to None (no limiter wired) so legacy tests
    that don't care about rate limiting see the factory's "no limiter"
    branch. Tests that exercise rate-limiter wiring pass concrete floats.
    """

    def __init__(
        self,
        *,
        llm_provider: str = "anthropic",
        anthropic_api_key: str = "sk-anthropic-stub",
        openai_api_key: str | None = None,
        google_api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
        anthropic_requests_per_second: float | None = None,
        openai_requests_per_second: float | None = None,
        google_requests_per_second: float | None = None,
        ollama_requests_per_second: float | None = None,
        llm_max_retries: int = 3,
    ):
        self.llm_provider = llm_provider
        self.anthropic_api_key = _secret(anthropic_api_key)
        self.openai_api_key = _secret(openai_api_key)
        self.google_api_key = _secret(google_api_key)
        self.ollama_base_url = ollama_base_url
        self.anthropic_requests_per_second = anthropic_requests_per_second
        self.openai_requests_per_second = openai_requests_per_second
        self.google_requests_per_second = google_requests_per_second
        self.ollama_requests_per_second = ollama_requests_per_second
        self.llm_max_retries = llm_max_retries


@pytest.fixture(autouse=True)
def _clear_factory_cache():
    """Drop cached clients between tests so per-test settings stick."""
    clear_cache()
    yield
    clear_cache()


# ---- lazy validation ----


def test_anthropic_missing_key_raises_config_error():
    settings = _FakeSettings(llm_provider="anthropic", anthropic_api_key="")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"),
    ):
        _validate_provider_config("anthropic")


def test_openai_missing_key_raises_config_error():
    settings = _FakeSettings(llm_provider="openai", openai_api_key=None)
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        pytest.raises(ConfigError, match="OPENAI_API_KEY"),
    ):
        _validate_provider_config("openai")


def test_google_missing_key_raises_config_error():
    settings = _FakeSettings(llm_provider="google", google_api_key=None)
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        pytest.raises(ConfigError, match="GOOGLE_API_KEY"),
    ):
        _validate_provider_config("google")


def test_ollama_missing_base_url_raises_config_error():
    settings = _FakeSettings(llm_provider="ollama", ollama_base_url="")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        pytest.raises(ConfigError, match="OLLAMA_BASE_URL"),
    ):
        _validate_provider_config("ollama")


def test_unknown_provider_raises_config_error():
    settings = _FakeSettings(llm_provider="anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        pytest.raises(ConfigError, match="unknown llm_provider"),
    ):
        _validate_provider_config("not-a-real-provider")


# ---- public supports_temperature (drives the GUI slider greying) ----


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5", "claude-fable-5"],
)
def test_supports_temperature_false_for_sampling_free_anthropic(model):
    assert supports_temperature("anthropic", model) is False


@pytest.mark.parametrize("model", ["claude-sonnet-4-6", "claude-haiku-4-5"])
def test_supports_temperature_true_for_older_anthropic(model):
    assert supports_temperature("anthropic", model) is True


def test_supports_temperature_true_for_non_anthropic_and_empty():
    # Non-Anthropic providers always accept temperature; an empty/unset
    # model is treated as "supported" so a blank picker stays enabled.
    assert supports_temperature("openai", "claude-opus-4-8") is True
    assert supports_temperature("anthropic", "") is True


# ---- dispatch ----


def test_get_llm_dispatches_anthropic_with_api_key():
    settings = _FakeSettings(llm_provider="anthropic", anthropic_api_key="sk-anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        mock_init.return_value = "fake-llm"
        result = get_llm("claude-sonnet-4-6", 0.0)
    assert result == "fake-llm"
    _args, kwargs = mock_init.call_args
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["model_provider"] == "anthropic"
    assert kwargs["api_key"] == "sk-anthropic"
    assert kwargs["temperature"] == 0.0


def test_get_llm_omits_temperature_for_sampling_free_anthropic_model():
    """The newest Anthropic models (Opus 4.8/4.7, Sonnet 5, Fable 5) reject
    `temperature` with a 400 — the factory must NOT forward it for them,
    while other kwargs still flow through."""
    settings = _FakeSettings(llm_provider="anthropic", anthropic_api_key="sk-anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("claude-opus-4-8", 0.3)
    _args, kwargs = mock_init.call_args
    assert "temperature" not in kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["api_key"] == "sk-anthropic"


def test_get_llm_keeps_temperature_for_older_anthropic_model():
    """Older Claude models still accept temperature — it must be forwarded."""
    settings = _FakeSettings(llm_provider="anthropic", anthropic_api_key="sk-anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("claude-haiku-4-5", 0.5)
    _args, kwargs = mock_init.call_args
    assert kwargs["temperature"] == 0.5


def test_get_llm_keeps_temperature_for_non_anthropic_provider():
    """The sampling-free rule is Anthropic-only — OpenAI/Ollama/etc. keep
    temperature even for identically-named models."""
    settings = _FakeSettings(llm_provider="openai", openai_api_key="sk-openai")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("gpt-5", 0.2)
    _args, kwargs = mock_init.call_args
    assert kwargs["temperature"] == 0.2


def test_get_llm_dispatches_openai_with_api_key():
    settings = _FakeSettings(
        llm_provider="openai",
        openai_api_key="sk-openai",
    )
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("gpt-4o", 0.7)
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "openai"
    assert kwargs["api_key"] == "sk-openai"
    assert kwargs["temperature"] == 0.7


def test_get_llm_dispatches_ollama_with_base_url_not_api_key():
    settings = _FakeSettings(
        llm_provider="ollama",
        ollama_base_url="http://gpu-box:11434",
    )
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("qwen2.5:7b", 0.0)
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "ollama"
    assert "api_key" not in kwargs
    assert kwargs["base_url"] == "http://gpu-box:11434"


# ---- cache ----


def test_get_llm_caches_by_model_and_temperature():
    settings = _FakeSettings(llm_provider="anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        mock_init.return_value = "fake-llm"
        get_llm("claude-sonnet-4-6", 0.0)
        get_llm("claude-sonnet-4-6", 0.0)  # cache hit
        get_llm("claude-sonnet-4-6", 0.7)  # cache miss (temp differs)
    assert mock_init.call_count == 2


def test_clear_cache_drops_cached_clients():
    settings = _FakeSettings(llm_provider="anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        mock_init.return_value = "fake-llm"
        get_llm("claude-sonnet-4-6", 0.0)
        clear_cache()
        get_llm("claude-sonnet-4-6", 0.0)
    assert mock_init.call_count == 2


# ---- provider:model reference helpers (provider-per-model transition) ----


def test_format_model_ref_composes():
    assert format_model_ref("anthropic", "claude-sonnet-5") == "anthropic:claude-sonnet-5"
    assert format_model_ref("ollama", "qwen2.5:7b") == "ollama:qwen2.5:7b"


def test_parse_model_ref_known_provider_prefix():
    assert parse_model_ref("anthropic:claude-sonnet-5") == ("anthropic", "claude-sonnet-5")
    # First-colon split, so a multi-colon Ollama tag survives intact.
    assert parse_model_ref("ollama:qwen2.5:7b") == ("ollama", "qwen2.5:7b")


def test_parse_model_ref_bare_or_non_provider_prefix():
    # No prefix (legacy) → provider None, the whole string is the model.
    assert parse_model_ref("claude-sonnet-5") == (None, "claude-sonnet-5")
    # A colon whose left side isn't a known provider (a bare Ollama tag) is
    # treated as bare, not mis-read as "provider qwen2.5".
    assert parse_model_ref("qwen2.5:7b") == (None, "qwen2.5:7b")


def test_format_parse_round_trip():
    for provider, model in [("anthropic", "claude-opus-4-8"), ("ollama", "qwen2.5:7b")]:
        assert parse_model_ref(format_model_ref(provider, model)) == (provider, model)


# ---- explicit provider override (foundation for cross-provider picks) ----


def test_get_llm_explicit_provider_overrides_global():
    """An explicit `provider=` dispatches through it, ignoring the global
    `settings.llm_provider` — the foundation for cross-provider model picks."""
    settings = _FakeSettings(llm_provider="anthropic", openai_api_key="sk-openai")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("gpt-4o", 0.2, provider="openai")
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "openai"
    assert kwargs["api_key"] == "sk-openai"


def test_get_llm_provider_none_falls_back_to_global():
    """`provider=None` keeps the legacy behavior — dispatch through the global
    active provider — so every existing call site is unaffected."""
    settings = _FakeSettings(llm_provider="openai", openai_api_key="sk-openai")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("gpt-4o", 0.2)  # no provider= → global
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "openai"


def test_get_llm_ref_dispatches_by_provider_prefix():
    """get_llm_ref reads the provider out of a 'provider:model' ref and
    dispatches through it — even when it differs from the global provider."""
    settings = _FakeSettings(llm_provider="anthropic", openai_api_key="sk-openai")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm_ref("openai:gpt-4o", 0.0)  # composite → openai, not global anthropic
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "openai"
    assert kwargs["model"] == "gpt-4o"


def test_get_llm_ref_bare_ref_uses_global_provider():
    """A bare/legacy ref (no provider prefix) falls back to the global provider,
    so pre-migration configs keep working unchanged."""
    settings = _FakeSettings(llm_provider="anthropic", anthropic_api_key="sk-anthropic")
    with (
        patch("knowledge_agent.llm_factory.get_settings", return_value=settings),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm_ref("claude-sonnet-4-6", 0.0)
    _args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "anthropic"
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_to_model_ref_wraps_bare_and_preserves_composite():
    """to_model_ref wraps a bare/legacy value with the fallback provider and
    leaves an already-composite ref untouched (the picker normalizer)."""
    assert to_model_ref("claude-sonnet-5", "anthropic") == "anthropic:claude-sonnet-5"
    assert to_model_ref("openai:gpt-4o", "anthropic") == "openai:gpt-4o"  # already composite
    assert to_model_ref("qwen2.5:7b", "ollama") == "ollama:qwen2.5:7b"  # bare Ollama tag wrapped
