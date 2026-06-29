"""Tests for `llm_factory` — provider-agnostic LLM dispatch.

Covers the lazy key validation, cache key shape, ConfigError messages,
and the dispatcher's provider→kwargs mapping. No real LangChain
clients are constructed — `init_chat_model` is patched so the tests
run without any provider's API key set in the environment.
"""

from unittest.mock import patch

import pytest

from knowledge_agent import llm_factory
from knowledge_agent.llm_factory import (
    ConfigError,
    _validate_provider_config,
    clear_cache,
    get_llm,
)


# init_chat_model is imported lazily inside `_build_llm`, so we patch
# the source module — not knowledge_agent.llm_factory — to intercept.
_INIT_PATCH = "langchain.chat_models.init_chat_model"


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
        self.anthropic_api_key = anthropic_api_key
        self.openai_api_key = openai_api_key
        self.google_api_key = google_api_key
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
    with patch(
        "knowledge_agent.llm_factory.get_settings", return_value=settings
    ):
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
            _validate_provider_config("anthropic")


def test_openai_missing_key_raises_config_error():
    settings = _FakeSettings(llm_provider="openai", openai_api_key=None)
    with patch(
        "knowledge_agent.llm_factory.get_settings", return_value=settings
    ):
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            _validate_provider_config("openai")


def test_google_missing_key_raises_config_error():
    settings = _FakeSettings(llm_provider="google", google_api_key=None)
    with patch(
        "knowledge_agent.llm_factory.get_settings", return_value=settings
    ):
        with pytest.raises(ConfigError, match="GOOGLE_API_KEY"):
            _validate_provider_config("google")


def test_ollama_missing_base_url_raises_config_error():
    settings = _FakeSettings(llm_provider="ollama", ollama_base_url="")
    with patch(
        "knowledge_agent.llm_factory.get_settings", return_value=settings
    ):
        with pytest.raises(ConfigError, match="OLLAMA_BASE_URL"):
            _validate_provider_config("ollama")


def test_unknown_provider_raises_config_error():
    settings = _FakeSettings(llm_provider="anthropic")
    with patch(
        "knowledge_agent.llm_factory.get_settings", return_value=settings
    ):
        with pytest.raises(ConfigError, match="unknown llm_provider"):
            _validate_provider_config("not-a-real-provider")


# ---- dispatch ----


def test_get_llm_dispatches_anthropic_with_api_key():
    settings = _FakeSettings(
        llm_provider="anthropic", anthropic_api_key="sk-anthropic"
    )
    with (
        patch(
            "knowledge_agent.llm_factory.get_settings", return_value=settings
        ),
        patch(_INIT_PATCH) as mock_init,
    ):
        mock_init.return_value = "fake-llm"
        result = get_llm("claude-sonnet-4-6", 0.0)
    assert result == "fake-llm"
    args, kwargs = mock_init.call_args
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["model_provider"] == "anthropic"
    assert kwargs["api_key"] == "sk-anthropic"
    assert kwargs["temperature"] == 0.0


def test_get_llm_dispatches_openai_with_api_key():
    settings = _FakeSettings(
        llm_provider="openai",
        openai_api_key="sk-openai",
    )
    with (
        patch(
            "knowledge_agent.llm_factory.get_settings", return_value=settings
        ),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("gpt-4o", 0.7)
    args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "openai"
    assert kwargs["api_key"] == "sk-openai"
    assert kwargs["temperature"] == 0.7


def test_get_llm_dispatches_ollama_with_base_url_not_api_key():
    settings = _FakeSettings(
        llm_provider="ollama",
        ollama_base_url="http://gpu-box:11434",
    )
    with (
        patch(
            "knowledge_agent.llm_factory.get_settings", return_value=settings
        ),
        patch(_INIT_PATCH) as mock_init,
    ):
        get_llm("qwen2.5:7b", 0.0)
    args, kwargs = mock_init.call_args
    assert kwargs["model_provider"] == "ollama"
    assert "api_key" not in kwargs
    assert kwargs["base_url"] == "http://gpu-box:11434"


# ---- cache ----


def test_get_llm_caches_by_model_and_temperature():
    settings = _FakeSettings(llm_provider="anthropic")
    with (
        patch(
            "knowledge_agent.llm_factory.get_settings", return_value=settings
        ),
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
        patch(
            "knowledge_agent.llm_factory.get_settings", return_value=settings
        ),
        patch(_INIT_PATCH) as mock_init,
    ):
        mock_init.return_value = "fake-llm"
        get_llm("claude-sonnet-4-6", 0.0)
        clear_cache()
        get_llm("claude-sonnet-4-6", 0.0)
    assert mock_init.call_count == 2
