"""Unit tests for `knowledge_agent.config`.

The Settings model owns every runtime knob the rest of the codebase
consults. These tests pin:

  - Required-no-default fields fail-fast when missing (API keys + the
    four DB credentials). Catching this here prevents the failure
    mode where a forgotten config silently falls back to a default
    pointing at the wrong instance.
  - The retrieval-window invariant
    (`top_k <= rrf_rank_window_size <= num_candidates`) catches
    misconfiguration before LanceDB sees a broken combination.
  - `disable_env_file()` is one-way and clears the `get_settings`
    cache (so a future GUI can refuse to inherit dev keys).
  - `load_test_env()` clears the cache (so a smoke that calls it
    before any `get_settings()` succeeds, AND a re-call gets fresh
    test credentials).
  - `get_settings()` returns the same instance on repeated calls
    (lru_cache behaviour the rest of the codebase relies on).

Notes on test isolation:

  - These tests pass `_env_file=None` to `Settings(...)` so they
    NEVER consult the developer's real `.env` at
    `C:\\Users\\kjosi\\dotenv\\.env`. Required values come in via
    kwargs.
  - Tests that hit `get_settings()` / `disable_env_file()` use
    `monkeypatch` to set env vars + clear the cache afterwards, so
    state doesn't bleed into other tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from knowledge_agent import config as config_module
from knowledge_agent.config import (
    Settings,
    disable_env_file,
    get_settings,
    load_test_env,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Shared minimal-required kwargs.
# ---------------------------------------------------------------------------


def _minimal_required(**overrides: object) -> dict[str, object]:
    """Build a complete set of required Settings kwargs, with optional
    per-test overrides. Forces `_env_file=None` so the developer's
    real `.env` is never consulted by tests."""
    base: dict[str, object] = {
        "_env_file": None,
        "anthropic_api_key": "test-anthropic",
        "voyage_api_key": "test-voyage",
        "neo4j_uri": "neo4j://127.0.0.1:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "test-password",
        "lancedb_path": Path("/tmp/test-lance"),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Test-state hygiene.
#
# `get_settings` is `@lru_cache(maxsize=1)` + `_env_file_disabled` is
# a module-level flag. Both must be reset between tests that touch
# them so state doesn't bleed.
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_config_state() -> Iterator[None]:
    """Save + restore `_env_file_disabled` and clear `get_settings`
    cache around the test."""
    saved_disabled = config_module._env_file_disabled
    get_settings.cache_clear()
    try:
        yield
    finally:
        config_module._env_file_disabled = saved_disabled
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Required-no-default fields.
# ---------------------------------------------------------------------------


def test_settings_minimal_required_construct_succeeds() -> None:
    s = Settings(**_minimal_required())  # type: ignore[arg-type]
    assert s.anthropic_api_key == "test-anthropic"
    assert s.voyage_api_key == "test-voyage"
    assert s.neo4j_uri == "neo4j://127.0.0.1:7687"
    assert s.neo4j_user == "neo4j"
    assert s.neo4j_password == "test-password"
    assert s.lancedb_path == Path("/tmp/test-lance")


@pytest.mark.parametrize(
    "missing_field",
    [
        # Per bundled-defaults removal 2026-06-29: API keys are NO
        # LONGER required — they're lazily validated in the factory
        # against the active `llm_provider` / `embedding_provider`.
        # The four DB credentials are still required-no-default.
        "neo4j_uri",
        "neo4j_user",
        "neo4j_password",
        "lancedb_path",
    ],
)
def test_settings_required_field_missing_raises(
    missing_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every required field must fail validation when absent.

    Also clears the corresponding env var so pydantic-settings doesn't
    fall back to it (case-insensitive match per SettingsConfigDict).
    """
    monkeypatch.delenv(missing_field.upper(), raising=False)
    kwargs = _minimal_required()
    del kwargs[missing_field]
    with pytest.raises(ValidationError) as exc_info:
        Settings(**kwargs)  # type: ignore[arg-type]
    assert missing_field in str(exc_info.value)


def test_settings_api_keys_are_optional_post_bundled_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """API keys (anthropic + voyage + openai + google) all default to
    None after the 2026-06-29 bundled-defaults removal. The factory
    raises ConfigError lazily if the active provider's key is unset;
    Settings construction itself doesn't enforce."""
    # Clear all key env vars so .env-fallback can't fill them in.
    for var in (
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    kwargs = _minimal_required()
    del kwargs["anthropic_api_key"]
    del kwargs["voyage_api_key"]
    s = Settings(**kwargs)  # type: ignore[arg-type]
    assert s.anthropic_api_key is None
    assert s.voyage_api_key is None
    assert s.openai_api_key is None
    assert s.google_api_key is None


# ---------------------------------------------------------------------------
# Default values for optional fields.
# ---------------------------------------------------------------------------


def test_settings_defaults_for_optional_fields() -> None:
    s = Settings(**_minimal_required())  # type: ignore[arg-type]

    # Embedding pin (must match across ingest + agent).
    assert s.embedding_model == "voyage-multimodal-3"
    assert s.embedding_dims == 1024

    # Retrieval mode default (the only one shipped today is lancedb_only).
    assert s.default_retrieval_mode == "lancedb_only"

    # Window defaults satisfy the invariant.
    assert s.top_k == 5
    assert s.lancedb_search_mode == "hybrid"
    assert s.num_candidates == 100
    assert s.rrf_rank_constant == 60
    assert s.rrf_rank_window_size == 50

    # LLM nodes.
    assert s.mode_classifier_model.startswith("claude-haiku")
    assert s.query_builder_model.startswith("claude-haiku")
    assert s.cypher_builder_model.startswith("claude-sonnet")
    assert s.synthesizer_model.startswith("claude-sonnet")

    # Behaviour toggles.
    assert s.skip_query_builder is False
    assert s.direct_retrieval is False

    # Parsing toggles.
    assert s.enable_pdf_ocr is False
    assert s.enable_image_ocr is True
    assert s.chunker_strategy == "hybrid"
    assert s.chunk_max_tokens == 512


# ---------------------------------------------------------------------------
# Retrieval-window invariant: top_k <= rrf_rank_window_size <= num_candidates.
# ---------------------------------------------------------------------------


def test_validate_retrieval_windows_happy_path_passes() -> None:
    """top_k=5 <= rrf_window=50 <= num_candidates=100 — defaults."""
    s = Settings(**_minimal_required())  # type: ignore[arg-type]
    assert s.top_k <= s.rrf_rank_window_size <= s.num_candidates


def test_validate_retrieval_windows_top_k_larger_than_window_fails() -> None:
    """top_k=60 > rrf_window=50 violates the invariant."""
    with pytest.raises(ValidationError, match="rrf_rank_window_size"):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(top_k=10, rrf_rank_window_size=5)
        )


def test_validate_retrieval_windows_num_candidates_smaller_than_window_fails() -> None:
    """num_candidates=40 < rrf_window=50 violates the invariant."""
    with pytest.raises(ValidationError, match="num_candidates"):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(num_candidates=40, rrf_rank_window_size=50)
        )


def test_validate_retrieval_windows_at_boundary_succeeds() -> None:
    """top_k == rrf_window == num_candidates is valid (equality OK)."""
    s = Settings(  # type: ignore[arg-type]
        **_minimal_required(top_k=20, rrf_rank_window_size=20, num_candidates=20)
    )
    assert s.top_k == s.rrf_rank_window_size == s.num_candidates == 20


# ---------------------------------------------------------------------------
# disable_env_file() + load_test_env() + get_settings() cache behaviour.
# ---------------------------------------------------------------------------


def test_disable_env_file_sets_flag_and_clears_cache(
    fresh_config_state: None,
) -> None:
    assert config_module._env_file_disabled is False
    disable_env_file()
    assert config_module._env_file_disabled is True
    # Subsequent get_settings() call must not return a stale cached
    # Settings that was built with the .env file active. We can't
    # easily verify "constructed with _env_file=None" directly, but
    # we CAN verify the cache was cleared.
    assert get_settings.cache_info().currsize == 0


def test_get_settings_caches_repeated_calls(
    fresh_config_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`@lru_cache(maxsize=1)` means two consecutive calls return the
    SAME instance, not just equal instances. Downstream caller
    `from_imports` rely on this for `_env_file_disabled` rebind
    detection."""
    # Set all required envs so Settings() can construct without the
    # .env file.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    monkeypatch.setenv("NEO4J_URI", "neo4j://test")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("LANCEDB_PATH", "/tmp/x")
    # Force env-only construction (no .env file).
    config_module._env_file_disabled = True
    get_settings.cache_clear()

    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_load_test_env_clears_cache(
    fresh_config_state: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`load_test_env()` must clear the `get_settings` cache so a
    smoke that calls it before any other config consumer gets fresh
    test-instance credentials instead of a cached production one.

    Uses `fresh_config_state` so the `_env_file_disabled=True` flip
    inside this test is restored at teardown — otherwise the flip
    bleeds into later unrelated tests (notably some in test_nodes.py)
    that didn't expect a globally-disabled .env file.
    """
    # Mock `load_dotenv` so we don't actually consult the .env.test
    # file (which may not exist in the test environment).
    called: list[Path] = []

    def fake_load_dotenv(path: Path, *, override: bool = False) -> bool:
        called.append(path)
        return True

    monkeypatch.setattr("dotenv.load_dotenv", fake_load_dotenv)
    # Warm the cache so we can verify it's cleared.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("VOYAGE_API_KEY", "y")
    monkeypatch.setenv("NEO4J_URI", "neo4j://test")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("LANCEDB_PATH", "/tmp/x")
    config_module._env_file_disabled = True
    get_settings.cache_clear()
    _ = get_settings()  # warm
    assert get_settings.cache_info().currsize == 1

    load_test_env()

    # Cache cleared by load_test_env regardless of file presence.
    assert get_settings.cache_info().currsize == 0
    # And the test env path was the target.
    assert called and ".env.test" in str(called[0])


# ---------------------------------------------------------------------------
# Negative cases on Literal / bounded fields.
# ---------------------------------------------------------------------------


def test_invalid_default_retrieval_mode_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(default_retrieval_mode="not_a_mode")
        )


def test_invalid_chunker_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(chunker_strategy="exotic")
        )


def test_top_k_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(**_minimal_required(top_k=0))  # type: ignore[arg-type]


def test_temperature_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(synthesizer_temperature=1.5)
        )


def test_mmr_lambda_below_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(mmr_lambda=-0.1)
        )
