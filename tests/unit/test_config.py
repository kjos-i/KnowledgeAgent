"""Unit tests for `knowledge_agent.config`.

The Settings model owns every runtime knob the rest of the codebase
consults. These tests pin:

  - Required-no-default fields fail-fast when missing (API keys + the
    four DB credentials). Catching this here prevents the failure
    mode where a forgotten config silently falls back to a default
    pointing at the wrong instance.
  - The retrieval-window invariant (`top_k <= num_candidates`) catches
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

import asyncio
from pathlib import Path
from types import SimpleNamespace
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


def test_ollama_base_url_strips_trailing_slash() -> None:
    """Trailing slash(es) on ollama_base_url are stripped so a downstream
    f-string join like f"{base}/api/tags" can't yield `.../api//tags`."""
    assert (
        Settings(  # type: ignore[arg-type]
            **_minimal_required(ollama_base_url="http://localhost:11434/")
        ).ollama_base_url
        == "http://localhost:11434"
    )
    # No trailing slash is unchanged; multiple trailing slashes all strip.
    assert (
        Settings(**_minimal_required(ollama_base_url="http://h:1")).ollama_base_url  # type: ignore[arg-type]
        == "http://h:1"
    )
    assert (
        Settings(**_minimal_required(ollama_base_url="http://h:1//")).ollama_base_url  # type: ignore[arg-type]
        == "http://h:1"
    )


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

    # Retrieval mode default: auto (the mode-classifier picks the store).
    assert s.default_retrieval_mode == "auto"

    # Window defaults satisfy the invariant.
    assert s.top_k == 5
    assert s.lancedb_search_mode == "hybrid"
    assert s.num_candidates == 100
    assert s.rrf_rank_constant == 60

    # LLM nodes.
    assert s.mode_classifier_model.startswith("claude-haiku")
    assert s.query_builder_model.startswith("claude-haiku")
    assert s.cypher_builder_model.startswith("claude-sonnet")
    assert s.synthesizer_model.startswith("claude-sonnet")

    # Behaviour toggles.
    assert s.skip_query_builder is False
    assert s.direct_retrieval is False


def test_settings_reads_embedding_dims_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-corpus bridge (`apply_corpus_embedding_to_env`) writes
    EMBEDDING_DIMS to carry a corpus's vector dimension; Settings must read
    that env var, else per-corpus dims silently fall back to the 1024
    default and LanceDB tables get the wrong shape. This path is new — the
    global `apply_embedding_to_env` never set EMBEDDING_DIMS — so pin it."""
    monkeypatch.setenv("EMBEDDING_DIMS", "768")
    s = Settings(**_minimal_required())  # type: ignore[arg-type]
    assert s.embedding_dims == 768


# ---------------------------------------------------------------------------
# Retrieval-window invariant: top_k <= num_candidates.
#
# `num_candidates` is the pre-truncation candidate pool the LanceDB
# retriever fetches; a pool smaller than the final `top_k` can't fill
# the requested results. The old `rrf_rank_window_size` middle knob was
# removed (LanceDB's RRF has no window param) and the rule simplified.
# ---------------------------------------------------------------------------


def test_validate_retrieval_windows_happy_path_passes() -> None:
    """top_k=5 <= num_candidates=100 — defaults."""
    s = Settings(**_minimal_required())  # type: ignore[arg-type]
    assert s.top_k <= s.num_candidates


def test_validate_retrieval_windows_num_candidates_smaller_than_top_k_fails() -> None:
    """num_candidates=3 < top_k=10 violates the invariant."""
    with pytest.raises(ValidationError, match="num_candidates"):
        Settings(  # type: ignore[arg-type]
            **_minimal_required(top_k=10, num_candidates=3)
        )


def test_validate_retrieval_windows_at_boundary_succeeds() -> None:
    """top_k == num_candidates is valid (equality OK)."""
    s = Settings(  # type: ignore[arg-type]
        **_minimal_required(top_k=20, num_candidates=20)
    )
    assert s.top_k == s.num_candidates == 20


def test_check_window_ordering_helper_returns_message_on_violation() -> None:
    """The shared helper (used by Settings + GUI + eval resolver) returns a
    human-readable message when num_candidates < top_k, else None."""
    from knowledge_agent.config import check_window_ordering

    assert check_window_ordering(10, 3) is not None
    assert "num_candidates" in check_window_ordering(10, 3)
    # Boundary + slack cases pass (return None).
    assert check_window_ordering(5, 5) is None
    assert check_window_ordering(5, 100) is None


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


def test_http_max_retries_negative_rejected() -> None:
    """A negative retry budget makes the request loop `range(retries + 1)`
    empty, so no attempt runs and `response` is unbound (UnboundLocalError on
    every request). `ge=0` fails it fast at config load instead."""
    with pytest.raises(ValidationError, match="http_max_retries"):
        Settings(**_minimal_required(http_max_retries=-1))  # type: ignore[arg-type]


def test_http_max_retries_zero_allowed() -> None:
    """0 is valid — it disables retries (a single attempt, no retry)."""
    s = Settings(**_minimal_required(http_max_retries=0))  # type: ignore[arg-type]
    assert s.http_max_retries == 0


# ---------------------------------------------------------------------------
# reset_after_settings_change() closes the evicted Neo4j driver (verify-11 HIGH,
# config.py:874). Evicting the get_kg_client cache without closing the old
# client leaked its AsyncDriver Bolt pool on every corpus switch — close() had
# no other caller, and neo4j 5.x's async __del__ only warns (GC never drains
# an async pool). Both bridge branches (running loop / no loop) must close it.
# ---------------------------------------------------------------------------


class _FakeKgClient:
    """A cached Neo4jClient stand-in with an open driver + a close() spy."""

    def __init__(self, closed: list[bool]) -> None:
        self._driver = object()  # a live, open driver
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(True)
        self._driver = None


class _FakeCachedGetter:
    """Mimic the lru_cache-wrapped get_kg_client: callable + cache_info +
    cache_clear, seeded with one already-cached client."""

    def __init__(self, client: _FakeKgClient) -> None:
        self._client = client
        self.cleared = False

    def __call__(self) -> _FakeKgClient:
        return self._client

    def cache_info(self) -> SimpleNamespace:
        return SimpleNamespace(currsize=1)

    def cache_clear(self) -> None:
        self.cleared = True


def _install_fake_kg_getter(
    monkeypatch: pytest.MonkeyPatch, closed: list[bool]
) -> _FakeCachedGetter:
    """Swap get_kg_client for a fake seeded with one open client. The local
    `from ...kg.client import get_kg_client` inside reset reads this attr."""
    from knowledge_agent.kg import client as kg_client_mod

    getter = _FakeCachedGetter(_FakeKgClient(closed))
    monkeypatch.setattr(kg_client_mod, "get_kg_client", getter)
    return getter


async def test_reset_closes_evicted_kg_driver_on_the_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI path: reset runs on the Flet event loop (a running loop), so the
    evicted client's async close() is scheduled on it and drains the Bolt pool.
    Fail-first: before the fix reset only cache_clear()'d, so close() was never
    called and the driver leaked."""
    closed: list[bool] = []
    getter = _install_fake_kg_getter(monkeypatch, closed)

    config_module.reset_after_settings_change()
    await asyncio.sleep(0)  # let the scheduled close task run to completion

    assert getter.cleared is True  # cache still evicted
    assert closed == [True]  # AND the old driver was closed (pool drained)


def test_reset_closes_evicted_kg_driver_without_a_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI path: no running loop, so the close runs to completion inline via
    asyncio.run. Same guarantee — the evicted driver is drained, not leaked."""
    closed: list[bool] = []
    _install_fake_kg_getter(monkeypatch, closed)

    config_module.reset_after_settings_change()

    assert closed == [True]
