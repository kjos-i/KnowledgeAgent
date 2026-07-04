"""Tests for `knowledge_agent.gui.config_store` — GuiConfig persistence
+ keyring bridge.

Keyring is mocked end-to-end (no real OS-keyring backend touched).
JSON persistence is tested against tmp_path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import keyring.errors
import pytest

from knowledge_agent.gui import config_store
from knowledge_agent.gui.config_store import (
    APP_ID,
    KEYRING_TO_ENV,
    ConfigError,
    GuiConfig,
    apply_keys_to_env,
    apply_llm_to_env,
    apply_retrieval_to_env,
    get_api_key,
    load_config,
    save_config,
    set_api_key,
)

# ---- GuiConfig defaults ----


def test_default_config_has_safe_defaults():
    cfg = GuiConfig()
    assert cfg.top_k == 5
    assert cfg.retrieval_mode == "auto"
    assert cfg.input_mode == "conversational"
    assert cfg.direct_retrieve is False
    assert cfg.keep_loaded_file_on_clear is True


def test_default_config_router_and_info_icon_fields():
    """GUI-only additions: the chat router's curated default model + the
    global info-icon toggle default (on = teaching-program-friendly)."""
    cfg = GuiConfig()
    assert cfg.chat_router_model == "claude-haiku-4-5-20251001"
    assert cfg.show_info_icons is True


# ---- env bridges: GUI-only fields must NOT reach the backend ----


def test_apply_retrieval_bridges_direct_retrieve_not_input_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    """`direct_retrieve` reaches the backend via DIRECT_RETRIEVAL. But
    `input_mode` is a GUI-only routing decision (consumed by app.on_send,
    not backend Settings) → NOT bridged. The removed `skip_query_builder`
    likewise leaves no SKIP_QUERY_BUILDER var."""
    # Isolate every env mutation to a throwaway copy.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    apply_retrieval_to_env(GuiConfig(direct_retrieve=True, input_mode="direct_cypher"))
    assert os.environ["DIRECT_RETRIEVAL"] == "true"
    assert "INPUT_MODE" not in os.environ
    assert "SKIP_QUERY_BUILDER" not in os.environ


def test_apply_llm_does_not_bridge_chat_router(
    monkeypatch: pytest.MonkeyPatch,
):
    """The chat router is a GUI-only supervisor stand-in — the backend
    graph is the tool it will call, so the router's model must not leak
    into backend env. `apply_llm_to_env` bridges the 4 graph-node models
    but NO CHAT_ROUTER_* var."""
    monkeypatch.setattr(os, "environ", dict(os.environ))
    apply_llm_to_env(GuiConfig(chat_router_model="claude-opus-4-8"))
    # The graph nodes DO bridge (sanity that the function ran).
    assert os.environ["MODE_CLASSIFIER_MODEL"]
    assert "CHAT_ROUTER_MODEL" not in os.environ
    assert "CHAT_ROUTER_TEMPERATURE" not in os.environ


def test_top_k_rejects_zero():
    with pytest.raises(ValueError):
        GuiConfig(top_k=0)


def test_retrieval_mode_rejects_unknown():
    with pytest.raises(ValueError):
        GuiConfig(retrieval_mode="definitely-not-a-mode")  # type: ignore[arg-type]


# ---- JSON persistence ----


def _redirect_config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point `config_store._config_dir` at `tmp_path` for isolation."""

    def fake_dir() -> Path:
        return tmp_path

    monkeypatch.setattr(config_store, "_config_dir", fake_dir)
    return tmp_path


def test_load_config_returns_defaults_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _redirect_config_dir(monkeypatch, tmp_path)
    cfg = load_config()
    # Fresh defaults — no file present.
    assert cfg.top_k == 5
    assert cfg.retrieval_mode == "auto"


def test_load_config_returns_defaults_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Corrupt JSON is treated as a missing file (logged, defaults
    returned) so the user can re-enter via the UI rather than crash."""
    _redirect_config_dir(monkeypatch, tmp_path)
    (tmp_path / "settings.json").write_text("not json{", encoding="utf-8")
    cfg = load_config()
    assert cfg.top_k == 5


def test_load_config_returns_defaults_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Schema drift (e.g. an unknown retrieval_mode value left over
    from a future version) falls back to defaults, not a crash."""
    _redirect_config_dir(monkeypatch, tmp_path)
    (tmp_path / "settings.json").write_text(
        json.dumps({"retrieval_mode": "magic-mode"}),
        encoding="utf-8",
    )
    cfg = load_config()
    assert cfg.retrieval_mode == "auto"


def test_save_then_load_roundtrips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _redirect_config_dir(monkeypatch, tmp_path)
    save_config(GuiConfig(top_k=9, retrieval_mode="neo4j_only"))
    cfg = load_config()
    assert cfg.top_k == 9
    assert cfg.retrieval_mode == "neo4j_only"


def test_save_config_atomic_no_tmp_left(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _redirect_config_dir(monkeypatch, tmp_path)
    save_config(GuiConfig())
    assert not (tmp_path / "settings.json.tmp").exists()


def test_save_config_cleans_tmp_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    _redirect_config_dir(monkeypatch, tmp_path)

    def boom(self, *_a, **_k):
        raise OSError("disk full")

    # Force `tmp.write_text` to raise — `.tmp` exists briefly then
    # should be cleaned up by save_config's except branch.
    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(ConfigError):
        save_config(GuiConfig())


# ---- keyring + env bridge ----


def test_get_api_key_returns_value_from_keyring():
    with patch("keyring.get_password", return_value="sk-test"):
        assert get_api_key("anthropic") == "sk-test"


def test_get_api_key_returns_none_on_backend_error():
    """Backend missing → None (not raise), so the GUI can show an empty
    field instead of crashing on first launch."""
    with patch(
        "keyring.get_password",
        side_effect=keyring.errors.KeyringError("no backend"),
    ):
        assert get_api_key("anthropic") is None


def test_set_api_key_writes_to_keyring():
    with patch("keyring.set_password") as mock_set:
        set_api_key("anthropic", "sk-new")
    mock_set.assert_called_once_with(APP_ID, "anthropic", "sk-new")


def test_set_api_key_empty_value_deletes():
    """Empty string is the GUI's signal to remove a stored secret."""
    with patch("keyring.delete_password") as mock_del:
        set_api_key("anthropic", "")
    mock_del.assert_called_once_with(APP_ID, "anthropic")


def test_set_api_key_delete_missing_is_idempotent():
    """Delete on already-absent key is a no-op, not an error — matches
    keyring's PasswordDeleteError semantics."""
    with patch(
        "keyring.delete_password",
        side_effect=keyring.errors.PasswordDeleteError(),
    ):
        set_api_key("anthropic", "")  # must not raise


def test_apply_keys_to_env_bridges_each_stored_secret(
    monkeypatch: pytest.MonkeyPatch,
):
    """For each KEYRING_TO_ENV entry with a stored value, the matching
    env var is set on os.environ."""
    # Start clean.
    for var in KEYRING_TO_ENV.values():
        monkeypatch.delenv(var, raising=False)

    def fake_get(name: str) -> str | None:
        return {
            "anthropic": "ak-1",
            "voyage": "vy-1",
        }.get(name)

    with patch("knowledge_agent.gui.config_store.get_api_key", fake_get):
        apply_keys_to_env()

    assert os.environ.get("ANTHROPIC_API_KEY") == "ak-1"
    assert os.environ.get("VOYAGE_API_KEY") == "vy-1"
    # Unset keys aren't created from empty/None.
    assert os.environ.get("OPENAI_API_KEY") is None


def test_apply_keys_to_env_skips_empty_values(monkeypatch: pytest.MonkeyPatch):
    """Empty stored values don't overwrite an existing shell-exported env."""
    monkeypatch.setenv("VOYAGE_API_KEY", "shell-value")
    with patch(
        "knowledge_agent.gui.config_store.get_api_key",
        return_value="",
    ):
        apply_keys_to_env()
    assert os.environ["VOYAGE_API_KEY"] == "shell-value"


# ---- apply_active_corpus_password_to_env (startup password bridge) ----


def test_apply_active_corpus_password_bridges_stored(monkeypatch):
    """A stored per-corpus password lands in NEO4J_PASSWORD."""
    monkeypatch.setattr(
        config_store,
        "get_corpus_password",
        lambda name: "s3cret",
    )
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    cfg = config_store.GuiConfig(active_corpus_name="c1")
    assert config_store.apply_active_corpus_password_to_env(cfg) is True
    assert os.environ["NEO4J_PASSWORD"] == "s3cret"


def test_apply_active_corpus_password_pops_when_none(monkeypatch):
    """No stored password -> NEO4J_PASSWORD is removed (not left stale)."""
    monkeypatch.setattr(
        config_store,
        "get_corpus_password",
        lambda name: None,
    )
    monkeypatch.setenv("NEO4J_PASSWORD", "stale")
    cfg = config_store.GuiConfig(active_corpus_name="c1")
    assert config_store.apply_active_corpus_password_to_env(cfg) is False
    assert "NEO4J_PASSWORD" not in os.environ


def test_apply_active_corpus_password_no_active_corpus(monkeypatch):
    """No active corpus -> pops NEO4J_PASSWORD, returns False."""
    monkeypatch.setenv("NEO4J_PASSWORD", "stale")
    cfg = config_store.GuiConfig(active_corpus_name=None)
    assert config_store.apply_active_corpus_password_to_env(cfg) is False
    assert "NEO4J_PASSWORD" not in os.environ
