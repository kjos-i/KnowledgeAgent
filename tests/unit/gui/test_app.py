"""Tests for `knowledge_agent.gui.app.GuiApp` coordinator logic.

Focuses on the pure logic the dataclass owns — `_missing_active_provider_key`,
`_load_corpus_config`, and the `on_clear` mode-handling — without
launching Flet. The Send pipeline is harder to test in isolation (it
chains chat-router + agent graph); we rely on integration tests for
that end-to-end path in later slices.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.gui import app as app_mod
from knowledge_agent.gui.app import GuiApp, _LoadedFile
from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.right_panel import MODE_FILE, MODE_LATEST


def _make_app(page: MagicMock) -> GuiApp:
    """Build a GuiApp without invoking build() — late-bound attributes
    stay un-set; tests that need them mock manually."""
    app = GuiApp(page=page)
    app.gui_config = GuiConfig()
    app.chat_panel = MagicMock(name="ChatPanel")
    app.right_panel = MagicMock(name="RightPanel")
    return app


# ---- _missing_active_provider_key ----


def test_missing_key_returns_env_var_when_anthropic_key_absent(
    fake_page: MagicMock, monkeypatch: pytest.MonkeyPatch,
):
    """Active provider == anthropic + no ANTHROPIC_API_KEY in env or
    keyring → returns the env var name so the chat panel surfaces it."""
    app = _make_app(fake_page)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_settings = MagicMock(
        llm_provider="anthropic", embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        result = app._missing_active_provider_key()
    assert result == "ANTHROPIC_API_KEY"


def test_missing_key_returns_none_when_local_provider(
    fake_page: MagicMock, monkeypatch: pytest.MonkeyPatch,
):
    """Ollama (local) doesn't need an API key. HuggingFace embedder
    same. So no env var is reported missing."""
    app = _make_app(fake_page)
    fake_settings = MagicMock(
        llm_provider="ollama", embedding_provider="huggingface",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        result = app._missing_active_provider_key()
    assert result is None


def test_missing_key_accepts_env_var_set(
    fake_page: MagicMock, monkeypatch: pytest.MonkeyPatch,
):
    """A shell-exported key (env var set) counts as present even when
    the keyring is empty — matches the bridging contract."""
    app = _make_app(fake_page)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    fake_settings = MagicMock(
        llm_provider="anthropic", embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        assert app._missing_active_provider_key() is None


def test_missing_key_checks_embedder_when_llm_ok(
    fake_page: MagicMock, monkeypatch: pytest.MonkeyPatch,
):
    """LLM key set but embedder key missing → embedder env var reported."""
    app = _make_app(fake_page)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_settings = MagicMock(
        llm_provider="anthropic", embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        assert app._missing_active_provider_key() == "VOYAGE_API_KEY"


# ---- _load_corpus_config ----


def test_load_corpus_config_uses_explicit_path_when_set(
    fake_page: MagicMock, tmp_path: Path,
):
    """When gui_config.corpus_config_path is set + exists, that file
    is loaded; CWD fallback is skipped."""
    app = _make_app(fake_page)
    target = tmp_path / "custom.toml"
    target.write_text("dummy", encoding="utf-8")
    app.gui_config.corpus_config_path = target

    fake_cfg = MagicMock(name="CorpusConfig")
    with patch(
        "knowledge_agent.gui.app.load_corpus_config", return_value=fake_cfg,
    ) as mock_load:
        result = app._load_corpus_config()
    assert result is fake_cfg
    mock_load.assert_called_once_with(target)


def test_load_corpus_config_falls_back_to_cwd_corpus_toml(
    fake_page: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When the explicit path is unset, CWD/corpus.toml is tried."""
    app = _make_app(fake_page)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.toml").write_text("dummy", encoding="utf-8")

    fake_cfg = MagicMock(name="CorpusConfig")
    with patch(
        "knowledge_agent.gui.app.load_corpus_config", return_value=fake_cfg,
    ):
        result = app._load_corpus_config()
    assert result is fake_cfg


def test_load_corpus_config_returns_none_when_no_candidate(
    fake_page: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """No explicit path AND no CWD corpus.toml → None. Caller surfaces
    a user-facing banner."""
    app = _make_app(fake_page)
    monkeypatch.chdir(tmp_path)
    assert app._load_corpus_config() is None


def test_load_corpus_config_falls_through_on_parse_failure(
    fake_page: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Explicit path exists but load_corpus_config raises → fall through
    to CWD; if that also fails (or doesn't exist), return None."""
    app = _make_app(fake_page)
    bad = tmp_path / "bad.toml"
    bad.write_text("dummy", encoding="utf-8")
    app.gui_config.corpus_config_path = bad
    monkeypatch.chdir(tmp_path)

    with patch(
        "knowledge_agent.gui.app.load_corpus_config",
        side_effect=ValueError("malformed"),
    ):
        assert app._load_corpus_config() is None


# ---- on_clear ----


def test_on_clear_wipes_session_state(fake_page: MagicMock):
    """Messages, last_answer, last_query all reset; loaded_file kept
    when the toggle is on (default)."""
    app = _make_app(fake_page)
    app.messages.append(MagicMock())
    app.last_answer = MagicMock()
    app.last_query = "q"
    app.loaded_file = _LoadedFile(name="x.md", content="x")
    app.right_panel.current_mode = MODE_LATEST

    app.on_clear(MagicMock())

    assert app.messages == []
    assert app.last_answer is None
    assert app.last_query is None
    # Default toggle: keep loaded file.
    assert app.loaded_file is not None


def test_on_clear_drops_loaded_file_when_toggle_off(fake_page: MagicMock):
    app = _make_app(fake_page)
    app.gui_config.keep_loaded_file_on_clear = False
    app.loaded_file = _LoadedFile(name="x.md", content="x")
    app.right_panel.current_mode = MODE_FILE

    app.on_clear(MagicMock())
    assert app.loaded_file is None
    app.right_panel.switch_mode.assert_called_with(MODE_LATEST)


# ---- on_save_answer ----


def test_save_answer_no_answer_appends_system_message(fake_page: MagicMock):
    """Save Answer with no current answer → user-friendly system message
    in chat, no exception."""
    app = _make_app(fake_page)
    app.last_answer = None
    app.last_query = None

    app.on_save_answer(MagicMock())

    app.chat_panel.append_system.assert_called_once()
    msg = app.chat_panel.append_system.call_args.args[0]
    assert "nothing to save" in msg


def test_save_chat_empty_chat_appends_system_message(fake_page: MagicMock):
    app = _make_app(fake_page)
    app.on_save_chat(MagicMock())
    msg = app.chat_panel.append_system.call_args.args[0]
    assert "empty" in msg.lower()
