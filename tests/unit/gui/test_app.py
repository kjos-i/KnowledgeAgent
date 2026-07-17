"""Tests for `knowledge_agent.gui.app.GuiApp` coordinator logic.

Focuses on the pure logic the dataclass owns — `_missing_active_provider_key`,
`_load_corpus_config`, and the `on_clear` mode-handling — without
launching Flet. The Send pipeline is harder to test in isolation (it
chains chat-router + agent graph); we rely on integration tests for
that end-to-end path in later slices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import flet as ft
from langchain_core.messages import HumanMessage

from knowledge_agent.artifacts import SaveError
from knowledge_agent.gui.app import GuiApp, _LoadedFile
from knowledge_agent.gui.config_store import GuiConfig, SwitchOutcome
from knowledge_agent.gui.right_panel import MODE_FILE, MODE_LATEST
from knowledge_agent.models import AgentAnswer

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


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
    fake_page: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Active provider == anthropic + no ANTHROPIC_API_KEY in env or
    keyring → returns the env var name so the chat panel surfaces it."""
    app = _make_app(fake_page)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_settings = MagicMock(
        llm_provider="anthropic",
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        result = app._missing_active_provider_key()
    assert result == "ANTHROPIC_API_KEY"


def test_missing_key_returns_none_when_local_provider(
    fake_page: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Ollama (local) doesn't need an API key. HuggingFace embedder
    same. So no env var is reported missing."""
    app = _make_app(fake_page)
    fake_settings = MagicMock(
        llm_provider="ollama",
        embedding_provider="huggingface",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        result = app._missing_active_provider_key()
    assert result is None


def test_missing_key_accepts_env_var_set(
    fake_page: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """A shell-exported key (env var set) counts as present even when
    the keyring is empty — matches the bridging contract."""
    app = _make_app(fake_page)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "vy-test")
    fake_settings = MagicMock(
        llm_provider="anthropic",
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        assert app._missing_active_provider_key() is None


def test_missing_key_checks_embedder_when_llm_ok(
    fake_page: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """LLM key set but embedder key missing → embedder env var reported."""
    app = _make_app(fake_page)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    fake_settings = MagicMock(
        llm_provider="anthropic",
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.gui.app.get_settings", return_value=fake_settings),
        patch("knowledge_agent.gui.app.get_api_key", return_value=None),
    ):
        assert app._missing_active_provider_key() == "VOYAGE_API_KEY"


# ---- _load_corpus_config ----


def test_load_corpus_config_uses_explicit_path_when_set(
    fake_page: MagicMock,
    tmp_path: Path,
):
    """When gui_config.corpus_config_path is set + exists, that file
    is loaded; CWD fallback is skipped."""
    app = _make_app(fake_page)
    target = tmp_path / "custom.toml"
    target.write_text("dummy", encoding="utf-8")
    app.gui_config.corpus_config_path = target

    fake_cfg = MagicMock(name="CorpusConfig")
    with patch(
        "knowledge_agent.gui.app.load_corpus_config",
        return_value=fake_cfg,
    ) as mock_load:
        result = app._load_corpus_config()
    assert result is fake_cfg
    mock_load.assert_called_once_with(target)


def test_load_corpus_config_falls_back_to_cwd_corpus_toml(
    fake_page: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When the explicit path is unset, CWD/corpus.toml is tried."""
    app = _make_app(fake_page)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus.toml").write_text("dummy", encoding="utf-8")

    fake_cfg = MagicMock(name="CorpusConfig")
    with patch(
        "knowledge_agent.gui.app.load_corpus_config",
        return_value=fake_cfg,
    ):
        result = app._load_corpus_config()
    assert result is fake_cfg


def test_load_corpus_config_returns_none_when_no_candidate(
    fake_page: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """No explicit path AND no CWD corpus.toml → None. Caller surfaces
    a user-facing banner."""
    app = _make_app(fake_page)
    monkeypatch.chdir(tmp_path)
    assert app._load_corpus_config() is None


def test_load_corpus_config_falls_through_on_parse_failure(
    fake_page: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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


async def test_save_answer_no_answer_appends_system_message(fake_page: MagicMock):
    """Save Answer with no current answer → user-friendly system message
    in chat, no exception. Handler is async since it awaits the OS
    folder picker; the early-return branch we exercise here doesn't
    actually open the picker but still has to be awaited."""
    app = _make_app(fake_page)
    app.last_answer = None
    app.last_query = None

    await app.on_save_answer(MagicMock())

    app.chat_panel.append_system.assert_called_once()
    msg = app.chat_panel.append_system.call_args.args[0]
    assert "nothing to save" in msg


async def test_save_chat_empty_chat_appends_system_message(fake_page: MagicMock):
    app = _make_app(fake_page)
    await app.on_save_chat(MagicMock())
    msg = app.chat_panel.append_system.call_args.args[0]
    assert "empty" in msg.lower()


async def test_save_answer_writes_configured_formats_to_default_dir(
    fake_page: MagicMock, tmp_path: Path
):
    """Default folder set + formats [md,txt] → both written, no picker, no hint."""
    app = _make_app(fake_page)
    app.gui_config.results_dir = tmp_path
    app.gui_config.save_formats = ["md", "txt"]
    app.last_answer = AgentAnswer(answer="hi", chunk_sources=[], kg_sources=[])
    app.last_query = "the q"
    app.file_picker = MagicMock()
    app.file_picker.get_directory_path = AsyncMock()

    await app.on_save_answer(MagicMock())

    app.file_picker.get_directory_path.assert_not_awaited()  # default dir → no picker
    assert list(tmp_path.glob("*.md")) and list(tmp_path.glob("*.txt"))
    msgs = [c.args[0] for c in app.chat_panel.append_system.call_args_list]
    assert sum("saved:" in m for m in msgs) == 2
    assert not any("Settings" in m for m in msgs)  # hint only on first (no-dir) save


async def test_save_answer_without_default_dir_asks_then_hints(
    fake_page: MagicMock, tmp_path: Path
):
    """No default folder → picker opens once, choice remembered, Settings hint shown."""
    app = _make_app(fake_page)
    app.gui_config.results_dir = None
    app.gui_config.save_formats = ["md"]
    app.last_answer = AgentAnswer(answer="hi", chunk_sources=[], kg_sources=[])
    app.last_query = "q"
    app.file_picker = MagicMock()
    app.file_picker.get_directory_path = AsyncMock(return_value=str(tmp_path))

    with patch("knowledge_agent.gui.app.save_config"):
        await app.on_save_answer(MagicMock())

    app.file_picker.get_directory_path.assert_awaited_once()
    assert app.gui_config.results_dir == tmp_path  # remembered
    assert list(tmp_path.glob("*.md"))
    msgs = [c.args[0] for c in app.chat_panel.append_system.call_args_list]
    assert any("Settings" in m for m in msgs)


async def test_save_chat_drops_json_and_falls_back_to_md(fake_page: MagicMock, tmp_path: Path):
    """json is answer-only — a chat save with json-only selection falls back to md."""
    app = _make_app(fake_page)
    app.gui_config.chat_dir = tmp_path
    app.gui_config.chat_save_formats = ["json"]
    app.messages = [HumanMessage(content="hi")]
    app.last_query = "q"
    app.file_picker = MagicMock()
    app.file_picker.get_directory_path = AsyncMock()

    await app.on_save_chat(MagicMock())

    assert list(tmp_path.glob("*.md"))  # fell back to md
    assert not list(tmp_path.glob("*.json"))  # json dropped for a transcript


async def test_save_chat_uses_its_own_dir_and_formats(fake_page: MagicMock, tmp_path: Path):
    """Save Chat reads chat_dir + chat_save_formats — independent of the results
    settings (which stay pointed elsewhere and are untouched)."""
    app = _make_app(fake_page)
    chat_dir = tmp_path / "chats"
    chat_dir.mkdir()
    app.gui_config.chat_dir = chat_dir
    app.gui_config.chat_save_formats = ["md", "txt"]
    app.gui_config.results_dir = tmp_path / "results"  # different — must be untouched
    app.gui_config.save_formats = ["json"]
    app.messages = [HumanMessage(content="hi")]
    app.last_query = "q"
    app.file_picker = MagicMock()
    app.file_picker.get_directory_path = AsyncMock()

    await app.on_save_chat(MagicMock())

    app.file_picker.get_directory_path.assert_not_awaited()  # chat_dir set → no picker
    assert list(chat_dir.glob("*.md")) and list(chat_dir.glob("*.txt"))
    assert not (tmp_path / "results").exists()  # results dir untouched


async def test_save_answer_surfaces_save_error(fake_page: MagicMock, tmp_path: Path):
    """A backend SaveError becomes a 'could not save' chat line, not a crash."""
    app = _make_app(fake_page)
    app.gui_config.results_dir = tmp_path
    app.gui_config.save_formats = ["md"]
    app.last_answer = AgentAnswer(answer="hi", chunk_sources=[], kg_sources=[])
    app.last_query = "q"
    app.file_picker = MagicMock()
    app.file_picker.get_directory_path = AsyncMock()

    with patch("knowledge_agent.gui.app.save_answer", side_effect=SaveError("disk full")):
        await app.on_save_answer(MagicMock())

    msgs = [c.args[0] for c in app.chat_panel.append_system.call_args_list]
    assert any("could not save" in m for m in msgs)


# ---- on_send: input-mode routing (the chat-router gating) ----
#
# on_send chains chat-router + agent graph. These two pin the branching
# the input-mode redesign hinges on: the router runs ONLY in
# conversational mode, and direct modes go straight to the graph. The
# collaborators (provider-key check, corpus load, diag, graph, router)
# are stubbed; `_invoke_state_for_input_mode` itself is covered in
# test_app_input_mode.


async def test_on_send_direct_cypher_skips_router_and_invokes_graph(fake_page: MagicMock):
    """Direct-Cypher mode bypasses the chat router entirely and invokes
    the graph with the user's text as `user_cypher` + KG-forced mode."""
    app = _make_app(fake_page)
    app.busy = False
    app.messages = []
    app.gui_config.input_mode = "direct_cypher"
    cypher = "MATCH (n) RETURN n LIMIT 5"
    app.chat_panel.get_input_text = MagicMock(return_value=cypher)

    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={})  # no final_answer branch
    with (
        patch.object(app, "_missing_active_provider_key", return_value=None),
        patch.object(app, "_load_corpus_config", return_value=MagicMock()),
        patch.object(app, "_diag"),
        patch("knowledge_agent.gui.app.get_chat_router") as get_router,
        patch("knowledge_agent.graph.graph", fake_graph),
    ):
        await app.on_send(MagicMock())

    get_router.assert_not_called()  # router skipped in direct modes
    fake_graph.ainvoke.assert_awaited_once()
    state = fake_graph.ainvoke.call_args.args[0]
    assert state["user_cypher"] == cypher
    assert state["retrieval_mode"] == "neo4j_only"


async def test_on_send_conversational_runs_router_and_gates_retrieval(fake_page: MagicMock):
    """Conversational mode runs the chat router; when it decides NOT to
    retrieve (ready_to_retrieve=False), the graph is never invoked."""
    app = _make_app(fake_page)
    app.busy = False
    app.messages = []
    app.gui_config.input_mode = "conversational"
    app.chat_panel.get_input_text = MagicMock(return_value="tell me about aspirin")

    router = MagicMock()
    router.ainvoke = AsyncMock(
        return_value=MagicMock(
            response="Could you clarify?",
            ready_to_retrieve=False,
            search_query="",
        )
    )
    fake_graph = MagicMock()
    fake_graph.ainvoke = AsyncMock(return_value={})
    with (
        patch.object(app, "_missing_active_provider_key", return_value=None),
        patch.object(app, "_load_corpus_config", return_value=MagicMock()),
        patch.object(app, "_diag"),
        patch("knowledge_agent.gui.app.get_chat_router", return_value=router) as get_router,
        patch("knowledge_agent.graph.graph", fake_graph),
    ):
        await app.on_send(MagicMock())

    get_router.assert_called_once()  # router ran in conversational mode
    router.ainvoke.assert_awaited_once()
    fake_graph.ainvoke.assert_not_awaited()  # not ready → no retrieval leg
    app.chat_panel.append_assistant.assert_called_once_with("Could you clarify?")


# ---- select_corpus (global corpus switch orchestration) ----


def test_build_corpus_selector_constructs(fake_page: MagicMock):
    """Guard the top-bar controls against a bad Flet kwarg (Flet 0.85's
    Dropdown uses on_select, not on_change; no hint_text/text_size). build()
    isn't unit-tested, so construct the real controls here."""
    app = _make_app(fake_page)
    dropdown, manage = app._build_corpus_selector()
    assert isinstance(dropdown, ft.Dropdown)
    assert isinstance(manage, ft.IconButton)


def test_select_corpus_noop_when_same_corpus(fake_page: MagicMock):
    """Selecting the already-active corpus does nothing (no switch)."""
    app = _make_app(fake_page)
    app.gui_config = GuiConfig(active_corpus_name="c1")
    with patch("knowledge_agent.gui.app.switch_active_corpus") as sw:
        app.select_corpus("c1")
    sw.assert_not_called()


def test_select_corpus_ok_resets_caches_and_broadcasts(fake_page: MagicMock):
    """A successful switch clears backend caches and refreshes the Library
    tabs (card + Ingest), and surfaces the confirmation message."""
    app = _make_app(fake_page)
    app.gui_config = GuiConfig(active_corpus_name="c1")
    app.library_tab = MagicMock(name="LibraryTab")
    app.corpus_dropdown = None  # selector sync is a no-op without the control
    with (
        patch(
            "knowledge_agent.gui.app.switch_active_corpus",
            return_value=SwitchOutcome(ok=True, message="Active corpus: c2"),
        ),
        patch("knowledge_agent.gui.app.reset_after_key_change") as reset,
    ):
        app.select_corpus("c2")
    reset.assert_called_once()
    app.library_tab.view.select_tab.refresh_after_switch.assert_called_once()
    app.library_tab.view.refresh_ingest.assert_called_once()
    app.chat_panel.append_system.assert_called()


def test_select_corpus_hard_failure_no_reset_no_broadcast(fake_page: MagicMock):
    """A hard failure (unknown corpus / save error) must NOT clear caches or
    refresh tabs — state is unchanged; only the message surfaces."""
    app = _make_app(fake_page)
    app.gui_config = GuiConfig(active_corpus_name="c1")
    app.library_tab = MagicMock(name="LibraryTab")
    app.corpus_dropdown = None
    with (
        patch(
            "knowledge_agent.gui.app.switch_active_corpus",
            return_value=SwitchOutcome(ok=False, message="could not save: boom"),
        ),
        patch("knowledge_agent.gui.app.reset_after_key_change") as reset,
    ):
        app.select_corpus("c2")
    reset.assert_not_called()
    app.library_tab.view.select_tab.refresh_after_switch.assert_not_called()
    app.chat_panel.append_system.assert_called()  # the error message
