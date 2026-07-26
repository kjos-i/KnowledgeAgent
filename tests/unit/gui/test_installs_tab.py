"""InstallsTab — a top-level tab that also owns provider (embedder + LLM) installs.

The widget lives in `gui/library/installs.py`; only its mount point moved out
of Library. It now also hosts the embedder- and LLM-provider Install/Uninstall
rows (moved here from the Embedding + LLM settings tabs), on the Installs
no-pre-check pattern: the handler always shows the confirm dialog, and button
visibility (+ the disabled active-Uninstall in `_sync`) covers the installed /
active cases — matching the existing extractor / parser handlers in the file.
The LLM install plan is async (it probes the Ollama daemon), so its handler
awaits the plan before the dialog; everything else is sync.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.library.installs import InstallsTab

_EMB_INSTALL = "knowledge_agent.gui.library.installs.install_embedder_provider_plan"
_EMB_UNINSTALL = "knowledge_agent.gui.library.installs.uninstall_embedder_provider_plan"
_LLM_INSTALL = "knowledge_agent.gui.library.installs.install_llm_provider_plan"
_LLM_UNINSTALL = "knowledge_agent.gui.library.installs.uninstall_llm_provider_plan"


def test_installs_tab_constructs(fake_app):
    assert InstallsTab(fake_app) is not None


# ---- embedder providers (sync install plan) ----


def test_embedder_provider_install_shows_confirm_dialog(fake_app):
    """No pre-check short-circuit (matches the extractor/parser handlers): the
    handler always shows the confirm dialog."""
    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    plan = MagicMock(display_name="Voyage AI", summary="install summary")
    with patch(_EMB_INSTALL, return_value=plan):
        tab._on_embedder_provider_install("voyage")
    fake_app.page.show_dialog.assert_called_once()


def test_embedder_provider_uninstall_shows_confirm_dialog(fake_app):
    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    plan = MagicMock(display_name="OpenAI Embeddings", summary="uninstall summary")
    with patch(_EMB_UNINSTALL, return_value=plan):
        tab._on_embedder_provider_uninstall("openai")
    fake_app.page.show_dialog.assert_called_once()


def test_embedder_provider_sync_hides_installed_and_disables_active_uninstall(fake_app):
    """Button visibility replaces the old pre-check: Install hidden when
    installed, and the ACTIVE embedder's Uninstall is disabled."""
    fake_app.gui_config = GuiConfig()
    fake_app.gui_config.embedding_provider = "voyage"
    tab = InstallsTab(fake_app)
    with patch.object(tab, "_safe_bool", return_value=True):  # all providers installed
        tab._sync_embedder_provider_state()
    assert tab.embedder_provider_install_buttons["voyage"].visible is False
    assert tab.embedder_provider_uninstall_buttons["voyage"].disabled is True  # active
    assert tab.embedder_provider_uninstall_buttons["openai"].disabled is False  # inactive


# ---- LLM providers (install plan is async — Ollama daemon probe) ----


def test_llm_provider_install_shows_confirm_dialog(fake_app):
    """The LLM install plan is async, so the handler awaits it before the
    dialog — exercised via the async `_prompt` directly."""
    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    plan = MagicMock(display_name="OpenAI GPT", summary="install summary")
    with patch(_LLM_INSTALL, AsyncMock(return_value=plan)):
        asyncio.run(tab._prompt_llm_provider_install("openai"))
    fake_app.page.show_dialog.assert_called_once()


def test_llm_provider_uninstall_shows_confirm_dialog(fake_app):
    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    plan = MagicMock(display_name="OpenAI GPT", summary="uninstall summary")
    with patch(_LLM_UNINSTALL, return_value=plan):
        tab._on_llm_provider_uninstall("openai")
    fake_app.page.show_dialog.assert_called_once()


def test_llm_provider_sync_disables_uninstall_for_in_use_provider(fake_app):
    """A provider a query-time node model uses can't be uninstalled; a provider
    that no node uses and isn't the default provider can. Each node picks its
    own provider."""
    fake_app.gui_config = GuiConfig()
    fake_app.gui_config.llm_provider = "anthropic"
    for attr in (
        "chat_router_model",
        "mode_classifier_model",
        "query_builder_model",
        "cypher_builder_model",
    ):
        setattr(fake_app.gui_config, attr, "anthropic:claude-haiku-4-5")
    fake_app.gui_config.synthesizer_model = "openai:gpt-4o"  # one node on OpenAI
    tab = InstallsTab(fake_app)
    with patch.object(tab, "_safe_bool", return_value=True):  # all providers installed
        tab._sync_llm_provider_state()
    assert tab.llm_provider_install_buttons["anthropic"].visible is False
    assert tab.llm_provider_uninstall_buttons["anthropic"].disabled is True  # in use
    assert tab.llm_provider_uninstall_buttons["openai"].disabled is True  # in use (synthesizer)
    assert tab.llm_provider_uninstall_buttons["google"].disabled is False  # unused


def test_llm_provider_sync_disables_uninstall_for_default_provider_not_in_nodes(fake_app):
    """Bug 2: the default provider (settings.llm_provider) can't be uninstalled
    even when NO node model references it — matching the lifecycle's is_active
    block, so the GUI never offers a button that would no-op."""
    fake_app.gui_config = GuiConfig()
    fake_app.gui_config.llm_provider = "google"  # the default/fallback provider
    for attr in (
        "chat_router_model",
        "mode_classifier_model",
        "query_builder_model",
        "cypher_builder_model",
        "synthesizer_model",
    ):
        setattr(fake_app.gui_config, attr, "anthropic:claude-haiku-4-5")  # all explicit, non-google
    tab = InstallsTab(fake_app)
    with patch.object(tab, "_safe_bool", return_value=True):  # all providers installed
        tab._sync_llm_provider_state()
    assert tab.llm_provider_uninstall_buttons["google"].disabled is True  # default, protected
    assert tab.llm_provider_uninstall_buttons["anthropic"].disabled is True  # node-used
    assert tab.llm_provider_uninstall_buttons["openai"].disabled is False  # neither


def test_llm_uninstall_noop_message_is_honest(fake_app):
    """Bug 1: a no-op uninstall (nothing removed, restart_required False) must
    NOT claim 'Uninstalled ... Restart'."""
    from knowledge_agent.gui.library import installs

    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    noop = MagicMock(uninstall_ok=True, restart_required=False)
    with (
        patch(_LLM_UNINSTALL, return_value=MagicMock()),
        patch.object(installs, "uninstall_llm_provider_execute", AsyncMock(return_value=noop)),
        patch.object(tab, "_sync_llm_provider_state"),
        patch.object(tab, "_safe_update"),
    ):
        asyncio.run(tab._run_llm_provider_uninstall("anthropic"))
    assert "not uninstalled" in tab.status.value.lower()
    assert "Restart" not in tab.status.value


def test_llm_uninstall_real_says_restart(fake_app):
    """A real uninstall (restart_required True) tells the user to restart."""
    from knowledge_agent.gui.library import installs

    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    done = MagicMock(uninstall_ok=True, restart_required=True)
    with (
        patch(_LLM_UNINSTALL, return_value=MagicMock()),
        patch.object(installs, "uninstall_llm_provider_execute", AsyncMock(return_value=done)),
        patch.object(tab, "_sync_llm_provider_state"),
        patch.object(tab, "_safe_update"),
    ):
        asyncio.run(tab._run_llm_provider_uninstall("anthropic"))
    assert "Restart" in tab.status.value


# ---- Ollama base URL + daemon status (moved here from the LLMs tab) ----


def test_ollama_url_blur_persists_and_bridges(fake_app):
    """Editing the Ollama base URL writes GuiConfig and bridges it to env via
    apply_llm_to_env (which sets OLLAMA_BASE_URL)."""
    from knowledge_agent.gui.library import installs

    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    tab.ollama_url_field.value = "http://localhost:22222"
    with (
        patch.object(installs, "save_config"),
        patch.object(installs, "apply_llm_to_env") as bridge,
        patch.object(installs, "reset_after_settings_change"),
    ):
        tab._on_ollama_url_blur(MagicMock())
    assert fake_app.gui_config.ollama_base_url == "http://localhost:22222"
    bridge.assert_called_once()


def test_ollama_daemon_status_reflects_state(fake_app):
    """The daemon line reports reachability when the adapter is installed, and
    otherwise points back to the install row (a separate, non-daemon state)."""
    fake_app.gui_config = GuiConfig()
    tab = InstallsTab(fake_app)
    # Adapter installed + daemon reachable → ✓ reachable.
    tab._ollama_daemon_ok = True
    with patch.object(tab, "_safe_bool", return_value=True):
        tab._sync_ollama_daemon_status()
    assert "reachable" in tab.ollama_daemon_status.value

    # Adapter not installed → not a daemon state; points at the install row.
    tab._ollama_daemon_ok = None
    with patch.object(tab, "_safe_bool", return_value=False):
        tab._sync_ollama_daemon_status()
    assert "not installed" in tab.ollama_daemon_status.value
