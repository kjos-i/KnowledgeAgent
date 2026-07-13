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


def test_llm_provider_sync_disables_active_uninstall(fake_app):
    fake_app.gui_config = GuiConfig()
    fake_app.gui_config.llm_provider = "anthropic"
    tab = InstallsTab(fake_app)
    with patch.object(tab, "_safe_bool", return_value=True):  # all providers installed
        tab._sync_llm_provider_state()
    assert tab.llm_provider_install_buttons["anthropic"].visible is False
    assert tab.llm_provider_uninstall_buttons["anthropic"].disabled is True  # active
    assert tab.llm_provider_uninstall_buttons["openai"].disabled is False  # inactive
