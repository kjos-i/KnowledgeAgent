"""Tests for the Embedding settings tab's provider install/uninstall wiring.

Mirrors test_llm_tab. The embedder install/uninstall plans are SYNC, so the
click handlers branch synchronously: no-op cases (bundled / active / already
installed / not installed) surface a status message; a real action shows the
confirm dialog.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.settings.embedding_tab import EmbeddingTab

_INSTALL = "knowledge_agent.gui.settings.embedding_tab.install_embedder_provider_plan"
_UNINSTALL = "knowledge_agent.gui.settings.embedding_tab.uninstall_embedder_provider_plan"


def _tab() -> EmbeddingTab:
    app = MagicMock()
    app.gui_config = GuiConfig()
    app.page = MagicMock()
    return EmbeddingTab(app)


def _plan(**kw) -> MagicMock:
    plan = MagicMock(**kw)
    plan.summary = kw.get("summary", "SUMMARY")
    return plan


def test_install_installable_shows_confirm_dialog():
    tab = _tab()
    plan = _plan(bundled=False, already_installed=False, display_name="Voyage")
    with patch(_INSTALL, return_value=plan):
        tab.on_install_clicked("voyage")
    tab.app.page.show_dialog.assert_called_once()


def test_install_already_installed_short_circuits_no_dialog():
    tab = _tab()
    plan = _plan(bundled=False, already_installed=True, summary="already installed")
    with patch(_INSTALL, return_value=plan):
        tab.on_install_clicked("voyage")
    assert tab.status.value == "already installed"
    tab.app.page.show_dialog.assert_not_called()


def test_uninstall_active_provider_short_circuits_no_dialog():
    tab = _tab()
    plan = _plan(bundled=False, is_active=True, installed=True, summary="X is ACTIVE")
    with patch(_UNINSTALL, return_value=plan):
        tab.on_uninstall_clicked("voyage")
    assert tab.status.value == "X is ACTIVE"
    tab.app.page.show_dialog.assert_not_called()


def test_uninstall_installed_inactive_shows_confirm_dialog():
    tab = _tab()
    plan = _plan(bundled=False, is_active=False, installed=True, display_name="Voyage")
    with patch(_UNINSTALL, return_value=plan):
        tab.on_uninstall_clicked("voyage")
    tab.app.page.show_dialog.assert_called_once()
