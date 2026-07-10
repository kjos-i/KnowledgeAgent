"""Tests for the LLM settings tab's chat-router row.

The router is GUI-only, but it reuses the per-node model+temp machinery
(keyed 'chat_router'). On a provider switch it must reset to the new
provider's curated model GUI-side — NOT via the backend registry, which
has no router entry.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.settings.llm_tab import LLM_AVAILABLE_MODELS, LlmTab


def _tab() -> LlmTab:
    app = MagicMock()
    app.gui_config = GuiConfig()  # real defaults (anthropic)
    app.page = MagicMock()
    return LlmTab(app)  # _create_controls() runs in __init__


def test_llm_tab_builds_chat_router_field():
    tab = _tab()
    assert "chat_router" in tab.node_model_fields
    assert tab.node_model_fields["chat_router"].value == tab.app.gui_config.chat_router_model
    assert "chat_router" in tab.node_temp_sliders


def test_provider_switch_resets_router_model_gui_side():
    tab = _tab()
    assert tab.app.gui_config.llm_provider == "anthropic"
    # Simulate the user picking a different provider in the radio.
    tab.active_provider_radio = MagicMock(value="openai")
    # Patch both persistence side-effects: save_config (disk) and
    # apply_llm_to_env (which would leak os.environ into later tests).
    with (
        patch("knowledge_agent.gui.settings.llm_tab.save_config"),
        patch("knowledge_agent.gui.settings.llm_tab.apply_llm_to_env"),
    ):
        tab.on_active_provider_changed(MagicMock())
    assert tab.app.gui_config.chat_router_model == LLM_AVAILABLE_MODELS["openai"][0]
    # And the visible dropdown value tracks it.
    assert tab.node_model_fields["chat_router"].value == LLM_AVAILABLE_MODELS["openai"][0]


# ---- temperature-slider greying (sampling-free models) ----


def test_temp_slider_greys_out_for_sampling_free_model():
    """A node whose model dropped temperature (e.g. Opus 4.8) has its temp
    slider disabled with a tooltip; switching to a temp-taking model
    (Haiku 4.5) re-enables it. The backend omits temperature for those
    models regardless — this just surfaces that."""
    tab = _tab()
    tab.app.gui_config.synthesizer_model = "claude-opus-4-8"
    tab._sync_temp_enabled("synthesizer")
    assert tab.node_temp_sliders["synthesizer"].disabled is True
    assert "temperature" in (tab.node_temp_sliders["synthesizer"].tooltip or "")

    tab.app.gui_config.synthesizer_model = "claude-haiku-4-5"
    tab._sync_temp_enabled("synthesizer")
    assert tab.node_temp_sliders["synthesizer"].disabled is False
    assert tab.node_temp_sliders["synthesizer"].tooltip is None


# ---- install / uninstall wiring ----


def _plan(**kw) -> MagicMock:
    plan = MagicMock(**kw)
    plan.summary = kw.get("summary", "SUMMARY")
    return plan


def test_uninstall_active_provider_short_circuits_no_dialog():
    """The active provider can't be uninstalled — the handler surfaces the
    plan summary as a status and shows NO confirm dialog."""
    tab = _tab()
    plan = _plan(bundled=False, is_active=True, installed=True, summary="X is ACTIVE")
    with patch(
        "knowledge_agent.gui.settings.llm_tab.uninstall_llm_provider_plan", return_value=plan
    ):
        tab.on_uninstall_clicked("anthropic")
    assert tab.status.value == "X is ACTIVE"
    tab.app.page.show_dialog.assert_not_called()


def test_uninstall_installed_inactive_shows_confirm_dialog():
    tab = _tab()
    plan = _plan(bundled=False, is_active=False, installed=True, display_name="OpenAI")
    with patch(
        "knowledge_agent.gui.settings.llm_tab.uninstall_llm_provider_plan", return_value=plan
    ):
        tab.on_uninstall_clicked("openai")
    tab.app.page.show_dialog.assert_called_once()


def test_prompt_install_already_installed_short_circuits_no_dialog():
    tab = _tab()
    plan = _plan(bundled=False, already_installed=True, summary="already installed")
    with patch(
        "knowledge_agent.gui.settings.llm_tab.install_llm_provider_plan",
        AsyncMock(return_value=plan),
    ):
        asyncio.run(tab._prompt_install("anthropic"))
    assert tab.status.value == "already installed"
    tab.app.page.show_dialog.assert_not_called()


def test_prompt_install_installable_shows_confirm_dialog():
    tab = _tab()
    plan = _plan(bundled=False, already_installed=False, display_name="OpenAI")
    with patch(
        "knowledge_agent.gui.settings.llm_tab.install_llm_provider_plan",
        AsyncMock(return_value=plan),
    ):
        asyncio.run(tab._prompt_install("openai"))
    tab.app.page.show_dialog.assert_called_once()
