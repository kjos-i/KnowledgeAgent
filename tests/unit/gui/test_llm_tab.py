"""Tests for the LLM settings tab — chat-router row, per-node temperature
greying, and the provider-switch model reset.

Provider install/uninstall moved to the Installs tab (see test_installs_tab).
The router is GUI-only but reuses the per-node model+temp machinery (keyed
'chat_router'); on a provider switch it resets to the new provider's curated
model GUI-side — NOT via the backend registry, which has no router entry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.config import PROVIDER_NODE_DEFAULTS
from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.settings.llm_tab import LlmTab


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
    # chat_router borrows mode_classifier's default from the single source
    # (config), NOT the menu's first item.
    expected_router = PROVIDER_NODE_DEFAULTS["openai"]["mode_classifier"]
    assert tab.app.gui_config.chat_router_model == expected_router
    # And the visible dropdown value tracks it.
    assert tab.node_model_fields["chat_router"].value == expected_router


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
