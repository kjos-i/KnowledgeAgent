"""Tests for the LLM settings tab's chat-router row.

The router is GUI-only, but it reuses the per-node model+temp machinery
(keyed 'chat_router'). On a provider switch it must reset to the new
provider's curated model GUI-side — NOT via the backend registry, which
has no router entry.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
