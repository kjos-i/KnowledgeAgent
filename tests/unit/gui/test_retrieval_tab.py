"""Tests for the Retrieval tab's input-mode radio (R4c).

The radio (Conversational / Direct query / Direct Cypher) persists the
user's choice to `GuiConfig.input_mode` via the standard commit-with-
rollback pattern. The downstream *consumer*
(`app._invoke_state_for_input_mode`) is covered in test_app_input_mode;
here we pin the *setter* — a radio change writes the config and rolls
back cleanly on a save failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import ConfigError, GuiConfig
from knowledge_agent.gui.settings.retrieval_tab import RetrievalTab

_SAVE = "knowledge_agent.gui.settings.retrieval_tab.save_config"
_APPLY = "knowledge_agent.gui.settings.retrieval_tab.apply_retrieval_to_env"
_RESET = "knowledge_agent.gui.settings.retrieval_tab.reset_after_key_change"


def _tab() -> RetrievalTab:
    app = MagicMock()
    app.gui_config = GuiConfig()
    app.page = MagicMock()
    return RetrievalTab(app)


def test_input_mode_radio_reflects_config_default():
    tab = _tab()
    assert tab.input_mode_radio is not None
    assert tab.input_mode_radio.value == "conversational"


def test_input_mode_change_persists():
    tab = _tab()
    tab.input_mode_radio.value = "direct_cypher"
    with patch(_SAVE), patch(_APPLY), patch(_RESET):
        tab.on_input_mode_changed(MagicMock())
    assert tab.app.gui_config.input_mode == "direct_cypher"


def test_input_mode_change_rolls_back_on_save_failure():
    tab = _tab()
    tab.input_mode_radio.value = "direct_query"
    # save_config raising ConfigError → _commit returns False → revert.
    with (
        patch(_SAVE, side_effect=ConfigError("disk full")),
        patch(_APPLY),
        patch(_RESET),
    ):
        tab.on_input_mode_changed(MagicMock())
    assert tab.app.gui_config.input_mode == "conversational"
    assert tab.input_mode_radio.value == "conversational"
