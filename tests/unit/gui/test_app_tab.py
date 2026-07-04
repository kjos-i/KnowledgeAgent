"""Tests for the App settings tab's info-icon toggle handler.

`on_show_info_icons_changed` persists the `show_info_icons` toggle and
flips every registered (i) icon live via the GuiApp registry (the
registry itself is covered in test_info_icon). Here we pin the handler:
it saves, drives the registry, and rolls back on a save failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import ConfigError, GuiConfig
from knowledge_agent.gui.settings.app_tab import AppTab

_SAVE = "knowledge_agent.gui.settings.app_tab.save_config"


def _tab() -> AppTab:
    app = MagicMock()
    app.gui_config = GuiConfig()  # show_info_icons defaults True
    app.page = MagicMock()
    return AppTab(app)


def test_toggle_off_persists_and_flips_icons():
    tab = _tab()
    tab.show_info_icons_checkbox.value = False
    with patch(_SAVE):
        tab.on_show_info_icons_changed(MagicMock())
    assert tab.app.gui_config.show_info_icons is False
    tab.app.set_info_icons_visible.assert_called_once_with(False)


def test_toggle_rolls_back_on_save_failure():
    tab = _tab()
    tab.show_info_icons_checkbox.value = False
    with patch(_SAVE, side_effect=ConfigError("disk full")):
        tab.on_show_info_icons_changed(MagicMock())
    # Save failed → config + checkbox revert; icons NOT flipped.
    assert tab.app.gui_config.show_info_icons is True
    assert tab.show_info_icons_checkbox.value is True
    tab.app.set_info_icons_visible.assert_not_called()
