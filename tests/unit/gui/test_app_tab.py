"""Tests for the App settings tab's info-icon toggle handler.

`on_show_info_icons_changed` persists the `show_info_icons` toggle and
flips every registered (i) icon live via the GuiApp registry (the
registry itself is covered in test_info_icon). Here we pin the handler:
it saves, drives the registry, and rolls back on a save failure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


# ---- Block 0: Save results & chat ----


def test_save_formats_default_is_md_only():
    assert GuiConfig().save_formats == ["md"]


def test_save_format_checkbox_persists_in_canonical_order():
    tab = _tab()  # md checked by default, others off
    tab.save_format_checkboxes["txt"].value = True
    with patch(_SAVE):
        tab.on_save_format_changed(MagicMock())
    assert tab.app.gui_config.save_formats == ["md", "txt"]


def test_save_format_keeps_at_least_one():
    tab = _tab()
    tab.save_format_checkboxes["md"].value = False  # uncheck the only one
    with patch(_SAVE) as save:
        tab.on_save_format_changed(MagicMock())
    save.assert_not_called()  # empty selection is never persisted
    assert tab.app.gui_config.save_formats == ["md"]
    assert tab.save_format_checkboxes["md"].value is True  # re-checked


async def test_browse_folder_persists_results_dir(tmp_path):
    tab = _tab()
    tab.app.file_picker = MagicMock()
    tab.app.file_picker.get_directory_path = AsyncMock(return_value=str(tmp_path))
    with patch(_SAVE):
        await tab.on_browse_folder(MagicMock())
    assert tab.app.gui_config.results_dir == tmp_path
    assert tab.results_dir_text.value == str(tmp_path)
