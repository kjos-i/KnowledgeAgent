"""Tests for `knowledge_agent.gui.right_panel.RightPanel`.

RightPanel is now the Search tab's **View · Retrieval · LLM** sub-tab strip.
The View sub-tab shows the latest answer or an opened file (`switch_mode`
between MODE_LATEST / MODE_FILE); Retrieval / LLM are the promoted settings
tabs. A real `GuiConfig` is used because building the panel builds those two
settings sub-tabs, which read config values at construct time.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.right_panel import (
    MODE_FILE,
    MODE_LATEST,
    SUB_TAB_LABELS,
    RightPanel,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def _panel(fake_app: MagicMock) -> RightPanel:
    fake_app.gui_config = GuiConfig()
    return RightPanel(fake_app)


def test_build_returns_view_retrieval_llm_subtabs(fake_app: MagicMock):
    panel = _panel(fake_app)
    ctl = panel.build()
    assert isinstance(ctl, ft.Tabs)
    assert ctl.length == len(SUB_TAB_LABELS) == 3
    assert SUB_TAB_LABELS == ("View", "Retrieval", "LLMs")
    assert panel.current_mode == MODE_LATEST


def test_view_subtab_has_save_and_open_buttons(fake_app: MagicMock):
    """The first sub-tab (View) holds the result display + a Save / Open row —
    no more mode buttons (View is the tab; Settings / Info are top-level)."""
    panel = _panel(fake_app)
    ctl = panel.build()
    # ft.Tabs -> Column[sub_bar, sub_bodies]; sub_bodies is a TabBarView.
    sub_bodies = ctl.content.controls[1]
    assert isinstance(sub_bodies, ft.TabBarView)
    view_body = sub_bodies.controls[0].content  # Column[view_container, button_row]
    button_row = view_body.controls[1]
    assert isinstance(button_row, ft.Row)
    labels = [b.content.value for b in button_row.controls]
    assert labels == ["Save Result", "Open Result"]


def test_switch_mode_file_shows_the_loaded_file(fake_app: MagicMock):
    panel = _panel(fake_app)
    fake_app.loaded_file = SimpleNamespace(name="answer.md", content="# hi")
    panel.build()
    panel.switch_mode(MODE_FILE)
    assert panel.current_mode == MODE_FILE
    assert panel.view_container is not None  # rebuilt without raising


def test_switch_mode_file_with_no_loaded_file_falls_through_to_latest(fake_app: MagicMock):
    """Defensive: Clear may wipe loaded_file while MODE_FILE is active — the
    view dispatch falls back to the Latest view instead of crashing."""
    panel = _panel(fake_app)
    fake_app.loaded_file = None
    panel.build()
    panel.switch_mode(MODE_FILE)  # must not raise
    assert panel.view_container is not None
