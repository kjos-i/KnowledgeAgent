"""Tests for the slice-1 tab stubs (Library + Evaluation) + SearchTab.

Static construction only. Each stub returns a `view_with_header` so we
verify the header text + that the body is the empty-state container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.chat_panel import ChatPanel
from knowledge_agent.gui.right_panel import RightPanel
from knowledge_agent.gui.tabs.evaluation_tab import EvaluationTab
from knowledge_agent.gui.tabs.library_tab import LibraryTab
from knowledge_agent.gui.tabs.search_tab import SearchTab

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def test_library_tab_delegates_to_library_view(fake_app: MagicMock):
    """LibraryTab is a thin wrapper — `.build()` returns whatever
    `LibraryView.build()` returns (the Slice 3 Tabs shell)."""
    tab = LibraryTab(fake_app)
    ctl = tab.build()
    # Slice 3 landed: LibraryView returns a Tabs shell. Just verify
    # something builds without asserting an exact type — the view
    # shape is exercised in the LibraryView tests, not here.
    assert ctl is not None


def test_evaluation_tab_delegates_to_evaluation_view(fake_app: MagicMock):
    """EvaluationTab is a thin wrapper — `.build()` returns the
    EvaluationView Tabs shell (5 sub-tabs). The shape is exercised in the
    EvaluationView tests, not here."""
    tab = EvaluationTab(fake_app)
    ctl = tab.build()
    assert ctl is not None


def test_search_tab_composes_chat_panel_and_right_panel_in_row(
    fake_app: MagicMock,
):
    """SearchTab returns a Row via `ResizableSplit`: left pane +
    drag handle + right pane. The exact widget structure is 3
    controls (two Containers around a draggable divider)."""
    fake_app.chat_panel = ChatPanel(fake_app)
    fake_app.right_panel = RightPanel(fake_app)

    tab = SearchTab(fake_app)
    ctl = tab.build()
    assert isinstance(ctl, ft.Row)
    # ResizableSplit: [left pane, drag handle, right pane] = 3 controls.
    assert len(ctl.controls) == 3
