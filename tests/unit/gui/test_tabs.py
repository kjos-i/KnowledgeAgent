"""Tests for the slice-1 tab stubs (Library + Evaluation) + SearchTab.

Static construction only. Each stub returns a `view_with_header` so we
verify the header text + that the body is the empty-state container.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.chat_panel import ChatPanel
from knowledge_agent.gui.right_panel import RightPanel
from knowledge_agent.gui.tabs.evaluation_tab import EvaluationTab
from knowledge_agent.gui.tabs.library_tab import LibraryTab
from knowledge_agent.gui.tabs.search_tab import SearchTab


def test_library_tab_stub_mentions_slice_3(fake_app: MagicMock):
    tab = LibraryTab(fake_app)
    ctl = tab.build()
    assert isinstance(ctl, ft.Column)
    # Body is the empty-state container.
    body = ctl.controls[1]
    text = body.content
    assert "slice 3" in text.value.lower()


def test_evaluation_tab_stub_mentions_eval_harness(fake_app: MagicMock):
    tab = EvaluationTab(fake_app)
    ctl = tab.build()
    body = ctl.controls[1]
    assert "eval" in body.content.value.lower()


def test_search_tab_composes_chat_panel_and_right_panel_in_row(
    fake_app: MagicMock,
):
    """SearchTab is a 45/55 Row composition; both panels are present
    inside Container wrappers with the expected `expand` ratios."""
    fake_app.chat_panel = ChatPanel(fake_app)
    fake_app.right_panel = RightPanel(fake_app)

    tab = SearchTab(fake_app)
    ctl = tab.build()
    assert isinstance(ctl, ft.Row)
    assert len(ctl.controls) == 2
    chat_container, right_container = ctl.controls
    assert isinstance(chat_container, ft.Container)
    assert isinstance(right_container, ft.Container)
    # Ratios are 45/55 — pinned because they match RA.
    assert chat_container.expand == 45
    assert right_container.expand == 55
