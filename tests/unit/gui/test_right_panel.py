"""Tests for `knowledge_agent.gui.right_panel.RightPanel`.

RightPanel is the Search tab's **View · Retrieval · LLM** sub-tab strip.
The View sub-tab shows a small in-memory history of recently-viewed results
(answers + opened files) with a `< n/N > ✕` pager; Retrieval / LLM are the
promoted settings tabs. A real `GuiConfig` is used because building the panel
builds those two settings sub-tabs, which read config values at construct time.
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
from knowledge_agent.models import AgentAnswer

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def _panel(fake_app: MagicMock) -> RightPanel:
    fake_app.gui_config = GuiConfig()
    fake_app.loaded_file = None
    return RightPanel(fake_app)


def _answer(text: str = "hi") -> AgentAnswer:
    return AgentAnswer(answer=text, chunk_sources=[], kg_sources=[])


def _button_row(ctl: ft.Tabs) -> ft.Row:
    # ft.Tabs -> Column[sub_bar, sub_bodies]; sub_bodies is a TabBarView whose
    # first body is the View Column[view_container, button_row].
    sub_bodies = ctl.content.controls[1]
    view_body = sub_bodies.controls[0].content
    return view_body.controls[1]


# ---- structure --------------------------------------------------------------


def test_build_returns_view_retrieval_llm_subtabs(fake_app: MagicMock):
    panel = _panel(fake_app)
    ctl = panel.build()
    assert isinstance(ctl, ft.Tabs)
    assert ctl.length == len(SUB_TAB_LABELS) == 4
    assert SUB_TAB_LABELS == ("View", "Retrieval", "LLMs", "Info")
    assert panel.current_mode == MODE_LATEST


def test_view_subtab_has_save_open_buttons_then_pager(fake_app: MagicMock):
    """The View sub-tab holds the display + a [Save, Open, pager] action row —
    the pager (prev / label / next / close) sits right of the two buttons."""
    panel = _panel(fake_app)
    ctl = panel.build()
    button_row = _button_row(ctl)
    assert isinstance(button_row, ft.Row)
    pager_row = button_row.controls[-1]
    assert isinstance(pager_row, ft.Row)
    assert len(pager_row.controls) == 4  # prev, label, next, close
    buttons = [c for c in button_row.controls if isinstance(c, ft.Button)]
    assert [b.content.value for b in buttons] == ["Save Result", "Open Result"]


def test_view_subtab_has_paste_path_field_between_open_and_pager(fake_app: MagicMock):
    """A paste-path TextField sits between Open Result and the pager, wired to
    `on_open_path`; row order is [Save, Open, path field, pager]."""
    panel = _panel(fake_app)
    ctl = panel.build()
    button_row = _button_row(ctl)
    assert isinstance(button_row.controls[2], ft.TextField)
    assert button_row.controls[2] is panel.path_field
    assert panel.path_field.on_submit is fake_app.on_open_path
    assert isinstance(button_row.controls[3], ft.Row)  # pager stays last


# ---- empty / initial state --------------------------------------------------


def test_history_starts_empty_with_greyed_pager(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    assert panel.history == []
    assert panel.history_index == -1
    assert panel.pager_label.value == "0/0"
    assert panel.pager_prev.disabled is True
    assert panel.pager_next.disabled is True
    assert panel.pager_close.disabled is True  # nothing to remove


# ---- push -------------------------------------------------------------------


def test_push_answer_appends_and_jumps(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    panel.push_answer(_answer("a"), "q1")
    panel.push_answer(_answer("b"), "q2")
    assert len(panel.history) == 2
    assert panel.history_index == 1  # jumped to newest
    assert panel.current_mode == MODE_LATEST
    assert panel.pager_label.value == "2/2"
    assert panel.pager_prev.disabled is False
    assert panel.pager_next.disabled is True  # at the newest end


def test_push_file_appends_a_file_slot(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    panel.push_file("answer.md", "# hi")
    assert len(panel.history) == 1
    item = panel.history[panel.history_index]
    assert item.kind == MODE_FILE
    assert item.name == "answer.md"
    assert panel.current_mode == MODE_FILE


def test_history_caps_at_five_dropping_oldest(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    for i in range(6):
        panel.push_answer(_answer(f"a{i}"), f"q{i}")
    assert len(panel.history) == 5
    assert panel.history_index == 4
    # The very first push (q0) fell off the front; q1 is now slot 0.
    assert panel.history[0].query == "q1"
    assert panel.history[-1].query == "q5"


# ---- navigation -------------------------------------------------------------


def test_prev_next_navigation(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    for i in range(3):
        panel.push_answer(_answer(), f"q{i}")
    assert panel.history_index == 2
    panel._on_prev(None)
    panel._on_prev(None)
    assert panel.history_index == 0
    panel._on_prev(None)  # clamped at the start
    assert panel.history_index == 0
    assert panel.pager_prev.disabled is True
    panel._on_next(None)
    assert panel.history_index == 1
    assert panel.pager_label.value == "2/3"


def test_close_removes_current_and_lands_on_previous(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    for i in range(3):
        panel.push_answer(_answer(), f"q{i}")
    # At the newest (index 2). Closing lands on the previous slot.
    panel._on_close_item(None)
    assert len(panel.history) == 2
    assert panel.history_index == 1
    panel._on_close_item(None)
    assert len(panel.history) == 1
    assert panel.history_index == 0
    panel._on_close_item(None)  # remove the last one → empty
    assert panel.history == []
    assert panel.history_index == -1
    assert panel.pager_label.value == "0/0"


def test_close_first_item_lands_on_new_first(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    for i in range(3):
        panel.push_answer(_answer(), f"q{i}")
    panel._on_prev(None)
    panel._on_prev(None)  # now at index 0 (q0)
    panel._on_close_item(None)
    assert len(panel.history) == 2
    assert panel.history_index == 0
    assert panel.history[0].query == "q1"  # old slot 1 slid to the front


# ---- reset (Clear) ----------------------------------------------------------


def test_reset_history_empties_when_no_loaded_file(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    panel.push_answer(_answer(), "q")
    panel.push_answer(_answer(), "q2")
    fake_app.loaded_file = None
    panel.reset_history()
    assert panel.history == []
    assert panel.history_index == -1


def test_reset_history_reseeds_surviving_loaded_file(fake_app: MagicMock):
    """A file kept across Clear (keep_loaded_file_on_clear) is re-seeded as the
    pager's sole slot so it stays visible."""
    panel = _panel(fake_app)
    panel.build()
    panel.push_answer(_answer(), "q")
    fake_app.loaded_file = SimpleNamespace(name="kept.md", content="# kept")
    panel.reset_history()
    assert len(panel.history) == 1
    assert panel.history_index == 0
    item = panel.history[0]
    assert item.kind == MODE_FILE
    assert item.name == "kept.md"


# ---- single-item pager states ----------------------------------------------


def test_single_item_greys_prev_next_but_allows_close(fake_app: MagicMock):
    panel = _panel(fake_app)
    panel.build()
    panel.push_answer(_answer(), "q")
    assert panel.pager_label.value == "1/1"
    assert panel.pager_prev.disabled is True
    assert panel.pager_next.disabled is True
    assert panel.pager_close.disabled is False
