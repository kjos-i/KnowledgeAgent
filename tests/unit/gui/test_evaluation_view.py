"""EvaluationView coordinator — static build of the 6-sub-tab shell.

Placeholder sub-tabs build without touching app state; the shape (a Tabs
shell with the fixed sub-tab labels) is what's asserted here. Per-sub-tab
behavior gets its own tests as each is fleshed out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.evaluation import EvaluationView
from knowledge_agent.gui.evaluation.evaluation_view import SUB_TAB_LABELS

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def test_evaluation_view_builds_six_sub_tabs(fake_app: MagicMock):
    view = EvaluationView(fake_app)
    ctl = view.build()
    assert isinstance(ctl, ft.Tabs)
    assert ctl.length == len(SUB_TAB_LABELS) == 6
    tab_bar = ctl.content.controls[0]
    assert [t.label for t in tab_bar.tabs] == list(SUB_TAB_LABELS)
    assert "Create test cases" in SUB_TAB_LABELS


def test_evaluation_view_selected_run_defaults_none(fake_app: MagicMock):
    assert EvaluationView(fake_app).selected_run_id is None
