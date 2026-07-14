"""EvaluationView coordinator — the 6-sub-tab shell, the view-tab tint, and the
auto-refresh-on-select dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.evaluation import EvaluationView
from knowledge_agent.gui.evaluation.evaluation_view import SUB_TAB_LABELS

if TYPE_CHECKING:
    from unittest.mock import MagicMock as MM  # noqa: F401


def _tab_label(t: ft.Tab) -> str:
    """A tab's label as text — the four view tabs carry a coloured `ft.Text`
    label, the authoring tabs a plain string."""
    return t.label.value if isinstance(t.label, ft.Text) else t.label


def test_evaluation_view_builds_six_sub_tabs(fake_app: MagicMock):
    view = EvaluationView(fake_app)
    ctl = view.build()
    assert isinstance(ctl, ft.Tabs)
    assert ctl.length == len(SUB_TAB_LABELS) == 6
    tab_bar = ctl.content.controls[0]
    assert [_tab_label(t) for t in tab_bar.tabs] == list(SUB_TAB_LABELS)
    assert "Create test cases" in SUB_TAB_LABELS


def test_view_tabs_are_tinted(fake_app: MagicMock):
    """The four eval-output tabs get a coloured label; the two authoring tabs
    keep a plain string label."""
    view = EvaluationView(fake_app)
    tab_bar = view.build().content.controls[0]
    by_label = {_tab_label(t): t for t in tab_bar.tabs}
    assert isinstance(by_label["Run Summary"].label, ft.Text)  # tinted view tab
    assert by_label["Trends"].label.color == ft.Colors.INDIGO_300
    assert by_label["Run evaluation"].label == "Run evaluation"  # plain authoring tab


def test_selecting_view_tab_auto_refreshes(fake_app: MagicMock):
    """Selecting a view tab auto-refreshes it (no manual Refresh); selecting an
    authoring tab refreshes no view tab."""
    view = EvaluationView(fake_app)
    view.build()
    view.run_summary_tab.refresh = MagicMock()

    view._tabs.selected_index = SUB_TAB_LABELS.index("Run Summary")
    view._on_subtab_change(MagicMock())
    view.run_summary_tab.refresh.assert_called_once()

    view.run_summary_tab.refresh.reset_mock()
    view._tabs.selected_index = SUB_TAB_LABELS.index("Run evaluation")
    view._on_subtab_change(MagicMock())
    view.run_summary_tab.refresh.assert_not_called()


def test_evaluation_view_selected_run_defaults_none(fake_app: MagicMock):
    assert EvaluationView(fake_app).selected_run_id is None
    assert EvaluationView(fake_app).selected_dataset is None
