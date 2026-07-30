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
    """A tab's label as text — the view tabs carry a coloured `ft.Text` label,
    the authoring tabs a plain string, and Ledger a `ft.Row` (edit icon + Text)."""
    label = t.label
    if isinstance(label, ft.Row):  # Ledger: icon + coloured Text
        label = next(c for c in label.controls if isinstance(c, ft.Text))
    return label.value if isinstance(label, ft.Text) else label


def test_evaluation_view_builds_nine_sub_tabs(fake_app: MagicMock):
    view = EvaluationView(fake_app)
    ctl = view.build()
    assert isinstance(ctl, ft.Tabs)
    assert ctl.length == len(SUB_TAB_LABELS) == 9
    tab_bar = ctl.content.controls[0]
    sub_bodies = ctl.content.controls[1]
    assert [_tab_label(t) for t in tab_bar.tabs] == list(SUB_TAB_LABELS)
    assert len(sub_bodies.controls) == len(SUB_TAB_LABELS)  # labels + bodies aligned
    # Info sits between the authoring tabs and the dashboards.
    assert SUB_TAB_LABELS.index("Info") == SUB_TAB_LABELS.index("Create test cases") + 1
    assert SUB_TAB_LABELS.index("Info") == SUB_TAB_LABELS.index("Run Summary") - 1
    # Ledger (the editing/management tab) sits last (rightmost).
    assert SUB_TAB_LABELS[-1] == "Ledger"


def test_view_tabs_are_tinted(fake_app: MagicMock):
    """The four eval-output tabs get a coloured label; the authoring tabs and the
    contextual Info tab keep a plain string label."""
    view = EvaluationView(fake_app)
    tab_bar = view.build().content.controls[0]
    by_label = {_tab_label(t): t for t in tab_bar.tabs}
    assert isinstance(by_label["Run Summary"].label, ft.Text)  # tinted view tab
    assert by_label["Trends"].label.color == ft.Colors.BLUE_300
    assert by_label["Run evaluation"].label == "Run evaluation"  # plain authoring tab
    assert by_label["Info"].label == "Info"  # plain — not a tinted view tab
    # Ledger: edit icon + amber Text label (the editing/management tab).
    ledger_label = by_label["Ledger"].label
    assert isinstance(ledger_label, ft.Row)
    assert any(isinstance(c, ft.Icon) for c in ledger_label.controls)
    text = next(c for c in ledger_label.controls if isinstance(c, ft.Text))
    assert text.color == ft.Colors.AMBER_300


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


def test_on_suite_complete_points_cascade_and_switches(fake_app: MagicMock):
    """A finished suite points the shared cascade at it (suite + a member run) and
    jumps to the Compare Datasets tab, which reads the run's suite_run_id."""
    view = EvaluationView(fake_app)
    view.build()
    view.compare_tab.refresh = MagicMock()
    suite = MagicMock(
        results=[
            MagicMock(run_id=1, report={"dataset_name": "facts_vector", "suite": "mode-sweep"}),
            MagicMock(run_id=2, report={"dataset_name": "facts_graph", "suite": "mode-sweep"}),
        ]
    )
    view.on_suite_complete(suite)
    assert view.selected_suite == "mode-sweep"  # the rail groups by suite NAME
    assert view.selected_dataset == "facts_vector"
    assert view.selected_run_id == 1  # first member drives Compare via its suite_run_id
    view.compare_tab.refresh.assert_called_once()
    assert view._tabs.selected_index == SUB_TAB_LABELS.index("Compare Datasets")
