"""Evaluation top-tab coordinator — 7 sub-tabs, full window (Library model).

Sub-tabs (fixed order), by display label:

  - Run evaluation — configure + trigger an eval run (dataset, metric
    groups, judge panel, max-cases) → `runner.run(cfg)` in-process with a
    progress bar; on completion, refresh + select the new run and hop to
    Run Summary.
  - Create test cases — browse/author the gold dataset: a scrollable case
    list + a full-field view of the selected case (editing / Add-Delete /
    capture-from-Search / LLM generation).
  - Run Summary — KPI cards + per-case table for the selected run.
  - Run Charts — metric-balance / distribution / correlation for the run.
  - Compare Datasets — several datasets' runs side by side (one kind), a
    metric × run table + grouped bars.
  - Trends — run-level metric trends over time (scoped per recipe hash).
  - Metrics Guide — the metrics reference (info_metrics.md).

Shared selected-run state lives here: the per-run tabs read
`self.selected_run_id`, and both the left-rail run selector and a completed
run in the Run tab set it. No auto-refresh polling — the GUI runs eval
in-process, so it refreshes exactly when a run finishes (plus a manual
Refresh in the result tabs). Modeled on `LibraryView`: each sub-tab owns
its internal layout; no shared splitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.evaluation._common import ALL_ORIGINS
from knowledge_agent.gui.evaluation.compare_tab import CompareDatasetsTab
from knowledge_agent.gui.evaluation.dataset_tab import DatasetTab
from knowledge_agent.gui.evaluation.metrics_guide_tab import MetricsGuideTab
from knowledge_agent.gui.evaluation.run_charts_tab import RunChartsTab
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab
from knowledge_agent.gui.evaluation.run_tab import RunTab
from knowledge_agent.gui.evaluation.trends_tab import TrendsTab
from knowledge_agent.gui.views.info_doc import InfoDoc, docs_dir

if TYPE_CHECKING:
    from knowledge_agent.evaluation.runner import SuiteRunResult
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = (
    "Run evaluation",
    "Create test cases",
    "Info",
    "Run Summary",
    "Run Charts",
    "Compare Datasets",
    "Trends",
    "Metrics Guide",
)

# The eval-output (view) tabs — tinted apart from the two authoring tabs, and
# auto-refreshed on select (they read the shared ledger; the authoring tabs
# own their own state).
_VIEW_TABS = frozenset({"Run Summary", "Run Charts", "Compare Datasets", "Trends", "Metrics Guide"})


class EvaluationView:
    """Top-tab coordinator for Evaluation — 5 sub-tabs, no shared splitter."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Shared dashboard selection, read by all five view tabs' left rails
        # (`DashboardRail`) so they stay in step. The rail is a Suite → Dataset →
        # Run cascade: `selected_suite` (a facts_hash) scopes the dataset list to
        # one suite's members; `selected_dataset` narrows the run picker + scopes
        # Trends (by that dataset's dataset_hash); `selected_run_id` drives the
        # per-run tabs (Run Summary / Run Charts) AND — via its suite_run_id —
        # Compare (the members of that suite execution). Set by the rails + the
        # Run tab on completion.
        self.selected_suite: str | None = None
        self.selected_dataset: str | None = None
        self.selected_run_id: int | None = None
        # Shared case-origin filter (the rail's checkboxes). All origins = no
        # filter; a subset restricts Run Summary / Run Charts / Compare to those
        # cases (Trends is unaffected). Read by those tabs via `filtered_run`.
        self.selected_origins: set[str] = set(ALL_ORIGINS)
        self._tabs: ft.Tabs | None = None
        self.run_tab = RunTab(app, coordinator=self)
        self.dataset_tab = DatasetTab(app, coordinator=self)
        self.run_summary_tab = RunSummaryTab(app, coordinator=self)
        self.run_charts_tab = RunChartsTab(app, coordinator=self)
        self.compare_tab = CompareDatasetsTab(app, coordinator=self)
        self.trends_tab = TrendsTab(app, coordinator=self)
        self.metrics_guide_tab = MetricsGuideTab(app, coordinator=self)

    def build(self) -> ft.Control:
        # The four view (eval-output) tabs get a tinted label so they read as a
        # distinct group from the two authoring tabs. (ft.Tab has no per-tab
        # background; a coloured label Control is the native way to tint one.)
        sub_bar = ft.TabBar(
            tabs=[
                ft.Tab(
                    label=ft.Text(label, color=ft.Colors.BLUE_300) if label in _VIEW_TABS else label
                )
                for label in SUB_TAB_LABELS
            ],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=self.run_tab.build(), padding=8, expand=True),
                ft.Container(content=self.dataset_tab.build(), padding=8, expand=True),
                ft.Container(
                    content=InfoDoc(
                        self.app,
                        docs_dir() / "info_evaluation.md",
                        missing_hint="Evaluation help is not written yet (info_evaluation.md).",
                    ).build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(content=self.run_summary_tab.build(), padding=8, expand=True),
                ft.Container(content=self.run_charts_tab.build(), padding=8, expand=True),
                ft.Container(content=self.compare_tab.build(), padding=8, expand=True),
                ft.Container(content=self.trends_tab.build(), padding=8, expand=True),
                ft.Container(content=self.metrics_guide_tab.build(), padding=8, expand=True),
            ],
            expand=True,
        )
        self._tabs = ft.Tabs(
            length=len(SUB_TAB_LABELS),
            selected_index=0,
            on_change=self._on_subtab_change,
            content=ft.Column(controls=[sub_bar, sub_bodies], expand=True, spacing=0),
            expand=True,
        )
        return self._tabs

    def _on_subtab_change(self, _e: ft.Event) -> None:
        """Auto-refresh a view tab when it's selected — the shared rail + body
        pick up the latest ledger without a manual Refresh. The two authoring
        tabs manage their own state, so they're skipped."""
        if self._tabs is None:
            return
        view_tabs = {
            SUB_TAB_LABELS.index("Run Summary"): self.run_summary_tab,
            SUB_TAB_LABELS.index("Run Charts"): self.run_charts_tab,
            SUB_TAB_LABELS.index("Compare Datasets"): self.compare_tab,
            SUB_TAB_LABELS.index("Trends"): self.trends_tab,
            SUB_TAB_LABELS.index("Metrics Guide"): self.metrics_guide_tab,
        }
        tab = view_tabs.get(self._tabs.selected_index)
        if tab is not None:
            tab.refresh()

    def on_run_complete(self, run_id: int) -> None:
        """A run finished in the Run tab: select it + jump to Run Summary.

        Refreshing the result tabs' content lands as those tabs are built;
        for now this records the run + switches to the Run Summary sub-tab.
        """
        self.selected_run_id = run_id
        self.run_summary_tab.refresh()
        if self._tabs is not None:
            self._tabs.selected_index = SUB_TAB_LABELS.index("Run Summary")
            self.app.page.update()

    def on_suite_complete(self, suite: SuiteRunResult) -> None:
        """A suite run finished: point the shared cascade at it (its suite + a
        member run) and jump to Compare, which shows that suite-run's members
        side by side (via the selected run's suite_run_id)."""
        if suite.results:
            first = suite.results[0]
            # The rail groups by suite NAME (not facts_hash), so point it there.
            self.selected_suite = first.report.get("suite")
            self.selected_dataset = first.report.get("dataset_name")
            self.selected_run_id = first.run_id
        self.compare_tab.refresh()
        if self._tabs is not None:
            self._tabs.selected_index = SUB_TAB_LABELS.index("Compare Datasets")
            self.app.page.update()
