"""Evaluation top-tab coordinator — 6 sub-tabs, full window (Library model).

Sub-tabs (fixed order), by display label:

  - Run evaluation — configure + trigger an eval run (dataset, metric
    groups, judge panel, max-cases) → `runner.run(cfg)` in-process with a
    progress bar; on completion, refresh + select the new run and hop to
    Run Summary.
  - Create test cases — browse/author the gold dataset: a scrollable case
    list + a full-field view of the selected case (editing / Add-Delete /
    capture-from-Search / LLM generation).
  - Run Summary — KPI cards + per-case table for the selected run.
  - Deep Analysis — metric-balance / distribution / correlation for the run.
  - Trends — run-level metric trends over time (scoped per dataset).
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

from knowledge_agent.gui.evaluation.dataset_tab import DatasetTab
from knowledge_agent.gui.evaluation.deep_analysis_tab import DeepAnalysisTab
from knowledge_agent.gui.evaluation.metrics_guide_tab import MetricsGuideTab
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab
from knowledge_agent.gui.evaluation.run_tab import RunTab
from knowledge_agent.gui.evaluation.trends_tab import TrendsTab

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = (
    "Run evaluation",
    "Create test cases",
    "Run Summary",
    "Deep Analysis",
    "Trends",
    "Metrics Guide",
)


class EvaluationView:
    """Top-tab coordinator for Evaluation — 5 sub-tabs, no shared splitter."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # The run the per-run result tabs display. None until a run exists /
        # is selected. Set by the Run tab on completion + the left-rail
        # selector.
        self.selected_run_id: int | None = None
        self._tabs: ft.Tabs | None = None
        self.run_tab = RunTab(app, coordinator=self)
        self.dataset_tab = DatasetTab(app, coordinator=self)
        self.run_summary_tab = RunSummaryTab(app, coordinator=self)
        self.deep_analysis_tab = DeepAnalysisTab(app, coordinator=self)
        self.trends_tab = TrendsTab(app, coordinator=self)
        self.metrics_guide_tab = MetricsGuideTab(app)

    def build(self) -> ft.Control:
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=self.run_tab.build(), padding=8, expand=True),
                ft.Container(content=self.dataset_tab.build(), padding=8, expand=True),
                ft.Container(content=self.run_summary_tab.build(), padding=8, expand=True),
                ft.Container(content=self.deep_analysis_tab.build(), padding=8, expand=True),
                ft.Container(content=self.trends_tab.build(), padding=8, expand=True),
                ft.Container(content=self.metrics_guide_tab.build(), padding=8, expand=True),
            ],
            expand=True,
        )
        self._tabs = ft.Tabs(
            length=len(SUB_TAB_LABELS),
            selected_index=0,
            content=ft.Column(controls=[sub_bar, sub_bodies], expand=True, spacing=0),
            expand=True,
        )
        return self._tabs

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
