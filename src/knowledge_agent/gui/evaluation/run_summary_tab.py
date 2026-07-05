"""Evaluation → Run Summary sub-tab — KPIs + per-case table for one run.

Left rail: run selector + selected-run metadata (judge models, cases,
enabled groups, thresholds, dataset). Body: registry-driven KPI cards
(derived from the metric groups, not hardcoded lists) + a per-case
`DataTable` with threshold-colored cells. Reads the run + its cases from
the `EvalLedger` readers (deferred to first show, not build time).

Placeholder — KPIs + table land next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


class RunSummaryTab:
    """Per-run KPI + per-case-table sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator

    def build(self) -> ft.Control:
        return view_with_header(
            "Run Summary",
            empty_state("KPI cards + per-case results table land next."),
        )
