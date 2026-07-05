"""Evaluation → Trends sub-tab — run-level metric trends over time.

Multi-series `ft.LineChart`s reading every `eval_runs` row (via
`EvalLedger.list_runs`), scoped to ONE dataset (left-rail dataset filter,
default = the selected run's dataset) so comparisons stay apples-to-apples.
Reads pre-aggregated `avg_*` columns — no in-GUI aggregation, no pandas.

Placeholder — Phase B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


class TrendsTab:
    """Cross-run trend charts sub-tab (Phase B)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator

    def build(self) -> ft.Control:
        return view_with_header(
            "Historical Trends",
            empty_state("Per-dataset metric trends over time land in Phase B."),
        )
