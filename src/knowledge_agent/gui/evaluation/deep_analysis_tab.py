"""Evaluation → Deep Analysis sub-tab — metric balance / spread / correlation.

For the selected run: metric-balance charts (native bar-of-means in place of
the reference's radar), a score-distribution histogram (bin in plain Python),
a correlation grid (`statistics.correlation`, no pandas), and latency / token
bars. All native Flet charts; no new dependencies.

Placeholder — Phase B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


class DeepAnalysisTab:
    """Per-run deep-analysis charts sub-tab (Phase B)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator

    def build(self) -> ft.Control:
        return view_with_header(
            "Deep Analysis",
            empty_state(
                "Metric balance, distribution, correlation, latency/tokens land in Phase B."
            ),
        )
