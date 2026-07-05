"""Evaluation → Run sub-tab — configure + trigger an eval run.

Collects the run config (dataset picker, metric-group toggles, judge-model
panel, max-cases, LangSmith toggle) into an `EvalConfig`, calls
`runner.run(cfg, on_progress=...)` in-process with a progress bar, and on
completion asks the coordinator to select the new run + refresh the result
tabs. Reads the active corpus (read-only) from `config_store`.

Placeholder — the form + async run wiring lands next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


class RunTab:
    """Run-configuration + trigger sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator

    def build(self) -> ft.Control:
        return view_with_header(
            "Run Evaluation",
            empty_state("Run configuration (dataset, groups, judge panel) + progress lands next."),
        )
