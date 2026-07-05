"""Evaluation top tab — dispatches to the full-window EvaluationView.

The real layout (5 sub-tabs: Run / Run Summary / Deep Analysis / Trends /
Metrics Guide) lives in `gui.evaluation.evaluation_view`. This module stays
thin so the existing `GuiApp` wiring (which constructs `EvaluationTab`)
doesn't change shape as the sub-tabs grow — same pattern as `LibraryTab`.

Top-level (full-window) tab, not a right-panel mode: eval result tables +
charts need the whole window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.evaluation import EvaluationView

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


class EvaluationTab:
    """Thin wrapper — delegates `build()` to `EvaluationView`."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.view = EvaluationView(app)

    def build(self) -> ft.Control:
        return self.view.build()
