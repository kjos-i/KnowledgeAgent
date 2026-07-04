"""Evaluation tab — eval-harness UI.

Slice 1 stub. The eval harness (`evaluation/` folder, planned per
[[evaluation-harness]]) lands as a backend module first; this tab
wires it into the UI after that ships:

  - Pick a dataset (eval-instance corpus, isolated from real data via
    `.env.eval` per [[test-instance-setup]])
  - Run a queryset → per-query results table (Hit@k, MRR, NDCG,
    faithfulness, answer_relevancy)
  - Aggregate metrics card
  - Optional LangSmith trace toggle (env-var-gated, eval-only)

Top-level tab because eval-result tables need the full window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


class EvaluationTab:
    """Evaluation tab — stub until the eval harness ships."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        return view_with_header(
            "Evaluation",
            empty_state(
                "Evaluation lands after the eval harness ships — "
                "queryset run, per-query metrics, aggregate dashboard."
            ),
        )
