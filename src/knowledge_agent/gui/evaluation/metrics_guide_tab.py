"""Evaluation → Metrics Guide sub-tab — the metrics reference.

Renders a KA-specific `info_metrics.md` via `ft.Markdown` so the in-app
reference never drifts from the source doc. The KA guide reuses the
reference's judge / source / chunk / keyword metric explanations, adds the
KG-metric group, drops the metrics KA doesn't compute, and rewrites the
verdict-logic section for KA's gates + the direct_retrieval pathway.

Placeholder — the KA `info_metrics.md` gets written next.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


class MetricsGuideTab:
    """Static metrics-reference sub-tab (renders info_metrics.md)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        return view_with_header(
            "Metrics Guide",
            empty_state("The metrics reference (info_metrics.md) lands next."),
        )
