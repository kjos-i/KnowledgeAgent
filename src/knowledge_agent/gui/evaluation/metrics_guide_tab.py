"""Evaluation → Metrics Guide sub-tab — the metrics reference.

Renders the harness's `info_metrics.md` via `ft.Markdown` so the in-app
reference never drifts from the source doc. The doc lives beside the harness
(`knowledge_agent/evaluation/info_metrics.md`) — the single source that the
CLI/docs can share too.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


def _guide_path() -> Path:
    import knowledge_agent.evaluation as ka_eval

    return Path(ka_eval.__file__).parent / "info_metrics.md"


class MetricsGuideTab:
    """Static metrics-reference sub-tab (renders info_metrics.md)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        path = _guide_path()
        if not path.exists():
            body: ft.Control = empty_state(f"Metrics guide not found at {path}.")
        else:
            body = ft.Column(
                controls=[
                    ft.Markdown(
                        path.read_text(encoding="utf-8"),
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    )
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        return view_with_header("Metrics Guide", body)
