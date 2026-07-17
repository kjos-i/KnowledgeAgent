"""Evaluation → Metrics Guide sub-tab — the metrics reference.

Renders the harness's `info_metrics.md` so the in-app reference never drifts from
the source doc. The doc lives beside the harness
(`knowledge_agent/evaluation/info_metrics.md`) — the single source the CLI/docs
can share too.

The rendering itself (load → split at `<a id>` anchors → themed markdown + native
tables → in-page anchor scroll) is the shared `views/info_doc.InfoDoc`, the same
widget every Info tab uses. This tab adds only the Evaluation chrome around it: a
left rail matching the other view tabs (its Refresh reloads the doc) + a header.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.gui.views.info_doc import InfoDoc

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


def _guide_path() -> Path:
    import knowledge_agent.evaluation as ka_eval

    return Path(ka_eval.__file__).parent / "info_metrics.md"


class MetricsGuideTab:
    """Static metrics-reference sub-tab (renders info_metrics.md)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.rail: DashboardRail | None = None
        self.body: ft.Container | None = None
        # The shared doc renderer; recreated on reload so the scroll handler tracks
        # the current body. Exposed for tests via `self.doc`.
        self.doc: InfoDoc | None = None

    def build(self) -> ft.Control:
        # The SAME shared rail as the other three view tabs (its selection isn't
        # used here — the guide is static — but the column stays consistent, and
        # its Refresh reloads the doc via `on_change`).
        self.rail = DashboardRail(self.app, self.coordinator, on_change=self._reload_guide)
        rail_ctl = self.rail.build()
        self.body = ft.Container(content=self._doc_body(), expand=True)
        return ft.Row(
            [
                rail_ctl,
                ft.Column(
                    [view_header("Metrics Guide"), self.body],
                    expand=True,
                    spacing=8,
                ),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def refresh(self) -> None:
        """Sync the shared rail (runs + selection). The guide body is static;
        it reloads via the rail's Refresh (`_reload_guide`)."""
        if self.rail is not None:
            self.rail.refresh()

    def _doc_body(self) -> ft.Control:
        path = _guide_path()
        self.doc = InfoDoc(self.app, path, missing_hint=f"Metrics guide not found at {path}.")
        return self.doc.build()

    def _reload_guide(self) -> None:
        """Reload the guide doc from disk (it's the single source; may have been
        edited). Fired by the shared rail's Refresh."""
        if self.body is not None:
            self.body.content = self._doc_body()
            self.app.page.update()
