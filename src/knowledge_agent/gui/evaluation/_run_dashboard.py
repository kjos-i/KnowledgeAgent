"""Shared scaffolding for the run-dashboard sub-tabs (Run Summary + Run Charts).

Both tabs frame a `DashboardRail` beside a scrolling body and, on refresh, resolve
the coordinator's selected run through the shared origin filter. This module is the
single source for that shell + run-resolution so the two tabs don't mirror it —
each tab owns only its own body-building.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView


def dashboard_shell(
    rail_ctl: ft.Control, title: str, body: ft.Column, *, trailing: ft.Control | None = None
) -> ft.Control:
    """The two-column dashboard frame: the run rail on the left, a titled scrolling
    body on the right. Shared by Run Summary + Run Charts so the shell lives in one
    place (the caller builds `body` with its own placeholder + spacing). Pass
    `trailing` (e.g. an info icon) to sit it beside the title."""
    return ft.Row(
        [
            rail_ctl,
            ft.Column([view_header(title, trailing=trailing), body], expand=True, spacing=8),
        ],
        expand=True,
        vertical_alignment=ft.CrossAxisAlignment.STRETCH,
    )


def resolve_run_or_empty(
    app: GuiApp, coordinator: EvaluationView, body: ft.Column
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Resolve the coordinator's selected run (re-scoped to the checked origins),
    or populate `body` with the right empty-state text and return None.

    The shared front half of both tabs' `_render_body`: the run-selected guard, the
    origin `filtered_run`, and the not-found fallback. On the empty/not-found paths
    it sets `body.controls` and calls `page.update()` itself; on success it returns
    `(run, cases)` and leaves body-building to the caller."""
    run_id = coordinator.selected_run_id
    if run_id is None:
        body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
        app.page.update()
        return None
    from knowledge_agent.gui.evaluation._common import filtered_run

    # The shared origin filter re-scopes the run to the checked provenances
    # (all checked = the stored run, unchanged).
    run, cases = filtered_run(app, run_id, coordinator.selected_origins)
    if run is None:
        body.controls = [ft.Text("Run not found — press Refresh.", italic=True)]
        app.page.update()
        return None
    return run, cases
