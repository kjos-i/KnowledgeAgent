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


def resolve_run_or_skeleton(
    app: GuiApp, coordinator: EvaluationView
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Resolve the coordinator's selected run (re-scoped to the checked origins) for
    a dashboard body, ALWAYS returning something to render.

    Returns `(run, cases, notice)`. For a real, found run: `(run, cases, None)` — the
    caller builds its full body. When no run is selected or the run isn't found,
    returns an empty skeleton `({}, [], guidance)`: the caller still renders its full
    chart/section structure (each empty spot labelled "Not evaluated") beneath the
    guidance line, per the always-show rule, instead of collapsing to one message.
    The origin filter re-scopes a found run to the checked provenances (all checked =
    the stored run, unchanged)."""
    run_id = coordinator.selected_run_id
    if run_id is None:
        return (
            {},
            [],
            (
                "No run selected yet. Run an evaluation from the Run tab, or press "
                "Refresh, to fill this dashboard."
            ),
        )
    from knowledge_agent.gui.evaluation._common import filtered_run

    run, cases = filtered_run(app, run_id, coordinator.selected_origins)
    if run is None:
        return {}, [], "Run not found. Press Refresh to reload the ledger."
    return run, cases, None


def empty_slot(note: str = "Not evaluated", *, height: int = 160) -> ft.Control:
    """A bordered placeholder standing in for a chart/section body with no data, so
    the section keeps its frame + label instead of collapsing to a bare line (the
    always-show rule). Used for the dashboard's non-chart empties (e.g. a per-case
    table with no rows); canvas charts draw their own empty frame instead."""
    return ft.Container(
        height=height,
        border=ft.Border.all(1, ft.Colors.GREY_800),
        border_radius=6,
        alignment=ft.Alignment(0, 0),
        content=ft.Text(note, italic=True, size=13, color=ft.Colors.GREY_500),
    )
