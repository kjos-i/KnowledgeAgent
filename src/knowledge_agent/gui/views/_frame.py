"""Shared view layout helpers — a bold title + divider header and a
centered empty-state body. Used by Latest / File / (future Library /
Settings / Diagnostics / Info) views so their visual structure stays
consistent.
"""

import flet as ft


def view_header(title: str) -> ft.Control:
    """The view-level title block: bold text + divider underneath."""
    return ft.Column(
        controls=[
            ft.Text(title, weight=ft.FontWeight.BOLD, size=18),
            ft.Divider(),
        ],
        spacing=4,
    )


def empty_state(message: str) -> ft.Control:
    """Centered italic placeholder body — used for empty / 'coming soon'
    views (slice 1 ships Latest + File live; the other tabs hold an
    empty-state until later slices replace them)."""
    return ft.Container(
        content=ft.Text(
            message,
            italic=True,
            color=ft.Colors.GREY_500,
            text_align=ft.TextAlign.CENTER,
        ),
        alignment=ft.Alignment(0, 0),
        expand=True,
    )


def view_with_header(title: str, body: ft.Control) -> ft.Control:
    """Header + body composition used by every view in the display panel.

    Header is the bold-title block; body fills the rest. Returning a
    Column with `expand=True` lets the parent container size both
    pieces — the divider hugs the title, body stretches.
    """
    return ft.Column(
        controls=[view_header(title), body],
        expand=True,
        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        spacing=8,
    )
