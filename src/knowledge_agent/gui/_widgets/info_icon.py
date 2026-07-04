"""Reusable `(i)` info icon — the house style for contextual help.

Press one and a small dialog explains the setting or feature it sits
next to. Visibility is gated by the global `gui_config.show_info_icons`
toggle (Settings -> App -> App behaviour): show rich help everywhere
(teaching mode) or hide every icon for a clean, expert UI.

ALWAYS build info icons with `info_icon(app, title=..., text=...)` —
never hand-roll an icon + dialog. That keeps the look, the dialog shape,
and the global show/hide behaviour in ONE place, so restyling or
changing the popup is a single edit here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


def info_icon(app: GuiApp, *, title: str, text: str) -> ft.IconButton:
    """Return a small `(i)` button that opens a help dialog on click.

    The button registers itself with `app` so the global show/hide
    toggle can flip every icon at once, live. Starts hidden when
    `gui_config.show_info_icons` is False.
    """

    def _open(_e: ft.Event) -> None:
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                width=440,
                content=ft.Text(text, size=13, selectable=True),
            ),
            actions=[
                ft.TextButton("Close", on_click=lambda _c: app.page.pop_dialog()),
            ],
        )
        app.page.show_dialog(dialog)
        app.page.update()

    button = ft.IconButton(
        icon=ft.Icons.INFO_OUTLINE,
        icon_size=16,
        icon_color=ft.Colors.BLUE_300,
        tooltip="What's this?",
        on_click=_open,
        visible=app.gui_config.show_info_icons,
    )
    app.register_info_icon(button)
    return button
