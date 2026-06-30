"""Search tab — the daily-use workflow.

Composes `ChatPanel` (left, 40%) + `RightPanel` (right, 60%) into a
single `ft.Row`. Owns no state itself — both child panels hold their
own controls, and `GuiApp` owns the session state they read from.

Library and Evaluation are separate top-level tabs because they need
the full window for dense data tables; Search keeps the side-by-side
split because reading the result alongside composing the next query
is the core workflow.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


# Column-width ratio: chat (left) takes 40%, results (right) takes 60%.
# The right panel is wider because it hosts answers / settings / info,
# but the chat side stays roomy enough for a multi-line query.
CHAT_COLUMN_EXPAND = 40
RIGHT_COLUMN_EXPAND = 60


class SearchTab:
    """Search tab: chat (left) + mode-switching display (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        return ft.Row(
            controls=[
                ft.Container(
                    content=self.app.chat_panel.build(),
                    expand=CHAT_COLUMN_EXPAND,
                ),
                ft.Container(
                    content=self.app.right_panel.build(),
                    expand=RIGHT_COLUMN_EXPAND,
                ),
            ],
            spacing=12,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
