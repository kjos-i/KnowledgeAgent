"""Search tab — the daily-use workflow.

Composes `ChatPanel` (left) + `RightPanel` (right) joined by the
shared `ResizableSplit` widget so the user can rebalance the two
panes for the current task.

Library and Evaluation are separate top-level tabs because they need
the full window for dense data tables; Search keeps the side-by-side
split because reading the result alongside composing the next query
is the core workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui._widgets import ResizableSplit

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


# Chat column starts ~40% of a 1200 px window. User drags to rebalance.
_INITIAL_CHAT_WIDTH = 480
_MIN_CHAT_WIDTH = 280
_MAX_CHAT_WIDTH = 900


class SearchTab:
    """Search tab: chat (left) + mode-switching display (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        split = ResizableSplit(
            page=self.app.page,
            first=self.app.chat_panel.build(),
            second=self.app.right_panel.build(),
            orientation="horizontal",
            initial_first_size=_INITIAL_CHAT_WIDTH,
            min_first_size=_MIN_CHAT_WIDTH,
            max_first_size=_MAX_CHAT_WIDTH,
        )
        return split.build()
