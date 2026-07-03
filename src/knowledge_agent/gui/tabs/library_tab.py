"""Library top tab — dispatches to the Slice 3 LibraryView.

The real layout (two-column shell with Select Dataset / Create New
Dataset / Installs sub-tabs + Documents browser) lives in
`gui.library.library_view`. This module stays thin so the existing
`GuiApp` wiring (which constructs LibraryTab) doesn't change shape
when new sub-tabs are added.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.library import LibraryView

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class LibraryTab:
    """Thin wrapper — delegates `build()` to `LibraryView`."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.view = LibraryView(app)

    def build(self) -> ft.Control:
        return self.view.build()
