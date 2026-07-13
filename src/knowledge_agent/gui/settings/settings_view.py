"""Settings view — coordinator for the Settings top-level tab's sub-tabs.

Settings is now a TOP-LEVEL tab (`app.py` mounts it), not a Search
right-panel mode. It holds the global, set-once-ish config:

  Keys / App

Retrieval and LLM were promoted to Search sub-tabs (they're per-query
knobs you tune beside the chat), so they no longer live here. Embedding
also left: the embedder is per-corpus now, so provider/model live in the
Ingest tab's per-corpus config editor and install/uninstall in the
Installs tab (workstream B, 2026-07-13).

Rendered as native Flet `ft.Tabs` with `secondary=True` so the sub-strip
is visually distinct from the primary top tab bar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.settings.app_tab import AppTab
from knowledge_agent.gui.settings.keys_tab import KeysTab

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = ("Keys", "App")


class SettingsView:
    """Sub-tab coordinator for the Settings top-level tab."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.keys_tab = KeysTab(app)
        self.app_tab = AppTab(app)

    def build(self) -> ft.Control:
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=self.keys_tab.build(), padding=8, expand=True),
                ft.Container(content=self.app_tab.build(), padding=8, expand=True),
            ],
            expand=True,
        )
        return ft.Tabs(
            length=len(SUB_TAB_LABELS),
            selected_index=0,
            content=ft.Column(
                controls=[sub_bar, sub_bodies],
                expand=True,
                spacing=0,
            ),
            expand=True,
        )
