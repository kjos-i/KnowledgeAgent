"""Settings view — coordinator for the 5 Settings sub-tabs.

The right panel's MODE_SETTINGS dispatches here. Internally we render
native Flet `ft.Tabs` with `secondary=True` so the sub-strip is
visually distinct from the primary top tab bar (Search / Library /
Evaluation).

Sub-tab order is fixed: Keys / Retrieval / LLM / Embedding / App.
Picked so the most-touched-first-launch tab (Keys) sits leftmost and
the App+Diagnostics tab is last (less-frequent reference). Order is
the binding decision; tab labels are provisional and may be tweaked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.settings.app_tab import AppTab
from knowledge_agent.gui.settings.embedding_tab import EmbeddingTab
from knowledge_agent.gui.settings.keys_tab import KeysTab
from knowledge_agent.gui.settings.llm_tab import LlmTab
from knowledge_agent.gui.settings.retrieval_tab import RetrievalTab

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = ("Keys", "Retrieval", "LLM", "Embedding", "App")


class SettingsView:
    """Sub-tab coordinator for the Settings right-panel mode."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.keys_tab = KeysTab(app)
        self.retrieval_tab = RetrievalTab(app)
        self.llm_tab = LlmTab(app)
        self.embedding_tab = EmbeddingTab(app)
        self.app_tab = AppTab(app)

    def build(self) -> ft.Control:
        # Native Flet Tabs with secondary indicator so the sub-strip
        # is visually distinct from the top primary tab bar. Sub-tab
        # bodies wrap in Containers with small padding so the form
        # inside breathes against the surrounding right-panel frame.
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=self.keys_tab.build(), padding=8, expand=True),
                ft.Container(
                    content=self.retrieval_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(content=self.llm_tab.build(), padding=8, expand=True),
                ft.Container(
                    content=self.embedding_tab.build(),
                    padding=8,
                    expand=True,
                ),
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
