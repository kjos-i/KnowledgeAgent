"""Library top-tab coordinator — 4 sub-tabs, fixed layouts.

Sub-tabs:

  Select     — dataset picker + info card + docs table (internal 2-col)
  Ingest     — per-corpus config editor + Ingest / bulk_ops actions
               (internal 2-col)
  Create New — new-corpus form + helper info (internal 2-col)
  Installs   — global install surface (full width)

No draggable splitter — each tab owns its internal layout with fixed
proportions. Splitters live in Search only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.library.create_new_dataset import CreateNewDatasetTab
from knowledge_agent.gui.library.ingest import IngestTab
from knowledge_agent.gui.library.installs import InstallsTab
from knowledge_agent.gui.library.select_dataset import SelectDatasetTab

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = ("Select", "Ingest", "Create New", "Installs")


class LibraryView:
    """Top-tab coordinator for Library — 4 sub-tabs, no shared splitter."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.select_tab = SelectDatasetTab(app)
        self.ingest_tab = IngestTab(app)
        self.create_tab = CreateNewDatasetTab(app)
        self.installs_tab = InstallsTab(app)
        # Cross-tab: after a successful ingest / bulk-op in the Ingest
        # sub-tab, refresh the Select sub-tab's card counts + Documents.
        self.ingest_tab.on_ingest_complete = self.select_tab.refresh_after_ingest

    def build(self) -> ft.Control:
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(
                    content=self.select_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.ingest_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.create_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.installs_tab.build(),
                    padding=8,
                    expand=True,
                ),
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
