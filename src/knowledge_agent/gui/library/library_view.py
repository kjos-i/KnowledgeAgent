"""Library top-tab coordinator — 3 sub-tabs, fixed layouts.

Sub-tabs (left → right, corpus lifecycle order):

  Create New — new-corpus form + helper info (internal 2-col)
  Ingest     — per-corpus config editor + Ingest / bulk_ops actions
               (internal 2-col)
  Metadata   — corpus info card + docs table (internal 2-col). (Corpus
               SELECTION lives in the global top-bar dropdown, so this tab
               is the selected corpus's metadata + documents, not a picker —
               hence "Metadata", renamed from "Select".)

Installs was promoted to its own top-level tab — it's a global,
machine-level install surface (providers / ontologies / weights), not
corpus-specific — so it no longer lives here. Its widget still lives in
`gui/library/installs.py`; `app.py` mounts it at the top level.

No draggable splitter — each tab owns its internal layout with fixed
proportions. Splitters live in Search only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.library.create_new_dataset import CreateNewDatasetTab
from knowledge_agent.gui.library.ingest import IngestTab
from knowledge_agent.gui.library.select_dataset import SelectDatasetTab
from knowledge_agent.gui.views.info_doc import InfoDoc, docs_dir

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


SUB_TAB_LABELS = ("Create New", "Ingest", "Metadata", "Info")


class LibraryView:
    """Top-tab coordinator for Library — 3 sub-tabs, no shared splitter."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.select_tab = SelectDatasetTab(app)
        self.ingest_tab = IngestTab(app)
        self.create_tab = CreateNewDatasetTab(app)
        # Ingest sub-tab body container — held so it can be rebuilt in
        # place when the active corpus changes (see `refresh_ingest`).
        self._ingest_body: ft.Container | None = None
        # Cross-tab: after a successful ingest / bulk-op in the Ingest
        # sub-tab, refresh the Select sub-tab's card counts + Documents.
        self.ingest_tab.on_ingest_complete = self.select_tab.refresh_after_ingest
        # Cross-tab: while editing config on the Ingest tab, keep the Select
        # card's "changed since last ingest" section live (the config editor
        # fires this after each draft-persist).
        self.ingest_tab.config_editor.on_draft_changed = self._on_config_draft_changed

    def _on_config_draft_changed(self) -> None:
        """A config-editor draft edit (or Discard) keeps BOTH live 'pending
        changes' views in sync: the Select card's section and the Ingest
        tab's own Ingestion summary (both read the same baseline-vs-draft
        diff, so they must never disagree). The Ingest summary previously
        went stale because only the Select card was wired here."""
        self.select_tab.refresh_pending_changes()
        self.ingest_tab.refresh_summary()
        self.ingest_tab.refresh_bulk_op_gating()

    def build(self) -> ft.Control:
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        self._ingest_body = ft.Container(
            content=self.ingest_tab.build(),
            padding=8,
            expand=True,
        )
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(
                    content=self.create_tab.build(),
                    padding=8,
                    expand=True,
                ),
                self._ingest_body,
                ft.Container(
                    content=self.select_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=InfoDoc(
                        self.app,
                        docs_dir() / "info_library.md",
                        missing_hint="Library help is not written yet (info_library.md).",
                    ).build(),
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

    def refresh_ingest(self) -> None:
        """Rebuild the Ingest sub-tab body for the current active corpus.

        The 4 sub-tab bodies are built once in `build()`; changing the
        active corpus (create / switch / remove / relocate) otherwise
        leaves Ingest frozen on the old corpus — a stale header and a
        config editor still pointed at the previous corpus.toml. Called
        from every corpus-mutation site so the header, config editor,
        and diff card track the new active corpus. No-op before the
        first `build()`.
        """
        if self._ingest_body is None:
            return
        # The editor caches by corpus NAME, so a relocate (same name,
        # new path) wouldn't reload on its own — force it.
        self.ingest_tab.config_editor.invalidate()
        self._ingest_body.content = self.ingest_tab.build()
        self.app.page.update()
