"""Library tab — dataset + KG management.

Slice 1 stub. Slice 3 fills in:
  - Dataset list (`corpora` in GuiConfig — add / edit / delete)
  - Active-dataset switcher (synced with the Search tab's queries)
  - Folder picker → ingest_folder_plan → confirm dialog → execute
    with per-file progress
  - Per-doc metadata table + bulk ops (delete_doc, re_embed,
    bulk_resolve_openalex)
  - KG backfill ops (chunks / entities / ontology / triples /
    cross_doc / cross_doc_xrefs)
  - Install dialogs for LLM / embedder / ontology / extractor
    providers (modal with progress + cancel; security warnings from
    sprint 0d render inline in the dialog summary)

Top-level tab (not a right-panel mode of Search) because every one
of those needs the full window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class LibraryTab:
    """Library tab — stub until slice 3."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        return view_with_header(
            "Library",
            empty_state(
                "Library lands in slice 3 — datasets, ingest from "
                "folder, per-doc metadata + bulk ops, KG backfills, "
                "and install dialogs for LLM / embedder / ontology / "
                "extractor providers."
            ),
        )
