"""Library right column — Documents browser.

Read-only documents browser for the active corpus. Mirrors
ResearchArticlesAgent's library/metadata pattern:

  - Single fetch on corpus open + Refresh button (no server-side
    pagination).
  - Coverage line (`N docs`) reflects the FULL loaded set, not the
    filtered count.
  - Sort once on load by `ingested_at desc`.
  - Client-side substring filter over `title` + `source_path`.
  - 1000-row sanity cap with banner above that.

Columns: Title / Source / Type / Date / Chunks / Layers (badges).

Per-row actions (below table): Edit metadata / Re-ingest / Remove
from corpus.

Edit metadata modal scope (5 fields, mirrors RA): title /
source_path / doc_type / ingested_at / status. doc_id stays
read-only.

Phase 4 of the Slice 3 plan fills this in.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class DocumentsView:
    """Documents browser for the active corpus."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self._create_controls()

    def _create_controls(self) -> None:
        """Placeholder — table lands in phase 4."""

    def build(self) -> ft.Control:
        return view_with_header(
            "Documents",
            empty_state(
                "Documents table lands in phase 4.\n\n"
                "Columns: Title / Source / Type / Date / Chunks /\n"
                "Layers (badges C E T X D).\n\n"
                "Per-row actions:\n"
                "  Edit metadata / Re-ingest / Remove."
            ),
        )
