"""Library right column — Documents browser.

Per-document browser for the active corpus, modelled on
ResearchArticlesAgent's library-metadata pattern (a card list, not a
rigid grid — rich cells + per-row actions, which Flet's DataTable
handles poorly).

Each document renders as a card:
  - status chip (`metadata_status`, colour-coded)
  - title (or filename fallback) + type + ingested date + chunk count
    (+ figure count for multimodal docs)
  - source path
  - per-row actions: **Edit** (rich metadata modal), **Re-ingest**
    (re-run the full pipeline on the source file with current settings),
    **Delete** (remove from corpus). Single-doc resolve lives in the
    Edit modal's "Look up DOI online" checkbox; corpus-wide "Resolve all"
    (with "Skip manually edited") lives in the Ingest tab's bulk-ops.

Edit modal (RAA parity) exposes every editable doc-level column:
Title / Authors / Year / Venue / DOI / Source URL / Language, plus a
"Look up DOI online" checkbox. Save stamps `metadata_status = manual`;
with the box checked + a DOI present it then runs `lookup_known_doi`
so OpenAlex enriches the rest (`status → enriched`). Chunk text +
embeddings are never touched.

Behaviour (mirrors RA):
  - Lazy fetch: loads once per active corpus, re-fetches on Refresh or
    when the active corpus changes.
  - Client-side substring filter over title + source_path; coverage
    line reflects the FULL loaded set.
  - Sort once on load by `ingested_at desc` (newest first).

Data source: LanceDB via `list_indexed_docs()` — every chunk row
carries the denormalised doc metadata, so one projection scan yields
the whole list + chunk counts. Edits patch the doc-level LanceDB
columns via `update_doc_metadata` (chunk text + embeddings untouched);
Delete goes through `bulk_ops.delete_doc_*` (LanceDB + Neo4j); Re-ingest
re-runs `pipeline.ingest_document` on the source file. Those backend
imports are local to the handlers (heavy-import-off-the-startup rule).

Deliberately deferred: Neo4j per-doc layer badges.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import FRAME_BORDER_COLOR, PANEL_BG, labeled_field
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.views._frame import empty_state

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp

logger = logging.getLogger(__name__)

# metadata_status -> chip colour. The four canonical values (see
# search.schema METADATA_STATUS_VALUES); unknown statuses fall back to grey.
_STATUS_COLORS: dict[str, str] = {
    "enriched": ft.Colors.GREEN_300,
    "pending": ft.Colors.ORANGE_300,
    "baseline": ft.Colors.CYAN_300,
    "manual": ft.Colors.BLUE_300,
}

# Idle placeholder for the per-doc op status line — always visible, so an
# empty status never leaves a blank gap above the card list.
_OP_STATUS_IDLE = "No re-ingest or delete running."

# Ordered (column, label, hint) spec for the Edit modal. Mirrors RAA's
# "Edit metadata" dialog. Every key is an editable doc-level LanceDB
# column (see search.schema). Excluded on purpose:
#   - `source_path`: structural (Sync uses it to detect moved/renamed
#     files); shown on the card but not hand-editable.
#   - `metadata_status`: derived, never typed — Save stamps 'manual',
#     a successful DOI lookup promotes it to 'enriched'.
#   - `sub_label` (type): would need a Neo4j label sync.
_EDIT_FIELD_SPEC: tuple[tuple[str, str, str], ...] = (
    ("title", "Title", ""),
    ("authors_display", "Authors", "e.g. Jane Doe; John Smith"),
    ("year", "Year", "e.g. 2020"),
    ("venue", "Venue", ""),
    ("doi", "DOI", "e.g. 10.1126/science.aaz5357"),
    ("source_url", "Source URL", "e.g. https://doi.org/…"),
    ("language", "Language", "e.g. en"),
)

# Fixed caption column for the Edit-metadata modal so every input's left edge
# lines up (the app-standard `labeled_field` layout).
_EDIT_LABEL_WIDTH = 110

# String columns written on every Save (full "set" semantics — a
# blanked field clears to ""). `year` is handled separately (int32).
_EDIT_STRING_KEYS: tuple[str, ...] = (
    "title",
    "authors_display",
    "venue",
    "doi",
    "source_url",
    "language",
)


class DocumentsView:
    """Documents browser for the active corpus."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Full loaded set (unfiltered) + which corpus it was loaded for.
        self._loaded_rows: list[dict[str, Any]] = []
        self._loaded_for: str | None = None
        # Strong refs to in-flight background tasks so they aren't GC'd
        # mid-run (they self-discard on completion).
        self._bg_tasks: set[asyncio.Task] = set()
        # Guards against a second per-doc op (re-ingest) starting mid-run.
        self._op_busy = False
        self._create_controls()

    def _spawn(self, coro) -> None:
        """Fire-and-forget a coroutine while keeping a strong reference
        until it completes (avoids the task being garbage-collected)."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _loop_running(self) -> bool:
        """True when an event loop is running.

        The sync click handlers create their coroutine only after this
        returns True, so unit tests (no running loop) never leave an
        un-awaited coroutine to close."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _create_controls(self) -> None:
        self.search_field = ft.TextField(
            hint_text="filename, title",
            on_change=self._on_search_changed,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
        )
        self.refresh_button = ft.Button(
            content="Refresh",
            on_click=self._on_refresh_clicked,
        )
        self.coverage_text = ft.Text("", size=14, color=ft.Colors.GREY_400)
        # Per-doc op (re-ingest / delete) status. Always visible; falls back
        # to the idle placeholder so the line never blanks into an empty gap.
        self.op_status = ft.Text(_OP_STATUS_IDLE, size=12, color=ft.Colors.GREY_400)
        # Plain Column — the parent (select_dataset right pane) already
        # scrolls, so nesting a scroll here would fight it.
        self.doc_list = ft.Column(spacing=6)

    # ----- build ---------------------------------------------------------

    def build(self) -> ft.Control:
        # Lazy: fetch when the active corpus changed since the last load.
        self._schedule_reload(force=False)
        # No own "Documents" header here — the Select tab draws the sticky
        # column title ("Documents (N)") above the pane and shows
        # `coverage_text` there, so this view is just the filter + list.
        return ft.Column(
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            expand=True,
            controls=[
                labeled_field("Filter", self.search_field, trailing=self.refresh_button),
                ft.Row(
                    controls=[
                        ft.Text("Table contents:", size=12, color=ft.Colors.GREY_400),
                        ft.Text("Cards and actions", size=12),
                        info(self.app, "select.documents_cards"),
                        ft.Container(width=8),
                        ft.Text("Edit dialog", size=12),
                        info(self.app, "select.documents_edit"),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.op_status,
                ft.Divider(),
                self.doc_list,
            ],
        )

    # ----- fetch ---------------------------------------------------------

    def _schedule_reload(self, *, force: bool) -> None:
        """Kick off an async reload if a loop is running. Guarded so
        `build()` is safe to call in tests (no running loop)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spawn(self.reload(force=force))

    async def reload(self, *, force: bool = False) -> None:
        """Fetch the active corpus's documents (lazy unless `force`)."""
        active = self.app.gui_config.active_corpus_name
        if active is None:
            self._loaded_rows = []
            self._loaded_for = None
            self.doc_list.controls = [empty_state("No corpus selected.")]
            self.coverage_text.value = ""
            self.app.page.update()
            return
        if not force and self._loaded_for == active:
            return
        self.doc_list.controls = [ft.Text("Loading …", italic=True, color=ft.Colors.GREY_500)]
        self.coverage_text.value = ""
        self.app.page.update()
        # Local import: keeps lancedb out of GUI startup (warmed in the
        # background by the graph pre-warm). See app.py `_prewarm_graph`.
        from knowledge_agent.search.client import get_search_client

        try:
            rows = await get_search_client().list_indexed_docs()
        except Exception as exc:
            logger.warning("DocumentsView: list_indexed_docs failed: %r", exc)
            self._loaded_rows = []
            self._loaded_for = active
            self.doc_list.controls = [self._fetch_error_control(exc)]
            self.coverage_text.value = ""
            self.app.page.update()
            return
        # Newest first. `ingested_at` may be datetime / str / None.
        rows.sort(key=self._sort_key, reverse=True)
        self._loaded_rows = rows
        self._loaded_for = active
        self._render_rows()
        self.app.page.update()

    @staticmethod
    def _sort_key(row: dict[str, Any]) -> str:
        v = row.get("ingested_at")
        return str(v) if v is not None else ""

    @staticmethod
    def _fetch_error_control(exc: Exception) -> ft.Control:
        """Friendly message for a failed fetch. The common case is a
        missing per-corpus Neo4j password (Settings validation) — the
        corpus index read only touches LanceDB, but building Settings
        requires the password, so surface an actionable hint rather than
        a raw pydantic dump."""
        msg = str(exc)
        if "neo4j_password" in msg.lower():
            return ft.Text(
                "This corpus has no Neo4j password configured, so its "
                "index can't be read yet. Set the password (Create New, "
                "or re-select the corpus in the picker on the left), "
                "then press Refresh.",
                color=ft.Colors.ORANGE_300,
            )
        return ft.Text(
            f"Could not read the corpus index: {exc}",
            color=ft.Colors.RED_300,
        )

    # ----- render + filter ----------------------------------------------

    def _on_search_changed(self, e: ft.Event) -> None:
        self._render_rows()
        self.app.page.update()

    def _on_refresh_clicked(self, e: ft.Event) -> None:
        self._schedule_reload(force=True)

    @staticmethod
    def _matches_filter(row: dict[str, Any], needle: str) -> bool:
        if not needle:
            return True
        hay = ((row.get("title") or "") + " " + (row.get("source_path") or "")).lower()
        return needle in hay

    def _render_rows(self) -> None:
        """Render `doc_list` from `_loaded_rows` + the search filter.

        Coverage reflects the FULL loaded set (the filter only changes
        the rendered subset). Empty-state messages differ between "no
        docs at all" and "filter excludes everything"."""
        if not self._loaded_rows:
            self.doc_list.controls = [
                empty_state(
                    "No documents indexed in this corpus yet.\n"
                    "Ingest a folder or file from the Ingest tab."
                )
            ]
            self.coverage_text.value = "(0 documents)"
            return
        needle = (self.search_field.value or "").strip().lower()
        filtered = [r for r in self._loaded_rows if self._matches_filter(r, needle)]
        total = len(self._loaded_rows)
        if needle:
            self.coverage_text.value = f"({len(filtered)} of {total} documents)"
        else:
            self.coverage_text.value = f"({total} document{'s' if total != 1 else ''})"
        if not filtered:
            self.doc_list.controls = [
                ft.Text(
                    f"No documents match '{needle}'.",
                    italic=True,
                    color=ft.Colors.GREY_500,
                )
            ]
            return
        self.doc_list.controls = [self._render_doc_card(r) for r in filtered]

    def _render_doc_card(self, row: dict[str, Any]) -> ft.Control:
        status = (row.get("metadata_status") or "").strip() or "unknown"
        color = _STATUS_COLORS.get(status, ft.Colors.GREY_500)
        title = row.get("title") or _basename(row.get("source_path")) or "(no title)"
        # Show "<main> <sub>" (e.g. "Document Paper") when a sub-label is set,
        # else just the main label (e.g. "Document"). Surfaces the type — and
        # specifically whether a doc is a Paper — on the card.
        main_lbl = row.get("main_label") or "—"
        sub_lbl = row.get("sub_label")
        doc_type = f"{main_lbl} {sub_lbl}" if sub_lbl else main_lbl
        date = _fmt_date(row.get("ingested_at"))
        n_chunks = row.get("n_chunks") or 0
        n_figures = row.get("n_figures") or 0
        source_path = row.get("source_path") or ""

        meta_parts = [doc_type]
        if date:
            meta_parts.append(date)
        chunk_str = f"{n_chunks} chunk{'s' if n_chunks != 1 else ''}"
        if n_figures:
            chunk_str += f" · {n_figures} figure{'s' if n_figures != 1 else ''}"
        meta_parts.append(chunk_str)
        meta_line = "  ·  ".join(meta_parts)

        status_chip = ft.Container(
            content=ft.Text(status, size=12, color=ft.Colors.BLACK),
            bgcolor=color,
            padding=ft.Padding.symmetric(horizontal=6, vertical=2),
            border_radius=8,
        )
        snap = dict(row)
        edit_button = ft.Button(
            content="Edit",
            height=28,
            on_click=lambda e, s=snap: self._open_edit_dialog(s),
        )
        reingest_button = ft.Button(
            content="Re-ingest",
            height=28,
            on_click=lambda e, s=snap: self._on_reingest_clicked(s),
        )
        delete_button = ft.Button(
            content="Delete",
            height=28,
            on_click=lambda e, s=snap: self._on_delete_clicked(s),
        )
        return ft.Container(
            border=ft.Border.all(1, ft.Colors.GREY_800),
            border_radius=6,
            padding=ft.Padding.symmetric(vertical=6, horizontal=8),
            content=ft.Column(
                spacing=2,
                controls=[
                    ft.Row(
                        controls=[
                            status_chip,
                            ft.Text(
                                title,
                                size=13,
                                weight=ft.FontWeight.BOLD,
                                expand=True,
                            ),
                            edit_button,
                            reingest_button,
                            delete_button,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Text(meta_line, size=12, color=ft.Colors.GREY_300),
                    ft.Text(
                        source_path,
                        size=12,
                        color=ft.Colors.GREY_500,
                        selectable=True,
                    ),
                ],
            ),
        )

    # ----- edit modal ----------------------------------------------------

    def _open_edit_dialog(self, row: dict[str, Any]) -> None:
        doc_id = row.get("doc_id") or ""
        fields = {
            key: ft.TextField(
                hint_text=hint or None,
                value=self._field_value(row, key),
            )
            for key, _label, hint in _EDIT_FIELD_SPEC
        }
        # App-standard "caption: [input]" rows — field name to the left, inputs
        # aligned down a fixed column.
        field_rows = [
            labeled_field(label, fields[key], label_width=_EDIT_LABEL_WIDTH)
            for key, label, _hint in _EDIT_FIELD_SPEC
        ]
        lookup_checkbox = ft.Checkbox(label="Look up DOI online", value=False)
        # Stash so tests + the save handler can read current values.
        self._active_edit = {
            "doc_id": doc_id,
            "fields": fields,
            "lookup_checkbox": lookup_checkbox,
        }
        heading = _basename(row.get("source_path")) or (row.get("title") or "") or f"{doc_id[:12]}…"

        def _close(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _save(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            patch = self._collect_edit_patch()
            doi = (fields["doi"].value or "").strip()
            lookup = doi if (lookup_checkbox.value and doi) else None
            self._schedule_save(doc_id, patch, lookup_doi=lookup)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Edit metadata — {heading}", size=14, weight=ft.FontWeight.BOLD),
            content=ft.Column(
                spacing=8,
                tight=True,
                width=480,
                scroll=ft.ScrollMode.AUTO,
                controls=[
                    *field_rows,
                    # Read-only — the OpenAlex work ID (set when a DOI resolves);
                    # shown for reference, not editable.
                    labeled_field(
                        "OpenAlex ID",
                        ft.Text(
                            (row.get("openalex_id") or "").strip() or "—",
                            size=14,
                            selectable=True,
                            color=ft.Colors.GREY_300,
                        ),
                        label_width=_EDIT_LABEL_WIDTH,
                    ),
                    lookup_checkbox,
                    ft.Text(
                        "Save stamps status = 'manual'. With 'Look up DOI "
                        "online' checked and a DOI present, OpenAlex "
                        "enriches the other fields afterward (status → "
                        "'enriched'). Chunk text + embeddings are never "
                        "touched.",
                        size=12,
                        color=ft.Colors.GREY_500,
                    ),
                ],
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_close),
                ft.Button(content="Save", on_click=_save),
            ],
        )
        self.app.page.show_dialog(dialog)

    @staticmethod
    def _field_value(row: dict[str, Any], key: str) -> str:
        """Prefill string for a modal field (int `year` → its digits)."""
        v = row.get(key)
        return "" if v is None else str(v)

    def _collect_edit_patch(self) -> dict[str, Any]:
        """Read the open edit dialog's fields into a patch dict.

        String columns use full "set" semantics — every one is written
        (a blanked field clears to ""), so an edit can correct OR clear.
        `year` is int32: written only when the entry parses as an int,
        otherwise left unchanged (blank / garbage → omitted). Every Save
        stamps `metadata_status = 'manual'` (a later DOI lookup may
        promote it to 'enriched').
        """
        active = getattr(self, "_active_edit", None)
        if not active:
            return {}
        fields = active["fields"]
        patch: dict[str, Any] = {}
        for key in _EDIT_STRING_KEYS:
            patch[key] = (fields[key].value or "").strip()
        year_raw = (fields["year"].value or "").strip()
        if year_raw:
            # leave `year` unchanged on unparseable input
            with contextlib.suppress(ValueError):
                patch["year"] = int(year_raw)
        patch["metadata_status"] = "manual"
        return patch

    def _schedule_save(
        self,
        doc_id: str,
        patch: dict[str, Any],
        *,
        lookup_doi: str | None = None,
    ) -> None:
        if not self._loop_running():
            return
        self._spawn(self._save_metadata(doc_id, patch, lookup_doi=lookup_doi))

    async def _save_metadata(
        self,
        doc_id: str,
        patch: dict[str, Any],
        *,
        lookup_doi: str | None = None,
    ) -> None:
        if not doc_id or not patch:
            return
        from knowledge_agent.search.client import get_search_client

        try:
            await get_search_client().update_doc_metadata(doc_id, patch)
        except Exception as exc:
            logger.warning(
                "DocumentsView: update_doc_metadata(%s) failed: %r",
                doc_id,
                exc,
            )
            return
        # RAA parity: with the box checked + a DOI, enrich from OpenAlex
        # (promotes status manual → enriched). Failure is non-fatal — the
        # manual edit already landed.
        if lookup_doi:
            from knowledge_agent.ingestion.metadata_resolution import (
                lookup_known_doi,
            )

            try:
                await lookup_known_doi(doc_id, lookup_doi)
            except Exception as exc:
                logger.warning(
                    "DocumentsView: lookup_known_doi(%s) failed: %r",
                    doc_id,
                    exc,
                )
        await self.reload(force=True)

    # ----- re-ingest -----------------------------------------------------

    def _set_op_status(self, msg: str) -> None:
        self.op_status.value = msg or _OP_STATUS_IDLE
        self.app.page.update()

    def _on_reingest_clicked(self, row: dict[str, Any]) -> None:
        if self._op_busy:
            self._set_op_status("A re-ingest is already running…")
            return
        self._open_reingest_confirm(row)

    def _open_reingest_confirm(self, row: dict[str, Any]) -> None:
        """Validate the source file exists, then confirm before re-running
        the full pipeline on this one document."""
        source_path = (row.get("source_path") or "").strip()
        name = (
            row.get("title")
            or _basename(source_path)
            or (row.get("doc_id") or "")[:12]
            or "document"
        )
        if not source_path:
            self._set_op_status(f"No source path recorded for '{name}' — can't re-ingest.")
            return
        if not Path(source_path).is_file():
            self._set_op_status(f"Source file not found: {source_path}")
            return

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _confirm(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            if self._loop_running():
                self._spawn(self._do_reingest(dict(row)))

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Re-ingest document"),
            content=ft.Text(
                f"Re-run the full pipeline on '{name}' with the corpus's "
                f"current settings (layers, extractors, ontologies)? This "
                f"replaces the existing version.\n\nSource: {source_path}",
                size=12,
                selectable=True,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content="Re-ingest", on_click=_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    async def _do_reingest(self, row: dict[str, Any]) -> None:
        from knowledge_agent.corpus_config import load_corpus_config
        from knowledge_agent.ingestion import pipeline

        source_path = (row.get("source_path") or "").strip()
        name = row.get("title") or _basename(source_path) or "document"
        cfg_path = self.app.gui_config.corpus_config_path
        if cfg_path is None:
            self._set_op_status("No active corpus.")
            return
        try:
            config = load_corpus_config(cfg_path)
        except Exception as exc:
            self._set_op_status(f"Could not load corpus.toml: {exc}")
            return
        self._op_busy = True
        self._set_op_status(f"Re-ingesting '{name}'…")
        try:
            # preserve_existing_labels: keep the doc's stored type on a
            # re-ingest of the same file (its labels don't change).
            await pipeline.ingest_document(
                Path(source_path),
                config,
                row.get("main_label") or "Document",
                row.get("sub_label"),
                preserve_existing_labels=True,
            )
        except Exception as exc:
            logger.warning(
                "DocumentsView: re-ingest(%s) failed: %r",
                source_path,
                exc,
            )
            self._op_busy = False
            self._set_op_status(f"Re-ingest of '{name}' failed: {exc}")
            return
        self._op_busy = False
        await self.reload(force=True)
        self._set_op_status(f"Re-ingested '{name}'.")

    # ----- delete --------------------------------------------------------

    def _on_delete_clicked(self, row: dict[str, Any]) -> None:
        if not self._loop_running():
            return
        self._spawn(self._open_delete_dialog(row.get("doc_id") or ""))

    async def _open_delete_dialog(self, doc_id: str) -> None:
        if not doc_id:
            return
        from knowledge_agent.ingestion.bulk_ops import delete_doc_plan

        try:
            plan = await delete_doc_plan(doc_id)
        except Exception as exc:
            logger.warning(
                "DocumentsView: delete_doc_plan(%s) failed: %r",
                doc_id,
                exc,
            )
            return

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _confirm(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            self._spawn(self._do_delete(plan))

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Delete document"),
            content=ft.Text(plan.summary, size=12, selectable=True),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content="Delete", on_click=_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    async def _do_delete(self, plan: Any) -> None:
        from knowledge_agent.ingestion.bulk_ops import delete_doc_execute

        try:
            result = await delete_doc_execute(plan)
        except Exception as exc:
            logger.warning(
                "DocumentsView: delete_doc_execute(%s) failed: %r",
                getattr(plan, "doc_id", "?"),
                exc,
            )
            # Surface the failure + reload so the card doesn't linger looking
            # deleted when it wasn't (previously this only logged and returned).
            self._set_op_status(f"Delete failed: {exc}")
            await self.reload(force=True)
            return
        if not result.ok:
            logger.warning(
                "DocumentsView: delete_doc_execute(%s) reported not-ok",
                getattr(plan, "doc_id", "?"),
            )
            self._set_op_status("Delete reported a problem — see log for details.")
        await self.reload(force=True)


def _basename(path: str | None) -> str:
    """Filename tail of a source path, or '' when absent."""
    if not path:
        return ""
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _fmt_date(value: Any) -> str:
    """Best-effort date string (YYYY-MM-DD) from a timestamp / datetime /
    str. Returns '' when absent or unparseable."""
    if value is None:
        return ""
    text = str(value)
    # ISO-ish timestamps start 'YYYY-MM-DD...'; take the date head.
    return text[:10] if len(text) >= 10 else text
