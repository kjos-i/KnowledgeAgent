"""Library → Ingest sub-tab — pure action flow (left) + config (right).

Internal 2-column layout (50/50, no dragger):

  Left column:  Corpus header + shared folder picker + Ingest /
                Re-ingest / Sync buttons + file picker + Ingest single
                file button + status. **Every configurable choice
                (main label, sub-label, overwrite, layers, ontologies,
                thresholds) lives on the right.**
  Right column: `CorpusConfigEditor` — 8 collapsible sections:
                Labels and sub-labels + per-layer sections
                (openalex L1-L4, chunks L5, entities L6, ontology
                linking L7, triples L8, cross-doc L9, cross-doc xrefs
                L10). Each layer section shows Current / New chips.

Column assignment follows the "action left, context right" rule —
strictly: the left column has no configuration surface, only actions.

Four action buttons:

  * **Ingest folder**    — add new files from the picked folder.
    Skip already-ingested via source_path dedup.
  * **Re-ingest**        — force re-run on the picked folder.
  * **Sync**             — bidirectional: add new + remove deleted +
    re-ingest changed. One-shot.
  * **Ingest single file** — pick one file and ingest it.

Empty state (no active corpus): both columns collapse to a single
hint pointing at Create New.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
    labeled_field,
    panel_box,
    panel_title,
    section_divider,
    section_title,
    sub_section_title,
    thin_rule,
)
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui.library.config_diff import config_diff
from knowledge_agent.gui.library.corpus_config_editor import (
    _ONTOLOGY_DISPLAY,
    CorpusConfigEditor,
)
from knowledge_agent.gui.library.session_state import (
    load_session,
    update_last_file,
    update_last_folder,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


class IngestTab:
    """Pure action flow (left) + full config surface (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.config_editor = CorpusConfigEditor(app)
        self.status: ft.Text | None = None

        # Shared folder picker (used by Ingest / Re-ingest / Sync).
        self.folder_field: ft.TextField | None = None
        self.folder_browse_button: ft.Button | None = None
        self.ingest_folder_button: ft.Button | None = None
        self.reingest_button: ft.Button | None = None
        self.sync_button: ft.Button | None = None

        # File picker (Ingest single file).
        self.file_field: ft.TextField | None = None
        self.file_browse_button: ft.Button | None = None
        self.ingest_file_button: ft.Button | None = None

        # Progress + busy state, shared across the 4 ingest actions.
        self.progress_ring: ft.ProgressRing | None = None
        # Determinate per-file bar shown during the execute phase (the
        # ring covers the indeterminate scan phase before the dialog).
        self.progress_bar: ft.ProgressBar | None = None
        self._busy = False
        self._bg_tasks: set[asyncio.Task] = set()
        # Bulk-ops: Skip-manually-edited toggle for bulk_resolve_openalex
        # (relocated here from the Documents table).
        self.skip_manual_checkbox: ft.Checkbox | None = None
        # Set by LibraryView — called after a successful ingest / bulk-op
        # so the Select sub-tab refreshes its counts + Documents list.
        self.on_ingest_complete: Callable[[], None] | None = None
        # Which corpus the folder / file pickers were last restored for, so
        # `build()` re-pre-fills them from the session sidecar only on a
        # corpus change (not on every render, which would clobber typing).
        self._session_restored_for: str | None = None

        self._create_controls()

    def _create_controls(self) -> None:
        # Idle placeholder — italic/dim "Empty" until a run writes a real
        # status via `_write_status` (which clears the placeholder styling).
        self.status = ft.Text(
            "Empty",
            size=12,
            italic=True,
            color=ft.Colors.GREY_500,
        )
        self.progress_ring = ft.ProgressRing(
            width=16,
            height=16,
            stroke_width=2,
            visible=False,
        )
        self.progress_bar = ft.ProgressBar(value=0, visible=False)
        self.skip_manual_checkbox = ft.Checkbox(
            label="Skip manually edited",
            value=True,
            tooltip="When resolving all, leave docs with status 'manual' "
            "untouched (protects your hand-edits).",
        )

        # ---- Folder picker + 3 folder-action buttons ----
        self.folder_field = ft.TextField(
            hint_text="Pick a folder of documents",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
            on_blur=self._on_folder_blur,
        )
        self.folder_browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_folder_browse_clicked,
        )
        self.ingest_folder_button = ft.Button(
            content=centered_label("Ingest folder"),
            on_click=self._on_ingest_folder_clicked,
        )
        self.reingest_button = ft.Button(
            content=centered_label("Re-ingest"),
            on_click=self._on_reingest_clicked,
        )
        self.sync_button = ft.Button(
            content=centered_label("Sync"),
            on_click=self._on_sync_clicked,
        )

        # ---- File picker + Ingest single file ----
        self.file_field = ft.TextField(
            hint_text="Pick a single file to ingest",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
            on_blur=self._on_file_blur,
        )
        self.file_browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_file_browse_clicked,
        )
        self.ingest_file_button = ft.Button(
            content=centered_label("Ingest single file"),
            on_click=self._on_ingest_file_clicked,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None or not self.app.gui_config.corpora:
            return ft.Column(
                spacing=10,
                controls=[
                    view_header("Ingest"),
                    ft.Text(
                        "No active corpus — open Create New to register "
                        "a corpus, then come back here to configure and "
                        "ingest into it.",
                        size=12,
                        color=ft.Colors.AMBER_300,
                        italic=True,
                    ),
                ],
            )

        # Config editor owns the Labels-section sub-label dropdown, which
        # reads `allowed_types` off the loaded config. Force a load here
        # so first render has state ready.
        self.config_editor.ensure_loaded()

        # Pre-fill the folder / file pickers from this corpus's saved
        # session — only when the corpus changed, so a plain re-render
        # never clobbers a path the user is mid-way through typing.
        self._restore_session_paths(active_name)

        left_pane = panel_box(
            ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    # (Panel title "Corpus ingestion" lives in the tab's fixed
                    # header above this pane, so it stays put while scrolling.)
                    # ============ Section: File selection ============
                    section_title("File selection"),
                    # ---- Sub-section: a folder ----
                    sub_section_title("Select a folder"),
                    labeled_field(
                        "Folder",
                        self.folder_field,
                        trailing=self.folder_browse_button,
                    ),
                    ft.Row(
                        controls=[
                            self.ingest_folder_button,
                            self.reingest_button,
                            self.sync_button,
                        ],
                        spacing=8,
                        wrap=True,
                    ),
                    # Thin rule between the two sub-sections.
                    thin_rule(),
                    # ---- Sub-section: a single file ----
                    sub_section_title("…or a single file"),
                    labeled_field(
                        "File",
                        self.file_field,
                        trailing=self.file_browse_button,
                    ),
                    ft.Row(
                        controls=[self.ingest_file_button],
                        spacing=8,
                    ),
                    section_divider(),
                    # ============ Section: Progress ============
                    section_title("Progress"),
                    ft.Row(
                        controls=[self.progress_ring, self.status],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.progress_bar,
                    section_divider(),
                    # ============ Section: Ingestion summary (flat, no box) ============
                    section_title("Ingestion summary"),
                    self._build_diff_card(),
                    # Discard reverts the pending config changes shown above.
                    ft.Row(controls=[self.config_editor.discard_button]),
                    section_divider(),
                    # ============ Section: Bulk operations ============
                    ft.Row(
                        controls=[
                            section_title("Bulk operations"),
                            info_icon(
                                self.app,
                                title="Bulk operations",
                                text=(
                                    "Retroactive per-layer refreshes for the "
                                    "already-ingested corpus — re-run one layer "
                                    "without re-ingesting the files."
                                ),
                            ),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self._build_bulk_ops_panel(),
                ],
            )
        )

        right_pane = panel_box(
            ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=8,
                controls=[self.config_editor.build()],
            )
        )

        # Column titles sit in a fixed header above the panes (aligned 50/50
        # over them), so each stays visible while its box scrolls. The right
        # title carries the config editor's unsaved-● + Discard button.
        header = ft.Row(
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=panel_title("Corpus ingestion"),
                ),
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=ft.Row(
                        controls=[
                            panel_title("Ingestion settings"),
                            self.config_editor.dirty_indicator,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                ),
            ],
        )
        body = ft.Row(
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=12,
            controls=[left_pane, right_pane],
        )
        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[header, body],
        )

    # ----- Diff card ------------------------------------------------------

    def _build_diff_card(self) -> ft.Control:
        """Compact `field: current → new` summary of pending config
        changes. Only rows where the saved baseline differs from the
        in-memory draft are shown. Empty state when nothing pending.

        The diff is computed by `config_diff` — shared with the Select
        tab's corpus card so the two views can never disagree.
        """
        baseline = self.config_editor._baseline_config
        inmem = self.config_editor._corpus_config
        if baseline is None or inmem is None:
            return ft.Container()

        diffs = config_diff(baseline, inmem)

        # Flat body only — the "Ingestion summary" section title + the
        # section dividers around it now provide the framing (no inner box).
        if not diffs:
            return ft.Text(
                "No pending changes — new ingest = previous.",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
        rows: list[ft.Control] = [
            ft.Text(
                f"{len(diffs)} pending change{'s' if len(diffs) != 1 else ''}:",
                size=12,
                color=ft.Colors.GREY_400,
            ),
        ]
        for name, cur, new in diffs:
            rows.append(
                ft.Row(
                    spacing=6,
                    controls=[
                        ft.Text(
                            name,
                            size=12,
                            color=ft.Colors.GREY_300,
                            width=180,
                        ),
                        ft.Text(
                            cur,
                            size=12,
                            color=ft.Colors.GREY_400,
                        ),
                        ft.Text(
                            "→",
                            size=12,
                            color=ft.Colors.GREY_500,
                        ),
                        ft.Text(
                            new,
                            size=12,
                            color=ft.Colors.AMBER_300,
                        ),
                    ],
                ),
            )
        return ft.Column(spacing=4, controls=rows)

    # ----- Bulk operations panel ------------------------------------------

    def _build_bulk_ops_panel(self) -> ft.Control:
        """Flat per-layer bulk-ops list — one small header per layer,
        buttons directly below it. Each button runs its
        `ingestion.bulk_ops` plan → confirm → execute flow via
        `_on_bulk_op_clicked`."""

        def op_button(op_name: str) -> ft.Control:
            return ft.Button(
                content=centered_label(op_name),
                on_click=lambda e, n=op_name: self._on_bulk_op_clicked(n),
            )

        def layer_group(title: str, ops: list[str]) -> list[ft.Control]:
            return [
                sub_section_title(title),
                ft.Row(
                    wrap=True,
                    spacing=8,
                    controls=[op_button(op) for op in ops],
                ),
            ]

        # openalex: the resolve button and its "Skip manually edited" toggle
        # share ONE line — the toggle governs what the button does, so they
        # read as a pair.
        openalex_row: list[ft.Control] = [op_button("bulk_resolve_openalex")]
        if self.skip_manual_checkbox is not None:
            openalex_row.append(self.skip_manual_checkbox)

        # One entry per layer sub-section — joined below with a thin rule.
        groups: list[list[ft.Control]] = [
            [
                sub_section_title("openalex_papers (L1–L4)"),
                ft.Row(
                    spacing=12,
                    controls=openalex_row,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            layer_group("Chunks (L5)", ["bulk_backfill_chunks", "bulk_re_embed"]),
            layer_group("Entities (L6)", ["bulk_backfill_entities"]),
            layer_group(
                "Ontology linking (L7)",
                ["bulk_backfill_ontology", "backfill_xrefs", "clear_xref_edges"],
            ),
            layer_group("Triples (L8)", ["bulk_backfill_triples"]),
            layer_group("Cross-doc (L9)", ["bulk_backfill_cross_doc"]),
            layer_group("Cross-doc xrefs (L10)", ["recompute_cross_doc_xrefs"]),
        ]
        controls: list[ft.Control] = []
        for i, group in enumerate(groups):
            if i:  # thin rule between sub-sections (none before the first)
                controls.append(thin_rule())
            controls.extend(group)
        return ft.Column(spacing=8, controls=controls)

    def _on_bulk_op_clicked(self, op_name: str) -> None:
        """Validate + save config, then run bulk op `op_name`:
        plan → confirm dialog → execute (with a spinner)."""
        if self._busy:
            return
        error = self.config_editor.try_save_and_get_error()
        if error is not None:
            self._show_invalid_config_dialog(op_name, error)
            return
        if not self._loop_running():
            return
        self._spawn(self._run_bulk_op(op_name))

    async def _run_bulk_op(self, op_name: str) -> None:
        from knowledge_agent.ingestion import bulk_ops
        from knowledge_agent.kg.corpus_config import load_corpus_config

        cfg_path = self.app.gui_config.corpus_config_path
        if cfg_path is None:
            self._set_status("No active corpus.")
            return
        try:
            config = load_corpus_config(cfg_path)
        except Exception as exc:
            self._set_status(f"could not load corpus.toml: {exc}")
            return

        # clear_xref_edges needs the user to pick WHICH ontology first.
        if op_name == "clear_xref_edges":
            self._prompt_clear_xref_ontology()
            return

        # op_name -> (plan_fn, execute_fn, plan_arg, execute_takes_config).
        # Dispatch through variables (not named calls) so the async-lint
        # guard doesn't flag the executor factory below.
        table = {
            "bulk_backfill_chunks": (
                bulk_ops.bulk_backfill_chunks_plan,
                bulk_ops.bulk_backfill_chunks_execute,
                "none",
                True,
            ),
            "bulk_re_embed": (
                bulk_ops.bulk_re_embed_plan,
                bulk_ops.bulk_re_embed_execute,
                "none",
                True,
            ),
            "bulk_backfill_entities": (
                bulk_ops.bulk_backfill_entities_plan,
                bulk_ops.bulk_backfill_entities_execute,
                "none",
                True,
            ),
            "bulk_backfill_ontology": (
                bulk_ops.bulk_backfill_ontology_plan,
                bulk_ops.bulk_backfill_ontology_execute,
                "none",
                True,
            ),
            "bulk_backfill_triples": (
                bulk_ops.bulk_backfill_triples_plan,
                bulk_ops.bulk_backfill_triples_execute,
                "none",
                True,
            ),
            "bulk_backfill_cross_doc": (
                bulk_ops.bulk_backfill_cross_doc_plan,
                bulk_ops.bulk_backfill_cross_doc_execute,
                "none",
                True,
            ),
            "backfill_xrefs": (
                bulk_ops.backfill_xrefs_plan,
                bulk_ops.backfill_xrefs_execute,
                "config",
                True,
            ),
            "recompute_cross_doc_xrefs": (
                bulk_ops.recompute_cross_doc_xrefs_plan,
                bulk_ops.recompute_cross_doc_xrefs_execute,
                "config",
                True,
            ),
            "bulk_resolve_openalex": (
                bulk_ops.bulk_resolve_openalex_plan,
                bulk_ops.bulk_resolve_openalex_execute,
                "skip_manual",
                False,
            ),
        }
        entry = table.get(op_name)
        if entry is None:
            self._set_status(f"{op_name}: not wired.")
            return
        plan_fn, exec_fn, plan_arg, exec_config = entry
        try:
            if plan_arg == "config":
                plan = await plan_fn(config)
            elif plan_arg == "skip_manual":
                skip = (
                    bool(self.skip_manual_checkbox.value)
                    if self.skip_manual_checkbox is not None
                    else True
                )
                plan = await plan_fn(skip_manual=skip)
            else:
                plan = await plan_fn()
        except Exception as exc:
            self._set_status(f"{op_name}: could not plan — {exc}")
            return

        if exec_config:

            def executor(pf=exec_fn, p=plan, c=config):
                return pf(p, c)
        else:

            def executor(pf=exec_fn, p=plan):
                return pf(p)

        self._show_ingest_confirm(
            op_name,
            plan.summary,
            lambda: self._spawn(self._execute_bulk_op(op_name, executor)),
        )

    def _prompt_clear_xref_ontology(self) -> None:
        """Ask which ontology's xref edges to clear, then plan + confirm."""
        dropdown = ft.Dropdown(
            label="Ontology",
            options=[
                ft.DropdownOption(key=key, text=text) for key, text in _ONTOLOGY_DISPLAY.items()
            ],
        )

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _go(_ev: ft.Event) -> None:
            name = dropdown.value
            self.app.page.pop_dialog()
            if not name:
                self._set_status("Pick an ontology to clear.")
                return
            self._spawn(self._run_clear_xref(name))

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Clear xref edges"),
            content=ft.Column(
                tight=True,
                spacing=8,
                controls=[
                    ft.Text(
                        "Which ontology's cross-ontology xref edges should be cleared?",
                        size=12,
                    ),
                    dropdown,
                ],
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content=centered_label("Continue"), on_click=_go),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    async def _run_clear_xref(self, ontology_name: str) -> None:
        from knowledge_agent.ingestion import bulk_ops

        try:
            plan = await bulk_ops.clear_xref_edges_plan(ontology_name)
        except Exception as exc:
            self._set_status(f"clear_xref_edges: could not plan — {exc}")
            return
        exec_fn = bulk_ops.clear_xref_edges_execute

        def executor(pf=exec_fn, p=plan):
            return pf(p)

        self._show_ingest_confirm(
            "clear_xref_edges",
            plan.summary,
            lambda: self._spawn(self._execute_bulk_op("clear_xref_edges", executor)),
        )

    async def _execute_bulk_op(self, op_name: str, executor) -> None:
        self._set_busy(True, f"{op_name}: working…")
        try:
            result = await executor()
        except Exception as exc:
            self._set_busy(False, f"{op_name} failed: {exc}")
            return
        self._set_busy(
            False,
            f"{op_name} done: {self._fmt_bulk_result(result)}.",
        )
        self._notify_ingest_complete()

    @staticmethod
    def _fmt_bulk_result(result: object) -> str:
        """Generic result summary: every int field + a failure count.
        Works across all the bulk_ops `*Result` dataclasses without
        hard-coding each one's fields."""
        from dataclasses import asdict, is_dataclass

        if not is_dataclass(result):
            return "done"
        parts: list[str] = []
        for key, val in asdict(result).items():
            if key == "failures":
                if val:
                    parts.append(f"{len(val)} failures")
            elif isinstance(val, int):  # bool is an int subclass — fine
                parts.append(f"{key}={val}")
        return ", ".join(parts) if parts else "done"

    # ----- handlers -------------------------------------------------------

    # ---- async plumbing ----

    def _loop_running(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _set_status(self, msg: str) -> None:
        self._write_status(msg)
        self.app.page.update()

    def _write_status(self, msg: str) -> None:
        """Write a real status message, clearing the idle 'Empty' placeholder
        styling (italic / dim grey) so live messages read at full weight."""
        if self.status is not None:
            self.status.value = msg
            self.status.italic = False
            self.status.color = ft.Colors.GREY_400

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        """Toggle the spinner + disable every action button during a run
        so a second ingest can't start on top of one in flight."""
        self._busy = busy
        if self.progress_ring is not None:
            self.progress_ring.visible = busy
        for btn in (
            self.ingest_folder_button,
            self.reingest_button,
            self.sync_button,
            self.ingest_file_button,
            self.folder_browse_button,
            self.file_browse_button,
        ):
            if btn is not None:
                btn.disabled = busy
        if msg is not None:
            self._write_status(msg)
        self.app.page.update()

    def _notify_ingest_complete(self) -> None:
        """Best-effort cross-tab refresh after a successful ingest / op
        (LibraryView wires this to the Select sub-tab's reload)."""
        if self.on_ingest_complete is None:
            return
        try:
            self.on_ingest_complete()
        except Exception as exc:
            logger.warning("on_ingest_complete callback failed: %r", exc)

    # ---- pickers ----

    def _on_folder_browse_clicked(self, e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        self._spawn(self._pick_folder())

    async def _pick_folder(self) -> None:
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title="Pick a folder to ingest",
            )
        except Exception as exc:
            self._set_status(f"folder picker error: {exc}")
            return
        if chosen and self.folder_field is not None:
            self.folder_field.value = chosen
            self._persist_folder(chosen)
            self.app.page.update()

    def _on_file_browse_clicked(self, e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        self._spawn(self._pick_file())

    async def _pick_file(self) -> None:
        try:
            files = await self.app.file_picker.pick_files(
                dialog_title="Pick a file to ingest",
            )
        except Exception as exc:
            self._set_status(f"file picker error: {exc}")
            return
        if files and files[0].path and self.file_field is not None:
            self.file_field.value = files[0].path
            self._persist_file(files[0].path)
            self.app.page.update()

    # ---- session persistence (folder / file pickers) ----

    def _corpus_toml_path(self) -> Path | None:
        """Active corpus's `corpus.toml` path (the sidecar lives beside it)."""
        raw = self.app.gui_config.corpus_config_path
        return Path(raw) if raw is not None else None

    def _restore_session_paths(self, active_name: str) -> None:
        """Pre-fill the folder / file pickers from the corpus's saved
        session — once per corpus, so re-renders don't clobber typing."""
        if active_name == self._session_restored_for:
            return
        self._session_restored_for = active_name
        toml_path = self._corpus_toml_path()
        if toml_path is None:
            return
        session = load_session(toml_path)
        if self.folder_field is not None:
            self.folder_field.value = session.last_folder or ""
        if self.file_field is not None:
            self.file_field.value = session.last_file or ""

    def _persist_folder(self, folder: str | None) -> None:
        toml_path = self._corpus_toml_path()
        if toml_path is not None:
            update_last_folder(toml_path, folder)

    def _persist_file(self, file: str | None) -> None:
        toml_path = self._corpus_toml_path()
        if toml_path is not None:
            update_last_file(toml_path, file)

    def _on_folder_blur(self, e: ft.Event) -> None:
        """Persist a hand-typed folder path when focus leaves the field."""
        if self.folder_field is not None:
            self._persist_folder(self.folder_field.value)

    def _on_file_blur(self, e: ft.Event) -> None:
        """Persist a hand-typed file path when focus leaves the field."""
        if self.file_field is not None:
            self._persist_file(self.file_field.value)

    # ---- action buttons ----

    def _on_ingest_folder_clicked(self, e: ft.Event) -> None:
        self._start_action("Ingest folder")

    def _on_reingest_clicked(self, e: ft.Event) -> None:
        self._start_action("Re-ingest")

    def _on_sync_clicked(self, e: ft.Event) -> None:
        self._start_action("Sync")

    def _on_ingest_file_clicked(self, e: ft.Event) -> None:
        self._start_action("Ingest single file")

    def _start_action(self, action: str) -> None:
        """Validate + save the config, then run `action` on the picked
        folder/file: plan → confirm dialog → execute (with a spinner).

        Validation failure surfaces as a warning dialog with the
        pydantic message; no pipeline runs.
        """
        if self._busy:
            return
        error = self.config_editor.try_save_and_get_error()
        if error is not None:
            self._show_invalid_config_dialog(action, error)
            return
        if not self._loop_running():
            return
        self._spawn(self._run_action(action))

    async def _run_action(self, action: str) -> None:
        """Load the saved config, build the plan for `action`, and show a
        confirm dialog whose OK button kicks off the execute."""
        from knowledge_agent.ingestion import bulk_ops
        from knowledge_agent.kg.corpus_config import load_corpus_config

        main, sub, overwrite = self.config_editor.get_ingest_args()
        cfg_path = self.app.gui_config.corpus_config_path
        if cfg_path is None:
            self._set_status("No active corpus.")
            return
        try:
            config = load_corpus_config(cfg_path)
        except Exception as exc:
            self._set_status(f"could not load corpus.toml: {exc}")
            return

        # Pre-flight: ingestion embeds every chunk (and may extract with an
        # LLM), so the required provider key(s) must be present. A missing
        # key aborts with a clear message instead of a silent "N succeeded"
        # that actually persisted nothing.
        missing_key = self._missing_ingest_key(config)
        if missing_key is not None:
            self._show_missing_key_dialog(action, missing_key)
            return

        if action == "Ingest single file":
            self._plan_single_file(config, main, sub, overwrite)
            return

        folder_str = (
            (self.folder_field.value or "").strip() if self.folder_field is not None else ""
        )
        if not folder_str:
            self._set_status("Pick a folder first (Browse).")
            return
        folder = Path(folder_str)

        # Building the plan hashes every file in the folder (~5-10s for a
        # 100-file folder). The *_plan functions run that hash off the
        # event loop (asyncio.to_thread), so the loop stays free and this
        # spinner both renders and animates for the whole scan. Cleared
        # the moment the dialog / empty / error result is ready.
        self._set_busy(True, f"{action}: scanning {folder.name}…")
        try:
            if action == "Ingest folder":
                plan = await bulk_ops.add_plan(folder, main, sub)
                empty = plan.n_new == 0
            elif action == "Re-ingest":
                plan = await bulk_ops.ingest_folder_plan(folder, main, sub)
                empty = plan.n_files == 0
            else:  # Sync
                plan = await bulk_ops.sync_plan(folder, main, sub)
                empty = (plan.n_new + plan.n_moved + plan.n_edited + plan.n_orphans) == 0
        except Exception as exc:
            self._set_busy(False, f"{action}: could not plan — {exc}")
            return

        if empty:
            self._set_busy(False, f"{action}: nothing to do in {folder}.")
            return

        self._set_busy(False, "")
        self._show_ingest_confirm(
            action,
            plan.summary,
            lambda: self._spawn(self._execute_action(action, plan, config, overwrite)),
        )

    def _plan_single_file(
        self,
        config: object,
        main: str,
        sub: str | None,
        overwrite: bool,
    ) -> None:
        path_str = (self.file_field.value or "").strip() if self.file_field is not None else ""
        if not path_str:
            self._set_status("Pick a file first (Browse).")
            return
        path = Path(path_str)
        if not path.is_file():
            self._set_status(f"not a file: {path}")
            return
        self._show_ingest_confirm(
            "Ingest single file",
            f"Ingest '{path.name}' into the active corpus?",
            lambda: self._spawn(
                self._execute_single_file(
                    path,
                    config,
                    main,
                    sub,
                    overwrite,
                )
            ),
        )

    def _show_ingest_confirm(
        self,
        action: str,
        summary: str,
        on_confirm,
    ) -> None:
        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _ok(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            on_confirm()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(action),
            content=ft.Text(summary, size=12, selectable=True),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content=centered_label(action), on_click=_ok),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    async def _execute_action(
        self,
        action: str,
        plan: object,
        config: object,
        overwrite: bool,
    ) -> None:
        from knowledge_agent.ingestion import bulk_ops

        preserve = not overwrite
        total = self._ingest_progress_total(action, plan)
        self._set_busy(True, f"{action}: working…")
        self._begin_progress(total)

        def progress(done: int, tot: int) -> None:
            self._on_ingest_progress(action, done, tot)

        cb = progress if total > 0 else None
        try:
            if action == "Ingest folder":
                result = await bulk_ops.add_execute(plan, config, preserve, progress_cb=cb)
                msg = self._fmt_ingest_result(action, result)
            elif action == "Re-ingest":
                result = await bulk_ops.ingest_folder_execute(
                    plan,
                    config,
                    preserve,
                    progress_cb=cb,
                )
                msg = self._fmt_ingest_result(action, result)
            else:  # Sync
                result = await bulk_ops.sync_execute(plan, config, preserve, progress_cb=cb)
                msg = self._fmt_sync_result(result)
        except Exception as exc:
            self._end_progress()
            self._set_busy(False, f"{action} failed: {exc}")
            return
        self._end_progress()
        self._set_busy(False, msg)
        self._notify_ingest_complete()

    # ----- progress bar (execute phase) -----------------------------------

    def _ingest_progress_total(self, action: str, plan: object) -> int:
        """Per-file step count `_execute_action` reports for the bar.

        Matches each executor's own loop length so the bar reaches 100%
        exactly: Ingest folder = new files, Re-ingest = all files, Sync =
        new + moved + edited + orphan items."""
        if action == "Ingest folder":
            return int(getattr(plan, "n_new", 0))
        if action == "Re-ingest":
            return int(getattr(plan, "n_files", 0))
        return int(
            getattr(plan, "n_new", 0)
            + getattr(plan, "n_moved", 0)
            + getattr(plan, "n_edited", 0)
            + getattr(plan, "n_orphans", 0)
        )

    def _begin_progress(self, total: int) -> None:
        """Enter the execute phase: hide the indeterminate ring, show the
        determinate bar at 0 (only when there's something to count)."""
        if self.progress_ring is not None:
            self.progress_ring.visible = False
        if self.progress_bar is not None:
            self.progress_bar.visible = total > 0
            self.progress_bar.value = 0.0
        self.app.page.update()

    def _on_ingest_progress(self, action: str, done: int, total: int) -> None:
        """Per-file callback from the executor — advance the bar + count."""
        if self.progress_bar is not None and total > 0:
            self.progress_bar.value = done / total
        self._write_status(f"{action}: {done} / {total} files")
        self.app.page.update()

    def _end_progress(self) -> None:
        """Leave the execute phase: hide + reset the bar."""
        if self.progress_bar is not None:
            self.progress_bar.visible = False
            self.progress_bar.value = 0.0
        self.app.page.update()

    async def _execute_single_file(
        self,
        path: Path,
        config: object,
        main: str,
        sub: str | None,
        overwrite: bool,
    ) -> None:
        from knowledge_agent.ingestion import pipeline

        self._set_busy(True, f"Ingesting {path.name}…")
        try:
            await pipeline.ingest_document(
                path,
                config,
                main,
                sub,
                preserve_existing_labels=not overwrite,
            )
        except Exception as exc:
            self._set_busy(False, f"Ingest failed: {exc}")
            return
        self._set_busy(False, f"Ingested '{path.name}'.")
        self._notify_ingest_complete()

    @staticmethod
    def _fmt_ingest_result(action: str, result: object) -> str:
        msg = f"{action} done: {result.n_succeeded} succeeded, {result.n_failed} failed."
        if result.failures:
            name, err = result.failures[0]
            msg += f"  First failure: {name} — {err}"
        return msg

    @staticmethod
    def _fmt_sync_result(result: object) -> str:
        return (
            f"Sync done: {result.n_new_ingested} new, "
            f"{result.n_edited_succeeded} re-ingested, "
            f"{result.n_orphans_deleted} removed, "
            f"{result.n_new_failed + result.n_edited_failed} failed."
        )

    def _missing_ingest_key(self, config: object) -> str | None:
        """Env var name of a provider key ingestion needs but the active
        provider lacks, or None when every needed key is present.

        Ingestion embeds every chunk, so the EMBEDDER key is always
        required; the LLM key is required only when this corpus extracts
        with an LLM (entities extractor includes 'llm', or triples on).
        Local providers (Ollama / HuggingFace) need no key, so they never
        report a miss. A key counts as present if it's in the keyring OR
        already in the env (a shell export still works)."""
        import os

        from knowledge_agent.config import get_settings
        from knowledge_agent.gui.config_store import get_api_key

        try:
            settings = get_settings()
        except Exception:
            # Can't resolve settings — don't block; the real error (if
            # any) surfaces downstream rather than here.
            return None

        embed_env = {
            "voyage": "VOYAGE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(settings.embedding_provider)
        if embed_env and not (os.getenv(embed_env) or get_api_key(settings.embedding_provider)):
            return embed_env

        uses_llm = bool(getattr(getattr(config, "layers", None), "triples", False))
        entities = getattr(config, "entities", None)
        if entities is not None and "llm" in (getattr(entities, "extractors", None) or []):
            uses_llm = True
        if uses_llm:
            llm_env = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
            }.get(settings.llm_provider)
            if llm_env and not (os.getenv(llm_env) or get_api_key(settings.llm_provider)):
                return llm_env
        return None

    def _show_missing_key_dialog(self, action: str, env_var: str) -> None:
        """Modal: the ingest can't start without a provider key. The GUI
        reads keys from the OS keyring (Settings → Keys), NOT `.env` — so
        a `.env` that works for the CLI won't help the GUI."""

        def _close(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Can't {action.lower()} — missing API key"),
            content=ft.Column(
                tight=True,
                spacing=6,
                controls=[
                    ft.Text(f"Ingestion needs {env_var}, but it isn't set.", size=12),
                    ft.Text(
                        "Add it in Settings → Keys, then try again. The GUI "
                        "reads keys from the OS keyring there, not from a "
                        ".env file.",
                        size=12,
                        italic=True,
                        color=ft.Colors.GREY_400,
                    ),
                ],
            ),
            actions=[ft.TextButton("OK", on_click=_close)],
        )
        self.app.page.show_dialog(dialog)

    def _show_invalid_config_dialog(self, action: str, message: str) -> None:
        """Modal explaining why the ingest can't start. One [OK]
        button — the config is untouched on disk (validation happens
        before the write)."""

        def _close(_ev):
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Can't run {action.lower()}"),
            content=ft.Column(
                spacing=6,
                tight=True,
                controls=[
                    ft.Text(
                        "The config editor has an invalid setting:",
                        size=12,
                    ),
                    ft.Text(message, size=12, selectable=True),
                    ft.Text(
                        "Fix it in the config editor on the right, then try again.",
                        size=12,
                        italic=True,
                        color=ft.Colors.GREY_400,
                    ),
                ],
            ),
            actions=[ft.TextButton("OK", on_click=_close)],
        )
        self.app.page.show_dialog(dialog)
