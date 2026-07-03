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

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)
from knowledge_agent.gui.library.corpus_config_editor import (
    CorpusConfigEditor,
    _ONTOLOGY_DISPLAY,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class IngestTab:
    """Pure action flow (left) + full config surface (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.config_editor = CorpusConfigEditor(app)
        self.status: ft.Text | None = None
        self.active_corpus_label: ft.Text | None = None

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

        self._create_controls()

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)
        self.active_corpus_label = ft.Text(
            "", size=13, weight=ft.FontWeight.BOLD,
        )

        # ---- Folder picker + 3 folder-action buttons ----
        self.folder_field = ft.TextField(
            label="Folder",
            hint_text="Pick a folder of documents",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
            disabled=True,  # Phase 6 wires this + Browse.
        )
        self.folder_browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_folder_browse_clicked,
            disabled=True,
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
            label="File",
            hint_text="Pick a single file to ingest",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            expand=True,
            disabled=True,
        )
        self.file_browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_file_browse_clicked,
            disabled=True,
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
                        size=12, color=ft.Colors.AMBER_300, italic=True,
                    ),
                ],
            )

        assert self.active_corpus_label is not None
        self.active_corpus_label.value = f"Corpus: {active_name}"

        # Config editor owns the Labels-section sub-label dropdown, which
        # reads `allowed_types` off the loaded config. Force a load here
        # so first render has state ready.
        self.config_editor.ensure_loaded()

        left_pane = ft.Container(
            expand=1,
            padding=12,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            bgcolor=PANEL_BG,
            border_radius=4,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self.active_corpus_label,
                    ft.Text(
                        "Every setting lives in the config on the right. "
                        "Pick a folder or file below and hit an action.",
                        size=11, color=ft.Colors.GREY_400,
                    ),
                    ft.Divider(),

                    # Folder picker + 3 folder actions.
                    ft.Row(
                        controls=[
                            self.folder_field,
                            self.folder_browse_button,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
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

                    ft.Divider(),

                    # File picker + single-file ingest.
                    ft.Row(
                        controls=[
                            self.file_field,
                            self.file_browse_button,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        controls=[self.ingest_file_button],
                        spacing=8,
                    ),

                    ft.Divider(),

                    self.status,
                    ft.Text(
                        "Pre-flight preview + progress + results land "
                        "with the pickers in phase 6.",
                        size=11, color=ft.Colors.GREY_500, italic=True,
                    ),

                    ft.Divider(),

                    self._build_diff_card(),

                    ft.Divider(),

                    ft.Text(
                        "Bulk operations",
                        size=13, weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(
                        "Retroactive per-layer refreshes for the "
                        "already-ingested corpus. Buttons stubbed until "
                        "phase 6 wires the dialogs.",
                        size=11, color=ft.Colors.GREY_500, italic=True,
                    ),
                    self._build_bulk_ops_panel(),
                ],
            ),
        )

        right_pane = ft.Container(
            expand=1,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=8,
                controls=[self.config_editor.build()],
            ),
        )

        body = ft.Row(
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            controls=[left_pane, right_pane],
        )
        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[view_header("Ingest"), body],
        )

    # ----- Diff card ------------------------------------------------------

    def _build_diff_card(self) -> ft.Control:
        """Compact `field: current → new` summary of pending config
        changes. Only rows where baseline differs from in-memory are
        shown. Empty state when nothing pending.
        """
        baseline = self.config_editor._baseline_config
        inmem = self.config_editor._corpus_config
        if baseline is None or inmem is None:
            return ft.Container()

        diffs: list[tuple[str, str, str]] = []
        # Layer bools.
        for field, label in (
            ("openalex_papers", "openalex_papers"),
            ("chunks", "chunks"),
            ("entities", "entities"),
            ("triples", "triples"),
            ("cross_doc", "cross_doc"),
            ("cross_doc_xrefs", "cross_doc_xrefs"),
        ):
            b = getattr(baseline.layers, field, False)
            i = getattr(inmem.layers, field, False)
            if b != i:
                diffs.append((label, _fmt_bool(b), _fmt_bool(i)))
        # xrefs (3-state).
        if baseline.layers.xrefs != inmem.layers.xrefs:
            diffs.append(("xrefs", baseline.layers.xrefs, inmem.layers.xrefs))
        # Per-ontology bools.
        for key, display in _ONTOLOGY_DISPLAY.items():
            b = getattr(baseline.layers, f"ontology_{key}", False)
            i = getattr(inmem.layers, f"ontology_{key}", False)
            if b != i:
                diffs.append(
                    (f"ontology_{key} ({display})",
                     _fmt_bool(b), _fmt_bool(i)),
                )
        # Extractor + entity_types.
        base_extractor = (
            baseline.entities.extractor if baseline.entities is not None
            else "—"
        )
        cur_extractor = (
            inmem.entities.extractor if inmem.entities is not None else "—"
        )
        if base_extractor != cur_extractor:
            diffs.append(("extractor", base_extractor, cur_extractor))
        base_types = (
            ", ".join(baseline.entities.entity_types)
            if baseline.entities is not None else "—"
        ) or "(default)"
        cur_types = (
            ", ".join(inmem.entities.entity_types)
            if inmem.entities is not None else "—"
        ) or "(default)"
        if base_types != cur_types:
            diffs.append(("entity_types", base_types, cur_types))
        # Chunks (L5) per-corpus fields (promoted 2026-07-02).
        # Entities (L6) per-corpus fields (promoted 2026-07-02).
        for name in (
            "chunker_strategy",
            "chunk_max_tokens",
            "merge_peers",
            "enable_pdf_ocr",
            "enable_image_ocr",
            "images_scale",
            "optimize_indexes_per_ingest",
            "entity_extractor_model",
            "entity_extractor_temperature",
            "triples_extractor_model",
            "triples_extractor_temperature",
        ):
            base_val = getattr(baseline, name)
            cur_val = getattr(inmem, name)
            if base_val != cur_val:
                if isinstance(base_val, bool):
                    diffs.append(
                        (name, _fmt_bool(base_val), _fmt_bool(cur_val)),
                    )
                else:
                    diffs.append((name, str(base_val), str(cur_val)))
        # Cross-doc thresholds.
        base_thr = (
            baseline.cross_doc.threshold
            if baseline.cross_doc is not None else 2
        )
        cur_thr = (
            inmem.cross_doc.threshold if inmem.cross_doc is not None else 2
        )
        if base_thr != cur_thr:
            diffs.append(("cross_doc.threshold", str(base_thr), str(cur_thr)))
        base_xthr = (
            baseline.cross_doc_xrefs.threshold
            if baseline.cross_doc_xrefs is not None else 2
        )
        cur_xthr = (
            inmem.cross_doc_xrefs.threshold
            if inmem.cross_doc_xrefs is not None else 2
        )
        if base_xthr != cur_xthr:
            diffs.append(
                ("cross_doc_xrefs.threshold", str(base_xthr), str(cur_xthr)),
            )

        header = ft.Text(
            "Ingestion summary",
            size=13, weight=ft.FontWeight.BOLD,
        )
        if not diffs:
            body: ft.Control = ft.Text(
                "No pending changes — new ingest = previous.",
                size=11, color=ft.Colors.GREY_500, italic=True,
            )
        else:
            rows: list[ft.Control] = [
                ft.Text(
                    f"{len(diffs)} pending change{'s' if len(diffs) != 1 else ''}:",
                    size=11, color=ft.Colors.GREY_400,
                ),
            ]
            for name, cur, new in diffs:
                rows.append(
                    ft.Row(
                        spacing=6,
                        controls=[
                            ft.Text(
                                name,
                                size=11,
                                color=ft.Colors.GREY_300,
                                width=180,
                            ),
                            ft.Text(
                                cur, size=11, color=ft.Colors.GREY_400,
                            ),
                            ft.Text(
                                "→", size=11, color=ft.Colors.GREY_500,
                            ),
                            ft.Text(
                                new, size=11, color=ft.Colors.AMBER_300,
                            ),
                        ],
                    ),
                )
            body = ft.Column(spacing=4, controls=rows)

        return ft.Container(
            padding=10,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=4,
            content=ft.Column(spacing=6, controls=[header, body]),
        )

    # ----- Bulk operations panel ------------------------------------------

    def _build_bulk_ops_panel(self) -> ft.Control:
        """Flat per-layer bulk-ops list — one small header per layer,
        buttons directly below it. Every button is stubbed until
        phase-6 wires the plan / execute dialogs; see
        `ingestion.bulk_ops` for the underlying async primitives."""
        def op_button(op_name: str) -> ft.Control:
            return ft.Button(
                content=centered_label(op_name),
                on_click=lambda e, n=op_name: self._on_bulk_op_clicked(n),
            )

        def layer_group(title: str, ops: list[str]) -> list[ft.Control]:
            return [
                ft.Text(
                    title, size=12, color=ft.Colors.GREY_300,
                ),
                ft.Row(
                    wrap=True, spacing=8,
                    controls=[op_button(op) for op in ops],
                ),
            ]

        controls: list[ft.Control] = []
        controls.extend(layer_group(
            "openalex_papers (L1–L4)",
            ["bulk_resolve_openalex"],
        ))
        controls.extend(layer_group(
            "Chunks (L5)",
            ["bulk_backfill_chunks", "bulk_re_embed"],
        ))
        controls.extend(layer_group(
            "Entities (L6)",
            ["bulk_backfill_entities"],
        ))
        controls.extend(layer_group(
            "Ontology linking (L7)",
            [
                "bulk_backfill_ontology",
                "backfill_xrefs",
                "clear_xref_edges",
            ],
        ))
        controls.extend(layer_group(
            "Triples (L8)",
            ["bulk_backfill_triples"],
        ))
        controls.extend(layer_group(
            "Cross-doc (L9)",
            ["bulk_backfill_cross_doc"],
        ))
        controls.extend(layer_group(
            "Cross-doc xrefs (L10)",
            ["recompute_cross_doc_xrefs"],
        ))
        return ft.Column(spacing=8, controls=controls)

    def _on_bulk_op_clicked(self, op_name: str) -> None:
        """Placeholder handler — phase-6 wires the plan → confirm →
        execute dialog for `op_name`."""
        if self.status is not None:
            self.status.value = (
                f"{op_name}: dialog + execution land in phase 6."
            )
            self.app.page.update()

    # ----- handlers -------------------------------------------------------

    def _on_folder_browse_clicked(self, e: ft.Event) -> None:
        pass

    def _on_file_browse_clicked(self, e: ft.Event) -> None:
        pass

    def _on_ingest_folder_clicked(self, e: ft.Event) -> None:
        self._start_action("Ingest folder")

    def _on_reingest_clicked(self, e: ft.Event) -> None:
        self._start_action("Re-ingest")

    def _on_sync_clicked(self, e: ft.Event) -> None:
        self._start_action("Sync")

    def _on_ingest_file_clicked(self, e: ft.Event) -> None:
        self._start_action("Ingest single file")

    def _start_action(self, action: str) -> None:
        """Validate current editor state + save corpus.toml, then
        (phase 6) kick off the pipeline for `action` with the Labels
        section's per-run args.

        Validation failure surfaces as a warning dialog with the
        pydantic message.
        """
        error = self.config_editor.try_save_and_get_error()
        if error is not None:
            self._show_invalid_config_dialog(action, error)
            return
        main, sub, overwrite = self.config_editor.get_ingest_args()
        if self.status is not None:
            sub_str = sub if sub is not None else "(none)"
            self.status.value = (
                f"{action}: config saved. "
                f"main={main}, sub={sub_str}, "
                f"overwrite_labels={overwrite}. "
                f"Pipeline wiring lands in phase 6."
            )
            self.app.page.update()

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
                        "Fix it in the config editor on the right, "
                        "then try again.",
                        size=11, italic=True, color=ft.Colors.GREY_400,
                    ),
                ],
            ),
            actions=[ft.TextButton("OK", on_click=_close)],
        )
        self.app.page.show_dialog(dialog)


def _fmt_bool(v: bool) -> str:
    return "on" if v else "off"
