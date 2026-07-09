"""Evaluation → Dataset sub-tab — author + edit a gold dataset.

Left column = the authoring surface: choose the dataset file, pick how to seed
a case (blank / from the last Search / an LLM draft), and edit EVERY field of
the current case in a form grouped like the `EvalCase` schema (so it doubles as
documentation of what a case contains).

Right column = two stacked sections:
  * **Preview** — a live card mirroring the left form (debounced) with the
    **Add case / Update case** commit button beneath it. Nothing is written
    until you commit.
  * **In this dataset** — the dataset's cases as cards; click one to load it
    into the form (commit becomes **Update**), or delete it on the card.

Dataset files are chosen by **Browse** (opens in the active corpus's folder)
or created by **New dataset** — no packaged-set dropdown. Persistence goes
through the backend `save_dataset` helper.

Two ways to draft LLM cases:
  * **Generate one** — draft a single candidate straight into the form for
    review; **Add case** keeps it. Nothing hits disk unreviewed.
  * **Generate multiple** — bulk-draft `count` candidates straight into the
    dataset (origin=llm); review/edit/delete them afterward from the list.

Dropdown options are derived from the model's own `Literal`s (via
`typing.get_args`) so the choices can never drift from the schema. List-valued
fields (expected_sources, keywords, …) are edited as one-item-per-line text.
"""

from __future__ import annotations

import asyncio
import typing
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    centered_label,
    labeled_field,
    panel_box,
    panel_title,
    section_divider,
    section_title,
    sub_section_title,
)
from knowledge_agent.gui._widgets.retrieval_form import (
    DEFAULT_VALUES,
    RetrievalControls,
    apply_gray_out,
    build_mmr_slider,
    build_search_mode_radios,
)
from knowledge_agent.gui.evaluation._case_view import case_card, render_case_cards
from knowledge_agent.gui.settings.llm_tab import LLM_AVAILABLE_MODELS

if TYPE_CHECKING:
    from knowledge_agent.evaluation.models import EvalCase, EvalDataset
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_NONE = ""  # the expected_mode "(none)" sentinel (maps to None on read)
_PREVIEW_DEBOUNCE_S = 0.3  # settle time before the live preview re-renders
# Fixed caption column for the case-edit form, so every field's input starts at
# the same x (aligned left edges) instead of hugging its variable-width caption.
# Wide enough for the longest label ("expected_answer_points").
_FORM_LABEL_WIDTH = 190


class DatasetTab:
    """Author + edit a gold dataset: form + live preview + case list."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.dataset_field: ft.TextField | None = None
        self.status_group: ft.RadioGroup | None = None
        self.case_list: ft.Column | None = None
        self.form: ft.Column | None = None
        self.status: ft.Text | None = None
        self.gen_count: ft.TextField | None = None
        self.gen_one_button: ft.Control | None = None
        self.gen_multi_button: ft.Control | None = None
        # Spinners shown beside each Generate button while its LLM calls run.
        self.gen_one_spinner: ft.ProgressRing | None = None
        self.gen_multi_spinner: ft.ProgressRing | None = None
        # LLM model + temperature used for generation (Generate one + multiple).
        self.gen_model_dropdown: ft.Dropdown | None = None
        self.gen_temp_slider: ft.Slider | None = None
        self.gen_temp_value_text: ft.Text | None = None
        # Right column: live preview of the form + its two commit buttons. Only
        # one is live at a time — Add for a new case, Update for an existing one.
        self.preview_holder: ft.Container | None = None
        self.add_button: ft.Button | None = None
        self.update_button: ft.Button | None = None
        # form field widgets (created in build)
        self.f: dict[str, ft.Control] = {}
        # state
        self._dataset: EvalDataset | None = None
        self._cases: list[EvalCase] = []
        self._selected: int | None = None
        self._path: Path | None = None
        # Pending debounced preview refresh (cancelled + replaced on each edit).
        self._preview_task: asyncio.Task | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.config import DEFAULT_DATASET_PATH
        from knowledge_agent.evaluation.models import (
            CaseOrigin,
            DatasetStatus,
            RetrievalMode,
        )

        origins = list(typing.get_args(CaseOrigin))
        statuses = list(typing.get_args(DatasetStatus))
        modes = list(typing.get_args(RetrievalMode))

        # Dataset file: a read-only path display + Browse / New dataset. No
        # packaged-set dropdown (an editable dropdown read as a confusing plain
        # field). Browse opens an existing file; New dataset starts an empty one.
        self.dataset_field = ft.TextField(
            read_only=True,
            value=str(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH.exists() else "",
            hint_text="Browse or New dataset to choose a file",
            expand=True,
        )
        browse_button = ft.Button(
            content=centered_label("Browse"),
            tooltip="Open an existing gold-dataset JSON (starts in the corpus folder)",
            on_click=self._on_browse_clicked,
        )
        new_dataset_button = ft.Button(
            content=centered_label("New dataset"),
            tooltip="Start a new empty dataset in the corpus folder",
            on_click=self._on_new_dataset,
        )
        # Status = the whole dataset's authoring state; 3 radios side by side.
        self.status_group = ft.RadioGroup(
            value="draft",
            content=ft.Row([ft.Radio(value=s, label=s) for s in statuses], spacing=12),
        )
        new_case_button = ft.TextButton("New case", icon=ft.Icons.ADD, on_click=self._on_new)
        capture_button = ft.TextButton(
            "From last search", icon=ft.Icons.SEARCH, on_click=self._on_capture_from_search
        )
        self.gen_one_button = ft.TextButton(
            "Generate one", icon=ft.Icons.AUTO_AWESOME, on_click=self._on_generate_one
        )
        self.gen_multi_button = ft.TextButton(
            "Generate multiple",
            icon=ft.Icons.AUTO_AWESOME_MOTION,
            on_click=self._on_generate_multiple,
        )
        self.gen_count = ft.TextField(value="5", width=80, dense=True)
        self.gen_one_spinner = ft.ProgressRing(visible=False, width=16, height=16, stroke_width=2)
        self.gen_multi_spinner = ft.ProgressRing(visible=False, width=16, height=16, stroke_width=2)
        provider = getattr(self.app.gui_config, "llm_provider", "")
        self.gen_model_dropdown = ft.Dropdown(
            editable=True,  # pick a known model OR type one; blank = backend default
            options=[
                ft.DropdownOption(key=m, text=m) for m in LLM_AVAILABLE_MODELS.get(provider, ())
            ],
            hint_text="default (cheap mode-classifier model)",
            expand=True,
        )
        # Temperature as a slider (0–1, 20 steps) — matches the LLM-tab temp
        # controls; value display trails the slider.
        self.gen_temp_slider = ft.Slider(
            value=0.3, min=0.0, max=1.0, divisions=20, on_change=self._on_temp_slide
        )
        self.gen_temp_value_text = ft.Text("0.30", size=12, color=ft.Colors.WHITE, width=42)
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_500)

        # No own scroll / expand — the right column scrolls the preview + list
        # together (avoids a nested-scroll region).
        self.case_list = ft.Column(
            controls=[ft.Text("No cases yet — New case to add one.", italic=True)],
            spacing=2,
        )

        # ---- form field widgets (each refreshes the live preview) ----
        oc = self._on_form_changed
        # Search-mode radios + MMR-λ slider come from the shared builders, so the
        # eval form renders the SAME widgets as Settings → Retrieval (radios show
        # all options at once; the slider replaces a raw number box).
        self._lancedb_radios, lancedb_mode_radio, self._lancedb_mode_box = build_search_mode_radios(
            "hybrid", oc
        )
        mmr_slider, self._mmr_value_text = build_mmr_slider(
            float(DEFAULT_VALUES["mmr_lambda"]), on_change=oc
        )
        self.f = {
            "id": ft.TextField(dense=True, expand=True, on_change=oc),
            "question": ft.TextField(
                multiline=True, min_lines=2, max_lines=4, expand=True, on_change=oc
            ),
            "origin": ft.Dropdown(
                options=[ft.DropdownOption(key=o, text=o) for o in origins],
                value="manual",
                expand=True,
                on_select=oc,
            ),
            "category": ft.TextField(dense=True, expand=True, on_change=oc),
            "notes": ft.TextField(
                multiline=True, min_lines=1, max_lines=3, expand=True, on_change=oc
            ),
            "retrieval_mode": ft.Dropdown(
                options=[ft.DropdownOption(key=m, text=m) for m in modes],
                value="lancedb_only",
                expand=True,
                on_select=oc,
            ),
            "lancedb_search_mode": lancedb_mode_radio,  # shared radios (see above)
            # Tuning knobs — pre-filled with the standard defaults (same look as
            # Settings → Retrieval); the ones the chosen mode can't use are
            # grayed out by `_sync_retrieval_gray_out` (shared with Settings) and
            # aren't fed to the search.
            "top_k": ft.TextField(
                value=str(DEFAULT_VALUES["top_k"]), width=100, dense=True, on_change=oc
            ),
            "num_candidates": ft.TextField(
                value=str(DEFAULT_VALUES["num_candidates"]), width=100, dense=True, on_change=oc
            ),
            "rrf_rank_constant": ft.TextField(
                value=str(DEFAULT_VALUES["rrf_rank_constant"]), width=100, dense=True, on_change=oc
            ),
            "use_mmr": ft.Checkbox(label="use_mmr", value=False, on_change=oc),
            "mmr_lambda": mmr_slider,  # shared slider (see above)
            "kg_max_rows": ft.TextField(
                value=str(DEFAULT_VALUES["kg_max_rows"]), width=100, dense=True, on_change=oc
            ),
            "input_mode": ft.RadioGroup(
                value="refined",
                on_change=oc,
                content=ft.Column(
                    spacing=2,
                    controls=[
                        ft.Radio(
                            value="refined",
                            label=(
                                "Refined query — the query-builder LLM rewrites your question, "
                                "then searches"
                            ),
                        ),
                        ft.Radio(
                            value="direct_query",
                            label="Direct query — your raw text goes straight to vector/hybrid search",
                        ),
                        ft.Radio(
                            value="direct_cypher",
                            label="Direct Cypher — run your text as raw Cypher on the knowledge graph",
                        ),
                    ],
                ),
            ),
            "direct_retrieval": ft.Checkbox(
                label="direct_retrieval — skip synthesizer, show raw chunks / rows",
                value=False,
                on_change=oc,
            ),
            "expected_sources": _list_field("one doc_id per line", oc),
            "expected_chunks": _list_field("one substring per line", oc),
            "required_keywords": _list_field("one per line", oc),
            "disallowed_keywords": _list_field("one per line", oc),
            "expected_answer_points": _list_field("one per line", oc),
            "expected_entities": _list_field("one per line", oc),
            "expected_mode": ft.Dropdown(
                options=[
                    ft.DropdownOption(key=_NONE, text="(none)"),
                    *(ft.DropdownOption(key=m, text=m) for m in modes),
                ],
                value=_NONE,
                expand=True,
                on_select=oc,
            ),
            "user_cypher": ft.TextField(
                multiline=True, min_lines=1, max_lines=4, expand=True, on_change=oc
            ),
        }

        self.form = ft.Column(
            controls=[
                *self._group("Identity", ["id", "question", "origin", "category", "notes"]),
                *self._retrieval_group(),
                *self._group("Retrieval / chunk gold", ["expected_sources", "expected_chunks"]),
                *self._group("Keyword checks", ["required_keywords", "disallowed_keywords"]),
                *self._group("Judge gold", ["expected_answer_points"]),
                *self._group("KG gold", ["expected_entities", "expected_mode"]),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
        )

        # Right column: the live preview card + its two commit buttons. Both
        # route through _on_save_case; the disabled state (set in _render_preview)
        # gates which applies, and _selected drives append-vs-overwrite.
        self.preview_holder = ft.Container()
        self.add_button = ft.Button("Add case", icon=ft.Icons.ADD, on_click=self._on_save_case)
        self.update_button = ft.Button(
            "Update case", icon=ft.Icons.SAVE, on_click=self._on_save_case
        )
        refresh_button = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh the preview from the form",
            on_click=self._on_refresh_preview,
        )

        # LEFT: dataset file + metadata + the case-edit form, in flat sections.
        left_pane = panel_box(
            ft.Column(
                [
                    # ============ Section: Dataset file ============
                    section_title("Dataset file"),
                    labeled_field(
                        "Dataset",
                        self.dataset_field,
                        trailing=ft.Row([browse_button, new_dataset_button], spacing=6),
                    ),
                    labeled_field("Status", self.status_group),
                    section_divider(),
                    # ============ Section: Progress ============
                    section_title("Progress"),
                    self.status,
                    section_divider(),
                    # ============ Section: Add cases ============
                    section_title("Add cases"),
                    # Sub-section titles here carry no rule — the "Add cases"
                    # section title already separates them.
                    sub_section_title("Generate new case by"),
                    ft.Row(
                        [
                            new_case_button,
                            capture_button,
                            self.gen_one_button,
                            self.gen_one_spinner,
                        ],
                        wrap=True,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [
                            self.gen_multi_button,
                            ft.Text("count:", size=14, color=ft.Colors.GREY_300),
                            self.gen_count,
                            self.gen_multi_spinner,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    labeled_field("Generate with model", self.gen_model_dropdown),
                    labeled_field(
                        "Temperature", self.gen_temp_slider, trailing=self.gen_temp_value_text
                    ),
                    # "Per case form" title only — no divider rule, per request.
                    ft.Container(
                        padding=ft.Padding.only(top=16, bottom=6),
                        content=sub_section_title("Per case form"),
                    ),
                    self.form,
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )
        # RIGHT: live preview + commit on top, the full case list below. The
        # whole column scrolls so the preview leads and the list follows.
        right_pane = panel_box(
            ft.Column(
                [
                    # ============ Section: Preview ============
                    section_title("Preview"),
                    self.preview_holder,
                    ft.Row([refresh_button, self.add_button, self.update_button], spacing=8),
                    section_divider(),
                    # ============ Section: In this dataset ============
                    section_title("In this dataset"),
                    self.case_list,
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )
        # Column titles sit in a fixed header above the panes (aligned 50/50
        # over them) so each stays visible while its box scrolls — the same
        # sticky-header idiom as the Ingest tab.
        header = ft.Row(
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=panel_title("Dataset editor"),
                ),
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=panel_title("Cases"),
                ),
            ],
        )

        # Auto-load the dataset shown in the field on first open (50a). The tree
        # isn't mounted yet, so no page.update — build returns it populated.
        default_path = self.dataset_field.value
        if default_path:
            self._load(Path(default_path), refresh_page=False)
        else:
            self._render_preview()

        body = ft.Row(
            [left_pane, right_pane],
            expand=True,
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return ft.Column(
            controls=[header, body],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
        )

    def _retrieval_group(self) -> list[ft.Control]:
        """The 'Retrieval settings' group — same fields + gray-out as
        Settings → Retrieval. Knobs the chosen mode can't use are grayed out
        (`_sync_retrieval_gray_out`); the fields come pre-filled with the
        standard defaults, which the curator can override per case."""
        f = self.f
        rows: list[ft.Control] = [
            ft.Container(
                padding=ft.Padding.only(top=12, bottom=4),
                content=sub_section_title("Retrieval settings"),
            ),
            labeled_field("retrieval_mode", f["retrieval_mode"], label_width=_FORM_LABEL_WIDTH),
            self._lancedb_mode_box,  # shared radios (carry their own caption)
            labeled_field("top_k", f["top_k"], label_width=_FORM_LABEL_WIDTH),
            labeled_field("num_candidates", f["num_candidates"], label_width=_FORM_LABEL_WIDTH),
            labeled_field(
                "rrf_rank_constant", f["rrf_rank_constant"], label_width=_FORM_LABEL_WIDTH
            ),
            f["use_mmr"],
            labeled_field(
                "mmr_lambda",
                f["mmr_lambda"],
                label_width=_FORM_LABEL_WIDTH,
                trailing=self._mmr_value_text,
            ),
            labeled_field("kg_max_rows", f["kg_max_rows"], label_width=_FORM_LABEL_WIDTH),
            ft.Container(
                padding=ft.Padding.only(top=12, bottom=4),
                content=sub_section_title("Input mode"),
            ),
            f["input_mode"],
            labeled_field("user_cypher", f["user_cypher"], label_width=_FORM_LABEL_WIDTH),
            f["direct_retrieval"],
        ]
        self._sync_retrieval_gray_out()
        return rows

    def _sync_retrieval_gray_out(self) -> None:
        """Gray out the knobs the current mode/toggles can't use, via the shared
        `apply_gray_out` — the same single source Settings → Retrieval uses, so
        the two forms can't disagree. A grayed knob isn't fed to the search."""
        f = self.f
        apply_gray_out(
            RetrievalControls(
                lancedb_mode_box=self._lancedb_mode_box,
                lancedb_radios=self._lancedb_radios,
                lancedb_mode_control=f["lancedb_search_mode"],
                num_candidates_field=f["num_candidates"],
                rrf_constant_field=f["rrf_rank_constant"],
                use_mmr_checkbox=f["use_mmr"],
                mmr_lambda_control=f["mmr_lambda"],
                kg_max_rows_field=f["kg_max_rows"],
            ),
            retrieval_mode=f["retrieval_mode"].value or "lancedb_only",
            lancedb_search_mode=f["lancedb_search_mode"].value or "hybrid",
            use_mmr=bool(f["use_mmr"].value),
        )
        # The Cypher box is only the input for Direct Cypher mode.
        f["user_cypher"].disabled = f["input_mode"].value != "direct_cypher"

    def _group(self, title: str, keys: list[str]) -> list[ft.Control]:
        # Each field group is a padded title (NO rule — consistent with the
        # other sub-section titles here) followed by its fields. Each field gets
        # its key as a caption via labeled_field (multiline boxes top-align their
        # caption); checkboxes keep their own built-in label.
        rows: list[ft.Control] = [
            ft.Container(
                padding=ft.Padding.only(top=12, bottom=4),
                content=sub_section_title(title),
            )
        ]
        for k in keys:
            control = self.f[k]
            if isinstance(control, ft.Checkbox):
                rows.append(control)
            else:
                # Fixed caption column so every input's left edge lines up.
                rows.append(labeled_field(k, control, label_width=_FORM_LABEL_WIDTH))
        return rows

    # ---- live preview -----------------------------------------------------

    def _on_form_changed(self, _e: ft.Event | None = None) -> None:
        """Debounced: re-render the live preview a beat after the last edit, so
        the card settles rather than flickering on every keystroke. The
        retrieval gray-out updates synchronously (cheap) and flushes with the
        same page update as the preview."""
        self._sync_retrieval_gray_out()
        if not self._loop_running():
            self._render_preview()
            self.app.page.update()
            return
        if self._preview_task is not None and not self._preview_task.done():
            self._preview_task.cancel()
        self._preview_task = asyncio.create_task(self._debounced_preview())

    async def _debounced_preview(self) -> None:
        try:
            await asyncio.sleep(_PREVIEW_DEBOUNCE_S)
        except asyncio.CancelledError:
            return
        self._render_preview()
        self.app.page.update()

    def _render_preview(self) -> None:
        """Fill the right-column preview card from the current form state, and
        label the commit button Add (new) or Update (editing an existing)."""
        if self.preview_holder is not None:
            try:
                content: ft.Control = case_card(self._read_form(lenient=True), 0, detailed=True)
            except Exception:  # broad: a half-typed form must never break the preview
                content = ft.Text(
                    "Fill the form to preview a case.", italic=True, color=ft.Colors.GREY_500
                )
            self.preview_holder.content = content
        # Exactly one commit button is live: Add for a new case, Update when an
        # existing case is loaded; the other greys out.
        editing = self._selected is not None
        if self.add_button is not None:
            self.add_button.disabled = editing
        if self.update_button is not None:
            self.update_button.disabled = not editing

    def _on_refresh_preview(self, _e: ft.Event | None = None) -> None:
        """Force an immediate preview re-render from the current form (it also
        updates live/debounced as you type; this is the manual refresh)."""
        self._render_preview()
        self.app.page.update()

    def _loop_running(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    # ---- load -------------------------------------------------------------

    async def _on_browse_clicked(self, _e: ft.Event) -> None:
        """Open an existing gold-dataset JSON — the picker starts in the active
        corpus's folder — and load it."""
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        corpus_dir = active_corpus_dir(self.app)
        initial = str(corpus_dir) if corpus_dir and corpus_dir.is_dir() else None
        try:
            files = await self.app.file_picker.pick_files(
                dialog_title="Pick a gold-dataset JSON",
                allowed_extensions=["json"],
                initial_directory=initial,
            )
        except Exception as exc:  # broad: surface any picker failure in-line
            self._set_status(f"file picker error: {exc}")
            return
        if files and files[0].path:
            self._load(Path(files[0].path))

    def _on_new_dataset(self, _e: ft.Event) -> None:
        """Name a NEW dataset and create it (an empty but valid JSON file) in
        the active corpus's folder, ready to fill with cases. Refuses to
        overwrite an existing file — that's what Browse is for."""
        from knowledge_agent.evaluation.config import DEFAULT_DATASET_PATH
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        folder = active_corpus_dir(self.app) or DEFAULT_DATASET_PATH.parent
        name_field = ft.TextField(label="Dataset file name", hint_text="e.g. my_gold_set")
        location = ft.Text(f"Created in {folder}", size=12, color=ft.Colors.GREY_500, italic=True)
        error = ft.Text("", size=12, color=ft.Colors.RED_300, visible=False)

        def _show_error(msg: str) -> None:
            error.value = msg
            error.visible = True
            self.app.page.update()

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _create(_ev: ft.Event) -> None:
            name = (name_field.value or "").strip()
            if not name:
                _show_error("Enter a name.")
                return
            if not name.endswith(".json"):
                name = f"{name}.json"
            path = folder / name
            if path.exists():
                _show_error(
                    f"{path.name} already exists — pick another name, or Browse to open it."
                )
                return
            try:
                self._start_new_dataset(path)
            except Exception as exc:  # broad: I/O failure → inline error, dialog stays open
                _show_error(f"could not create: {exc}")
                return
            self.app.page.pop_dialog()
            self._set_status(f"created {path.name} — add cases, then Add case")

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("New dataset"),
            content=ft.Column([name_field, location, error], tight=True, width=420),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content="Create", on_click=_create),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    def _start_new_dataset(self, path: Path) -> None:
        """Write an empty dataset to `path` and make it the current one (blank
        form, empty list). Raises on I/O failure — the caller surfaces it."""
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        if self._is_shipped_dataset(path):
            raise ValueError(
                "cannot create a dataset in the packaged datasets folder — select a corpus first"
            )
        ds = EvalDataset()
        path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(ds, path)
        self._dataset = ds
        self._cases = []
        self._path = path
        self._selected = None
        if self.dataset_field is not None:
            self.dataset_field.value = str(path)
        if self.status_group is not None:
            self.status_group.value = ds.status
        self._clear_form()
        self._render_list()
        self._render_preview()

    def _load(self, path: Path, *, refresh_page: bool = True) -> None:
        from knowledge_agent.evaluation.models import load_dataset

        try:
            ds = load_dataset(path)
        except Exception as exc:  # broad: surface any parse/validation error in-line
            if self.status is not None:
                self.status.value = f"could not load: {exc}"
            if refresh_page:
                self.app.page.update()
            return
        self._dataset = ds
        self._cases = ds.cases
        self._path = path
        self._selected = None
        if self.dataset_field is not None:
            self.dataset_field.value = str(path)
        if self.status_group is not None:
            self.status_group.value = ds.status
        self._clear_form()
        self._render_list()
        self._render_preview()
        if self.status is not None:
            self.status.value = f"{len(self._cases)} case(s)"
        if refresh_page:
            self.app.page.update()

    # ---- list -------------------------------------------------------------

    def _render_list(self) -> None:
        if self.case_list is None:
            return
        # Shared renderer (same cards the Run tab shows read-only); here it's
        # editable — click a card or Edit to load it into the form, Delete to
        # remove it.
        self.case_list.controls = render_case_cards(
            self._cases,
            selected=self._selected,
            detailed=True,  # show every gold field on each card, not just a summary
            on_edit=self._select,
            on_delete=self._delete_case,
            on_cancel=self._on_cancel_edit,
            empty_hint="No cases yet — New case to add one.",
        )

    def _select(self, idx: int) -> None:
        self._selected = idx
        self._fill_form(self._cases[idx])
        self._render_list()
        self._render_preview()
        self.app.page.update()

    def _on_cancel_edit(self, _idx: int | None = None) -> None:
        """Cancel editing (the selected card's Cancel button): blank the form
        and deselect, back to new-case mode."""
        self._selected = None
        self._clear_form()
        self._render_list()
        self._render_preview()
        self._set_status("edit cancelled")

    # ---- form <-> case ----------------------------------------------------

    def _fill_form(self, case: EvalCase) -> None:
        f = self.f
        f["id"].value = case.id
        f["question"].value = case.question
        f["origin"].value = case.origin
        f["category"].value = case.category
        f["notes"].value = case.notes
        f["retrieval_mode"].value = case.retrieval.retrieval_mode
        f["lancedb_search_mode"].value = case.retrieval.lancedb_search_mode
        f["top_k"].value = str(case.retrieval.top_k)
        f["num_candidates"].value = _num_or_default(case.retrieval.num_candidates, "num_candidates")
        f["rrf_rank_constant"].value = _num_or_default(
            case.retrieval.rrf_rank_constant, "rrf_rank_constant"
        )
        mmr_val = (
            case.retrieval.mmr_lambda
            if case.retrieval.mmr_lambda is not None
            else float(DEFAULT_VALUES["mmr_lambda"])
        )
        f["mmr_lambda"].value = mmr_val
        self._mmr_value_text.value = f"{mmr_val:.2f}"
        f["use_mmr"].value = case.retrieval.use_mmr
        f["kg_max_rows"].value = _num_or_default(case.retrieval.kg_max_rows, "kg_max_rows")
        # Derive the input-mode radio from the case's skip_query_builder / cypher.
        if case.user_cypher:
            f["input_mode"].value = "direct_cypher"
        elif case.retrieval.skip_query_builder:
            f["input_mode"].value = "direct_query"
        else:
            f["input_mode"].value = "refined"
        f["direct_retrieval"].value = case.retrieval.direct_retrieval
        f["expected_sources"].value = _text(case.expected_sources)
        f["expected_chunks"].value = _text(case.expected_chunks)
        f["required_keywords"].value = _text(case.required_keywords)
        f["disallowed_keywords"].value = _text(case.disallowed_keywords)
        f["expected_answer_points"].value = _text(case.expected_answer_points)
        f["expected_entities"].value = _text(case.expected_entities)
        f["expected_mode"].value = case.expected_mode or _NONE
        f["user_cypher"].value = case.user_cypher or ""
        self._sync_retrieval_gray_out()

    def _clear_form(self) -> None:
        f = self.f
        for key in ("id", "question", "category", "notes", "user_cypher"):
            f[key].value = ""
        for key in (
            "expected_sources",
            "expected_chunks",
            "required_keywords",
            "disallowed_keywords",
            "expected_answer_points",
            "expected_entities",
        ):
            f[key].value = ""
        f["origin"].value = "manual"
        f["retrieval_mode"].value = "lancedb_only"
        f["lancedb_search_mode"].value = "hybrid"
        f["top_k"].value = str(DEFAULT_VALUES["top_k"])
        f["num_candidates"].value = str(DEFAULT_VALUES["num_candidates"])
        f["rrf_rank_constant"].value = str(DEFAULT_VALUES["rrf_rank_constant"])
        f["use_mmr"].value = False
        f["mmr_lambda"].value = float(DEFAULT_VALUES["mmr_lambda"])
        self._mmr_value_text.value = f"{float(DEFAULT_VALUES['mmr_lambda']):.2f}"
        f["kg_max_rows"].value = str(DEFAULT_VALUES["kg_max_rows"])
        f["input_mode"].value = "refined"
        f["direct_retrieval"].value = False
        f["expected_mode"].value = _NONE
        self._sync_retrieval_gray_out()

    def _read_form(self, *, lenient: bool = False) -> EvalCase:
        """Build an `EvalCase` from the form. Strict (default) raises on bad
        input — the commit surfaces it. `lenient=True` fills placeholders for
        the required id / question and a safe top_k, so a half-filled form still
        renders in the live preview instead of erroring."""
        from knowledge_agent.evaluation.models import EvalCase

        f = self.f
        case_id = (f["id"].value or "").strip()
        question = (f["question"].value or "").strip()
        if lenient:
            case_id = case_id or "(unnamed)"
            question = question or "(no question yet)"

        def _num(parse: typing.Callable[[str], typing.Any], raw: str | None) -> typing.Any:
            """Blank → None (knob not pinned); else parse. In lenient (preview)
            mode a malformed entry degrades to None instead of raising."""
            s = (raw or "").strip()
            if not s:
                return None
            try:
                return parse(s)
            except ValueError:
                if not lenient:
                    raise
                return None

        try:
            top_k = int((f["top_k"].value or "5").strip())
        except ValueError:
            if not lenient:
                raise
            top_k = 5
        # Input-mode radio → the case's skip_query_builder / user_cypher.
        input_mode = f["input_mode"].value or "refined"
        cypher = None
        if input_mode == "direct_cypher":
            cypher = (f["user_cypher"].value or "").strip() or None
        return EvalCase(
            id=case_id,
            question=question,
            origin=f["origin"].value or "manual",
            category=(f["category"].value or "").strip(),
            notes=(f["notes"].value or "").strip(),
            expected_sources=_lines(f["expected_sources"].value),
            expected_chunks=_lines(f["expected_chunks"].value),
            required_keywords=_lines(f["required_keywords"].value),
            disallowed_keywords=_lines(f["disallowed_keywords"].value),
            expected_answer_points=_lines(f["expected_answer_points"].value),
            expected_entities=_lines(f["expected_entities"].value),
            expected_mode=f["expected_mode"].value or None,
            user_cypher=cypher,
            retrieval={
                "retrieval_mode": f["retrieval_mode"].value,
                "lancedb_search_mode": f["lancedb_search_mode"].value,
                "top_k": top_k,
                "num_candidates": _num(int, f["num_candidates"].value),
                "rrf_rank_constant": _num(int, f["rrf_rank_constant"].value),
                "mmr_lambda": float(f["mmr_lambda"].value),  # slider → float
                "use_mmr": bool(f["use_mmr"].value),
                "kg_max_rows": _num(int, f["kg_max_rows"].value),
                "skip_query_builder": input_mode == "direct_query",
                "direct_retrieval": bool(f["direct_retrieval"].value),
            },
        )

    # ---- CRUD -------------------------------------------------------------

    def _current_path(self) -> Path | None:
        # The displayed path is the source of truth; it and `_path` are set
        # together on load / New dataset.
        v = self.dataset_field.value if self.dataset_field else None
        if v:
            return Path(v)
        return self._path

    def _on_new(self, _e: ft.Event) -> None:
        self._selected = None
        self._clear_form()
        self._render_list()
        self._render_preview()
        self._set_status("new case — fill the form, then Add case")

    def _on_capture_from_search(self, _e: ft.Event) -> None:
        """Pre-fill a NEW case from the last Search result — the question +
        the retrieved doc_ids (deduped, order-preserved) as expected_sources,
        `origin=search`. The user reviews (keywords / answer points) + commits.

        This is the authoring-side of the 'capture from search' flow: no
        cross-view navigation — the user is already here to curate the set."""
        answer = getattr(self.app, "last_answer", None)
        query = getattr(self.app, "last_query", None)
        if answer is None or not query:
            self._set_status("No search result to capture — run a query in Search first.")
            return
        seen: set[str] = set()
        sources: list[str] = []
        for cs in getattr(answer, "chunk_sources", None) or []:
            doc_id = getattr(cs, "doc_id", None)
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                sources.append(doc_id)
        self._selected = None
        self._clear_form()
        self.f["question"].value = query
        self.f["expected_sources"].value = _text(sources)
        self.f["origin"].value = "search"
        self._render_list()
        self._render_preview()
        self._set_status(f"captured from search ({len(sources)} source(s)) — review, then Add case")

    def _selected_gen_model(self) -> str | None:
        """The chosen LLM model for case generation, or None to let the backend
        use its default (the cheap mode-classifier model)."""
        v = (self.gen_model_dropdown.value or "").strip() if self.gen_model_dropdown else ""
        return v or None

    def _selected_gen_temp(self) -> float:
        """The generation temperature from the slider (0–1)."""
        return float(self.gen_temp_slider.value) if self.gen_temp_slider is not None else 0.3

    def _on_temp_slide(self, _e: ft.Event | None = None) -> None:
        """Update the inline temperature value display as the slider drags."""
        if self.gen_temp_slider is not None and self.gen_temp_value_text is not None:
            self.gen_temp_value_text.value = f"{self.gen_temp_slider.value:.2f}"
            self.app.page.update()

    async def _on_generate_one(self, _e: ft.Event) -> None:
        """Draft ONE LLM candidate straight into the form for review — nothing
        is written until you Add case (unlike Generate multiple, which bulk-adds
        unreviewed). The candidate is flagged `origin=llm`."""
        from knowledge_agent.evaluation.generator import generate_from_corpus

        self._set_busy(self.gen_one_button, self.gen_one_spinner, True)
        self._set_status("drafting one candidate from the active corpus…")
        try:
            cases = await generate_from_corpus(
                1, model=self._selected_gen_model(), temperature=self._selected_gen_temp()
            )
        except Exception as exc:  # broad: provider / corpus / LLM errors → status
            self._set_busy(self.gen_one_button, self.gen_one_spinner, False)
            self._set_status(f"generation failed: {exc}")
            return
        self._set_busy(self.gen_one_button, self.gen_one_spinner, False)
        if not cases:
            self._set_status("no candidate generated (empty corpus or all passages too short)")
            return
        # Load it into the form as a NEW case for review — do NOT save.
        self._selected = None
        self._fill_form(cases[0])
        self._render_list()
        self._render_preview()
        self._set_status("generated one candidate (origin=llm) — review, then Add case")

    def _on_generate_multiple(self, _e: ft.Event) -> None:
        """Validate the count, then CONFIRM before bulk-drafting: these cost LLM
        calls and land straight in the dataset unreviewed, so ask first."""
        path = self._current_path()
        if path is None:
            self._set_status("Choose a dataset (Browse or New dataset) first.")
            return
        try:
            n = int((self.gen_count.value or "5").strip()) if self.gen_count else 5
        except ValueError:
            self._set_status("Count must be a whole number.")
            return
        if n <= 0:
            self._set_status("Count must be at least 1.")
            return

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        async def _confirm(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            await self._run_generate_multiple(n, path)

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Generate cases with the LLM?"),
            content=ft.Text(
                f"This will draft {n} case(s) with the LLM and add them straight to "
                f"{path.name} (origin=llm). You review, edit, or delete them afterward.",
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content=f"Generate {n}", on_click=_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    async def _run_generate_multiple(self, n: int, path: Path) -> None:
        """Bulk-draft `n` LLM candidates straight into the dataset (origin=llm)
        and save — no per-case review; edit/delete them afterward from the list.
        A spinner shows beside the button while the LLM calls run.

        `generator.generate_from_corpus` samples one passage per corpus doc and
        drafts a question + answer points + keywords via the active provider's
        model."""
        from knowledge_agent.evaluation.generator import generate_from_corpus
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        if self._is_shipped_dataset(path):
            self._set_status(
                "Packaged read-only dataset — use New dataset to make an editable copy."
            )
            return
        self._set_busy(self.gen_multi_button, self.gen_multi_spinner, True)
        self._set_status(f"generating {n} candidate case(s) from the active corpus…")
        try:
            cases = await generate_from_corpus(
                n, model=self._selected_gen_model(), temperature=self._selected_gen_temp()
            )
        except Exception as exc:  # broad: provider / corpus / LLM errors → status
            self._set_busy(self.gen_multi_button, self.gen_multi_spinner, False)
            self._set_status(f"generation failed: {exc}")
            return
        self._set_busy(self.gen_multi_button, self.gen_multi_spinner, False)

        if not cases:
            self._set_status("no candidates generated (empty corpus or all passages too short)")
            return

        if self._dataset is None:
            self._dataset = EvalDataset()
            if self.status_group is not None:
                self.status_group.value = self._dataset.status
        self._dataset.cases.extend(cases)
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status
            self._set_status(f"generated {len(cases)} but save failed: {exc}")
            return
        self._cases = self._dataset.cases
        self._path = path
        self._selected = None
        self._render_list()
        self._render_preview()
        self._set_status(
            f"generated {len(cases)} LLM candidate(s) — review each (origin=llm), keep/edit/delete"
        )

    def _is_shipped_dataset(self, path: Path) -> bool:
        """True if `path` is a packaged/shipped gold dataset (lives under the
        package datasets dir). Those are read-only in the GUI so editing the
        auto-loaded default can't silently overwrite the shipped file."""
        from knowledge_agent.evaluation.config import DEFAULT_DATASET_PATH

        try:
            path.resolve().relative_to(DEFAULT_DATASET_PATH.parent.resolve())
            return True
        except ValueError:
            return False

    def _on_save_case(self, _e: ft.Event) -> None:
        """Commit the form: append a new case (Add) or overwrite the selected
        one (Update), then persist. Wired to the right-column commit button."""
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        path = self._current_path()
        if path is None:
            self._set_status("Choose a dataset (Browse or New dataset) first.")
            return
        if self._is_shipped_dataset(path):
            self._set_status(
                "Packaged read-only dataset — use New dataset to make an editable copy."
            )
            return
        try:
            case = self._read_form()
        except Exception as exc:  # broad: validation / int-parse errors → status line
            self._set_status(f"invalid case: {exc}")
            return
        if self._dataset is None:
            self._dataset = EvalDataset()
        # Status edit rides along with any save (name is the filename now —
        # the human `name` header field was dropped from the UI).
        if self.status_group is not None:
            self._dataset.status = self.status_group.value or "draft"

        if self._selected is not None and 0 <= self._selected < len(self._dataset.cases):
            self._dataset.cases[self._selected] = case
            verb = "updated"
        else:
            self._dataset.cases.append(case)
            self._selected = len(self._dataset.cases) - 1
            verb = "added"
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"save failed: {exc}")
            return
        self._cases = self._dataset.cases
        self._path = path
        self._render_list()
        self._render_preview()
        self._set_status(f"{verb} — saved {len(self._cases)} case(s) to {path.name}")

    def _delete_case(self, idx: int) -> None:
        """Delete case `idx` (wired to each card's Delete button) and persist."""
        from knowledge_agent.evaluation.models import save_dataset

        if self._dataset is None or not (0 <= idx < len(self._dataset.cases)):
            return
        path = self._current_path()
        if path is None:
            self._set_status("No dataset path to save to.")
            return
        if self._is_shipped_dataset(path):
            self._set_status(
                "Packaged read-only dataset — use New dataset to make an editable copy."
            )
            return
        del self._dataset.cases[idx]
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"save failed: {exc}")
            return
        # Keep the form + selection sensible after the delete.
        if self._selected == idx:
            self._selected = None
            self._clear_form()
        elif self._selected is not None and self._selected > idx:
            self._selected -= 1
        self._cases = self._dataset.cases
        self._render_list()
        self._render_preview()
        self._set_status(f"deleted — {len(self._cases)} case(s) remain")

    # ---- helpers ----------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        if self.status is not None:
            self.status.value = msg
            self.app.page.update()

    def _set_busy(self, button: ft.Control | None, spinner: ft.Control | None, busy: bool) -> None:
        """Toggle a Generate button's disabled state and its spinner together."""
        if button is not None:
            button.disabled = busy
        if spinner is not None:
            spinner.visible = busy
        self.app.page.update()


def _list_field(hint: str, on_change=None) -> ft.TextField:
    return ft.TextField(
        hint_text=hint,
        multiline=True,
        min_lines=2,
        max_lines=5,
        expand=True,
        on_change=on_change,
    )


def _lines(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _text(values: list[str]) -> str:
    return "\n".join(values)


def _num_or_default(value: int | float | None, key: str) -> str:
    """A knob field's text: the case's value, or the standard default when the
    case leaves it unset — so the form is always pre-filled, never blank."""
    return str(value) if value is not None else str(DEFAULT_VALUES[key])
