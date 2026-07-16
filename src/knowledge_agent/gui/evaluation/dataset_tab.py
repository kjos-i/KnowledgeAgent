"""Evaluation → Dataset sub-tab — author + edit a gold dataset.

Left column = the authoring surface: choose the dataset file, pick how to seed
a case (blank / from the last Search / an LLM draft), and edit EVERY field of
the current case in a form grouped like the `EvalCase` schema (so it doubles as
documentation of what a case contains).

The form is backed by a **draft buffer** (602): a seed action replaces the
buffer with its output — **+ Case form** → one blank page, **From last search**
→ one page from the last Search, **LLM** → `nr. of cases` pages (1 → a page,
N → a batch). A batch shows a `< n/N >` pager + delete-X in the form header;
**Blank** clears to a fresh page. Seeding / editing an existing case first
warns if the form holds unsaved content.

Right column = two stacked sections:
  * **Commit** — **Add case(s)** appends every draft page at once; loading a
    dataset case flips it to **Update**. Nothing is written until you commit.
  * **Dataset cases** — the dataset's cases as cards; click one to load it
    into the form (commit becomes **Update**), or delete it on the card.

Dataset files are chosen by **Browse** (opens in the active corpus's folder)
or created by **New dataset** — no packaged-set dropdown. Persistence goes
through the backend `save_dataset` helper. Drafts never hit disk until
**Add case(s)** (origin=llm for the LLM ones).

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
    FRAME_BORDER_COLOR,
    LEFT_COLUMN_WIDTH,
    apply_input_box,
    centered_label,
    labeled_field,
    panel_box,
    panel_title,
    section_divider,
    section_title,
    sub_section_header,
    sub_section_title,
)
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui._widgets.retrieval_form import (
    DEFAULT_VALUES,
    RetrievalControls,
    apply_gray_out,
    build_input_mode_radios,
    build_mmr_slider,
    build_search_mode_radios,
    knobs_to_query_mode,
    query_mode_to_knobs,
    store_forced_by_mode,
)
from knowledge_agent.gui.evaluation._case_view import render_case_cards
from knowledge_agent.gui.settings.llm_tab import model_options
from knowledge_agent.llm_factory import parse_model_ref, supports_temperature

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

    from knowledge_agent.evaluation.models import (
        ConversationTurn,
        EvalCase,
        EvalDataset,
        RetrievalSettings,
    )
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_NONE = ""  # the expected_mode "(none)" sentinel (maps to None on read)
_MAX_GEN_CASES = 20  # upper bound on the LLM "nr. of cases" per generate (602)
_PREVIEW_DEBOUNCE_S = 0.3  # settle time before the live preview re-renders
_IDLE_STATUS = "No activity yet."  # the Progress line's default when nothing has happened
# Fixed caption column for the case-edit form, so every field's input starts at
# the same x (aligned left edges) instead of hugging its variable-width caption.
# Wide enough for the longest label ("expected_answer_points").
_FORM_LABEL_WIDTH = 190

# The exact prefixes of the bracketed status notes app.on_send appends to the
# chat history ("(Answered from N chunk…)", "(Retrieved N raw chunk…)"). They're
# UI chatter, not real turns, so a captured chat conversation drops them.
_STATUS_NOTE_PREFIXES = ("(Answered from", "(Retrieved ")


def _conversation_from_messages(messages: list[BaseMessage]) -> list[ConversationTurn]:
    """The user/assistant turns of a chat as `ConversationTurn`s — provenance for
    an `origin="chat"` case. Skips empty messages and the bracketed status notes
    (see `_STATUS_NOTE_PREFIXES`) that on_send adds to the history."""
    from knowledge_agent.evaluation.models import ConversationTurn

    role_by_type = {"human": "user", "ai": "assistant", "system": "system"}
    turns: list[ConversationTurn] = []
    for m in messages or []:
        content = getattr(m, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith(_STATUS_NOTE_PREFIXES):
            continue
        turns.append(
            ConversationTurn(
                role=role_by_type.get(getattr(m, "type", ""), "assistant"), content=content
            )
        )
    return turns


def _dedup_doc_ids(chunk_sources: list) -> list[str]:
    """The distinct doc_ids from an answer's chunk_sources, order-preserved —
    the gold `expected_sources` for a captured search/chat case."""
    seen: set[str] = set()
    out: list[str] = []
    for cs in chunk_sources or []:
        doc_id = getattr(cs, "doc_id", None)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


class DatasetTab:
    """Author + edit a gold dataset: form + live preview + case list."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.dataset_field: ft.TextField | None = None
        self.status_group: ft.RadioGroup | None = None
        # Freeze/unfreeze affordance (Run-tab style). The dataset's run settings
        # (metric groups + judge panel + gate thresholds) are edited on the Run
        # tab; here we only show + toggle the dataset-level frozen lock.
        self.unfreeze_button: ft.TextButton | None = None
        self.frozen_indicator: ft.Text | None = None
        self.case_list: ft.Column | None = None
        self.form: ft.Column | None = None
        self.status: ft.Text | None = None
        self.gen_count: ft.TextField | None = None
        # One "LLM" generate button (602): drafts `gen_count` cases into the form
        # buffer (was two buttons — Generate one / multiple).
        self.gen_button: ft.Control | None = None
        # Spinner shown beside the LLM button while its calls run.
        self.gen_spinner: ft.ProgressRing | None = None
        # LLM model + temperature used for case generation.
        self.gen_model_dropdown: ft.Dropdown | None = None
        self.gen_temp_slider: ft.Slider | None = None
        self.gen_temp_value_text: ft.Text | None = None
        # Draft-buffer pager (602): < n/N > + delete-X over the multi-case buffer,
        # plus a Blank button — all created in build().
        self.pager: ft.Row | None = None
        self.page_label: ft.Text | None = None
        self.page_prev: ft.IconButton | None = None
        self.page_next: ft.IconButton | None = None
        self.page_delete: ft.IconButton | None = None
        self.blank_button: ft.TextButton | None = None
        # Right column: two commit buttons. Only one is live at a time — Add for
        # a new case, Update for an existing one (Add is also grayed in suite mode).
        self.add_button: ft.Button | None = None
        self.update_button: ft.Button | None = None
        # Suite-mode widgets (601) — created in build(); referenced by handlers.
        self.generate_suite_check: ft.Checkbox | None = None
        self.suite_panel: ft.Container | None = None
        self.suite_members_list: ft.Column | None = None
        self.suite_name_field: ft.TextField | None = None
        self.suite_status: ft.Text | None = None
        # form field widgets (created in build)
        self.f: dict[str, ft.Control] = {}
        # "Query from chat" toggle: when on, a seeded case takes its QUESTION from
        # the last Search chat's distilled query (origin=chat) and the seeds
        # (From-search / LLM) only fill the rest.
        self.from_chat_check: ft.Checkbox | None = None
        # state
        self._dataset: EvalDataset | None = None
        self._cases: list[EvalCase] = []
        self._selected: int | None = None
        self._path: Path | None = None
        # Draft buffer (602): unsaved draft "pages" as raw form snapshots — a
        # blank/half-filled page isn't a valid EvalCase (id/question are required),
        # so we snapshot form state, not cases. In draft mode `_selected` is None
        # and the form shows `_drafts[_draft_index]`; editing an existing case is
        # the other mode (`_selected` set, buffer idle) with `_edit_baseline` the
        # snapshot taken on load for the unsaved-changes guard.
        self._drafts: list[dict] = []
        self._draft_index: int = 0
        self._edit_baseline: dict | None = None
        # Chat provenance carried onto an origin="chat" case (not editable form
        # fields): the captured conversation turns + the router model that
        # distilled the query. Set on capture-from-chat, restored on load.
        self._chat_conversation: list[ConversationTurn] = []
        self._chat_router_model: str | None = None
        # Pending debounced preview refresh (cancelled + replaced on each edit).
        self._preview_task: asyncio.Task | None = None
        # Suite mode: the knob-sets assembled for a Generate-suite run, each a
        # (label, RetrievalSettings) — from the premade dropdowns or the form.
        self._suite_members: list[tuple[str, RetrievalSettings]] = []

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.models import (
            CaseOrigin,
            DatasetStatus,
            RetrievalMode,
        )
        from knowledge_agent.evaluation.suite_gen import PRESET_GROUPS, STRATEGY_LABELS

        origins = list(typing.get_args(CaseOrigin))
        statuses = list(typing.get_args(DatasetStatus))
        modes = list(typing.get_args(RetrievalMode))

        # Dataset file: a read-only path display + Browse / New dataset. No
        # packaged-set dropdown (an editable dropdown read as a confusing plain
        # field). Browse opens an existing file; New dataset starts an empty one.
        self.dataset_field = ft.TextField(
            read_only=True,
            value="",
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
        # Status = the whole dataset's authoring state; radios side by side.
        # Dropping to draft clears any frozen lock (frozen requires final).
        self.status_group = ft.RadioGroup(
            value="draft",
            on_change=self._on_status_change,
            content=ft.Row([ft.Radio(value=s, label=s) for s in statuses], spacing=12),
        )
        new_case_button = ft.TextButton("Case form", icon=ft.Icons.ADD, on_click=self._on_new)
        capture_button = ft.TextButton(
            "From last search", icon=ft.Icons.SEARCH, on_click=self._on_capture_from_search
        )
        # One LLM button (602): drafts `gen_count` cases into the form buffer —
        # 1 → a single page, N → an N-page batch with the pager. Replaces the old
        # Generate-one / Generate-multiple pair.
        self.gen_button = ft.TextButton(
            "LLM", icon=ft.Icons.AUTO_AWESOME, on_click=self._on_generate
        )
        # Orthogonal to the three seeds: when on, a seeded case's QUESTION is the
        # last chat's distilled query (origin=chat) and the seeds only fill the
        # rest. A chat is one conversation → one case, so it forces the count to 1.
        self.from_chat_check = ft.Checkbox(
            label="Query from chat",
            value=False,
            tooltip=(
                "Take the question from the last Search chat's distilled query "
                "(origin=chat). From-search then only adds sources and LLM only "
                "writes the gold — neither overwrites the question. Run a "
                "Conversational search first."
            ),
            on_change=self._on_from_chat_toggled,
        )
        # nr. of cases the LLM button drafts (default 1 → a single page).
        self.gen_count = ft.TextField(value="1", width=80, dense=True)
        self.gen_spinner = ft.ProgressRing(visible=False, width=16, height=16, stroke_width=2)
        # Draft-buffer pager + delete-X (602): shown only for a multi-page batch.
        self.page_prev = ft.IconButton(
            ft.Icons.CHEVRON_LEFT, tooltip="Previous case", on_click=self._on_prev_page
        )
        self.page_label = ft.Text("", size=12)
        self.page_next = ft.IconButton(
            ft.Icons.CHEVRON_RIGHT, tooltip="Next case", on_click=self._on_next_page
        )
        self.page_delete = ft.IconButton(
            ft.Icons.CLOSE,
            icon_color=ft.Colors.RED_300,
            tooltip="Delete this case from the batch",
            on_click=self._on_delete_page,
        )
        self.pager = ft.Row(
            [self.page_prev, self.page_label, self.page_next, self.page_delete],
            spacing=0,
            tight=True,
            visible=False,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.blank_button = ft.TextButton(
            "Blank", icon=ft.Icons.CLEAR, tooltip="Clear the form", on_click=self._on_blank
        )
        self.gen_model_dropdown = ft.Dropdown(
            editable=True,  # pick a known model OR type one; required for LLM generation
            # Any installed provider's models (stored as a 'provider:model' ref
            # the generator dispatches per-ref).
            options=model_options(),
            hint_text="required — pick a model for LLM case generation",
            expand=True,
            on_blur=self._on_gen_model_changed,
        )
        # Temperature as a slider (0–1, 20 steps) — matches the LLM-tab temp
        # controls; value display trails the slider.
        self.gen_temp_slider = ft.Slider(
            value=0.3, min=0.0, max=1.0, divisions=20, on_change=self._on_temp_slide
        )
        self.gen_temp_value_text = ft.Text("0.30", size=12, color=ft.Colors.WHITE, width=42)
        # A sampling-free model (e.g. Opus 4.8) greys the temp slider out.
        self._sync_gen_temp_enabled()
        self.status = ft.Text(_IDLE_STATUS, size=12, italic=True, color=ft.Colors.GREY_600)

        # Frozen badge + Unfreeze, Run-tab style: the badge shows only when the
        # dataset is frozen; Unfreeze is ALWAYS shown, just disabled (greyed)
        # until the dataset is frozen. The run settings it locks are edited on
        # the Run tab.
        self.frozen_indicator = ft.Text(
            "\U0001f512 Run settings frozen",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )
        self.unfreeze_button = ft.TextButton(
            "Unfreeze",
            icon=ft.Icons.LOCK_OPEN,
            tooltip="Unlock the run settings so they can change again (asks to confirm)",
            on_click=self._on_unfreeze_clicked,
            disabled=True,  # always shown; enabled only when the dataset is frozen
        )

        # No own scroll / expand — the right column scrolls the preview + list
        # together (avoids a nested-scroll region).
        self.case_list = ft.Column(
            controls=[ft.Text("No cases yet — add one to get started.", italic=True)],
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
                    # Shared with Settings → Retrieval (SSOT). Eval has no chat
                    # router, so no 'Conversational' option — the harness runs the
                    # backend graph directly.
                    controls=build_input_mode_radios(),
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

        # Three sections mirroring the case model's hash structure:
        #   * Information — bookkeeping, NOT hashed (id / origin / category / notes).
        #   * Facts — the gold that forms `facts_hash`; SHARED across a suite's
        #     members (question + user_cypher + every expected_*).
        #   * Retrieval settings — the knobs that form `knob_hash`; they VARY
        #     across a suite's members (built by `_retrieval_group`).
        # Field captions stay as their exact schema keys for now (friendlier
        # labels come later); this only regroups + reorders the same fields.
        self.form = ft.Column(
            controls=[
                *self._group(
                    "Information",
                    ["id", "origin", "category", "notes"],
                    first=True,
                ),
                *self._group(
                    "Facts",
                    [
                        "question",
                        "user_cypher",
                        "expected_sources",
                        "expected_chunks",
                        "required_keywords",
                        "disallowed_keywords",
                        "expected_answer_points",
                        "expected_entities",
                        "expected_mode",
                    ],
                ),
                *self._retrieval_group(),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
        )

        # Right column: two commit buttons. Both route through _on_save_case; the
        # disabled state (set in _sync_commit_buttons) gates which applies, and
        # _selected drives append-vs-overwrite. Add is also grayed in suite mode.
        self.add_button = ft.Button("Add case(s)", icon=ft.Icons.ADD, on_click=self._on_save_case)
        self.update_button = ft.Button(
            "Update case", icon=ft.Icons.SAVE, on_click=self._on_save_case
        )
        refresh_button = ft.Button(
            "Refresh",
            tooltip="Reload the dataset from disk and refresh the case list",
            on_click=self._on_refresh,
        )

        # ----- Suite mode (601): assemble a knob-sweep suite -----
        # The checkbox reveals the suite panel + grays "Add case": in suite
        # mode you add the shared facts via "Add Information and Fact" (the form's
        # Info + Fact → a case) and assemble the knob-sets below. Generate then
        # clones those facts once per knob-set.
        self.generate_suite_check = ft.Checkbox(
            label="Generate suite", value=False, on_change=self._on_suite_toggled
        )
        self.add_info_fact_button = ft.Button(
            "Add case(s) info and fact",
            icon=ft.Icons.ADD,
            tooltip="Add the form's case(s) as shared suite facts (the whole batch at once)",
            on_click=self._on_save_case,
        )
        # Two premade dropdowns (Hybrid / KG groups) — picking one drops it into
        # the knob-set list (read-only), like adding a judge to the judge panel.
        # Each preset picker gets the app-standard outlined box (a bare Flet
        # Dropdown has no border) + a fixed width so the pickers and "Add from
        # form" stay grouped rather than stretching apart on a wide window (614).
        self.hybrid_preset_dd = apply_input_box(
            ft.Dropdown(
                options=[
                    ft.DropdownOption(key=k, text=STRATEGY_LABELS[k])
                    for k in PRESET_GROUPS["Hybrid"]
                ],
                on_select=lambda _e: self._add_preset_member(self.hybrid_preset_dd),
            )
        )
        self.kg_preset_dd = apply_input_box(
            ft.Dropdown(
                options=[
                    ft.DropdownOption(key=k, text=STRATEGY_LABELS[k]) for k in PRESET_GROUPS["KG"]
                ],
                on_select=lambda _e: self._add_preset_member(self.kg_preset_dd),
            )
        )
        self.add_from_form_button = ft.Button(
            "Add from form",
            icon=ft.Icons.ADD,
            tooltip="Add the left form's current retrieval settings as a knob-set",
            on_click=self._on_add_form_member,
        )
        self.suite_members_list = ft.Column(spacing=2)
        self.suite_name_field = ft.TextField(hint_text="e.g. escrt-sweep")
        self.suite_status = ft.Text("", size=12, color=ft.Colors.GREY_500, italic=True)
        self.suite_generate_button = ft.Button(
            "Generate suite", icon=ft.Icons.PLAY_ARROW, on_click=self._on_generate_suite_clicked
        )
        self.suite_panel = ft.Container(
            visible=False,
            padding=12,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=4,
            content=ft.Column(
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    # Suite name at the top, with "Add Info and Fact" (+ its help
                    # (i)) to the RIGHT of the field — that button seeds each
                    # shared suite fact from the form (613/614).
                    labeled_field(
                        "Suite name",
                        self.suite_name_field,
                        trailing=ft.Row(
                            [
                                self.add_info_fact_button,
                                info_icon(
                                    self.app,
                                    title="Case information and facts",
                                    text=(
                                        "The suite's facts are this dataset's "
                                        "cases. Add them from the form with 'Add "
                                        "case(s) info and fact'; Generate then "
                                        "clones every fact once per retrieval "
                                        "knob-set below."
                                    ),
                                ),
                            ],
                            spacing=4,
                            tight=True,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                    # Plain title, no rule, but keep the sub-section's top
                    # breathing room (613 removed the rule, not the spacing).
                    ft.Container(
                        content=sub_section_title("Retrieval settings"),
                        padding=ft.Padding.only(top=16),
                    ),
                    # Both preset pickers + "Add from form" share one row (614).
                    # The pickers expand + STRETCH so each dropdown fills its half
                    # and they narrow with the window; the button keeps its width
                    # on the right, vertically centered against the dropdowns.
                    ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text("Hybrid preset", size=12, color=ft.Colors.GREY_400),
                                    self.hybrid_preset_dd,
                                ],
                                spacing=2,
                                expand=1,
                                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                            ),
                            ft.Column(
                                [
                                    ft.Text("KG preset", size=12, color=ft.Colors.GREY_400),
                                    self.kg_preset_dd,
                                ],
                                spacing=2,
                                expand=1,
                                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                            ),
                            # A blank caption above the button mirrors the dropdown
                            # columns' caption + spacing, so CENTER lines the
                            # button's middle up with the dropdowns' middle rather
                            # than with the captions above them.
                            ft.Column(
                                [
                                    ft.Text(" ", size=12),
                                    self.add_from_form_button,
                                ],
                                spacing=2,
                            ),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    # The assembled knob-sets heading, with "Generate suite" on
                    # its right, and extra top space setting it off from the
                    # pickers above (614).
                    ft.Container(
                        padding=ft.Padding.only(top=16),
                        content=ft.Row(
                            [
                                sub_section_title("Selected settings for datasets in suite"),
                                self.suite_generate_button,
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                    self.suite_members_list,
                    self.suite_status,
                ],
            ),
        )

        # LEFT: dataset file + metadata + the case-edit form, in flat sections.
        left_pane = panel_box(
            ft.Column(
                [
                    # ============ Section: Evaluation cases ============
                    section_title("Evaluation cases"),
                    # Dataset field spans the full width; its file actions sit on
                    # the row beneath (three trailing buttons used to squash the
                    # field to a sliver).
                    labeled_field("Dataset", self.dataset_field),
                    # Browse / New dataset + the freeze affordance (Run-tab style)
                    # on ONE row: Unfreeze sits right of New dataset (always shown,
                    # greyed until frozen) and the frozen badge trails it (visible
                    # only when frozen). The run settings it locks live on the Run
                    # tab.
                    ft.Row(
                        [
                            browse_button,
                            new_dataset_button,
                            self.unfreeze_button,
                            self.frozen_indicator,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                    labeled_field("Status", self.status_group),
                    section_divider(),
                    # ============ Section: Progress ============
                    section_title("Progress"),
                    self.status,
                    section_divider(),
                    # ============ Section: Add cases ============
                    section_title("Add cases"),
                    # + Case form / From last search on the first row; the LLM
                    # button + its nr.-of-cases field on the row below (602: one
                    # LLM button; count 1 → a page, N → a batch).
                    sub_section_title("Generate new case by"),
                    self.from_chat_check,
                    ft.Row(
                        [new_case_button, capture_button],
                        wrap=True,
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [
                            self.gen_button,
                            ft.Text("nr. of cases:", size=14, color=ft.Colors.GREY_300),
                            self.gen_count,
                            self.gen_spinner,
                        ],
                        wrap=True,
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    sub_section_header("LLM model for case generation"),
                    labeled_field("Model", self.gen_model_dropdown),
                    labeled_field(
                        "Temperature", self.gen_temp_slider, trailing=self.gen_temp_value_text
                    ),
                    section_divider(),
                    # ============ Section: Per case form ============
                    # Bespoke treatment (by request): a centered, panel-size (16)
                    # "--- Per case form ---" title + bold sub-section titles,
                    # all wrapped in a thin bordered box.
                    ft.Container(
                        border=ft.Border.all(1, FRAME_BORDER_COLOR),
                        border_radius=4,
                        padding=12,
                        content=ft.Column(
                            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                            spacing=8,
                            controls=[
                                # Header: the batch pager (left, shown only for a
                                # multi-page batch), the centered title, and the
                                # Blank button on the right (602).
                                ft.Row(
                                    [
                                        self.pager,
                                        ft.Container(
                                            content=ft.Text(
                                                "--- Per case form ---",
                                                size=16,
                                                weight=ft.FontWeight.BOLD,
                                                text_align=ft.TextAlign.CENTER,
                                            ),
                                            expand=True,
                                        ),
                                        self.blank_button,
                                    ],
                                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                                ),
                                self.form,
                            ],
                        ),
                    ),
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            ),
            width=LEFT_COLUMN_WIDTH,
        )
        # RIGHT: commit buttons + the suite toggle/panel on top, the full case
        # list below. The form itself is the preview now (no separate card).
        right_pane = panel_box(
            ft.Column(
                [
                    # ============ Section: Commit / suite ============
                    # The suite toggle rides the commit-button row, kept on ONE
                    # line (no wrap) so it never drops below the buttons (613).
                    ft.Row(
                        [
                            refresh_button,
                            self.update_button,
                            self.add_button,
                            self.generate_suite_check,
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.suite_panel,
                    section_divider(),
                    # ============ Section: Dataset cases ============
                    section_title("Dataset cases"),
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
                    width=LEFT_COLUMN_WIDTH,
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

        # Start with one blank draft page (602) so the form is immediately
        # editable + committable; seed actions replace this buffer.
        self._load_drafts([self._blank_snapshot()])

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
            sub_section_header("Retrieval settings", bold=True),
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
            # Input mode is part of Retrieval settings, not its own section — a
            # plain bold label (no dividing rule) rather than a sub_section_header.
            sub_section_title("Input mode", bold=True),
            f["input_mode"],
            f["direct_retrieval"],
        ]
        self._sync_retrieval_gray_out()
        return rows

    def _sync_retrieval_gray_out(self) -> None:
        """Gray out the knobs the current mode/toggles can't use, via the shared
        `apply_gray_out` — the same single source Settings → Retrieval uses, so
        the two forms can't disagree. A grayed knob isn't fed to the search."""
        f = self.f
        # direct_cypher runs only on the graph → pin + lock the store to neo4j
        # (store_forced_by_mode), so the gray-out below reflects the real legs
        # and the form can't author a Cypher case that never reaches Neo4j.
        forced_store = store_forced_by_mode(f["input_mode"].value or "refined")
        if forced_store:
            f["retrieval_mode"].value = forced_store
        f["retrieval_mode"].disabled = forced_store is not None
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

    def _group(self, title: str, keys: list[str], *, first: bool = False) -> list[ft.Control]:
        # Each field group is headed by a sub-section header (thin rule + title)
        # so the groups read as distinct sub-sections. The first group hugs the
        # section title instead (no rule directly under it). Each field gets its
        # key as a caption via labeled_field (multiline boxes top-align their
        # caption); checkboxes keep their own built-in label.
        header: ft.Control = (
            ft.Container(
                padding=ft.Padding.only(top=12, bottom=4),
                content=sub_section_title(title, bold=True),
            )
            if first
            else sub_section_header(title, bold=True)
        )
        rows: list[ft.Control] = [header]
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
        """Sync the commit buttons + the draft-batch pager (the form itself is the
        preview). Exactly one commit button is live: Add case(s) for drafts,
        Update when editing an existing case; Add is ALSO grayed in suite mode
        (you build a suite via the panel). The pager shows only for a multi-page
        batch."""
        editing = self._selected is not None
        suite = self.generate_suite_check is not None and bool(self.generate_suite_check.value)
        if self.add_button is not None:
            self.add_button.disabled = editing or suite
        if self.update_button is not None:
            self.update_button.disabled = not editing
        self._render_pager(editing)

    def _render_pager(self, editing: bool) -> None:
        """Show < n/N > + the delete-X only for a multi-page draft batch (not
        while editing an existing case); disable the ends."""
        n = len(self._drafts)
        show = (not editing) and n > 1
        if self.pager is not None:
            self.pager.visible = show
        if show and self.page_label is not None:
            self.page_label.value = f"{self._draft_index + 1}/{n}"
            if self.page_prev is not None:
                self.page_prev.disabled = self._draft_index == 0
            if self.page_next is not None:
                self.page_next.disabled = self._draft_index >= n - 1

    def _on_refresh(self, _e: ft.Event | None = None) -> None:
        """Reload the current dataset from disk and refresh the case list — picks
        up external edits + regenerated suite files. No-op without a saved file."""
        if self._path is not None and self._path.exists():
            self._load(self._path)
        else:
            self._render_list()
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
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        folder = active_corpus_dir(self.app) or Path.cwd()
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
            self._set_status(f"created {path.name} — add cases, then Add case(s)")

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
        self._apply_frozen_ui()
        self._clear_form()
        self._render_list()
        self._render_preview()

    # ---- suite mode (601) -------------------------------------------------

    def _on_suite_toggled(self, _e: ft.Event) -> None:
        """Reveal/hide the suite panel; suite mode also grays "Add case"
        (you build the suite via the panel, not per-case Add)."""
        on = self.generate_suite_check is not None and bool(self.generate_suite_check.value)
        if self.suite_panel is not None:
            self.suite_panel.visible = on
        if on:
            self._render_suite_members()
        self._render_preview()  # re-syncs Add/Update (suite grays Add)
        self.app.page.update()

    def _add_preset_member(self, dropdown: ft.Dropdown) -> None:
        """A premade preset was picked → append it to the knob-set list (keyed by
        its slug so filenames stay clean; deduped), then clear the dropdown so
        the next pick fires again. The nice label is shown at render time."""
        from knowledge_agent.evaluation.suite_gen import STRATEGIES

        key = dropdown.value
        dropdown.value = None  # clear so the next pick re-fires
        if key in STRATEGIES and not any(k == key for k, _ in self._suite_members):
            self._suite_members.append((key, STRATEGIES[key].model_copy()))
            self._render_suite_members()
        self.app.page.update()

    def _on_add_form_member(self, _e: ft.Event) -> None:
        """Capture the left form's current retrieval settings as a custom knob-set."""
        try:
            settings = self._read_form(lenient=True).retrieval
        except Exception as exc:  # broad: a half-typed form shouldn't crash the add
            self._set_suite_status(f"could not read the form's retrieval settings: {exc}")
            return
        n = sum(1 for label, _ in self._suite_members if label.startswith("custom-")) + 1
        self._suite_members.append((f"custom-{n}", settings.model_copy()))
        self._render_suite_members()
        self.app.page.update()

    def _remove_suite_member(self, index: int) -> None:
        if 0 <= index < len(self._suite_members):
            del self._suite_members[index]
            self._render_suite_members()
            self.app.page.update()

    @staticmethod
    def _knob_summary(settings: RetrievalSettings) -> str:
        """A COMPLETE read-only description of a knob-set — every value it pins
        for the legs it runs (incl. the default numbers), so two presets are
        fully distinguishable on their list line."""
        parts = [f"mode={settings.retrieval_mode}", f"top_k={settings.top_k}"]
        if settings.retrieval_mode != "neo4j_only":  # LanceDB leg runs
            parts.append(f"search={settings.lancedb_search_mode}")
            if settings.num_candidates is not None:
                parts.append(f"num_candidates={settings.num_candidates}")
            if settings.rrf_rank_constant is not None:
                parts.append(f"rrf={settings.rrf_rank_constant}")
            parts.append(f"mmr={'on' if settings.use_mmr else 'off'}")
            if settings.use_mmr and settings.mmr_lambda is not None:
                parts.append(f"mmr_lambda={settings.mmr_lambda}")
        if settings.kg_max_rows is not None:  # Neo4j leg runs
            parts.append(f"kg_max_rows={settings.kg_max_rows}")
        if settings.skip_query_builder:
            parts.append("skip_query_builder")
        if settings.direct_retrieval:
            parts.append("direct_retrieval")
        return " · ".join(parts)

    def _render_suite_members(self) -> None:
        """Re-render the knob-set list — one line each: label + read-only settings
        summary + remove (mirrors the judge-panel judge list). A preset's key is
        shown via its nice `STRATEGY_LABELS` text; a custom knob-set as-is."""
        from knowledge_agent.evaluation.suite_gen import STRATEGY_LABELS

        if self.suite_members_list is None:
            return
        if not self._suite_members:
            self.suite_members_list.controls = [
                ft.Text(
                    "No knob-sets yet — pick a preset or add one from the form.",
                    size=12,
                    italic=True,
                    color=ft.Colors.GREY_500,
                )
            ]
            return
        self.suite_members_list.controls = [
            ft.Row(
                [
                    ft.Icon(ft.Icons.CIRCLE, size=6, color=ft.Colors.GREY_500),
                    ft.Text(STRATEGY_LABELS.get(label, label), size=12, color=ft.Colors.GREY_200),
                    ft.Text(
                        self._knob_summary(settings),
                        size=12,
                        color=ft.Colors.GREY_500,
                        italic=True,
                        expand=True,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        icon_size=16,
                        tooltip="Remove this knob-set",
                        on_click=lambda _e, i=idx: self._remove_suite_member(i),
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=6,
            )
            for idx, (label, settings) in enumerate(self._suite_members)
        ]

    def _set_suite_status(self, msg: str) -> None:
        if self.suite_status is not None:
            self.suite_status.value = msg
            self.app.page.update()

    def _on_generate_suite_clicked(self, _e: ft.Event) -> None:
        """Generate the suite: clone the dataset's cases (the shared facts) once
        per assembled knob-set, writing one file each to the corpus folder."""
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        if self._dataset is None or not self._dataset.cases:
            self._set_suite_status("Add at least one fact first (Add Information and Fact).")
            return
        suite = (self.suite_name_field.value or "").strip() if self.suite_name_field else ""
        if not suite:
            self._set_suite_status("Enter a suite name.")
            return
        if not self._suite_members:
            self._set_suite_status("Add at least one knob-set (a preset or from the form).")
            return
        folder = active_corpus_dir(self.app) or (self._path.parent if self._path else Path.cwd())
        try:
            written = self._write_suite(self._dataset, suite, list(self._suite_members), folder)
        except Exception as exc:  # broad: bad input / I-O → inline suite status
            self._set_suite_status(f"could not generate: {exc}")
            return
        self._set_suite_status(f"generated {len(written)} file(s): {', '.join(written)}")
        self._set_status(f"generated suite “{suite}” — {len(written)} files")

    def _write_suite(
        self,
        master: EvalDataset,
        suite_name: str,
        members: list[tuple[str, RetrievalSettings]],
        folder: Path,
    ) -> list[str]:
        """Generate + write the suite files to `folder`; return the written
        filenames. `members` = (label, knob-set) pairs — premade presets and/or
        custom form knob-sets. Raises on bad input / I-O (the caller surfaces it)."""
        from knowledge_agent.evaluation.models import save_dataset
        from knowledge_agent.evaluation.suite_gen import generate_suite

        written: list[str] = []
        for stem, ds in generate_suite(master, suite_name, members):
            path = folder / f"{stem}.json"
            save_dataset(ds, path)
            written.append(path.name)
        return written

    def _load(self, path: Path, *, refresh_page: bool = True) -> None:
        from knowledge_agent.evaluation.models import load_dataset

        try:
            ds = load_dataset(path)
        except Exception as exc:  # broad: surface any parse/validation error in-line
            self._write_status(f"could not load: {exc}")
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
        self._apply_frozen_ui()
        self._clear_form()
        self._render_list()
        self._render_preview()
        self._write_status(f"{len(self._cases)} case(s)")
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
            empty_hint="No cases yet — add one to get started.",
        )

    def _select(self, idx: int) -> None:
        """Load an existing case for editing — guarded: warn first if the form
        holds unsaved drafts or edits."""
        self._guarded(lambda: self._do_select(idx))

    def _do_select(self, idx: int) -> None:
        self._selected = idx
        self._drafts = []
        self._draft_index = 0
        self._fill_form(self._cases[idx])
        self._edit_baseline = self._snapshot_form()  # baseline for the unsaved guard
        self._render_list()
        self._render_preview()
        self.app.page.update()

    def _on_cancel_edit(self, _idx: int | None = None) -> None:
        """Cancel editing (the selected card's Cancel button): back to a fresh
        blank draft page."""
        self._load_drafts([self._blank_snapshot()])
        self._set_status("edit cancelled")

    # ---- draft buffer (602) ----------------------------------------------
    # The form is backed by a buffer of unsaved draft "pages" (raw form
    # snapshots — a blank/half-filled page is not a valid EvalCase). Seed
    # actions replace the buffer; the pager moves between pages; Add case(s)
    # commits the whole buffer. Editing an existing case is the other mode
    # (`_selected` set, buffer idle).

    _CONTENT_FIELDS = (
        "id",
        "question",
        "expected_sources",
        "expected_chunks",
        "required_keywords",
        "disallowed_keywords",
        "expected_answer_points",
        "expected_entities",
        "user_cypher",
        "notes",
        "category",
    )

    def _snapshot_form(self) -> dict:
        """Capture the whole form as a raw snapshot (form state, not a case)."""
        return {
            "f": {k: v.value for k, v in self.f.items()},
            "chat_conversation": list(self._chat_conversation),
            "chat_router_model": self._chat_router_model,
            "mmr": float(self.f["mmr_lambda"].value),
        }

    def _restore_form(self, snap: dict) -> None:
        """Load a form snapshot back into the widgets + the derived controls."""
        for key, val in snap["f"].items():
            if key in self.f:
                self.f[key].value = val
        self._chat_conversation = list(snap["chat_conversation"])
        self._chat_router_model = snap["chat_router_model"]
        self._mmr_value_text.value = f"{float(snap['mmr']):.2f}"
        if self.from_chat_check is not None:
            self.from_chat_check.value = (self.f["origin"].value or "") == "chat"
        self._sync_gen_count_for_chat()
        self._sync_retrieval_gray_out()

    def _blank_snapshot(self) -> dict:
        """A snapshot of the blank + default form. Clears the live form as a side
        effect; every caller then shows a page, so the form is re-rendered."""
        self._clear_form()
        return self._snapshot_form()

    def _snapshots_from_cases(self, cases: list[EvalCase]) -> list[dict]:
        """Turn generated cases into draft-page snapshots via the case→form map
        (so the round-trip matches editing). Leaves the form on the last case;
        the caller shows page 0."""
        snaps: list[dict] = []
        for case in cases:
            self._fill_form(case)
            snaps.append(self._snapshot_form())
        return snaps

    def _load_drafts(self, snaps: list[dict]) -> None:
        """Replace the draft buffer and show its first page (draft mode)."""
        self._selected = None
        self._edit_baseline = None
        self._drafts = list(snaps)
        self._draft_index = 0
        if self._drafts:
            self._restore_form(self._drafts[0])
        else:
            self._clear_form()
        self._render_list()
        self._render_preview()

    def _capture_page(self) -> None:
        """Sync the live form back into the current draft page (draft mode)."""
        if self._selected is None and 0 <= self._draft_index < len(self._drafts):
            self._drafts[self._draft_index] = self._snapshot_form()

    def _show_page(self, i: int) -> None:
        if not (0 <= i < len(self._drafts)):
            return
        self._capture_page()
        self._draft_index = i
        self._restore_form(self._drafts[i])
        self._render_preview()
        self.app.page.update()

    def _on_prev_page(self, _e: ft.Event) -> None:
        self._show_page(self._draft_index - 1)

    def _on_next_page(self, _e: ft.Event) -> None:
        self._show_page(self._draft_index + 1)

    def _on_delete_page(self, _e: ft.Event) -> None:
        """Drop the current page from the batch (the X by the pager). Only shown
        for a multi-page batch, so the buffer never empties here."""
        if self._selected is not None or not self._drafts:
            return
        del self._drafts[self._draft_index]
        self._draft_index = min(self._draft_index, len(self._drafts) - 1)
        if self._drafts:
            self._restore_form(self._drafts[self._draft_index])
        else:
            self._clear_form()
        self._render_preview()
        self._set_status("removed a case from the batch")

    @staticmethod
    def _snap_has_content(snap: dict) -> bool:
        f = snap.get("f", {})
        return any((f.get(k) or "").strip() for k in DatasetTab._CONTENT_FIELDS)

    def _has_unsaved(self) -> bool:
        """True when the form holds work a reload/reseed would lose — unsaved
        edits to a loaded case, or any draft page with real content. A pristine
        blank page counts as empty."""
        if self._selected is not None:
            return self._edit_baseline is not None and self._snapshot_form() != self._edit_baseline
        self._capture_page()
        return any(self._snap_has_content(s) for s in self._drafts)

    def _discard_dialog(self, on_confirm) -> None:
        """Show the 'discard unsaved content?' warning; `on_confirm` is the
        confirm button's on_click (sync or async — Flet awaits async on_click)."""

        def _cancel(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Discard unsaved content?"),
            content=ft.Text("The form has unsaved content that will be lost. Continue?", size=12),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button("Discard & continue", on_click=on_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    def _guarded(self, proceed) -> None:
        """Run `proceed` now, or warn first when there's unsaved content."""
        if not self._has_unsaved():
            proceed()
            return

        def _confirm(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            proceed()

        self._discard_dialog(_confirm)

    def _on_blank(self, _e: ft.Event) -> None:
        """Blank button: wipe the form to a fresh blank page (guarded)."""
        self._guarded(self._do_blank)

    def _do_blank(self) -> None:
        self._load_drafts([self._blank_snapshot()])
        self._set_status("form cleared")

    def _sync_gen_count_for_chat(self) -> None:
        """A chat is one conversation → one case: with 'Query from chat' on, pin
        the nr.-of-cases field to 1 and disable it."""
        on = bool(self.from_chat_check and self.from_chat_check.value)
        if self.gen_count is not None:
            if on:
                self.gen_count.value = "1"
            self.gen_count.disabled = on

    # ---- form <-> case ----------------------------------------------------

    def _set_retrieval_fields(
        self, rs: RetrievalSettings, *, user_cypher: str | None = None
    ) -> None:
        """Populate the form's retrieval knobs from a `RetrievalSettings` — shared
        by load-a-case (`_fill_form`) and pin-what-the-search-ran-under
        (`_pin_retrieval`) so the two can't drift. Nullable knobs fall back to the
        standard default display (never blank). The input-mode radio is derived
        from skip_query_builder + user_cypher (inverse of query_mode_to_knobs)."""
        f = self.f
        f["retrieval_mode"].value = rs.retrieval_mode
        f["lancedb_search_mode"].value = rs.lancedb_search_mode
        f["top_k"].value = str(rs.top_k)
        f["num_candidates"].value = _num_or_default(rs.num_candidates, "num_candidates")
        f["rrf_rank_constant"].value = _num_or_default(rs.rrf_rank_constant, "rrf_rank_constant")
        mmr_val = (
            rs.mmr_lambda if rs.mmr_lambda is not None else float(DEFAULT_VALUES["mmr_lambda"])
        )
        f["mmr_lambda"].value = mmr_val
        self._mmr_value_text.value = f"{mmr_val:.2f}"
        f["use_mmr"].value = rs.use_mmr
        f["kg_max_rows"].value = _num_or_default(rs.kg_max_rows, "kg_max_rows")
        f["input_mode"].value = knobs_to_query_mode(
            skip_query_builder=rs.skip_query_builder, user_cypher=user_cypher
        )
        f["direct_retrieval"].value = rs.direct_retrieval
        self._sync_retrieval_gray_out()

    def _pin_retrieval(self, rs: RetrievalSettings | None) -> None:
        """Pin the form's retrieval knobs to a real search's settings (the app's
        `last_retrieval` snapshot) instead of the blank-form defaults, so a
        captured case reproduces the search it came from. No-op when None."""
        if rs is not None:
            self._set_retrieval_fields(rs)

    def _fill_form(self, case: EvalCase) -> None:
        f = self.f
        f["id"].value = case.id
        f["question"].value = case.question
        f["origin"].value = case.origin
        f["category"].value = case.category
        f["notes"].value = case.notes
        self._set_retrieval_fields(case.retrieval, user_cypher=case.user_cypher)
        f["expected_sources"].value = _text(case.expected_sources)
        f["expected_chunks"].value = _text(case.expected_chunks)
        f["required_keywords"].value = _text(case.required_keywords)
        f["disallowed_keywords"].value = _text(case.disallowed_keywords)
        f["expected_answer_points"].value = _text(case.expected_answer_points)
        f["expected_entities"].value = _text(case.expected_entities)
        f["expected_mode"].value = case.expected_mode or _NONE
        f["user_cypher"].value = case.user_cypher or ""
        # Restore chat provenance so editing an origin="chat" case keeps it, and
        # reflect it in the checkbox (+ the bulk-generator lock it drives).
        self._chat_conversation = list(case.source_conversation)
        self._chat_router_model = case.chat_router_model
        if self.from_chat_check is not None:
            self.from_chat_check.value = case.origin == "chat"
            self._sync_gen_count_for_chat()

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
        # Drop any chat provenance — a fresh/blank case is not chat-sourced until
        # capture-from-chat sets it again. (origin resets to "manual" above.)
        self._chat_conversation = []
        self._chat_router_model = None
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
        # Input-mode radio → the case's skip_query_builder / user_cypher, via the
        # shared mapping (SSOT with the chat dispatch). direct_cypher pins the
        # store to neo4j (store_forced_by_mode) — Cypher only runs on the graph.
        input_mode = f["input_mode"].value or "refined"
        knobs = query_mode_to_knobs(input_mode, cypher_text=f["user_cypher"].value)
        retrieval_mode = store_forced_by_mode(input_mode) or f["retrieval_mode"].value
        origin = f["origin"].value or "manual"
        return EvalCase(
            id=case_id,
            question=question,
            origin=origin,
            category=(f["category"].value or "").strip(),
            notes=(f["notes"].value or "").strip(),
            expected_sources=_lines(f["expected_sources"].value),
            expected_chunks=_lines(f["expected_chunks"].value),
            required_keywords=_lines(f["required_keywords"].value),
            disallowed_keywords=_lines(f["disallowed_keywords"].value),
            expected_answer_points=_lines(f["expected_answer_points"].value),
            expected_entities=_lines(f["expected_entities"].value),
            expected_mode=f["expected_mode"].value or None,
            user_cypher=knobs["user_cypher"],
            # Chat provenance rides along ONLY for origin="chat" (ignored by
            # scoring, like `notes`; see EvalCase.source_conversation).
            source_conversation=(self._chat_conversation if origin == "chat" else []),
            chat_router_model=(self._chat_router_model if origin == "chat" else None),
            retrieval={
                "retrieval_mode": retrieval_mode,
                "lancedb_search_mode": f["lancedb_search_mode"].value,
                "top_k": top_k,
                "num_candidates": _num(int, f["num_candidates"].value),
                "rrf_rank_constant": _num(int, f["rrf_rank_constant"].value),
                "mmr_lambda": float(f["mmr_lambda"].value),  # slider → float
                "use_mmr": bool(f["use_mmr"].value),
                "kg_max_rows": _num(int, f["kg_max_rows"].value),
                "skip_query_builder": knobs["skip_query_builder"],
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

    def _stamp_header(self) -> None:
        """Copy the dataset-level header control (status) onto the in-memory
        dataset so the next `save_dataset` persists it. Called by every save
        path — status rides along with any dataset write, the same way a case
        edit does. The recipe (metric groups / judge panel / gate thresholds)
        is edited on the Run tab; here it's left exactly as loaded, so a save
        from this tab preserves it."""
        if self._dataset is None:
            return
        if self.status_group is not None:
            self._dataset.status = self.status_group.value or "draft"
        # Invariant: only a final dataset can be frozen.
        if self._dataset.status != "final":
            self._dataset.frozen = False

    def _apply_frozen_ui(self) -> None:
        """Sync the freeze affordance to the dataset's frozen flag (Run-tab
        style): the badge shows only when frozen, and Unfreeze is always shown
        but disabled until frozen. The run settings it locks live on the Run
        tab; cases stay editable regardless."""
        frozen = bool(self._dataset is not None and self._dataset.frozen)
        if self.unfreeze_button is not None:
            self.unfreeze_button.disabled = not frozen
        if self.frozen_indicator is not None:
            self.frozen_indicator.visible = frozen

    def _on_status_change(self, _e: ft.Event) -> None:
        """Dropping the status below 'final' clears any frozen lock (invariant:
        only a final dataset can be frozen) and re-enables the recipe. Persists
        on the next save via `_stamp_header`."""
        if self._dataset is None or self.status_group is None:
            return
        if self.status_group.value != "final" and self._dataset.frozen:
            self._dataset.frozen = False
            self._apply_frozen_ui()
        self.app.page.update()

    def _on_unfreeze_clicked(self, _e: ft.Event) -> None:
        """Unfreeze — a confirmed action that clears the frozen flag so the
        recipe can change again."""
        if self._dataset is None:
            return
        from knowledge_agent.gui.evaluation._common import confirm_dialog

        confirm_dialog(
            self.app,
            title="Unfreeze run settings?",
            message=(
                "This clears the dataset's frozen lock so its run settings can "
                "be changed again on the Run tab."
            ),
            confirm_label="Unfreeze",
            on_confirm=self._do_unfreeze,
        )

    def _do_unfreeze(self) -> None:
        from knowledge_agent.evaluation.models import save_dataset

        path = self._current_path()
        if self._dataset is None or path is None:
            return
        self._dataset.frozen = False
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"could not unfreeze: {exc}")
            return
        self._apply_frozen_ui()
        self._set_status("Unfroze — run settings can change again (edit them on the Run tab).")
        self.app.page.update()

    def _on_new(self, _e: ft.Event) -> None:
        """+ Case form: reset to a fresh blank draft page (guarded)."""
        self._guarded(self._do_new)

    def _do_new(self) -> None:
        """Reset the buffer to one fresh blank page. With 'Query from chat' on it
        starts with the chat's distilled question; revert the toggle if there's
        no chat to draw from."""
        self._load_drafts([self._blank_snapshot()])
        if self.from_chat_check and self.from_chat_check.value:
            if self._chat_available():
                self._apply_chat_source()
                self._capture_page()
            else:
                self.from_chat_check.value = False
                self._sync_gen_count_for_chat()
        msg = (
            "new chat-sourced case — question from the last chat; add the gold, then Add case(s)"
            if (self.f["origin"].value or "") == "chat"
            else "new case — fill the form, then Add case(s)"
        )
        self._set_status(msg)

    def _chat_available(self) -> bool:
        """True when the last Search send was a conversational chat that produced
        a distilled query to seed a case from."""
        return bool(getattr(self.app, "last_search_query", None))

    def _apply_chat_source(self) -> None:
        """Fill the question + `origin=chat` + pinned retrieval + stored
        conversation from the last chat. Leaves the gold (sources / answer points)
        for a seed or manual entry. No-op when there's no chat query."""
        query = getattr(self.app, "last_search_query", None)
        if not query:
            return
        self.f["question"].value = query
        self.f["origin"].value = "chat"
        self._pin_retrieval(getattr(self.app, "last_retrieval", None))
        self._chat_conversation = _conversation_from_messages(self.app.messages)
        self._chat_router_model = getattr(self.app.gui_config, "chat_router_model", None)

    def _on_from_chat_toggled(self, _e: ft.Event) -> None:
        """Turn 'Query from chat' on/off. On: seed the question from the last
        chat's distilled query (warn + revert if there's no chat yet) and pin the
        nr.-of-cases to 1 (a chat is one conversation → one case). Off: drop the
        chat provenance and reset origin to manual."""
        on = bool(self.from_chat_check and self.from_chat_check.value)
        if on and not self._chat_available():
            self.from_chat_check.value = False
            self._set_status("No chat query yet — run a Conversational search in Search first.")
            self.app.page.update()
            return
        self._sync_gen_count_for_chat()
        if on:
            self._apply_chat_source()
        else:
            self._chat_conversation = []
            self._chat_router_model = None
            if (self.f["origin"].value or "") == "chat":
                self.f["origin"].value = "manual"
        self._render_preview()
        self.app.page.update()

    def _on_capture_from_search(self, _e: ft.Event) -> None:
        """Pre-fill a NEW case from the last Search result — the question + the
        retrieved doc_ids (deduped, order-preserved) as expected_sources, pinning
        the retrieval settings the search ACTUALLY ran under (not form defaults).

        With 'Query from chat' on, the question is the router's DISTILLED query
        and `origin=chat` (+ the conversation is stored); otherwise the raw query
        and `origin=search`. Guarded (warn if the form holds unsaved content) —
        but only after confirming a result exists, so we never warn then no-op."""
        app = self.app
        from_chat = bool(self.from_chat_check and self.from_chat_check.value)
        answer = getattr(app, "last_answer", None)
        query = getattr(app, "last_search_query" if from_chat else "last_query", None)
        if answer is None or not query:
            need = "a Conversational search" if from_chat else "a query"
            self._set_status(f"No search result to capture — run {need} in Search first.")
            return
        self._guarded(self._do_capture_from_search)

    def _do_capture_from_search(self) -> None:
        app = self.app
        from_chat = bool(self.from_chat_check and self.from_chat_check.value)
        answer = getattr(app, "last_answer", None)
        query = getattr(app, "last_search_query" if from_chat else "last_query", None)
        if answer is None or not query:
            return
        sources = _dedup_doc_ids(getattr(answer, "chunk_sources", None) or [])
        self._load_drafts([self._blank_snapshot()])  # fresh single page
        self.f["question"].value = query
        self.f["expected_sources"].value = _text(sources)
        self._pin_retrieval(getattr(app, "last_retrieval", None))
        if from_chat:
            self.f["origin"].value = "chat"
            self._chat_conversation = _conversation_from_messages(app.messages)
            self._chat_router_model = getattr(app.gui_config, "chat_router_model", None)
        else:
            self.f["origin"].value = "search"
        if self.from_chat_check is not None:
            self.from_chat_check.value = from_chat
        self._capture_page()  # fold the capture into the single draft page
        self._render_preview()
        kind = "chat" if from_chat else "search"
        self._set_status(
            f"captured from {kind} ({len(sources)} source(s)) — review, then Add case(s)"
        )

    def _selected_gen_model(self) -> str | None:
        """The chosen LLM model for case generation, or None to let the backend
        use its default (the cheap mode-classifier model)."""
        v = (self.gen_model_dropdown.value or "").strip() if self.gen_model_dropdown else ""
        return v or None

    def _selected_gen_temp(self) -> float:
        """The generation temperature from the slider (0–1)."""
        return float(self.gen_temp_slider.value) if self.gen_temp_slider is not None else 0.3

    def _require_gen_model(self) -> bool:
        """True when an LLM model for case generation is chosen. When blank,
        pop an info dialog (LLM case generation must not silently fall back to a
        default model) and return False so the caller aborts."""
        if self._selected_gen_model() is not None:
            return True

        def _close(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Pick a model first"),
            content=ft.Text(
                "Choose an LLM model under “LLM model for case generation” before "
                "generating cases — LLM generation has no default model.",
                size=12,
            ),
            actions=[ft.TextButton("OK", on_click=_close)],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()
        return False

    def _sync_gen_temp_enabled(self) -> None:
        """Grey out the generation temperature slider when the selected model
        doesn't accept a temperature (e.g. Opus 4.8). The backend omits
        temperature for those models regardless; this makes it visible. An
        empty selection keeps the slider active."""
        if self.gen_temp_slider is None:
            return
        raw = (self.gen_model_dropdown.value or "").strip() if self.gen_model_dropdown else ""
        provider, model = parse_model_ref(raw)
        if provider is None:  # bare / legacy / empty → global fallback provider
            provider = getattr(self.app.gui_config, "llm_provider", "")
        takes_temp = supports_temperature(provider, model)
        self.gen_temp_slider.disabled = not takes_temp
        self.gen_temp_slider.tooltip = None if takes_temp else f"{model} ignores temperature"
        if self.gen_temp_value_text is not None:
            self.gen_temp_value_text.color = ft.Colors.WHITE if takes_temp else ft.Colors.WHITE_38

    def _on_gen_model_changed(self, _e: ft.Event | None = None) -> None:
        """Re-evaluate temp-slider greying when the generation model changes."""
        self._sync_gen_temp_enabled()
        self.app.page.update()

    def _on_temp_slide(self, _e: ft.Event | None = None) -> None:
        """Update the inline temperature value display as the slider drags."""
        if self.gen_temp_slider is not None and self.gen_temp_value_text is not None:
            self.gen_temp_value_text.value = f"{self.gen_temp_slider.value:.2f}"
            self.app.page.update()

    async def _on_generate(self, _e: ft.Event) -> None:
        """LLM button: draft `nr. of cases` candidates into the form buffer for
        review — nothing is written until Add case(s). 1 → a single page, N → an
        N-page batch. With 'Query from chat' on it drafts ONE case (the LLM writes
        only the gold for the chat's distilled question, origin=chat); otherwise
        it invents whole cases from corpus passages (origin=llm). Guarded: warn
        first if the form holds unsaved content."""
        if not self._require_gen_model():
            return
        from_chat = bool(self.from_chat_check and self.from_chat_check.value)
        n = 1 if from_chat else self._gen_count_value()
        if n is None:
            return
        if not self._has_unsaved():
            await self._do_generate(n, from_chat)
            return

        async def _confirm(_ev: ft.Event) -> None:
            self.app.page.pop_dialog()
            await self._do_generate(n, from_chat)

        self._discard_dialog(_confirm)

    def _gen_count_value(self) -> int | None:
        """Parse + bound the nr.-of-cases field (1..`_MAX_GEN_CASES`); on bad
        input set the status line and return None."""
        try:
            n = int((self.gen_count.value or "1").strip()) if self.gen_count else 1
        except ValueError:
            self._set_status("Count must be a whole number.")
            return None
        if n <= 0:
            self._set_status("Count must be at least 1.")
            return None
        if n > _MAX_GEN_CASES:
            self._set_status(f"Count is capped at {_MAX_GEN_CASES}.")
            return _MAX_GEN_CASES
        return n

    async def _do_generate(self, n: int, from_chat: bool) -> None:
        """Draft `n` cases and load them into the form buffer for review (nothing
        saved). from_chat routes to the single gold-for-the-chat path."""
        if from_chat:
            await self._generate_gold_from_chat()
            return
        from knowledge_agent.evaluation.generator import (
            EvalGenerationConnectionError,
            generate_from_corpus,
        )

        self._set_busy(self.gen_button, self.gen_spinner, True)
        self._set_status(f"drafting {n} candidate case(s) from the active corpus…")
        try:
            cases = await generate_from_corpus(
                n, model=self._selected_gen_model(), temperature=self._selected_gen_temp()
            )
        except EvalGenerationConnectionError as exc:
            # Network/connection failure — show the clear, retryable message.
            self._set_busy(self.gen_button, self.gen_spinner, False)
            self._set_status(str(exc))
            return
        except Exception as exc:  # broad: provider / corpus / LLM errors → status
            self._set_busy(self.gen_button, self.gen_spinner, False)
            self._set_status(f"generation failed: {exc}")
            return
        self._set_busy(self.gen_button, self.gen_spinner, False)
        if not cases:
            self._set_status("no candidate generated (empty corpus or all passages too short)")
            return
        # Load the batch into the form buffer for review — nothing saved yet.
        self._load_drafts(self._snapshots_from_cases(cases))
        shortfall = ""
        if len(cases) < n:
            shortfall = (
                f" of {n} requested (one case per document — the corpus has "
                "fewer usable docs, or a passage was skipped)"
            )
        tail = (
            "review the batch, then Add case(s)" if len(cases) > 1 else "review, then Add case(s)"
        )
        self._set_status(f"drafted {len(cases)} LLM candidate(s){shortfall} (origin=llm) — {tail}")

    async def _generate_gold_from_chat(self) -> None:
        """The 'Query from chat' + LLM path: keep the chat's distilled query as the
        question, and have the LLM write the gold (answer points + keywords) FOR it
        from the chat's own retrieved passages (not a fresh corpus sample, and not
        the agent's own answer — that would be circular). origin=chat."""
        from knowledge_agent.evaluation.generator import (
            EvalGenerationConnectionError,
            generate_gold_for_question,
            passages_from_sources,
        )

        query = getattr(self.app, "last_search_query", None)
        answer = getattr(self.app, "last_answer", None)
        if not query:
            self._set_status("No chat query — run a Conversational search in Search first.")
            return
        self._set_busy(self.gen_button, self.gen_spinner, True)
        self._set_status("writing gold for the chat question…")
        try:
            passages = await passages_from_sources(getattr(answer, "chunk_sources", None) or [])
            gold = await generate_gold_for_question(
                query,
                passages,
                model=self._selected_gen_model(),
                temperature=self._selected_gen_temp(),
            )
        except EvalGenerationConnectionError as exc:
            self._set_busy(self.gen_button, self.gen_spinner, False)
            self._set_status(str(exc))
            return
        except Exception as exc:  # broad: provider / LLM errors → status
            self._set_busy(self.gen_button, self.gen_spinner, False)
            self._set_status(f"generation failed: {exc}")
            return
        self._set_busy(self.gen_button, self.gen_spinner, False)
        # Single page: chat question + provenance + pinned retrieval, then the
        # LLM-written gold + the chat's own retrieved sources.
        self._load_drafts([self._blank_snapshot()])
        self._apply_chat_source()
        self.f["expected_answer_points"].value = _text(gold.answer_points)
        self.f["required_keywords"].value = _text(gold.keywords)
        self.f["expected_sources"].value = _text(
            _dedup_doc_ids(getattr(answer, "chunk_sources", None) or [])
        )
        self._capture_page()
        self._render_preview()
        self._set_status(
            "wrote gold for the chat question (origin=chat) — review, then Add case(s)"
        )

    def _on_save_case(self, _e: ft.Event) -> None:
        """Commit the form: Update the selected existing case, or Add ALL draft
        pages as new cases (602 — one Add case(s) commits the whole batch), then
        persist. Wired to both right-column commit buttons."""
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        path = self._current_path()
        if path is None:
            self._set_status("Choose a dataset (Browse or New dataset) first.")
            return
        if self._dataset is None:
            self._dataset = EvalDataset()
        # The status header rides along with any save; the recipe is left exactly
        # as loaded (it's edited on the Run tab). `name` is the filename now — the
        # human `name` header field was dropped from the UI.
        self._stamp_header()

        if self._selected is not None and 0 <= self._selected < len(self._dataset.cases):
            # Edit mode: overwrite the selected case with the form.
            try:
                case = self._read_form()
            except Exception as exc:  # broad: validation / int-parse → status line
                self._set_status(f"invalid case: {exc}")
                return
            self._dataset.cases[self._selected] = case
            verb = "updated 1"
        else:
            # Draft mode: read every page (strict) and append the valid batch —
            # a page that fails validation aborts the whole add (all-or-nothing).
            self._capture_page()
            cases, err = self._cases_from_drafts()
            if not cases:
                self._set_status(err or "nothing to add — fill in the case first.")
                return
            self._dataset.cases.extend(cases)
            verb = f"added {len(cases)}"
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"save failed: {exc}")
            return
        self._cases = self._dataset.cases
        self._path = path
        # Fresh blank page, ready for the next case; nothing left selected.
        self._load_drafts([self._blank_snapshot()])
        self._set_status(f"{verb} — saved {len(self._cases)} case(s) to {path.name}")

    def _cases_from_drafts(self) -> tuple[list[EvalCase], str | None]:
        """Read every draft page (strict) into EvalCases; restore the displayed
        page afterward. Returns (cases, error-or-None); a page that fails
        validation aborts with a message naming its position (all-or-nothing)."""
        saved = self._draft_index
        cases: list[EvalCase] = []
        err: str | None = None
        for i, snap in enumerate(self._drafts):
            self._restore_form(snap)
            try:
                cases.append(self._read_form())
            except Exception as exc:  # broad: validation / parse → abort the add
                where = f"case {i + 1}/{len(self._drafts)}" if len(self._drafts) > 1 else "case"
                err = f"invalid {where}: {exc}"
                cases = []
                break
        if self._drafts:
            self._restore_form(self._drafts[min(saved, len(self._drafts) - 1)])
        return cases, err

    def _delete_case(self, idx: int) -> None:
        """Delete case `idx` (wired to each card's Delete button) and persist."""
        from knowledge_agent.evaluation.models import save_dataset

        if self._dataset is None or not (0 <= idx < len(self._dataset.cases)):
            return
        path = self._current_path()
        if path is None:
            self._set_status("No dataset path to save to.")
            return
        del self._dataset.cases[idx]
        self._stamp_header()  # persist the status header alongside the deletion
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

    def _write_status(self, msg: str) -> None:
        """Set the Progress line's text + style without flushing the page: a real
        message reads GREY_500; a blank message falls back to the muted italic
        idle placeholder so the line never collapses to nothing."""
        if self.status is None:
            return
        idle = not msg.strip()
        self.status.value = _IDLE_STATUS if idle else msg
        self.status.italic = idle
        self.status.color = ft.Colors.GREY_600 if idle else ft.Colors.GREY_500

    def _set_status(self, msg: str) -> None:
        self._write_status(msg)
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
