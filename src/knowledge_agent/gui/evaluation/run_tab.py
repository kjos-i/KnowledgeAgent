"""Evaluation → Run sub-tab — configure + trigger an eval run.

Collects the run config (dataset, metric-group toggles, judge-model panel,
max-cases) into an `EvalConfig`, calls `runner.run(cfg, on_progress=...)`
in-process with a progress bar, and on completion tells the coordinator to
select the new run + jump to Run Summary. The active corpus is read-only
here (switch it in Library); its stores + keys are already bridged into the
environment at GUI startup, so `run()` works in-process.

Async idiom mirrors the Ingest tab: a sync on_click spawns a tracked task
that flips a busy flag (progress bar + disabled Run button) around the
awaited run. `on_progress(done, total)` drives the bar.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    LEFT_COLUMN_WIDTH,
    centered_label,
    labeled_field,
    panel_box,
    panel_title,
    section_divider,
    section_title,
    sub_section_header,
    sub_section_title,
    thin_rule,
)
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui.config_store import get_api_key
from knowledge_agent.gui.evaluation._case_view import render_case_cards
from knowledge_agent.gui.evaluation._recipe_form import RecipeForm

if TYPE_CHECKING:
    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

logger = logging.getLogger(__name__)


class RunTab:
    """Run-configuration + trigger sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self._bg_tasks: set[asyncio.Task] = set()
        self._busy = False
        self.dataset_field: ft.TextField | None = None
        # The recipe controls (metric groups + judge panel + gate thresholds +
        # profile) — one shared widget with the Create-test-cases tab. Loaded
        # from the selected dataset; editable when the dataset is not frozen,
        # read-only when it is.
        self.recipe_form: RecipeForm | None = None
        # Freeze controls. `freeze_check` (next to Run) is the opt-in that locks
        # the recipe onto the dataset when the run finishes; `unfreeze_button` +
        # `frozen_indicator` sit at the top and show only when frozen. The
        # dataset's status + frozen flag drive their enabled/visible state.
        self.freeze_check: ft.Checkbox | None = None
        self.freeze_hint: ft.Text | None = None
        self.unfreeze_button: ft.TextButton | None = None
        self.frozen_indicator: ft.Text | None = None
        # Shown when the selected SUITE's members carry different recipes (R4) —
        # hand-edited divergence; the first member's recipe loads + this warns.
        self.divergence_warning: ft.Text | None = None
        # Aggregate freeze state of the current selection (a single file, or ALL
        # members of the selected suite): "final" only when every member is, and
        # frozen only when every member is. Drives the shared freeze UI.
        self._dataset_status: str = "draft"
        self._dataset_frozen: bool = False
        self._suite_diverged: bool = False
        self.max_cases_field: ft.TextField | None = None
        self.trace_check: ft.Checkbox | None = None
        self.project_field: ft.TextField | None = None
        self._project_row: ft.Row | None = None
        self.trace_warning: ft.Text | None = None
        # Shown when tracing is ticked but no LangSmith key is set yet.
        self.trace_key_hint: ft.Text | None = None
        # Link next to the trace checkbox → opens LangSmith in the browser.
        # The key itself is set in Settings → Keys, not here.
        self.open_langsmith_button: ft.TextButton | None = None
        # Resolves this run's project (name + key → URL) and opens its traces.
        self.open_project_button: ft.TextButton | None = None
        self.run_button: ft.Button | None = None
        self.progress: ft.ProgressBar | None = None
        self.status: ft.Text | None = None
        # Read-only echoes derived from the ACTIVE corpus — refreshed live on
        # an app-wide corpus switch (see `refresh_active_corpus`).
        self.output_line: ft.Text | None = None
        # Right column: read-only case preview — a TabBar with one tab per
        # selected file (each suite member, or the single file), rebuilt on every
        # selection change. The container holds the current tabs widget.
        self.case_pane: ft.Container | None = None
        # Selection: EITHER a named suite (the Suite dropdown) OR a single file
        # (Browse) — mutually exclusive, Ingest-style. A suite is the corpus
        # files sharing a named `suites` tag (same facts, swept knobs); the
        # dropdown lists every such name in the corpus, and picking one runs its
        # members together (run_suite). Browsing a file runs that one file
        # (run). `_suite_name` is the selected suite (None ⇒ single-file mode);
        # `_suite_paths` its members. Both recomputed on every selection change.
        self.suite_dd: ft.Dropdown | None = None
        self.selection_hint: ft.Text | None = None
        self._suite_paths: list[Path] = []
        self._suite_name: str | None = None

    def refresh_active_corpus(self) -> None:
        """Update the read-only output-path echo from the current active corpus.

        Called on an app-wide corpus switch (via
        `GuiApp.refresh_after_corpus_change`) so the Run tab — built once and
        kept mounted in the TabBarView — doesn't show a stale output path. No
        `page.update()` here; the app broadcast issues one.
        """
        from knowledge_agent.gui.evaluation._common import active_output_dir

        if self.output_line is not None:
            self.output_line.value = f"Results save to: {active_output_dir(self.app)}"
        # The named suites belong to the corpus, so re-scan them on a switch.
        self._refresh_suite_options()

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.config import DEFAULT_LANGSMITH_PROJECT
        from knowledge_agent.gui.evaluation._common import active_output_dir

        # Dataset file: a read-only path display + Browse (opens in the active
        # corpus's folder). Starts empty — there's no baked-in default; the
        # user Browses to a gold JSON in the corpus folder. The right column
        # previews whatever's chosen.
        self.dataset_field = ft.TextField(
            read_only=True,
            value="",
            hint_text="Browse for a gold-dataset JSON",
            expand=True,
        )
        browse_button = ft.Button(
            content=centered_label("Browse"),
            tooltip="Browse for a gold-dataset JSON (starts in the corpus folder)",
            on_click=self._on_browse_clicked,
        )

        # Read-only: where this run's ledger + JSON/CSV reports will land —
        # derived from the active corpus, exactly as `_build_config` resolves
        # it. Selectable so the path can be copied; it is NOT an input (override
        # only via --output-dir / KA_EVAL_OUTPUT_DIR).
        self.output_line = ft.Text(
            f"Results save to: {active_output_dir(self.app)}",
            size=12,
            color=ft.Colors.GREY_600,
            selectable=True,
        )

        # The recipe editor, shared with the Create-test-cases tab. Built here;
        # `_load_dataset_state` sets it read-only vs editable per the dataset's
        # frozen flag on select. A fresh form (no dataset) is editable.
        self.recipe_form = RecipeForm(self.app)
        recipe_body = self.recipe_form.build()
        # "Freeze run settings" — the opt-in next to Run: tick it + Run and the
        # finished run persists frozen=true, locking the recipe onto the
        # dataset. Disabled unless the dataset is final AND not already frozen.
        self.freeze_check = ft.Checkbox(
            label="Freeze run settings",
            value=False,
            disabled=True,
            tooltip="Lock this recipe onto the dataset when the run finishes (final datasets only)",
        )
        self.freeze_hint = ft.Text(
            "Set the dataset's status to “final” (in Create test cases) to freeze it.",
            size=12,
            color=ft.Colors.GREY_600,
            italic=True,
            visible=False,
        )
        # Unfreeze + the frozen badge live at the TOP (Evaluation cases section).
        # The badge shows only when frozen; Unfreeze is always present, just
        # disabled (greyed) until the dataset is frozen.
        self.frozen_indicator = ft.Text(
            "\U0001f512 Recipe frozen — read-only",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )
        self.unfreeze_button = ft.TextButton(
            "Unfreeze",
            icon=ft.Icons.LOCK_OPEN,
            tooltip="Unlock the recipe so it can change again (asks to confirm)",
            on_click=self._on_unfreeze_clicked,
            disabled=True,
        )
        # Warns when a picked suite's members don't share one recipe (R4). The
        # first member's recipe is loaded; Freeze (which re-writes all members)
        # re-syncs them. Hidden unless a divergent suite is selected.
        self.divergence_warning = ft.Text(
            "⚠ This suite's members have different run settings (likely hand-edited). "
            "Showing the first member's recipe — Freeze re-syncs all members.",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )

        self.max_cases_field = ft.TextField(
            width=200,
            keyboard_type=ft.KeyboardType.NUMBER,
        )

        # Tracing — OFF by default. The data-safety warning + project field
        # only appear once the box is checked (see `_on_trace_toggle`).
        self.trace_check = ft.Checkbox(
            label="Trace to LangSmith",
            value=False,
            on_change=self._on_trace_toggle,
        )
        self.open_langsmith_button = ft.TextButton(
            "Open LangSmith",
            icon=ft.Icons.OPEN_IN_NEW,
            url="https://smith.langchain.com",
            tooltip=(
                "Open LangSmith in your browser — sign in to view traces or copy "
                "your API key (set the key in Settings → Keys)"
            ),
        )
        self.open_project_button = ft.TextButton(
            "Open project in LangSmith",
            icon=ft.Icons.OPEN_IN_NEW,
            tooltip=(
                "Open this run's LangSmith project traces — resolved from the "
                "project name + your key. The project exists only after a traced run."
            ),
            on_click=self._on_open_project_clicked,
        )
        self.trace_warning = ft.Text(
            "⚠ Tracing uploads this run's data — your queries, the retrieved "
            "chunk text, and the answers — to LangSmith's cloud. Enable ONLY "
            "for a non-sensitive corpus (e.g. the test corpus); never for "
            "private documents.",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )
        self.trace_key_hint = ft.Text(
            "⚠ No LangSmith API key set — add it in Settings → Keys first, "
            "then this run can trace.",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )
        self.project_field = ft.TextField(
            value=DEFAULT_LANGSMITH_PROJECT,
            width=300,
        )
        # Wrapped in a labeled_field row. Always visible; the field greys out
        # (disabled) until tracing is ticked (toggled in `_on_trace_toggle`).
        self._project_row = labeled_field("LangSmith project", self.project_field)
        self._project_row.disabled = True

        # Suite dropdown — the corpus's named suites (scanned from files' `suites`
        # tags). Picking one runs its members together; disabled when the corpus
        # has none. Fires on_select (Flet 0.85), NOT on_change.
        self.suite_dd = ft.Dropdown(label="Suite", options=[], width=260)
        self.suite_dd.on_select = self._on_suite_dd_change
        # One-line summary of what's currently selected (suite + members, or the
        # single file). Sits under the two fields.
        self.selection_hint = ft.Text("", size=12, color=ft.Colors.GREY_600, italic=True)

        self.run_button = ft.Button(
            "Run evaluation",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run_clicked,
        )
        self.progress = ft.ProgressBar(visible=False, width=460)
        self.status = ft.Text("", color=ft.Colors.GREY_600)

        form = ft.Column(
            controls=[
                # ============ Section: Evaluation cases ============
                # Pick EITHER a named suite OR a single file (mutually exclusive,
                # Ingest-style "…or…"). The choice decides suite vs single run.
                section_title("Evaluation cases"),
                sub_section_title("Run a suite"),
                labeled_field("Suite", self.suite_dd),
                thin_rule(),
                sub_section_title("…or a single file"),
                labeled_field("Dataset", self.dataset_field, trailing=browse_button),
                self.selection_hint,
                self.divergence_warning,
                ft.Row(
                    [self.frozen_indicator, self.unfreeze_button],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=8,
                ),
                # The dataset's saved recipe (metric groups + judge panel + gate
                # thresholds + profile). Editable when the dataset is not frozen;
                # read-only when it is (Unfreeze up top to edit).
                recipe_body,
                section_divider(),
                # ============ Section: Run options ============
                section_title("Run options"),
                labeled_field("Max cases (blank = all)", self.max_cases_field),
                # ---- Sub-section: Tracing (opt-in LangSmith) ----
                sub_section_header(
                    "Tracing (optional)",
                    trailing=info_icon(
                        self.app,
                        title="Tracing (optional)",
                        text=(
                            "Opt-in per run, OFF by default. When on, this run's "
                            "data — your queries, the retrieved chunk text, and the "
                            "answers — is uploaded to LangSmith's cloud. Enable ONLY "
                            "for a non-sensitive corpus (e.g. the test corpus), never "
                            "for private documents. The key is stored in your OS "
                            "keyring (same as Settings → Keys)."
                        ),
                    ),
                ),
                ft.Row(
                    [
                        self.trace_check,
                        self.open_langsmith_button,
                        self.open_project_button,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    wrap=True,
                ),
                self.trace_key_hint,
                self.trace_warning,
                self._project_row,
                section_divider(),
                # ============ Section: Run ============
                # Scope (suite vs single) comes from the selection at the top, so
                # one button runs whichever is set.
                section_title("Run"),
                ft.Row(
                    [self.run_button, self.freeze_check, self.progress],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.freeze_hint,
                # Where this run's ledger + reports land — sits under the Run
                # button as a caption (derived from the active corpus).
                self.output_line,
                self.status,
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=12,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        # RIGHT column: read-only preview of the selected dataset's cases —
        # "exactly what will run". Populated now + on every dataset change.
        # Read-only (no on_edit/on_delete) keeps runs comparable; how many
        # actually run is the deterministic top-N set by Max cases.
        self.case_pane = ft.Container(content=self._build_case_tabs(), expand=True)
        left_pane = panel_box(form, width=LEFT_COLUMN_WIDTH)
        right_pane = panel_box(
            ft.Column(
                [self.case_pane],
                expand=True,
                spacing=8,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        )
        # Column titles sit in a fixed header above the panes (aligned 50/50
        # over them) so each stays visible while its box scrolls — the same
        # sticky-header idiom as the Ingest tab. The right title's (i) carries
        # the read-only preview note the old inline title spelled out.
        header = ft.Row(
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[
                ft.Container(
                    width=LEFT_COLUMN_WIDTH,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=panel_title("Run configuration"),
                ),
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=ft.Row(
                        controls=[
                            panel_title("Test cases"),
                            info_icon(
                                self.app,
                                title="Test cases",
                                text=(
                                    "Read-only preview of what will run — one tab "
                                    "per suite member (labelled by its retrieval "
                                    "strategy), or a single tab for a single file. "
                                    "The tabs share the same facts; the strategy "
                                    "label is the knob difference. Max cases runs "
                                    "the first N, top-down; leave it blank to run all."
                                ),
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=8,
                    ),
                ),
            ],
        )
        body = ft.Row(
            [left_pane, right_pane],
            expand=True,
            spacing=12,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        # Populate the Suite dropdown from the active corpus's named suites.
        self._refresh_suite_options()
        return ft.Column(
            controls=[header, body],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
        )

    # ---- dataset case preview (right column) ------------------------------

    @staticmethod
    def _member_label(path: Path) -> str:
        """A short tab label for a preview member — its strategy suffix (a
        generated member is `<suite>__<strategy>.json`, so `…__vector` → 'vector')
        when present, else the file stem."""
        stem = path.stem
        if "__" in stem:
            suffix = stem.rsplit("__", 1)[-1]
            if suffix:
                return suffix
        return stem

    def _case_cards_for(self, path: Path) -> list[ft.Control]:
        """One member's cases as read-only cards (or an inline error). Best-effort
        — a bad path shows a message, never raises."""
        from knowledge_agent.evaluation.models import load_dataset

        try:
            ds = load_dataset(path)
        except Exception as exc:  # broad: any parse/IO error → inline message
            return [ft.Text(f"Could not load dataset: {exc}", color=ft.Colors.RED_300)]
        # detailed=True: no edit form on this tab, so surface every gold field on
        # the card ("all the information for each case").
        return render_case_cards(ds.cases, detailed=True, empty_hint="This dataset has no cases.")

    def _build_case_tabs(self) -> ft.Control:
        """The read-only case preview: a TabBar with one tab per selected file —
        each suite member (labelled by its strategy) or the single file. Nothing
        selected ⇒ a hint. Rebuilt whole on every selection change."""
        members = self._selected_paths()
        if not members:
            return ft.Column(
                controls=[
                    ft.Text(
                        "Pick a suite or a dataset to preview its cases.",
                        italic=True,
                        color=ft.Colors.GREY_500,
                    )
                ],
                expand=True,
            )
        bar = ft.TabBar(tabs=[ft.Tab(label=self._member_label(p)) for p in members], secondary=True)
        bodies = ft.TabBarView(
            controls=[
                ft.Column(
                    controls=self._case_cards_for(p),
                    scroll=ft.ScrollMode.AUTO,
                    expand=True,
                    spacing=6,
                )
                for p in members
            ],
            expand=True,
        )
        return ft.Tabs(
            length=len(members),
            selected_index=0,
            content=ft.Column([bar, bodies], expand=True, spacing=0),
            expand=True,
        )

    def _refresh_case_view(self) -> None:
        """Rebuild the per-member case-preview tabs for the current selection."""
        if self.case_pane is None:
            return
        self.case_pane.content = self._build_case_tabs()
        self.app.page.update()

    # ---- recipe freeze/unfreeze -------------------------------------------

    def _load_dataset_state(self, path: Path) -> None:
        """Load the selected dataset's recipe + status + frozen flag, then sync
        the freeze UI. A bad / legacy dataset falls back to harness-default
        recipe + draft / unfrozen. Called on every dataset change."""
        from knowledge_agent.evaluation.models import load_dataset

        status, frozen, recipe = "draft", False, None
        try:
            ds = load_dataset(path)
            status, frozen, recipe = ds.status, ds.frozen, ds.recipe
        except Exception:  # broad: a bad dataset already surfaces in the case view
            pass
        self._dataset_status = status
        self._dataset_frozen = frozen
        self._suite_diverged = False  # single file — no members to diverge
        if self.recipe_form is not None:
            self.recipe_form.load(recipe)
        self._apply_frozen_ui()
        self._sync_selection_hint()
        self._sync_divergence_warning()

    def _apply_frozen_ui(self) -> None:
        """Sync every run setting to the selection's aggregate status + frozen
        flag (a single file, or ALL members of the selected suite). Frozen ⇒ the
        whole tab locks read-only (recipe, Max cases, Tracing); the freeze
        checkbox is enabled only when final AND not already frozen; Unfreeze is
        always present but disabled until frozen. Freeze acts on the whole scope
        (1 file or N members), so no suite-specific greying (R3)."""
        final = self._dataset_status == "final"
        frozen = self._dataset_frozen
        if self.recipe_form is not None:
            self.recipe_form.set_enabled(not frozen)
        # Freezing greys out all the other run settings too; unfreezing restores
        # each — the project field falls back to following the tracing toggle.
        if self.max_cases_field is not None:
            self.max_cases_field.disabled = frozen
        if self.trace_check is not None:
            self.trace_check.disabled = frozen
        if self._project_row is not None:
            trace_on = bool(self.trace_check and self.trace_check.value)
            self._project_row.disabled = frozen or not trace_on
        if self.freeze_check is not None:
            self.freeze_check.value = frozen
            self.freeze_check.disabled = (not final) or frozen
        if self.freeze_hint is not None:
            self.freeze_hint.visible = not final and not frozen
        if self.unfreeze_button is not None:
            self.unfreeze_button.disabled = not frozen
        if self.frozen_indicator is not None:
            self.frozen_indicator.visible = frozen

    # ---- selection: named suite vs single file ----------------------------

    @staticmethod
    def _dataset_suites(path: Path) -> list[str]:
        """The named suites a dataset file declares in its header (empty on a bad
        / untagged file)."""
        from knowledge_agent.evaluation.models import load_dataset

        try:
            return list(load_dataset(path).suites)
        except Exception:  # broad: a bad file has no suites
            return []

    def _corpus_suites(self) -> list[str]:
        """Every named suite declared across the active corpus's dataset files —
        the sorted union of their `suites` tags. Backs the Suite dropdown."""
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        corpus_dir = active_corpus_dir(self.app)
        if corpus_dir is None or not corpus_dir.is_dir():
            return []
        names: set[str] = set()
        for p in corpus_dir.glob("*.json"):
            names.update(self._dataset_suites(p))
        return sorted(names)

    def _members_for_suite(self, tag: str) -> list[Path]:
        """Corpus files whose `suites` header contains `tag` — the members of that
        named suite. Sorted for a stable run order."""
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        corpus_dir = active_corpus_dir(self.app)
        if corpus_dir is None or not corpus_dir.is_dir():
            return []
        return [p for p in sorted(corpus_dir.glob("*.json")) if tag in self._dataset_suites(p)]

    def _refresh_suite_options(self) -> None:
        """Repopulate the Suite dropdown from the active corpus's named suites
        (called on build + corpus switch). Disabled when the corpus has none;
        drops a selection the corpus no longer offers."""
        if self.suite_dd is None:
            return
        suites = self._corpus_suites()
        self.suite_dd.options = [ft.DropdownOption(key=s, text=s) for s in suites]
        self.suite_dd.disabled = not suites
        if self._suite_name is not None and self._suite_name not in suites:
            self._suite_name, self._suite_paths = None, []
            self.suite_dd.value = None
            self._sync_selection_hint()
            self._apply_frozen_ui()

    def _on_suite_dd_change(self, _e: ft.Event) -> None:
        """A named suite was picked → clear the single-file field (mutually
        exclusive) and load the suite's members + shared recipe."""
        if self.suite_dd is None or not self.suite_dd.value:
            return
        if self.dataset_field is not None:
            self.dataset_field.value = ""  # mutual exclusion
        self._load_suite_state(self.suite_dd.value)
        self._refresh_case_view()
        self.app.page.update()

    def _load_suite_state(self, name: str) -> None:
        """Select the named suite `name`: resolve its members, compute the suite's
        AGGREGATE freeze state (final only when EVERY member is; frozen only when
        every member is), and load the shared recipe into the form. Members should
        share one recipe (the generator stamps them equal; freeze re-syncs). If a
        hand-edit made them diverge, the FIRST member's recipe is loaded and the
        divergence banner shows (R4)."""
        from knowledge_agent.evaluation.models import compute_recipe_hash, load_dataset

        self._suite_name = name
        self._suite_paths = self._members_for_suite(name)
        recipes, statuses, frozens = [], [], []
        for p in self._suite_paths:
            try:
                ds = load_dataset(p)
            except Exception:  # broad: a bad member counts as a default draft
                recipes.append(None)
                statuses.append("draft")
                frozens.append(False)
                continue
            recipes.append(ds.recipe)
            statuses.append(ds.status)
            frozens.append(ds.frozen)
        # A suite shares one recipe; >1 distinct hash ⇒ hand-edited divergence.
        self._suite_diverged = len({compute_recipe_hash(r) for r in recipes}) > 1
        self._dataset_status = (
            "final" if statuses and all(s == "final" for s in statuses) else "draft"
        )
        self._dataset_frozen = bool(frozens) and all(frozens)
        if self.recipe_form is not None:
            self.recipe_form.load(recipes[0] if recipes else None)
        self._apply_frozen_ui()
        self._sync_selection_hint()
        self._sync_divergence_warning()

    def _sync_selection_hint(self) -> None:
        """One-line summary of the current selection — the suite + its members,
        or the single file, or blank when nothing is picked."""
        if self.selection_hint is None:
            return
        if self._suite_name is not None:
            names = ", ".join(p.stem for p in self._suite_paths)
            n = len(self._suite_paths)
            self.selection_hint.value = f"Suite “{self._suite_name}” — {n} member(s): {names}"
        elif self.dataset_field and self.dataset_field.value:
            self.selection_hint.value = f"Single file: {Path(self.dataset_field.value).name}"
        else:
            self.selection_hint.value = ""

    def _sync_divergence_warning(self) -> None:
        """Show the divergence banner only for a selected suite whose members
        carry different recipes (R4)."""
        if self.divergence_warning is not None:
            self.divergence_warning.visible = self._suite_name is not None and self._suite_diverged

    def _selected_paths(self) -> list[Path]:
        """The dataset files the current selection refers to: every member of the
        selected suite, or the single browsed file (empty when nothing is
        selected). Drives BOTH the case preview (a tab each) and freeze/unfreeze."""
        if self._suite_name is not None:
            return list(self._suite_paths)
        if self.dataset_field and self.dataset_field.value:
            return [Path(self.dataset_field.value)]
        return []

    def _suite_selected(self) -> bool:
        """True when a named suite is selected (→ run_suite over its members)."""
        return self._suite_name is not None and len(self._suite_paths) >= 1

    def _on_unfreeze_clicked(self, _e: ft.Event) -> None:
        """Unfreeze the selection — a deliberate, confirmed action that clears the
        frozen flag across the scope (a single file, or every member of the
        selected suite) so the recipe can change again."""
        paths = self._selected_paths()
        if not paths:
            return
        from knowledge_agent.gui.evaluation._common import confirm_dialog

        scope = f"all {len(paths)} suite members" if self._suite_name is not None else "the dataset"
        confirm_dialog(
            self.app,
            title="Unfreeze run settings?",
            message=(
                "This unlocks the recipe so it can be changed again, and clears "
                f"the frozen state saved on {scope}."
            ),
            confirm_label="Unfreeze",
            on_confirm=self._do_unfreeze,
        )

    def _do_unfreeze(self) -> None:
        from knowledge_agent.evaluation.models import load_dataset, save_dataset

        paths = self._selected_paths()
        if not paths:
            return
        try:
            for path in paths:
                ds = load_dataset(path)
                ds.frozen = False
                save_dataset(ds, path)
        except Exception as exc:  # broad: I/O / parse failure → status line
            self._set_status(f"could not unfreeze: {exc}")
            return
        self._dataset_frozen = False
        self._apply_frozen_ui()
        self._set_status("Unfroze run settings — the recipe is editable again.")
        self.app.page.update()

    def _freeze_dataset(self) -> None:
        """Persist the current recipe + frozen=true across the SELECTED scope's
        members — the 'Freeze run settings' opt-in, run after a successful run
        when the box is ticked. One file for a single run, or EVERY member of the
        selected suite (which also re-syncs any divergent recipes). Only when all
        are final (the checkbox enforces it too)."""
        from knowledge_agent.evaluation.models import load_dataset, save_dataset

        paths = self._selected_paths()
        if not paths:
            return
        recipe = self.recipe_form.to_recipe() if self.recipe_form else None
        try:
            for path in paths:
                ds = load_dataset(path)
                if ds.status != "final":
                    return  # invariant: only a final dataset can be frozen
                if recipe is not None:
                    ds.recipe = recipe.model_copy(deep=True)
                ds.frozen = True
                save_dataset(ds, path)
        except Exception as exc:  # broad: I/O / parse failure → status line
            self._set_status(f"run done, but freeze failed: {exc}")
            return
        self._dataset_frozen = True
        self._suite_diverged = False  # freeze re-wrote every member identical
        self._apply_frozen_ui()
        self._sync_divergence_warning()

    # ---- tracing ----------------------------------------------------------

    def _refresh_trace_key_hint(self) -> None:
        """Missing-key hint shows only when tracing is on AND no key is set."""
        on = bool(self.trace_check and self.trace_check.value)
        if self.trace_key_hint is not None:
            self.trace_key_hint.visible = on and not get_api_key("langsmith")

    def _on_trace_toggle(self, _e: ft.Event) -> None:
        """Reveal the data-safety warning when tracing is enabled — so it's
        impossible to miss at opt-in time. The project field stays visible
        throughout, just greyed out (disabled) until tracing is ticked."""
        on = bool(self.trace_check and self.trace_check.value)
        if self.trace_warning is not None:
            self.trace_warning.visible = on
        # Surface a missing LangSmith key the moment tracing is ticked, not
        # only when the run is blocked later.
        self._refresh_trace_key_hint()
        if self._project_row is not None:
            self._project_row.disabled = not on
        self.app.page.update()

    def _on_open_project_clicked(self, _e: ft.Event) -> None:
        """Open this run's LangSmith project traces. Needs a key (Settings →
        Keys) + a project name; the project only exists after a traced run."""
        if not self._loop_running():
            return
        key = get_api_key("langsmith")
        if not key:
            self._set_status("Set a LangSmith API key in Settings → Keys to open the project.")
            return
        project = (self.project_field.value or "").strip() if self.project_field else ""
        if not project:
            self._set_status("Enter a LangSmith project name first.")
            return
        self._spawn(self._open_project(key, project))

    async def _open_project(self, key: str, project: str) -> None:
        """Resolve the project's URL (name + key → URL via the LangSmith SDK)
        and open it in the browser. Best-effort — a missing project / network
        error surfaces in the status line."""
        try:
            from langsmith import Client
        except ImportError:
            self._set_status("LangSmith isn't installed — install the tracing extra first.")
            return
        self._set_status(f"Opening LangSmith project {project!r}…")
        try:
            session = await asyncio.to_thread(
                lambda: Client(api_key=key).read_project(project_name=project)
            )
        except Exception as exc:  # broad: not-found / network / auth → status line
            self._set_status(
                f"Couldn't open project {project!r}: {exc} (it exists only after a traced run)."
            )
            return
        url = session.url
        if not url:
            self._set_status(f"LangSmith returned no URL for {project!r}.")
            return
        await self.app.page.launch_url(url)

    # ---- run --------------------------------------------------------------

    def _on_browse_clicked(self, _e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        self._spawn(self._pick_dataset())

    async def _pick_dataset(self) -> None:
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        corpus_dir = active_corpus_dir(self.app)
        initial = str(corpus_dir) if corpus_dir and corpus_dir.is_dir() else None
        try:
            files = await self.app.file_picker.pick_files(
                dialog_title="Pick a gold-dataset JSON",
                allowed_extensions=["json"],
                initial_directory=initial,
            )
        except Exception as exc:  # broad: surface any picker failure in the status line
            self._set_status(f"file picker error: {exc}")
            return
        if files and files[0].path and self.dataset_field is not None:
            self.dataset_field.value = files[0].path
            if self.suite_dd is not None:
                self.suite_dd.value = None  # mutual exclusion
            self._suite_name, self._suite_paths = None, []
            self._load_dataset_state(Path(files[0].path))
            self._refresh_case_view()
            self.app.page.update()

    def _on_run_clicked(self, _e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        if not (self.recipe_form and self.recipe_form.any_group_selected()):
            self._set_status("Select at least one metric group.")
            return
        if not self._suite_selected() and not (self.dataset_field and self.dataset_field.value):
            self._set_status("Select a suite or a dataset file.")
            return
        if self.trace_check and self.trace_check.value and not get_api_key("langsmith"):
            self._set_status("Set a LangSmith API key in Settings → Keys to trace.")
            return
        # R4: a divergent suite runs the FIRST member's recipe for EVERY member —
        # make the user acknowledge that before spending tokens.
        if self._suite_selected() and self._suite_diverged:
            from knowledge_agent.gui.evaluation._common import confirm_dialog

            confirm_dialog(
                self.app,
                title="Members have different run settings",
                message=(
                    "This suite's members don't share one recipe (likely "
                    "hand-edited). The first member's run settings will be used "
                    "for all of them. Run anyway?"
                ),
                confirm_label="Run anyway",
                on_confirm=lambda: self._spawn(self._execute_run()),
            )
            return
        self._spawn(self._execute_run())

    async def _execute_run(self) -> None:
        from knowledge_agent.evaluation import runner

        suite = self._suite_selected()
        trace = bool(self.trace_check and self.trace_check.value)
        project = (self.project_field.value or "").strip() if self.project_field else ""
        try:
            # Same recipe/overrides for every member; only the dataset path differs.
            cfgs = [self._build_config(p) for p in self._suite_paths] if suite else None
            cfg = None if suite else self._build_config()
        except Exception as exc:  # broad: report any bad form input in the status line
            self._set_status(f"config error: {exc}")
            return
        self._set_busy(True, "starting…")
        if self.progress is not None:
            self.progress.value = 0.0
        try:
            if suite:
                outcome = await runner.run_suite(
                    cfgs,
                    on_run_complete=self._on_suite_progress,
                    suite=self._suite_name,
                    trace=trace,
                    langsmith_project=project or None,
                )
            else:
                result = await runner.run(
                    cfg,
                    on_progress=self._on_progress,
                    trace=trace,
                    langsmith_project=project or None,
                )
        except Exception as exc:  # broad: one failed run must not crash the GUI
            logger.exception("eval run failed")
            self._set_busy(False, f"run failed: {exc}")
            return
        # Opt-in freeze: lock this recipe onto the scope (single file or every
        # suite member) now the run(s) succeeded (R3).
        froze = bool(self.freeze_check and self.freeze_check.value)
        if suite:
            self._set_busy(
                False,
                f"done: suite of {len(outcome.results)} runs — opening Compare."
                + (" Recipe frozen." if froze else ""),
            )
            if froze:
                self._freeze_dataset()
            self.coordinator.on_suite_complete(outcome)
            return
        summary = result.report.get("summary", {})
        self._set_busy(
            False,
            f"done: run {result.run_id} — "
            f"{summary.get('pass_count')}/{summary.get('case_count')} pass."
            + (" Recipe frozen." if froze else ""),
        )
        if froze:
            self._freeze_dataset()
        self.coordinator.on_run_complete(result.run_id)

    def _on_suite_progress(self, done: int, total: int, _result: object) -> None:
        """Advance the bar per completed member of a suite run (file-level)."""
        if self.progress is not None:
            self.progress.value = done / total if total else None
        self._set_status(f"ran dataset {done}/{total}…")

    def _build_config(self, dataset_path: Path | None = None) -> EvalConfig:
        from knowledge_agent.evaluation.config import load_eval_config
        from knowledge_agent.gui.evaluation._common import active_corpus_config_path

        # The (possibly deviated) recipe on the form drives the run's metric
        # groups, judge panel, and the three gate thresholds.
        recipe = self.recipe_form.to_recipe() if self.recipe_form else None
        overrides: dict = {}
        if recipe is not None:
            overrides["enabled_groups"] = frozenset(recipe.enabled_groups)
            overrides["judge_models"] = tuple(recipe.judge_models)
            overrides["judge_threshold"] = recipe.judge_threshold
            overrides["metadata_match_threshold"] = recipe.metadata_match_threshold
            overrides["required_keyword_threshold"] = recipe.required_keyword_threshold
        # A suite run passes each member's path explicitly; a single run reads
        # the browsed dataset field. Same recipe/overrides for every member.
        ds_path = dataset_path
        if ds_path is None and self.dataset_field and self.dataset_field.value:
            ds_path = Path(self.dataset_field.value)
        if ds_path is not None:
            overrides["dataset_path"] = ds_path
        corpus_path = active_corpus_config_path(self.app)
        if corpus_path:
            overrides["corpus_config_path"] = corpus_path
        raw_max = (self.max_cases_field.value or "").strip() if self.max_cases_field else ""
        if raw_max:
            overrides["max_cases"] = int(raw_max)
        return load_eval_config(**overrides)

    def _on_progress(self, done: int, total: int) -> None:
        if self.progress is not None:
            self.progress.value = done / total if total else None
        self._set_status(f"evaluating case {done}/{total}…")

    # ---- async plumbing (mirrors IngestTab) -------------------------------

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
        if self.status is not None:
            self.status.value = msg
            self.app.page.update()

    def _set_busy(self, busy: bool, msg: str | None = None) -> None:
        """Toggle the progress bar + disable the Run button so a second run
        can't start on top of one in flight."""
        self._busy = busy
        if self.progress is not None:
            self.progress.visible = busy
        if self.run_button is not None:
            self.run_button.disabled = busy
        if msg is not None and self.status is not None:
            self.status.value = msg
        self.app.page.update()
