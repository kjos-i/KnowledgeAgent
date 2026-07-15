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
        self._dataset_status: str = "draft"
        self._dataset_frozen: bool = False
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
        # Right column: read-only preview of the selected dataset's cases.
        self.case_list: ft.Column | None = None
        # Run scope: run just the picked file, OR its whole SUITE — the corpus
        # files sharing its facts_hash (same facts, swept knobs). `_suite_paths`
        # is recomputed on every dataset load; the radio + hint reflect it.
        self.suite_mode: ft.RadioGroup | None = None
        self.suite_hint: ft.Text | None = None
        self._suite_paths: list[Path] = []

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

        # Run scope: this file vs the whole suite (siblings sharing facts_hash).
        # The 'Whole suite' radio is disabled until a dataset with ≥2 members is
        # loaded (see `_refresh_suite_mode`); the hint spells out the members.
        self.suite_mode = ft.RadioGroup(
            value="single",
            on_change=self._on_suite_mode_change,
            content=ft.Row(
                [
                    ft.Radio(value="single", label="This file"),
                    ft.Radio(value="suite", label="Whole suite"),
                ],
                spacing=12,
            ),
        )
        self.suite_hint = ft.Text("", size=12, color=ft.Colors.GREY_600, italic=True)

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
                section_title("Evaluation cases"),
                labeled_field("Dataset", self.dataset_field, trailing=browse_button),
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
                section_title("Run"),
                labeled_field("Run scope", self.suite_mode),
                self.suite_hint,
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
        self.case_list = ft.Column(
            controls=self._case_controls(),
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=6,
        )
        left_pane = panel_box(form, width=LEFT_COLUMN_WIDTH)
        right_pane = panel_box(
            ft.Column(
                [self.case_list],
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
                                    "Read-only preview of the selected dataset's "
                                    "cases — exactly what will run. Max cases runs "
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
        # No dataset yet → 'Whole suite' disabled, hint blank.
        self._refresh_suite_mode()
        return ft.Column(
            controls=[header, body],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
        )

    # ---- dataset case preview (right column) ------------------------------

    def _case_controls(self) -> list[ft.Control]:
        """Load the selected dataset's cases as read-only cards (or a hint /
        error). Best-effort — a bad path shows a message, never raises."""
        path_str = (self.dataset_field.value or "").strip() if self.dataset_field else ""
        if not path_str:
            return [
                ft.Text(
                    "Pick a dataset to preview its cases.",
                    italic=True,
                    color=ft.Colors.GREY_500,
                )
            ]
        from knowledge_agent.evaluation.models import load_dataset

        try:
            ds = load_dataset(Path(path_str))
        except Exception as exc:  # broad: any parse/IO error → inline message
            return [ft.Text(f"Could not load dataset: {exc}", color=ft.Colors.RED_300)]
        # detailed=True: no edit form on this tab, so surface every gold field
        # on the card ("all the information for each case").
        return render_case_cards(ds.cases, detailed=True, empty_hint="This dataset has no cases.")

    def _refresh_case_view(self) -> None:
        """Re-render the read-only case preview from the current dataset path."""
        if self.case_list is None:
            return
        self.case_list.controls = self._case_controls()
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
        if self.recipe_form is not None:
            self.recipe_form.load(recipe)
        self._apply_frozen_ui()
        self._refresh_suite_mode()

    def _apply_frozen_ui(self) -> None:
        """Sync every run setting to the dataset's status + frozen flag. Frozen
        ⇒ the whole tab locks read-only (recipe, Max cases, Tracing); the freeze
        checkbox is enabled only when the dataset is final AND not already
        frozen; Unfreeze is always present but disabled until frozen."""
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

    # ---- run scope: single file vs whole suite ----------------------------

    def _suite_members(self, picked: Path) -> list[Path]:
        """The dataset files in the active corpus folder that share `picked`'s
        facts_hash — the members of its test-dataset suite (same facts, swept
        knobs). Always includes `picked`; returns just [picked] when it has no
        siblings. Unparseable / hashless files are skipped."""
        from knowledge_agent.evaluation.models import compute_facts_hash, load_cases
        from knowledge_agent.gui.evaluation._common import active_corpus_dir

        def _facts(path: Path) -> str | None:
            try:
                return compute_facts_hash(load_cases(path))
            except Exception:  # broad: a bad file just isn't a suite member
                return None

        target = _facts(picked)
        corpus_dir = active_corpus_dir(self.app)
        if target is None or corpus_dir is None or not corpus_dir.is_dir():
            return [picked]
        members = [p for p in sorted(corpus_dir.glob("*.json")) if _facts(p) == target]
        if not any(p.resolve() == picked.resolve() for p in members):
            members.append(picked)
        return members

    def _refresh_suite_mode(self) -> None:
        """Recompute the current dataset's suite members + reflect them in the
        run-scope radio/hint. <2 members ⇒ 'Whole suite' is disabled and the
        mode snaps back to single. Freezing doesn't apply to a suite, so the
        freeze checkbox greys while 'Whole suite' is selected."""
        path_str = (self.dataset_field.value or "").strip() if self.dataset_field else ""
        self._suite_paths = self._suite_members(Path(path_str)) if path_str else []
        n = len(self._suite_paths)
        available = n >= 2
        if self.suite_mode is not None:
            for radio in self.suite_mode.content.controls:
                if radio.value == "suite":
                    radio.disabled = not available
            if not available:
                self.suite_mode.value = "single"
        if self.suite_hint is not None:
            if available:
                names = ", ".join(p.stem for p in self._suite_paths)
                self.suite_hint.value = f"Suite: {n} files with the same facts — {names}"
            elif path_str:
                self.suite_hint.value = "No sibling files with the same facts — single-file run."
            else:
                self.suite_hint.value = ""
        self._sync_suite_freeze()

    def _sync_suite_freeze(self) -> None:
        """Freeze locks a recipe onto ONE dataset, so it's meaningless for a
        suite run — grey the checkbox while 'Whole suite' is selected. Single
        mode leaves the enabled state to `_apply_frozen_ui`."""
        if self.freeze_check is None:
            return
        if self.suite_mode is not None and self.suite_mode.value == "suite":
            self.freeze_check.value = False
            self.freeze_check.disabled = True

    def _on_suite_mode_change(self, _e: ft.Event | None = None) -> None:
        # Restore the freeze/enabled baseline, then re-grey freeze if suite.
        self._apply_frozen_ui()
        self._sync_suite_freeze()
        self.app.page.update()

    def _suite_selected(self) -> bool:
        """True when 'Whole suite' is chosen AND there are ≥2 members to run."""
        return (
            self.suite_mode is not None
            and self.suite_mode.value == "suite"
            and len(self._suite_paths) >= 2
        )

    def _on_unfreeze_clicked(self, _e: ft.Event) -> None:
        """Unfreeze the dataset — a deliberate, confirmed action that clears the
        frozen flag on disk so the recipe can change again."""
        if not (self.dataset_field and self.dataset_field.value):
            return
        from knowledge_agent.gui.evaluation._common import confirm_dialog

        confirm_dialog(
            self.app,
            title="Unfreeze run settings?",
            message=(
                "This unlocks the recipe so it can be changed again, and clears "
                "the frozen state saved on the dataset."
            ),
            confirm_label="Unfreeze",
            on_confirm=self._do_unfreeze,
        )

    def _do_unfreeze(self) -> None:
        from knowledge_agent.evaluation.models import load_dataset, save_dataset

        path = Path(self.dataset_field.value) if self.dataset_field else None
        if path is None:
            return
        try:
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
        """Persist the current recipe + frozen=true on the dataset — the 'Freeze
        run settings' opt-in, run after a successful run when the box is ticked.
        Only a final dataset can be frozen (the checkbox enforces it too)."""
        from knowledge_agent.evaluation.models import load_dataset, save_dataset

        path = Path(self.dataset_field.value) if self.dataset_field else None
        if path is None:
            return
        try:
            ds = load_dataset(path)
            if ds.status != "final":
                return  # invariant: only a final dataset can be frozen
            if self.recipe_form is not None:
                ds.recipe = self.recipe_form.to_recipe()
            ds.frozen = True
            save_dataset(ds, path)
        except Exception as exc:  # broad: I/O / parse failure → status line
            self._set_status(f"run done, but freeze failed: {exc}")
            return
        self._dataset_frozen = True
        self._apply_frozen_ui()

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
            self._load_dataset_state(Path(files[0].path))
            self._refresh_case_view()
            self.app.page.update()

    def _on_run_clicked(self, _e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        if not (self.recipe_form and self.recipe_form.any_group_selected()):
            self._set_status("Select at least one metric group.")
            return
        if not (self.dataset_field and self.dataset_field.value):
            self._set_status("Select a dataset.")
            return
        if self.trace_check and self.trace_check.value and not get_api_key("langsmith"):
            self._set_status("Set a LangSmith API key in Settings → Keys to trace.")
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
        if suite:
            self._set_busy(False, f"done: suite of {len(outcome.results)} runs — opening Compare.")
            self.coordinator.on_suite_complete(outcome)
            return
        summary = result.report.get("summary", {})
        froze = bool(self.freeze_check and self.freeze_check.value)
        self._set_busy(
            False,
            f"done: run {result.run_id} — "
            f"{summary.get('pass_count')}/{summary.get('case_count')} pass."
            + (" Recipe frozen." if froze else ""),
        )
        # Opt-in freeze: lock this recipe onto the dataset now the run succeeded.
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
