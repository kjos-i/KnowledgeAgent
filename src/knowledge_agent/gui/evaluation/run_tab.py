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
        # profile) — one shared widget with the Create-test-cases tab. Here it
        # is loaded from the selected dataset's saved recipe and rendered FROZEN;
        # Unfreeze deviates for a single run (never written back).
        self.recipe_form: RecipeForm | None = None
        self.unfreeze_button: ft.TextButton | None = None
        self.deviation_note: ft.Text | None = None
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

        # The recipe editor, shared with the Create-test-cases tab. Built here,
        # loaded + frozen when a dataset is chosen (`_load_recipe_from`).
        self.recipe_form = RecipeForm(self.app)
        recipe_body = self.recipe_form.build()
        # Frozen until Unfreeze — the recipe belongs to the dataset; the Run tab
        # only displays it (and lets a single run deviate).
        self.recipe_form.set_enabled(False)
        self.unfreeze_button = ft.TextButton(
            "Unfreeze for a one-off run",
            icon=ft.Icons.LOCK_OPEN,
            tooltip="Edit the recipe for THIS run only — the dataset's saved recipe is unchanged",
            on_click=self._on_unfreeze_clicked,
        )
        self.deviation_note = ft.Text(
            "Editing the recipe for THIS run only — not saved to the dataset.",
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
                section_divider(),
                # ============ Section: Recipe ============
                # The dataset's saved recipe (metric groups + judge panel + gate
                # thresholds + profile), loaded read-only. Unfreeze to deviate
                # for one run without touching the dataset.
                section_title("Recipe"),
                recipe_body,
                ft.Row(
                    [self.unfreeze_button, self.deviation_note],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=8,
                    wrap=True,
                ),
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
                ft.Row(
                    [self.run_button, self.progress],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
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

    def _load_recipe_from(self, path: Path) -> None:
        """Load the dataset's saved recipe into the (frozen) recipe form. A
        legacy dataset with no recipe loads the harness defaults. Re-freezes +
        resets the Unfreeze affordance on every dataset change."""
        from knowledge_agent.evaluation.models import load_dataset

        recipe = None
        try:
            recipe = load_dataset(path).recipe
        except Exception:  # broad: a bad dataset already surfaces in the case view
            recipe = None
        if self.recipe_form is not None:
            self.recipe_form.load(recipe)
            self.recipe_form.set_enabled(False)
        if self.unfreeze_button is not None:
            self.unfreeze_button.visible = True
        if self.deviation_note is not None:
            self.deviation_note.visible = False

    def _on_unfreeze_clicked(self, _e: ft.Event) -> None:
        """Make the recipe editable for a ONE-OFF run — the change is used by
        this run only and never written back to the dataset."""
        if self.recipe_form is not None:
            self.recipe_form.set_enabled(True)
        if self.unfreeze_button is not None:
            self.unfreeze_button.visible = False
        if self.deviation_note is not None:
            self.deviation_note.visible = True
        self.app.page.update()

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
            self._load_recipe_from(Path(files[0].path))
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

        try:
            cfg = self._build_config()
        except Exception as exc:  # broad: report any bad form input in the status line
            self._set_status(f"config error: {exc}")
            return
        self._set_busy(True, "starting…")
        if self.progress is not None:
            self.progress.value = 0.0
        trace = bool(self.trace_check and self.trace_check.value)
        project = (self.project_field.value or "").strip() if self.project_field else ""
        try:
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
        summary = result.report.get("summary", {})
        self._set_busy(
            False,
            f"done: run {result.run_id} — "
            f"{summary.get('pass_count')}/{summary.get('case_count')} pass.",
        )
        self.coordinator.on_run_complete(result.run_id)

    def _build_config(self) -> EvalConfig:
        from knowledge_agent.evaluation.config import load_eval_config
        from knowledge_agent.gui.evaluation._common import active_corpus_config_path

        # The (possibly deviated) recipe on the form drives the run's metric
        # groups, judge panel, and the three gate thresholds. dataset_kind is
        # NOT read here — it's provenance the runner stamps from the dataset's
        # SAVED recipe, so the ledger records which canonical recipe this run
        # is associated with even when the form deviated.
        recipe = self.recipe_form.to_recipe() if self.recipe_form else None
        overrides: dict = {}
        if recipe is not None:
            overrides["enabled_groups"] = frozenset(recipe.enabled_groups)
            overrides["judge_models"] = tuple(recipe.judge_models)
            overrides["judge_threshold"] = recipe.judge_threshold
            overrides["metadata_match_threshold"] = recipe.metadata_match_threshold
            overrides["required_keyword_threshold"] = recipe.required_keyword_threshold
        if self.dataset_field and self.dataset_field.value:
            overrides["dataset_path"] = Path(self.dataset_field.value)
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
