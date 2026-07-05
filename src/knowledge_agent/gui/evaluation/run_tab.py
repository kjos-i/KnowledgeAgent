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

from knowledge_agent.gui.config_store import get_api_key
from knowledge_agent.gui.settings.llm_tab import LLM_AVAILABLE_MODELS
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

logger = logging.getLogger(__name__)

# Metric-group toggles + their defaults. judge is OFF by default — it calls
# LLM judges and costs money; source/chunk/kg are deterministic + free.
_GROUP_DEFAULTS: dict[str, bool] = {"source": True, "chunk": True, "kg": True, "judge": False}


class RunTab:
    """Run-configuration + trigger sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self._bg_tasks: set[asyncio.Task] = set()
        self._busy = False
        self.dataset_dropdown: ft.Dropdown | None = None
        self.group_checks: dict[str, ft.Checkbox] = {}
        self.judge_panel: ft.Column | None = None
        self.judge_dropdowns: list[ft.Dropdown] = []
        self.add_judge_button: ft.TextButton | None = None
        self.max_cases_field: ft.TextField | None = None
        self.trace_check: ft.Checkbox | None = None
        self.project_field: ft.TextField | None = None
        self.trace_warning: ft.Text | None = None
        self.run_button: ft.Button | None = None
        self.progress: ft.ProgressBar | None = None
        self.status: ft.Text | None = None
        self.output_line: ft.Text | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.config import (
            DEFAULT_DATASET_PATH,
            DEFAULT_LANGSMITH_PROJECT,
        )
        from knowledge_agent.gui.evaluation._common import active_output_dir

        datasets_dir = DEFAULT_DATASET_PATH.parent
        options = [
            ft.DropdownOption(key=str(p), text=p.name) for p in sorted(datasets_dir.glob("*.json"))
        ]
        self.dataset_dropdown = ft.Dropdown(
            label="Dataset",
            editable=True,  # editable = pick a packaged set OR type/paste a path
            options=options,
            value=str(DEFAULT_DATASET_PATH),
            width=460,
        )
        browse_button = ft.IconButton(
            icon=ft.Icons.FOLDER_OPEN,
            tooltip="Browse for a gold-dataset JSON",
            on_click=self._on_browse_clicked,
        )

        active = getattr(self.app.gui_config, "active_corpus_name", None)
        corpus_line = ft.Text(
            f"Active corpus: {active}" if active else "Active corpus: none — select one in Library",
            italic=not active,
            color=None if active else ft.Colors.ORANGE,
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

        self.group_checks = {
            group: ft.Checkbox(
                label=group,
                value=_GROUP_DEFAULTS[group],
                on_change=self._on_judge_toggle if group == "judge" else None,
            )
            for group in _GROUP_DEFAULTS
        }

        self.judge_dropdowns = []
        judge_on = self.group_checks["judge"].value
        self.judge_panel = ft.Column(controls=[], visible=judge_on, spacing=6)
        self.add_judge_button = ft.TextButton(
            "Add judge model",
            icon=ft.Icons.ADD,
            on_click=self._on_add_judge_clicked,
            visible=judge_on,
        )

        self.max_cases_field = ft.TextField(
            label="Max cases (blank = all)",
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
        self.trace_warning = ft.Text(
            "⚠ Tracing uploads this run's data — your queries, the retrieved "
            "chunk text, and the answers — to LangSmith's cloud. Enable ONLY "
            "for a non-sensitive corpus (e.g. the test corpus); never for "
            "private documents.",
            size=12,
            color=ft.Colors.ORANGE,
            visible=False,
        )
        self.project_field = ft.TextField(
            label="LangSmith project",
            value=DEFAULT_LANGSMITH_PROJECT,
            width=300,
            visible=False,
        )

        self.run_button = ft.Button(
            "Run evaluation",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._on_run_clicked,
        )
        self.progress = ft.ProgressBar(visible=False, width=460)
        self.status = ft.Text("", color=ft.Colors.GREY_600)

        form = ft.Column(
            controls=[
                corpus_line,
                self.output_line,
                ft.Row([self.dataset_dropdown, browse_button], spacing=4),
                ft.Text("Metric groups", weight=ft.FontWeight.BOLD),
                ft.Row([self.group_checks[g] for g in _GROUP_DEFAULTS], wrap=True),
                ft.Text(
                    "Judge is off by default — it calls LLM judges and costs money. Leave the "
                    "panel empty to use one default judge from your active provider.",
                    size=12,
                    color=ft.Colors.GREY_600,
                ),
                self.judge_panel,
                self.add_judge_button,
                self.max_cases_field,
                ft.Text("Tracing (optional)", weight=ft.FontWeight.BOLD),
                self.trace_check,
                self.trace_warning,
                self.project_field,
                ft.Row(
                    [self.run_button, self.progress],
                    spacing=12,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.status,
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=12,
        )
        return ft.Column(controls=[view_header("Run Evaluation"), form], expand=True, spacing=8)

    # ---- judge panel ------------------------------------------------------

    def _judge_options(self) -> list[ft.DropdownOption]:
        provider = getattr(self.app.gui_config, "llm_provider", "")
        return [ft.DropdownOption(key=m, text=m) for m in LLM_AVAILABLE_MODELS.get(provider, ())]

    def _on_judge_toggle(self, _e: ft.Event) -> None:
        on = self.group_checks["judge"].value
        if self.judge_panel is not None:
            self.judge_panel.visible = on
        if self.add_judge_button is not None:
            self.add_judge_button.visible = on
        self.app.page.update()

    def _on_add_judge_clicked(self, _e: ft.Event) -> None:
        self._add_judge_row()
        self.app.page.update()

    def _add_judge_row(self) -> None:
        dropdown = ft.Dropdown(
            label="Judge model",
            editable=True,
            options=self._judge_options(),
            width=380,
        )
        self.judge_dropdowns.append(dropdown)
        row = ft.Row(spacing=4)
        row.controls = [
            dropdown,
            ft.IconButton(
                icon=ft.Icons.CLOSE,
                tooltip="Remove this judge",
                on_click=lambda _e, d=dropdown, r=row: self._remove_judge_row(d, r),
            ),
        ]
        if self.judge_panel is not None:
            self.judge_panel.controls.append(row)

    def _remove_judge_row(self, dropdown: ft.Dropdown, row: ft.Row) -> None:
        if dropdown in self.judge_dropdowns:
            self.judge_dropdowns.remove(dropdown)
        if self.judge_panel is not None and row in self.judge_panel.controls:
            self.judge_panel.controls.remove(row)
        self.app.page.update()

    # ---- tracing ----------------------------------------------------------

    def _on_trace_toggle(self, _e: ft.Event) -> None:
        """Reveal the data-safety warning + project field only when tracing
        is enabled — so the warning is impossible to miss at opt-in time."""
        on = bool(self.trace_check and self.trace_check.value)
        if self.trace_warning is not None:
            self.trace_warning.visible = on
        if self.project_field is not None:
            self.project_field.visible = on
        self.app.page.update()

    # ---- run --------------------------------------------------------------

    def _on_browse_clicked(self, _e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        self._spawn(self._pick_dataset())

    async def _pick_dataset(self) -> None:
        try:
            files = await self.app.file_picker.pick_files(
                dialog_title="Pick a gold-dataset JSON",
                allowed_extensions=["json"],
            )
        except Exception as exc:  # broad: surface any picker failure in the status line
            self._set_status(f"file picker error: {exc}")
            return
        if files and files[0].path and self.dataset_dropdown is not None:
            self.dataset_dropdown.value = files[0].path
            self.app.page.update()

    def _on_run_clicked(self, _e: ft.Event) -> None:
        if self._busy or not self._loop_running():
            return
        if not any(cb.value for cb in self.group_checks.values()):
            self._set_status("Select at least one metric group.")
            return
        if not (self.dataset_dropdown and self.dataset_dropdown.value):
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

        groups = frozenset(g for g, cb in self.group_checks.items() if cb.value)
        judge_models = tuple(
            d.value.strip() for d in self.judge_dropdowns if d.value and d.value.strip()
        )
        overrides: dict = {
            "dataset_path": Path(self.dataset_dropdown.value),  # type: ignore[union-attr]
            "enabled_groups": groups,
            "judge_models": judge_models,
        }
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
