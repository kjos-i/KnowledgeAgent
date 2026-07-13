"""The shared left column for the four Evaluation dashboard tabs.

ONE widget — a **Dataset → Run** drill-down + Refresh + a read-only context
panel (the selected run's recipe + short hash) — mounted identically on Run
Summary, Deep Analysis, Trends, AND Metrics Guide. Even where a tab's body
doesn't consume the selection (Metrics Guide), the column still shows so the
context is always visible. SSOT: the run/dataset selectors + selected-run
metadata that used to be copied into each tab live here once.

A control can only mount in one place, so each tab holds its OWN `DashboardRail`
instance; they share the selection through the coordinator (`selected_dataset`
+ `selected_run_id`) and re-sync from it on `refresh()`. The host tab passes
`on_change`, fired when the selection changes or Refresh is pressed, so it can
re-render its own body from the shared selection.

The context panel reads the RUN's row in the ledger — the immutable snapshot of
what that run actually used (groups, thresholds, judges, kind, `recipe_hash`) —
NOT the (mutable) dataset JSON, so it stays accurate even after the dataset is
edited or deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import FIELD_LABEL_SIZE, PANEL_BG_RAISED, PANEL_RADIUS

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_RAIL_WIDTH = 280


def dataset_of(run: dict[str, Any]) -> str:
    """The run's dataset name — the C1 `dataset_name` column, or the dataset
    file's stem for an older row that predates it."""
    name = run.get("dataset_name")
    if name:
        return str(name)
    return Path(run.get("dataset_path") or "").stem or "?"


def _run_label(run: dict[str, Any]) -> str:
    ts = (run.get("run_timestamp") or "")[:16]
    return f"Run {run['run_id']} — {ts}"


def _parse(raw: Any) -> Any:
    """A ledger JSON column → Python (list / dict), or the raw value untouched."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


class DashboardRail:
    """Shared Dataset→Run selector + Refresh + read-only recipe/hash context."""

    def __init__(
        self, app: GuiApp, coordinator: EvaluationView, *, on_change: Callable[[], None]
    ) -> None:
        self.app = app
        self.coordinator = coordinator
        self._on_change = on_change
        self.dataset_dd: ft.Dropdown | None = None
        self.run_dd: ft.Dropdown | None = None
        self.context: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # Flet 0.85's Dropdown fires `on_select` (NOT on_change) — wiring
        # on_change silently never fires, so a picked run/dataset would be lost
        # and the next refresh would snap back to the newest. Mirror the app's
        # other Dropdowns (corpus selector, dataset-tab dropdowns).
        self.dataset_dd = ft.Dropdown(
            label="Dataset", options=[], width=_RAIL_WIDTH - 24, text_size=FIELD_LABEL_SIZE
        )
        self.dataset_dd.on_select = self._on_dataset_change
        self.run_dd = ft.Dropdown(
            label="Run", options=[], width=_RAIL_WIDTH - 24, text_size=FIELD_LABEL_SIZE
        )
        self.run_dd.on_select = self._on_run_change
        refresh_button = ft.TextButton("Refresh", icon=ft.Icons.REFRESH, on_click=self._on_refresh)
        self.context = ft.Column(controls=[], spacing=2)
        return ft.Container(
            width=_RAIL_WIDTH,
            bgcolor=PANEL_BG_RAISED,
            padding=12,
            border_radius=PANEL_RADIUS,
            content=ft.Column(
                controls=[
                    self.dataset_dd,
                    self.run_dd,
                    refresh_button,
                    ft.Divider(),
                    self.context,
                ],
                spacing=8,
            ),
        )

    # ---- data / sync ------------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload runs from the active corpus's ledger + sync the dropdowns and
        context to the shared selection. Safe before any run exists."""
        self._runs = self._ledger().list_runs()
        self._sync_controls()

    def _datasets(self) -> list[str]:
        out: list[str] = []
        for r in self._runs:
            d = dataset_of(r)
            if d not in out:
                out.append(d)
        return out

    def _sync_controls(self) -> None:
        """Populate the dropdowns from the loaded runs, honouring the shared
        selection (falling back to the newest dataset / run), then render the
        context panel for the selected run."""
        if self.dataset_dd is None or self.run_dd is None:
            return
        datasets = self._datasets()
        self.dataset_dd.options = [ft.DropdownOption(key=d, text=d) for d in datasets]
        # If a run is already selected (e.g. just finished, or user-picked), its
        # dataset wins — so a fresh run shows even if the filter pointed
        # elsewhere. Otherwise honour the dataset filter, falling back to newest.
        sel_run = next(
            (r for r in self._runs if r["run_id"] == self.coordinator.selected_run_id), None
        )
        if sel_run is not None:
            ds = dataset_of(sel_run)
        else:
            ds = self.coordinator.selected_dataset
            if ds not in datasets:
                ds = datasets[0] if datasets else None
        self.coordinator.selected_dataset = ds
        self.dataset_dd.value = ds
        # Runs scoped to the selected dataset (list_runs is newest-first).
        runs = [r for r in self._runs if dataset_of(r) == ds]
        self.run_dd.options = [
            ft.DropdownOption(key=str(r["run_id"]), text=_run_label(r)) for r in runs
        ]
        run_ids = [r["run_id"] for r in runs]
        run_id = self.coordinator.selected_run_id
        if run_id not in run_ids:
            run_id = run_ids[0] if run_ids else None
        self.coordinator.selected_run_id = run_id
        self.run_dd.value = str(run_id) if run_id is not None else None
        self._render_context(self._ledger().get_run(run_id) if run_id is not None else None)

    def _render_context(self, run: dict[str, Any] | None) -> None:
        """The read-only recipe panel for the selected run — sourced from the
        run's ledger row (what it actually used), not the dataset JSON."""
        if self.context is None:
            return
        if run is None:
            self.context.controls = [
                ft.Text("No run selected.", size=12, italic=True, color=ft.Colors.GREY_500)
            ]
            return
        groups = _parse(run.get("enabled_groups")) or []
        thresholds = _parse(run.get("gate_thresholds")) or {}
        judges = _parse(run.get("judge_models")) or []
        rhash = run.get("recipe_hash")
        lines: list[ft.Control] = [
            ft.Text("Recipe (this run)", weight=ft.FontWeight.BOLD, size=12),
            ft.Text(f"Dataset: {dataset_of(run)}", size=12),
            ft.Text(f"Type: {run.get('dataset_kind') or 'custom'}", size=12),
            ft.Text(f"Recipe hash: {rhash[:8] if rhash else '—'}", size=12),
            ft.Text(f"Groups: {', '.join(groups) if groups else '(none)'}", size=12),
            ft.Text(f"Judges: {', '.join(judges) if judges else '(default)'}", size=12),
        ]
        lines += [ft.Text(f"{k}: {v}", size=12) for k, v in thresholds.items()]
        lines.append(ft.Text(f"Cases: {run.get('case_count')}", size=12))
        model = run.get("synthesizer_model")
        if model:
            lines.append(ft.Text(f"Model: {run.get('llm_provider') or '?'}/{model}", size=12))
        self.context.controls = lines

    # ---- handlers ---------------------------------------------------------

    def _on_dataset_change(self, _e: ft.Event) -> None:
        if self.dataset_dd is None:
            return
        self.coordinator.selected_dataset = self.dataset_dd.value
        self.coordinator.selected_run_id = None  # re-pick the newest run in it
        self._sync_controls()
        self.app.page.update()
        self._on_change()

    def _on_run_change(self, _e: ft.Event) -> None:
        if self.run_dd and self.run_dd.value:
            self.coordinator.selected_run_id = int(self.run_dd.value)
            self._render_context(self._ledger().get_run(self.coordinator.selected_run_id))
        self.app.page.update()
        self._on_change()

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()
        self.app.page.update()
        self._on_change()
