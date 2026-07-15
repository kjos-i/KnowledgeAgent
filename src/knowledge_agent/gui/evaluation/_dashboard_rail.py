"""The shared left column for the five Evaluation dashboard tabs.

ONE widget — the shared left column for EVERY dashboard tab (Run Summary, Run
Charts, Compare Datasets, Trends, Metrics Guide), so the column is IDENTICAL
everywhere. It is a **Suite → Dataset → Run** cascade + Refresh + a read-only
run-info panel:

  - **Suite** — a group of dataset files sharing a `facts_hash` (the same facts,
    swept knobs). Picking one scopes the Dataset list to that suite's members.
  - **Dataset** — one member file. Scopes the Run picker + Trends (by that
    dataset's `dataset_hash`).
  - **Run** — one run of the selected dataset. Drives Run Summary / Run Charts,
    and — via its `suite_run_id` — Compare (that suite execution's members).

Each tab's BODY consumes only what it needs; the selectors + selected-run
metadata live here once (SSOT). A control can only mount in one place, so each
tab holds its OWN `DashboardRail` instance; they share state through the
coordinator (`selected_suite` / `selected_dataset` / `selected_run_id`) and
re-sync from it on `refresh()`. The host tab passes `on_change`, fired when any
selection changes or Refresh is pressed, so it re-renders its own body.

The context panel reads the RUN's row in the ledger — the immutable snapshot of
what that run actually used (groups, thresholds, judges, hashes) — NOT the
(mutable) dataset JSON, so it stays accurate even after the dataset is edited or
deleted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import (
    FIELD_LABEL_SIZE,
    PANEL_BG_RAISED,
    PANEL_RADIUS,
    sub_section_header,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_RAIL_WIDTH = 280
# The suite key for legacy runs recorded before facts_hash existed (NULL). A
# real facts_hash is 64 hex chars, so this sentinel never collides.
_NO_SUITE = "∅"


def dataset_of(run: dict[str, Any]) -> str:
    """The run's dataset name — the C1 `dataset_name` column, or the dataset
    file's stem for an older row that predates it."""
    name = run.get("dataset_name")
    if name:
        return str(name)
    return Path(run.get("dataset_path") or "").stem or "?"


def _suite_key(run: dict[str, Any]) -> str:
    """The run's suite identity — its `facts_hash`, or the legacy sentinel."""
    return run.get("facts_hash") or _NO_SUITE


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
    """Shared Suite→Dataset→Run selector + Refresh + read-only run context."""

    def __init__(
        self, app: GuiApp, coordinator: EvaluationView, *, on_change: Callable[[], None]
    ) -> None:
        self.app = app
        self.coordinator = coordinator
        self._on_change = on_change
        self.suite_dd: ft.Dropdown | None = None
        self.dataset_dd: ft.Dropdown | None = None
        self.run_dd: ft.Dropdown | None = None
        self.context: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # Flet 0.85's Dropdown fires `on_select` (NOT on_change) — wiring
        # on_change silently never fires, so a picked value would be lost and the
        # next refresh would snap back to the newest. Mirror the app's other
        # Dropdowns (corpus selector, dataset-tab dropdowns).
        self.suite_dd = ft.Dropdown(
            label="Suite", options=[], width=_RAIL_WIDTH - 24, text_size=FIELD_LABEL_SIZE
        )
        self.suite_dd.on_select = self._on_suite_change
        self.dataset_dd = ft.Dropdown(
            label="Dataset", options=[], width=_RAIL_WIDTH - 24, text_size=FIELD_LABEL_SIZE
        )
        self.dataset_dd.on_select = self._on_dataset_change
        self.run_dd = ft.Dropdown(
            label="Run", options=[], width=_RAIL_WIDTH - 24, text_size=FIELD_LABEL_SIZE
        )
        self.run_dd.on_select = self._on_run_change
        refresh_button = ft.TextButton("Refresh", on_click=self._on_refresh)
        self.context = ft.Column(controls=[], spacing=2)
        return ft.Container(
            width=_RAIL_WIDTH,
            bgcolor=PANEL_BG_RAISED,
            padding=12,
            border_radius=PANEL_RADIUS,
            content=ft.Column(
                controls=[
                    refresh_button,
                    sub_section_header("Selection"),
                    self.suite_dd,
                    self.dataset_dd,
                    self.run_dd,
                    self.context,
                ],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
        )

    # ---- data / sync ------------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload runs from the active corpus's ledger + sync the cascade and
        context to the shared selection. Safe before any run exists."""
        self._runs = self._ledger().list_runs()
        self._sync_controls()

    def _suites(self) -> list[tuple[str, str]]:
        """The distinct suites in the corpus as (key, label). A suite groups the
        runs sharing a `facts_hash` (same facts, swept knobs); its label lists the
        member dataset names. Legacy runs with no facts_hash fall under one
        '(no suite)' group. Order follows first appearance (list_runs is newest
        first)."""
        order: list[str] = []
        members: dict[str, list[str]] = {}
        for r in self._runs:
            key = _suite_key(r)
            d = dataset_of(r)
            if key not in members:
                members[key] = []
                order.append(key)
            if d not in members[key]:
                members[key].append(d)
        out: list[tuple[str, str]] = []
        for key in order:
            names = ", ".join(members[key])
            if len(names) > 40:
                names = names[:39] + "…"
            out.append((key, f"(no suite) {names}" if key == _NO_SUITE else names))
        return out

    def _sync_controls(self) -> None:
        """Populate the Suite → Dataset → Run cascade from the loaded runs,
        honouring the shared selection (a selected run's suite/dataset win, else
        fall back to the newest), then render the context for the selected run."""
        if self.suite_dd is None or self.dataset_dd is None or self.run_dd is None:
            return
        sel_run = next(
            (r for r in self._runs if r["run_id"] == self.coordinator.selected_run_id), None
        )
        # ---- Suite ----
        suites = self._suites()
        suite_keys = [k for k, _ in suites]
        self.suite_dd.options = [ft.DropdownOption(key=k, text=t) for k, t in suites]
        if sel_run is not None:
            suite_key = _suite_key(sel_run)
        else:
            suite_key = self.coordinator.selected_suite
            if suite_key not in suite_keys:
                suite_key = suite_keys[0] if suite_keys else None
        self.coordinator.selected_suite = suite_key
        self.suite_dd.value = suite_key
        suite_runs = [r for r in self._runs if _suite_key(r) == suite_key]
        # ---- Dataset (scoped to the suite's members) ----
        datasets: list[str] = []
        for r in suite_runs:
            d = dataset_of(r)
            if d not in datasets:
                datasets.append(d)
        self.dataset_dd.options = [ft.DropdownOption(key=d, text=d) for d in datasets]
        if sel_run is not None:
            ds = dataset_of(sel_run)
        else:
            ds = self.coordinator.selected_dataset
            if ds not in datasets:
                ds = datasets[0] if datasets else None
        self.coordinator.selected_dataset = ds
        self.dataset_dd.value = ds
        # ---- Run (scoped to the dataset) ----
        runs = [r for r in suite_runs if dataset_of(r) == ds]
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
        fhash = run.get("facts_hash")
        lines: list[ft.Control] = [
            ft.Text("Run Information", weight=ft.FontWeight.BOLD, size=12),
            ft.Text(f"Dataset: {dataset_of(run)}", size=12),
            ft.Text(f"Facts hash: {fhash[:8] if fhash else '—'}", size=12),
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

    def _on_suite_change(self, _e: ft.Event) -> None:
        if self.suite_dd is None:
            return
        self.coordinator.selected_suite = self.suite_dd.value
        self.coordinator.selected_dataset = None  # re-pick within the new suite
        self.coordinator.selected_run_id = None
        self._sync_controls()
        self.app.page.update()
        self._on_change()

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
