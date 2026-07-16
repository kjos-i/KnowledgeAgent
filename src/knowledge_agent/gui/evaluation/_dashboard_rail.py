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
    section_divider,
    sub_section_header,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_RAIL_WIDTH = 280
# Grouping key for every run NOT executed as a named suite — single-file runs
# and legacy rows alike. Suite is a by-name concept (R7), so anything without a
# `suite` name falls in this one "— No suite —" bucket.
_NO_SUITE = "∅"


def dataset_of(run: dict[str, Any]) -> str:
    """The run's dataset name — the C1 `dataset_name` column, or the dataset
    file's stem for an older row that predates it."""
    name = run.get("dataset_name")
    if name:
        return str(name)
    return Path(run.get("dataset_path") or "").stem or "?"


def _suite_key(run: dict[str, Any]) -> str:
    """The run's suite for grouping — the named `suite` it was executed as, else
    the '— No suite —' sentinel. Suite is a by-name concept (R7): a single-file
    or legacy run belongs to no suite, so it groups under the sentinel rather
    than by facts_hash."""
    return run.get("suite") or _NO_SUITE


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
        # Origin filter — one checkbox per case provenance (Manual/LLM/Search/
        # Chat). Ticking a subset re-scopes the per-run view to those cases; all
        # ticked = no filter. Shared through the coordinator like the cascade.
        self.origin_checks: dict[str, ft.Checkbox] = {}
        self.context: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # Flet 0.85's Dropdown fires `on_select` (NOT on_change) — wiring
        # on_change silently never fires, so a picked value would be lost and the
        # next refresh would snap back to the newest. Mirror the app's other
        # Dropdowns (corpus selector, dataset-tab dropdowns).
        # `hint_text` shows only when a dropdown has no value — which, given the
        # newest-run auto-fallback, happens exactly when the ledger is empty. So
        # "No runs yet" reads as the empty-ledger default across all three.
        self.suite_dd = ft.Dropdown(
            label="Suite",
            options=[],
            width=_RAIL_WIDTH - 24,
            text_size=FIELD_LABEL_SIZE,
            hint_text="No runs yet",
        )
        self.suite_dd.on_select = self._on_suite_change
        self.dataset_dd = ft.Dropdown(
            label="Dataset",
            options=[],
            width=_RAIL_WIDTH - 24,
            text_size=FIELD_LABEL_SIZE,
            hint_text="No runs yet",
        )
        self.dataset_dd.on_select = self._on_dataset_change
        self.run_dd = ft.Dropdown(
            label="Run",
            options=[],
            width=_RAIL_WIDTH - 24,
            text_size=FIELD_LABEL_SIZE,
            hint_text="No runs yet",
        )
        self.run_dd.on_select = self._on_run_change
        refresh_button = ft.TextButton("Refresh", on_click=self._on_refresh)
        # Origin filter — sits below the Run picker, above the info panel. Metrics
        # recompute over the checked origins (Run Summary / Charts / Compare).
        from knowledge_agent.gui.evaluation._common import ALL_ORIGINS, ORIGIN_LABELS

        self.origin_checks = {
            origin: ft.Checkbox(
                label=ORIGIN_LABELS[origin], value=True, on_change=self._on_origin_toggle
            )
            for origin in ALL_ORIGINS
        }
        origin_filter = ft.Column(
            controls=[
                sub_section_header("Filter by test case origin"),
                *self.origin_checks.values(),
            ],
            spacing=0,
        )
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
                    origin_filter,
                    # Divider between the origin filter and the read-only info panel.
                    section_divider(),
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
        """The distinct suites among the corpus's runs as (key, label): each NAMED
        suite by its name, plus a single '— No suite —' bucket for every run not
        executed as a named suite (single-file / legacy). Order follows first
        appearance (list_runs is newest first)."""
        order: list[str] = []
        seen: set[str] = set()
        for r in self._runs:
            key = _suite_key(r)
            if key not in seen:
                seen.add(key)
                order.append(key)
        return [(k, "— No suite —" if k == _NO_SUITE else k) for k in order]

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
        self._sync_origin_checks()

    def _sync_origin_checks(self) -> None:
        """Reflect the coordinator's shared origin filter onto this rail's
        checkboxes (each view tab holds its own rail; they sync on refresh)."""
        selected = self.coordinator.selected_origins
        for origin, check in self.origin_checks.items():
            check.value = origin in selected

    def _render_context(self, run: dict[str, Any] | None) -> None:
        """The read-only info panel for the selected run — sourced from the run's
        ledger row (what it actually used), not the dataset JSON. Split into three
        sections, each anchored by one of the three hashes: Suite (facts hash),
        Dataset (knobs hash), Run (run-settings hash)."""
        if self.context is None:
            return
        if run is None:
            self.context.controls = [
                ft.Text("No runs yet.", size=12, italic=True, color=ft.Colors.GREY_500)
            ]
            return
        groups = _parse(run.get("enabled_groups")) or []
        thresholds = _parse(run.get("gate_thresholds")) or {}
        judges = _parse(run.get("judge_models")) or []

        def _hash(h: Any) -> str:
            return h[:8] if h else "—"

        def _header(text: str) -> ft.Text:
            return ft.Text(text, weight=ft.FontWeight.BOLD, size=12)

        # Judge panel: the resolved models that ran; "(off)" when the judge group
        # wasn't run; "(default)" for a legacy run that didn't record the panel.
        if "judge" not in groups:
            judge_line = "(off)"
        elif judges:
            judge_line = ", ".join(judges)
        else:
            judge_line = "(default)"

        lines: list[ft.Control] = [
            # ---- Suite (facts hash = the suite identity) ----
            _header("Suite Information"),
            ft.Text(f"Suite: {run.get('suite') or '— No suite —'}", size=12),
            ft.Text(f"Facts hash: {_hash(run.get('facts_hash'))}", size=12),
            ft.Container(height=6),
            # ---- Dataset (knobs hash = what tells suite members apart) ----
            _header("Dataset Information"),
            ft.Text(f"Dataset: {dataset_of(run)}", size=12),
            ft.Text(f"Knobs hash: {_hash(run.get('knob_hash'))}", size=12),
            ft.Container(height=6),
            # ---- Run (run-settings hash = the recipe that scored it) ----
            _header("Run Information"),
            ft.Text(f"Run settings hash: {_hash(run.get('recipe_hash'))}", size=12),
            ft.Text(f"Metric groups: {', '.join(groups) if groups else '(none)'}", size=12),
            ft.Text(f"Judge panel: {judge_line}", size=12),
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

    def _on_origin_toggle(self, _e: ft.Event) -> None:
        """A provenance checkbox flipped — update the shared filter + re-render
        the host tab's body (which recomputes its metrics over the subset)."""
        self.coordinator.selected_origins = {
            origin for origin, check in self.origin_checks.items() if check.value
        }
        self.app.page.update()
        self._on_change()
