"""Evaluation → Run Summary sub-tab — KPIs + per-case table for one run.

Left rail: run selector + selected-run metadata (dataset, enabled groups,
thresholds, cases) + Refresh. Body: registry-driven KPI sections (derived
from `METRICS` display groups, not hardcoded lists) + a per-case `DataTable`
with threshold-colored status/score cells.

Reads the ledger via the `EvalLedger` readers — but only on `refresh()`
(run-complete, run-selection, or the Refresh button), never at build time
(GUI startup rule: no fetches in `build`). No polling: the GUI runs eval
in-process, so it refreshes exactly when a run finishes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import FRAME_BORDER_COLOR
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

# Display-group → section header + order. Registry `MetricGroup` values that
# carry run-level averages worth a KPI section.
_GROUP_HEADERS: dict[str, str] = {
    "retrieval": "Source Retrieval",
    "chunk": "Chunk Retrieval",
    "kg": "Knowledge Graph",
    "llm": "Judge Scores",
    "keyword": "Keywords",
    "citation": "Grounding",
    "tokens": "Tokens",
    "latency": "Latency",
}
_GROUP_ORDER: tuple[str, ...] = tuple(_GROUP_HEADERS)

# Per-case table: a compact, cross-group set of columns (key, label).
_CASE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("hit_at_k", "Hit@k"),
    ("chunk_hit_at_k", "Chunk Hit@k"),
    ("kg_hit_at_k", "KG Hit@k"),
    ("avg_judge_score", "Judge"),
    ("latency_seconds", "Latency"),
)


def _score_color(value: Any) -> str:
    """Green ≥ 0.8, orange ≥ 0.5, red below, grey when not evaluated."""
    if not isinstance(value, (int, float)):
        return ft.Colors.GREY_500
    if value >= 0.8:
        return ft.Colors.GREEN
    if value >= 0.5:
        return ft.Colors.ORANGE
    return ft.Colors.RED


class RunSummaryTab:
    """Per-run KPI + per-case-table sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.run_dropdown: ft.Dropdown | None = None
        self.metadata: ft.Column | None = None
        self.body: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # Flet 0.85's Dropdown takes on_change as an attribute, not a ctor kwarg.
        self.run_dropdown = ft.Dropdown(label="Run", options=[], width=240)
        self.run_dropdown.on_change = self._on_select_run
        self.metadata = ft.Column(controls=[], spacing=2)
        refresh_button = ft.TextButton("Refresh", icon=ft.Icons.REFRESH, on_click=self._on_refresh)
        rail = ft.Container(
            width=280,
            content=ft.Column(
                controls=[
                    ft.Text("Run", weight=ft.FontWeight.BOLD),
                    self.run_dropdown,
                    refresh_button,
                    ft.Divider(),
                    self.metadata,
                ],
                spacing=8,
            ),
        )
        self.body = ft.Column(
            controls=[
                ft.Text("Run an evaluation, or press Refresh to load past runs.", italic=True)
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=14,
        )
        return ft.Column(
            controls=[
                view_header("Run Summary"),
                ft.Row(
                    [rail, self.body], expand=True, vertical_alignment=ft.CrossAxisAlignment.START
                ),
            ],
            expand=True,
            spacing=8,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload runs from the ledger + render the selected (or newest) run.
        Safe before any run exists (shows an empty state)."""
        if self.body is None or self.run_dropdown is None:
            return
        self._runs = self._ledger().list_runs()
        if not self._runs:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.run_dropdown.options = []
            self._render_metadata(None)
            self.app.page.update()
            return

        self.run_dropdown.options = [
            ft.DropdownOption(key=str(r["run_id"]), text=self._run_label(r)) for r in self._runs
        ]
        run_id = self.coordinator.selected_run_id or self._runs[0]["run_id"]
        self.coordinator.selected_run_id = run_id
        self.run_dropdown.value = str(run_id)

        ledger = self._ledger()
        run = ledger.get_run(run_id)
        cases = ledger.get_run_cases(run_id)
        self._render_metadata(run)
        self.body.controls = self._build_body(run, cases)
        self.app.page.update()

    @staticmethod
    def _run_label(run: dict[str, Any]) -> str:
        from pathlib import Path

        ts = (run.get("run_timestamp") or "")[:16]
        dataset = Path(run.get("dataset_path") or "").name or "?"
        return f"Run {run['run_id']} — {ts} ({dataset})"

    def _render_metadata(self, run: dict[str, Any] | None) -> None:
        if self.metadata is None:
            return
        if run is None:
            self.metadata.controls = []
            return
        import json
        from pathlib import Path

        def _parse(raw: Any) -> Any:
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except ValueError:
                    return raw
            return raw

        groups = _parse(run.get("enabled_groups")) or []
        thresholds = _parse(run.get("gate_thresholds")) or {}
        lines = [
            ft.Text("Run details", weight=ft.FontWeight.BOLD),
            ft.Text(f"Dataset: {Path(run.get('dataset_path') or '').name or '?'}", size=12),
            ft.Text(f"Cases: {run.get('case_count')}", size=12),
            ft.Text(f"Groups: {', '.join(groups) if groups else '(none)'}", size=12),
        ]
        lines += [ft.Text(f"{k}: {v}", size=12) for k, v in thresholds.items()]
        if run.get("git_commit"):
            lines.append(ft.Text(f"commit: {run['git_commit'][:8]}", size=12))
        self.metadata.controls = lines

    # ---- body: KPIs + table ----------------------------------------------

    def _build_body(self, run: dict[str, Any], cases: list[dict[str, Any]]) -> list[ft.Control]:
        return [self._headline(run), *self._kpi_sections(run), self._case_table(cases)]

    def _headline(self, run: dict[str, Any]) -> ft.Control:
        pr = run.get("pass_rate")
        pr_text = f"{pr:.0%}" if isinstance(pr, (int, float)) else "—"
        return ft.Row(
            [
                self._kpi_card("Pass rate", pr_text, _score_color(pr)),
                self._kpi_card("Passed", f"{run.get('pass_count')}/{run.get('case_count')}", None),
            ],
            wrap=True,
            spacing=10,
        )

    def _kpi_sections(self, run: dict[str, Any]) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import METRICS, metric_fmts

        fmts = metric_fmts()
        sections: list[ft.Control] = []
        for group in _GROUP_ORDER:
            metrics = [m for m in METRICS if m.group == group and m.summary_avg_key]
            if not metrics:
                continue
            cards = [
                self._kpi_card(
                    m.summary_avg_label or m.label,
                    self._fmt(run.get(m.summary_avg_key), fmts.get(m.summary_avg_key, ".2f")),
                    _score_color(run.get(m.summary_avg_key))
                    if group not in ("tokens", "latency")
                    else None,
                )
                for m in metrics
            ]
            sections.append(
                ft.Column(
                    [
                        ft.Text(_GROUP_HEADERS[group], weight=ft.FontWeight.BOLD),
                        ft.Row(cards, wrap=True, spacing=10),
                    ],
                    spacing=6,
                )
            )
        return sections

    @staticmethod
    def _fmt(value: Any, fmt: str) -> str:
        if value is None:
            return "Not evaluated"
        try:
            return format(value, fmt)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _kpi_card(label: str, value: str, color: str | None) -> ft.Control:
        return ft.Container(
            width=150,
            padding=10,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=8,
            content=ft.Column(
                [
                    ft.Text(label, size=11, color=ft.Colors.GREY_600),
                    ft.Text(value, size=18, weight=ft.FontWeight.BOLD, color=color),
                ],
                spacing=2,
            ),
        )

    def _case_table(self, cases: list[dict[str, Any]]) -> ft.Control:
        from knowledge_agent.evaluation.registry import metric_fmts

        fmts = metric_fmts()
        columns = [
            ft.DataColumn(ft.Text("Case")),
            ft.DataColumn(ft.Text("Category")),
            ft.DataColumn(ft.Text("Status")),
        ]
        columns += [ft.DataColumn(ft.Text(label)) for _, label in _CASE_COLUMNS]

        rows: list[ft.DataRow] = []
        for c in cases:
            status = c.get("status") or ""
            status_color = ft.Colors.GREEN if status == "PASS" else ft.Colors.RED
            cells = [
                ft.DataCell(ft.Text(c.get("case_id") or "")),
                ft.DataCell(ft.Text(c.get("category") or "")),
                ft.DataCell(ft.Text(status, color=status_color, weight=ft.FontWeight.BOLD)),
            ]
            for key, _ in _CASE_COLUMNS:
                val = c.get(key)
                cells.append(
                    ft.DataCell(
                        ft.Text(self._fmt(val, fmts.get(key, ".2f")), color=_score_color(val))
                    )
                )
            rows.append(ft.DataRow(cells=cells))

        return ft.Column(
            [
                ft.Text("Per-case results", weight=ft.FontWeight.BOLD),
                ft.Row([ft.DataTable(columns=columns, rows=rows)], scroll=ft.ScrollMode.AUTO),
            ],
            spacing=6,
        )

    # ---- handlers ---------------------------------------------------------

    def _on_select_run(self, _e: ft.Event) -> None:
        if self.run_dropdown and self.run_dropdown.value:
            self.coordinator.selected_run_id = int(self.run_dropdown.value)
            self.refresh()

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()
