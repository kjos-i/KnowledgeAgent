"""Evaluation → Deep Analysis sub-tab — metric balance / spread / correlation.

For the selected run, all native Flet widgets (no canvas, no deps):
  - Metric Balance — horizontal bar-of-means per score group (the reference's
    radar, as bars): each metric's run-level average as a proportional bar.
  - Score Distribution — a histogram (10 bins) of a chosen metric's per-case
    values, binned in plain Python.
  - Correlation — pairwise Pearson (`statistics.correlation`, stdlib) between
    the varying score metrics, as a color-coded grid. Zero-variance metrics
    are flagged, not correlated.

Uses the coordinator's `selected_run_id` (set in Run Summary); reads the
ledger on `refresh()` only.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui.evaluation.run_summary_tab import _score_color
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_SCORE_GROUPS: tuple[str, ...] = ("retrieval", "chunk", "kg", "llm")
_GROUP_HEADERS: dict[str, str] = {
    "retrieval": "Source Retrieval",
    "chunk": "Chunk Retrieval",
    "kg": "Knowledge Graph",
    "llm": "Judge Scores",
}
_BAR_W = 200


def _hbar(value: float, y_max: float, color: str) -> ft.Control:
    """A left-filled proportional bar on a track."""
    frac = 0.0 if not y_max else max(0.0, min(1.0, value / y_max))
    return ft.Container(
        width=_BAR_W,
        height=14,
        bgcolor=ft.Colors.GREY_800,
        border_radius=3,
        alignment=ft.Alignment(-1, 0),
        content=ft.Container(
            width=max(2.0, frac * _BAR_W), height=14, bgcolor=color, border_radius=3
        ),
    )


class DeepAnalysisTab:
    """Per-run deep-analysis views (bar-of-means, histogram, correlation)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.body: ft.Column | None = None
        self.hist_dropdown: ft.Dropdown | None = None
        self.hist_container: ft.Container | None = None
        self._cases: list[dict[str, Any]] = []
        self._hist_metric: str | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        refresh_button = ft.TextButton("Refresh", icon=ft.Icons.REFRESH, on_click=self._on_refresh)
        self.body = ft.Column(
            [ft.Text("Run an evaluation, or press Refresh to analyze a run.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        return ft.Column(
            [view_header("Deep Analysis"), refresh_button, self.body],
            expand=True,
            spacing=8,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        if self.body is None:
            return
        ledger = self._ledger()
        runs = ledger.list_runs()
        if not runs:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.app.page.update()
            return
        run_id = self.coordinator.selected_run_id or runs[0]["run_id"]
        self.coordinator.selected_run_id = run_id
        run = ledger.get_run(run_id)
        self._cases = ledger.get_run_cases(run_id)

        self.body.controls = [
            ft.Text(f"Run {run_id}", weight=ft.FontWeight.BOLD),
            *self._balance_sections(run),
            self._distribution_section(),
            self._correlation_section(),
        ]
        self.app.page.update()

    # ---- metric balance (bar-of-means) ------------------------------------

    def _balance_sections(self, run: dict[str, Any]) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        sections: list[ft.Control] = []
        for group in _SCORE_GROUPS:
            metrics = [m for m in METRICS if m.group == group and m.summary_avg_key]
            rows = []
            for m in metrics:
                v = run.get(m.summary_avg_key)
                if not isinstance(v, (int, float)):
                    continue
                rows.append(
                    ft.Row(
                        [
                            ft.Text(labels.get(m.summary_avg_key, m.label), width=150, size=12),
                            _hbar(float(v), 1.0, _score_color(v)),
                            ft.Text(f"{v:.2f}", size=12),
                        ],
                        spacing=8,
                    )
                )
            if rows:
                sections.append(
                    ft.Column(
                        [ft.Text(_GROUP_HEADERS[group], weight=ft.FontWeight.BOLD), *rows],
                        spacing=4,
                    )
                )
        if not sections:
            sections.append(ft.Text("No score metrics for this run.", italic=True))
        return [ft.Text("Metric Balance (run averages)", weight=ft.FontWeight.BOLD), *sections]

    # ---- score distribution (histogram) -----------------------------------

    def _score_keys_present(self) -> list[tuple[str, str]]:
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        keys = [
            (m.key, labels.get(m.key, m.label))
            for m in METRICS
            if m.group in _SCORE_GROUPS and m.sql_column
        ]
        return [(k, lbl) for k, lbl in keys if any(c.get(k) is not None for c in self._cases)]

    def _distribution_section(self) -> ft.Control:
        present = self._score_keys_present()
        if not present:
            return ft.Text("No per-case score metrics to distribute.", italic=True)
        if self._hist_metric not in {k for k, _ in present}:
            self._hist_metric = present[0][0]
        self.hist_dropdown = ft.Dropdown(
            label="Metric",
            options=[ft.DropdownOption(key=k, text=lbl) for k, lbl in present],
            value=self._hist_metric,
            width=260,
        )
        self.hist_dropdown.on_change = self._on_hist_metric_change
        self.hist_container = ft.Container(content=self._histogram(self._hist_metric))
        return ft.Column(
            [
                ft.Text("Score Distribution", weight=ft.FontWeight.BOLD),
                self.hist_dropdown,
                self.hist_container,
            ],
            spacing=6,
        )

    def _histogram(self, key: str) -> ft.Control:
        values = [float(c[key]) for c in self._cases if isinstance(c.get(key), (int, float))]
        if not values:
            return ft.Text("No values.", italic=True)
        bins = [0] * 10
        for v in values:
            idx = min(9, max(0, int(v * 10)))  # 0..1 -> bin 0..9 (1.0 -> 9)
            bins[idx] += 1
        top = max(bins) or 1
        rows = [
            ft.Row(
                [
                    ft.Text(f"{i / 10:.1f}–{(i + 1) / 10:.1f}", width=70, size=12),
                    _hbar(count, top, ft.Colors.BLUE),
                    ft.Text(str(count), size=12),
                ],
                spacing=8,
            )
            for i, count in enumerate(bins)
        ]
        return ft.Column(rows, spacing=2)

    def _on_hist_metric_change(self, _e: ft.Event) -> None:
        if self.hist_dropdown and self.hist_container and self.hist_dropdown.value:
            self._hist_metric = self.hist_dropdown.value
            self.hist_container.content = self._histogram(self._hist_metric)
            self.app.page.update()

    # ---- correlation grid -------------------------------------------------

    def _correlation_section(self) -> ft.Control:
        # Only metrics that vary across cases (constant series can't correlate).
        varying: list[tuple[str, str]] = []
        for key, label in self._score_keys_present():
            vals = [c[key] for c in self._cases if isinstance(c.get(key), (int, float))]
            if len(set(vals)) > 1:
                varying.append((key, label))
        if len(varying) < 2:
            return ft.Text(
                "Correlation needs at least 2 metrics that vary across cases.", italic=True
            )

        header = [ft.DataColumn(ft.Text(""))] + [
            ft.DataColumn(ft.Text(lbl, size=12)) for _, lbl in varying
        ]
        rows: list[ft.DataRow] = []
        for key_r, label_r in varying:
            cells = [ft.DataCell(ft.Text(label_r, size=12, weight=ft.FontWeight.BOLD))]
            for key_c, _ in varying:
                cells.append(ft.DataCell(self._corr_cell(key_r, key_c)))
            rows.append(ft.DataRow(cells=cells))
        return ft.Column(
            [
                ft.Text("Metric Correlations (Pearson)", weight=ft.FontWeight.BOLD),
                ft.Row([ft.DataTable(columns=header, rows=rows)], scroll=ft.ScrollMode.AUTO),
            ],
            spacing=6,
        )

    def _corr_cell(self, key_r: str, key_c: str) -> ft.Control:
        xs, ys = [], []
        for c in self._cases:
            a, b = c.get(key_r), c.get(key_c)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                xs.append(float(a))
                ys.append(float(b))
        if len(xs) < 2:
            return ft.Text("—", size=12, color=ft.Colors.GREY_500)
        try:
            r = statistics.correlation(xs, ys)
        except statistics.StatisticsError:
            return ft.Text("—", size=12, color=ft.Colors.GREY_500)
        color = ft.Colors.BLUE if r >= 0.5 else ft.Colors.RED if r <= -0.5 else ft.Colors.GREY_400
        return ft.Text(f"{r:.2f}", size=12, color=color)

    # ---- handlers ---------------------------------------------------------

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()
