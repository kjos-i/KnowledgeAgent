"""Evaluation → Trends sub-tab — run-level metric trends over time.

Multi-series line charts drawn on `flet.canvas` (Flet 0.85 has no built-in
LineChart) — zero new dependencies. Reads every `eval_runs` row via
`EvalLedger.list_runs`, scoped to ONE dataset (left-rail dataset filter,
default = the selected run's dataset) so comparisons stay apples-to-apples.
Plots the pre-aggregated `avg_*` columns straight from the ledger — no
in-GUI aggregation, no pandas.

Reads on `refresh()` only (never at build time; GUI startup rule), no
polling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft
from flet import canvas as cv

from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

# Series colours, cycled within a chart.
_PALETTE: tuple[str, ...] = (
    ft.Colors.BLUE,
    ft.Colors.GREEN,
    ft.Colors.ORANGE,
    ft.Colors.PURPLE,
    ft.Colors.RED,
    ft.Colors.TEAL,
    ft.Colors.PINK,
    ft.Colors.BROWN,
)

# Chart geometry (px).
_W, _H = 620, 220
_ML, _MR, _MT, _MB = 46, 12, 12, 26

# 0–1 score groups get a fixed y-axis; tokens/latency auto-scale.
_SCORE_GROUPS: tuple[str, ...] = ("retrieval", "chunk", "kg", "llm")
_GROUP_HEADERS: dict[str, str] = {
    "retrieval": "Source Retrieval",
    "chunk": "Chunk Retrieval",
    "kg": "Knowledge Graph",
    "llm": "Judge Scores",
    "tokens": "Tokens",
    "latency": "Latency",
}


def _paint(color: str, width: float = 2.0, *, fill: bool = False) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.FILL if fill else ft.PaintingStyle.STROKE,
    )


def _line_chart(
    series: list[tuple[str, str, list[float | None]]],
    x_labels: list[str],
    y_max: float,
) -> ft.Control:
    """A multi-series line chart on a fixed-size canvas + a widget legend.

    `series` = (label, colour, values-aligned-to-x); None values break the
    line (a gap). `y_max` sets the top of the y-axis.
    """
    n = len(x_labels)
    plot_w = _W - _ML - _MR
    plot_h = _H - _MT - _MB
    y_max = y_max or 1.0

    def px(i: int) -> float:
        return _ML + (i / max(1, n - 1)) * plot_w

    def py(v: float) -> float:
        return _MT + plot_h - (v / y_max) * plot_h

    shapes: list[cv.Shape] = [
        cv.Line(_ML, _MT, _ML, _MT + plot_h, _paint(ft.Colors.GREY_500, 1.0)),
        cv.Line(_ML, _MT + plot_h, _ML + plot_w, _MT + plot_h, _paint(ft.Colors.GREY_500, 1.0)),
    ]
    label_style = ft.TextStyle(size=12, color=ft.Colors.GREY_500)
    for frac in (0.0, 0.5, 1.0):
        y = py(y_max * frac)
        shapes.append(cv.Line(_ML, y, _ML + plot_w, y, _paint(ft.Colors.GREY_800, 1.0)))
        tick = f"{y_max * frac:.2f}" if y_max <= 1 else f"{y_max * frac:.0f}"
        shapes.append(cv.Text(2, y - 6, tick, label_style))
    for i, lbl in enumerate(x_labels):
        shapes.append(cv.Text(px(i) - 4, _MT + plot_h + 4, lbl, label_style))

    for _label, color, values in series:
        line_paint = _paint(color, 2.0)
        dot_paint = _paint(color, 1.0, fill=True)
        prev: tuple[float, float] | None = None
        for i, v in enumerate(values):
            if v is None:
                prev = None
                continue
            cx, cy = px(i), py(float(v))
            if prev is not None:
                shapes.append(cv.Line(prev[0], prev[1], cx, cy, line_paint))
            shapes.append(cv.Circle(cx, cy, 2.5, dot_paint))
            prev = (cx, cy)

    legend = ft.Row(
        [
            ft.Row(
                [
                    ft.Container(width=12, height=12, bgcolor=color, border_radius=2),
                    ft.Text(label, size=12),
                ],
                spacing=3,
            )
            for label, color, _ in series
        ],
        wrap=True,
        spacing=12,
    )
    return ft.Column([cv.Canvas(shapes=shapes, width=_W, height=_H), legend], spacing=4)


class TrendsTab:
    """Cross-run trend charts sub-tab (scoped per dataset)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.dataset_dropdown: ft.Dropdown | None = None
        self.body: ft.Column | None = None
        self._dataset: str | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        self.dataset_dropdown = ft.Dropdown(label="Dataset", options=[], width=240)
        self.dataset_dropdown.on_change = self._on_dataset_change
        refresh_button = ft.TextButton("Refresh", icon=ft.Icons.REFRESH, on_click=self._on_refresh)
        rail = ft.Container(
            width=280,
            content=ft.Column(
                [
                    ft.Text("Dataset", weight=ft.FontWeight.BOLD),
                    self.dataset_dropdown,
                    refresh_button,
                ],
                spacing=8,
            ),
        )
        self.body = ft.Column(
            [ft.Text("Run evaluations, or press Refresh to load trends.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        return ft.Column(
            [
                view_header("Historical Trends"),
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
        if self.body is None or self.dataset_dropdown is None:
            return
        runs = self._ledger().list_runs()
        datasets = list(dict.fromkeys(self._dataset_name(r) for r in runs))
        self.dataset_dropdown.options = [ft.DropdownOption(key=d, text=d) for d in datasets]

        if not datasets:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.app.page.update()
            return

        if self._dataset not in datasets:
            self._dataset = self._default_dataset(runs, datasets)
        self.dataset_dropdown.value = self._dataset

        chrono = [r for r in reversed(runs) if self._dataset_name(r) == self._dataset]
        if len(chrono) < 2:
            self.body.controls = [
                ft.Text(
                    f"Need at least 2 runs of '{self._dataset}' for a trend "
                    f"({len(chrono)} recorded).",
                    italic=True,
                )
            ]
            self.app.page.update()
            return

        self.body.controls = self._build_charts(chrono)
        self.app.page.update()

    @staticmethod
    def _dataset_name(run: dict[str, Any]) -> str:
        return Path(run.get("dataset_path") or "").name or "?"

    def _default_dataset(self, runs: list[dict[str, Any]], datasets: list[str]) -> str:
        selected = self.coordinator.selected_run_id
        for r in runs:
            if r["run_id"] == selected:
                return self._dataset_name(r)
        return datasets[0]

    # ---- charts -----------------------------------------------------------

    def _build_charts(self, chrono: list[dict[str, Any]]) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        x_labels = [str(r["run_id"]) for r in chrono]
        charts: list[ft.Control] = []

        # Summary: pass rate + avg judge score (0–1).
        summary_specs = [("pass_rate", "Pass Rate")] + [
            (m.summary_avg_key, m.summary_avg_label or m.label)
            for m in METRICS
            if m.group == "summary" and m.summary_avg_key
        ]
        charts.append(self._chart("Pass Rate & Judge Score", summary_specs, chrono, x_labels, 1.0))

        for group, header in _GROUP_HEADERS.items():
            metrics = [m for m in METRICS if m.group == group and m.summary_avg_key]
            if not metrics:
                continue
            specs = [(m.summary_avg_key, labels.get(m.summary_avg_key, m.label)) for m in metrics]
            y_max = 1.0 if group in _SCORE_GROUPS else self._auto_max(specs, chrono)
            if y_max <= 0:
                continue
            charts.append(self._chart(header, specs, chrono, x_labels, y_max))
        return charts

    def _chart(
        self,
        header: str,
        specs: list[tuple[str, str]],
        chrono: list[dict[str, Any]],
        x_labels: list[str],
        y_max: float,
    ) -> ft.Control:
        series = [
            (label, _PALETTE[i % len(_PALETTE)], [self._num(r.get(col)) for r in chrono])
            for i, (col, label) in enumerate(specs)
        ]
        return ft.Column(
            [ft.Text(header, weight=ft.FontWeight.BOLD), _line_chart(series, x_labels, y_max)],
            spacing=6,
        )

    @staticmethod
    def _num(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    def _auto_max(self, specs: list[tuple[str, str]], chrono: list[dict[str, Any]]) -> float:
        vals = [self._num(r.get(col)) for col, _ in specs for r in chrono]
        top = max((v for v in vals if v is not None), default=0.0)
        return top * 1.1 if top > 0 else 0.0

    # ---- handlers ---------------------------------------------------------

    def _on_dataset_change(self, _e: ft.Event) -> None:
        if self.dataset_dropdown and self.dataset_dropdown.value:
            self._dataset = self.dataset_dropdown.value
            self.refresh()

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()
