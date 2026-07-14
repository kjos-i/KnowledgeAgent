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

from typing import TYPE_CHECKING, Any

import flet as ft
from flet import canvas as cv

from knowledge_agent.gui._styles import dashboard_section_header, section_divider
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail, dataset_of
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

# Chart geometry (px). Extra left/bottom margin for the axis titles + angled
# time labels on the x-axis.
_W, _H = 640, 262
_ML, _MR, _MT, _MB = 72, 16, 14, 76

# 0–1 score groups get a fixed y-axis; tokens/latency auto-scale.
_SCORE_GROUPS: tuple[str, ...] = ("retrieval", "chunk", "kg", "llm")
# Chart order matches the reference (Judge Scores right after the summary, then
# retrieval/chunk, then latency); Knowledge Graph is the KA-specific addition,
# tokens last.
_GROUP_HEADERS: dict[str, str] = {
    "llm": "Judge Scores Over Time",
    "retrieval": "Source Retrieval Quality Over Time",
    "chunk": "Chunk Retrieval Quality Over Time",
    "kg": "Knowledge Graph Over Time",
    "latency": "Latency Over Time",
    "tokens": "Tokens Over Time",
}
# Y-axis title per group (0–1 score groups default to "Score").
_Y_LABELS: dict[str, str] = {"latency": "Seconds", "tokens": "Tokens"}


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
    y_label: str,
    x_label: str = "Time",
) -> ft.Control:
    """A multi-series line chart on a fixed-size canvas + a widget legend, with
    named x/y axes. `series` = (label, colour, values-aligned-to-x); None values
    break the line (a gap). `y_max` sets the top of the y-axis.
    """
    n = len(x_labels)
    plot_w = _W - _ML - _MR
    plot_h = _H - _MT - _MB
    y_max = y_max or 1.0

    def px(i: int) -> float:
        return _ML + (i / max(1, n - 1)) * plot_w

    def py(v: float) -> float:
        return _MT + plot_h - (v / y_max) * plot_h

    label_style = ft.TextStyle(size=12, color=ft.Colors.GREY_500)
    title_style = ft.TextStyle(size=12, color=ft.Colors.GREY_400)
    shapes: list[cv.Shape] = [
        cv.Line(_ML, _MT, _ML, _MT + plot_h, _paint(ft.Colors.GREY_500, 1.0)),
        cv.Line(_ML, _MT + plot_h, _ML + plot_w, _MT + plot_h, _paint(ft.Colors.GREY_500, 1.0)),
    ]
    # Axis titles: y rotated up the left edge, x centred below the time labels.
    shapes.append(
        cv.Text(
            14,
            _MT + plot_h / 2,
            y_label,
            title_style,
            rotate=-1.5708,
            alignment=ft.Alignment(0, 0),
        )
    )
    shapes.append(
        cv.Text(_ML + plot_w / 2, _H - 6, x_label, title_style, alignment=ft.Alignment(0, 0))
    )
    for frac in (0.0, 0.5, 1.0):
        y = py(y_max * frac)
        shapes.append(cv.Line(_ML, y, _ML + plot_w, y, _paint(ft.Colors.GREY_800, 1.0)))
        tick = f"{y_max * frac:.2f}" if y_max <= 1 else f"{y_max * frac:g}"
        shapes.append(cv.Text(_ML - 6, y, tick, label_style, alignment=ft.Alignment(1, 0)))
    for i, lbl in enumerate(x_labels):
        shapes.append(
            cv.Text(
                px(i),
                _MT + plot_h + 6,
                lbl,
                label_style,
                rotate=-0.5,
                alignment=ft.Alignment(1, -1),
            )
        )

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

    legend = ft.Column(
        [
            ft.Row(
                [
                    ft.Container(width=12, height=12, bgcolor=color, border_radius=2),
                    ft.Text(label, size=12),
                ],
                spacing=6,
            )
            for label, color, _ in series
        ],
        spacing=4,
    )
    # Legend on the RIGHT of the chart (chart scrolls horizontally if narrow).
    return ft.Row(
        [
            ft.Row(
                [cv.Canvas(shapes=shapes, width=_W, height=_H)],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            ),
            ft.Container(content=legend, height=_H, padding=ft.Padding.only(left=8)),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
        spacing=8,
    )


class TrendsTab:
    """Cross-run trend charts sub-tab (scoped per dataset)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.rail: DashboardRail | None = None
        self.body: ft.Column | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        self.rail = DashboardRail(self.app, self.coordinator, on_change=self._render_body)
        rail_ctl = self.rail.build()
        self.body = ft.Column(
            [ft.Text("Run evaluations, or press Refresh to load trends.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        return ft.Row(
            [
                rail_ctl,
                ft.Column(
                    [view_header("Historical Trends"), self.body],
                    expand=True,
                    spacing=8,
                ),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload the rail (runs + selection) then render the trend for the
        shared selected dataset."""
        if self.rail is not None:
            self.rail.refresh()
        self._render_body()

    def _render_body(self) -> None:
        """Render the cross-run trend charts for the coordinator's selected
        dataset — the rail owns the selectors; this owns the charts. Uses the
        SAME `dataset_of` identity the rail selects by, so they stay in step."""
        if self.body is None:
            return
        ds = self.coordinator.selected_dataset
        if ds is None:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.app.page.update()
            return
        runs = self._ledger().list_runs()
        chrono = [r for r in reversed(runs) if dataset_of(r) == ds]
        if len(chrono) < 2:
            self.body.controls = [
                ft.Text(
                    f"Need at least 2 runs of '{ds}' for a trend ({len(chrono)} recorded).",
                    italic=True,
                )
            ]
            self.app.page.update()
            return
        self.body.controls = self._build_charts(chrono)
        self.app.page.update()

    # ---- charts -----------------------------------------------------------

    def _build_charts(self, chrono: list[dict[str, Any]]) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        x_labels = [self._ts_label(r) for r in chrono]  # time points, not run ids
        charts: list[ft.Control] = []

        # Summary: pass rate + avg judge score (0–1).
        summary_specs = [("pass_rate", "Pass Rate")] + [
            (m.summary_avg_key, m.summary_avg_label or m.label)
            for m in METRICS
            if m.group == "summary" and m.summary_avg_key
        ]
        charts.append(
            self._chart(
                "Pass Rate and Avg Judge Score Over Time",
                summary_specs,
                chrono,
                x_labels,
                1.0,
                "Score",
            )
        )

        for group, header in _GROUP_HEADERS.items():
            metrics = [m for m in METRICS if m.group == group and m.summary_avg_key]
            if not metrics:
                continue
            specs = [(m.summary_avg_key, labels.get(m.summary_avg_key, m.label)) for m in metrics]
            y_max = 1.0 if group in _SCORE_GROUPS else self._auto_max(specs, chrono)
            if y_max <= 0:
                continue
            charts.append(
                self._chart(header, specs, chrono, x_labels, y_max, _Y_LABELS.get(group, "Score"))
            )
        # Interleave section dividers so the trend charts read as distinct
        # sections — the same common style as Run Summary / Run Charts.
        out: list[ft.Control] = []
        for i, chart in enumerate(charts):
            if i:
                out.append(section_divider())
            out.append(chart)
        return out

    def _chart(
        self,
        header: str,
        specs: list[tuple[str, str]],
        chrono: list[dict[str, Any]],
        x_labels: list[str],
        y_max: float,
        y_label: str,
    ) -> ft.Control:
        series = [
            (label, _PALETTE[i % len(_PALETTE)], [self._num(r.get(col)) for r in chrono])
            for i, (col, label) in enumerate(specs)
        ]
        return ft.Column(
            [
                dashboard_section_header(header),
                _line_chart(series, x_labels, y_max, y_label),
            ],
            spacing=6,
        )

    @staticmethod
    def _ts_label(run: dict[str, Any]) -> str:
        """A compact time-point label for the x-axis: 'MM-DD HH:MM' from the
        run's ISO timestamp (falls back to the date, then '?')."""
        ts = run.get("run_timestamp") or ""
        if "T" in ts:
            date, time = ts.split("T", 1)
            return f"{date[5:10]} {time[:5]}"
        return ts[:10] or "?"

    @staticmethod
    def _num(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    def _auto_max(self, specs: list[tuple[str, str]], chrono: list[dict[str, Any]]) -> float:
        vals = [self._num(r.get(col)) for col, _ in specs for r in chrono]
        top = max((v for v in vals if v is not None), default=0.0)
        return top * 1.1 if top > 0 else 0.0
