"""Evaluation → Deep Analysis sub-tab — metric balance / spread / correlation.

Left rail: run selector + selected-run details (same rail as Run Summary).
Body (all native Flet — canvas ships with Flet, no extra deps):
  - Metric Balance — horizontal bar-of-means per score group (the reference's
    radar, as bars), the groups laid out side by side, each with a description;
    lower-is-better metrics (e.g. hallucination) are pulled out below.
  - Score Distribution — a vertical column-chart histogram (10 bins) of a chosen
    metric's per-case values, drawn on flet.canvas.
  - Correlation — a diverging Pearson heatmap on flet.canvas (hallucination
    excluded; zero-variance metrics shown as '= X').
  - Latency by Case — total latency per case (KA has no retrieval/LLM split).
  - Token Usage by Case — stacked agent input + output tokens per case.

Reads the ledger on `refresh()` only. (SSOT TODO: the run rail + run-details
rendering is mirrored from RunSummaryTab — extract a shared helper.)
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any

import flet as ft
from flet import canvas as cv

from knowledge_agent.gui._styles import (
    FIELD_LABEL_SIZE,
    dashboard_section_header,
    section_divider,
)
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail
from knowledge_agent.gui.evaluation.run_summary_tab import _score_color
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

# The score-metric groups (membership test for distribution / correlation /
# balance). Metric Balance lays its cards out in an explicit 2-column grid, so
# this tuple's order doesn't drive that.
_SCORE_GROUPS: tuple[str, ...] = ("retrieval", "chunk", "kg", "llm")
# Metric-Balance group titles + descriptions, matching the reference's charts.
_GROUP_HEADERS: dict[str, str] = {
    "retrieval": "Source Retrieval Metrics",
    "chunk": "Chunk Retrieval Metrics",
    "kg": "Knowledge Graph Metrics",
    "llm": "Judge Metrics",
}
_GROUP_DESCRIPTIONS: dict[str, str] = {
    "retrieval": "Average scores across all cases in the selected run. "
    "Highlights weak spots in source retrieval.",
    "chunk": "Average scores across all cases in the selected run. Highlights whether "
    "the right passages were retrieved, not just the right files.",
    "kg": "Average scores across all cases in the selected run. "
    "Highlights knowledge-graph retrieval quality.",
    "llm": "Average scores across all cases in the selected run. Identifies weak spots "
    "in answer quality. Hallucination is shown separately below.",
}
_BAR_W = 160
_GROUP_WIDTH = 340  # fixed width so descriptions wrap + groups sit side by side

# Score-distribution histogram geometry (px) — a vertical column chart on
# flet.canvas (native, zero-dep), matching the reference's histogram.
_HIST_W, _HIST_H = 560, 240
_HIST_ML, _HIST_MR, _HIST_MT, _HIST_MB = 40, 12, 16, 34

# By-case stacked column charts (Latency; Token Usage).
_CASE_CHART_H = 300
_CASE_ML, _CASE_MR, _CASE_MT, _CASE_MB = 52, 12, 18, 110
_CASE_BAR_W = 22
_CASE_GAP = 16
_CASE_MIN_W = 480
_CASE_LABEL_ANGLE = -0.62  # radians (~ -35°) for angled case-name labels


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


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _stroke(color: str, width: float = 1.0) -> ft.Paint:
    return ft.Paint(color=color, stroke_width=width, style=ft.PaintingStyle.STROKE)


def _trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# Diverging endpoints as #rrggbb (no hash): red = +1, blue = −1.
_CORR_POS = "b2182b"
_CORR_NEG = "2166ac"


def _corr_color(r: float) -> str:
    """See-through diverging fill for a correlation, as an ``#aarrggbb`` string:
    red for positive, blue for negative, with OPACITY = |r|. So r≈0 is fully
    transparent (the dark app shows through) and ±1 is solid — strong
    correlations stay bold while weak ones melt into the background."""
    r = max(-1.0, min(1.0, r))
    base = _CORR_POS if r >= 0 else _CORR_NEG
    alpha = round(abs(r) * 255)
    return f"#{alpha:02x}{base}"


def _nice_axis(y_max: float, *, integer: bool = False) -> tuple[float, float]:
    """A rounded axis top + tick step giving ~5 ticks for `y_max` (>0). With
    `integer`, the step is forced to a whole number so a count axis never shows
    fractional ticks."""
    if y_max <= 0:
        return 1.0, 1.0
    raw = y_max / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = mag
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if raw <= step:
            break
    if integer:
        step = max(1.0, float(round(step)))
    top = math.ceil(y_max / step) * step
    return top, step


def _by_case_series(cases: list[dict[str, Any]], keys: list[str]) -> list[tuple[str, list[float]]]:
    """(case_id, [value per key]) for cases with ≥1 numeric value; a missing
    value becomes 0 so a stacked bar still totals correctly."""
    out: list[tuple[str, list[float]]] = []
    for c in cases:
        raw = [c.get(k) for k in keys]
        if any(isinstance(v, (int, float)) for v in raw):
            out.append(
                (
                    c.get("case_id") or "",
                    [float(v) if isinstance(v, (int, float)) else 0.0 for v in raw],
                )
            )
    return out


def _stacked_by_case_chart(
    cases: list[dict[str, Any]], series: list[tuple[str, str, str]]
) -> ft.Control:
    """A (possibly stacked) vertical column chart per case on a native canvas —
    one bar per case, one segment per `series` entry (label, colour, case
    column), with angled case labels + a right-side legend. Scrolls horizontally.
    A single-entry `series` is just a plain bar chart (e.g. total latency)."""
    data = _by_case_series(cases, [key for _, _, key in series])
    if not data:
        return ft.Text("No per-case data for this chart.", italic=True)
    y_top, step = _nice_axis(max((sum(vals) for _, vals in data), default=0.0))

    n = len(data)
    pitch = _CASE_BAR_W + _CASE_GAP
    width = max(_CASE_MIN_W, _CASE_ML + _CASE_MR + n * pitch)
    plot_h = _CASE_CHART_H - _CASE_MT - _CASE_MB
    baseline = _CASE_MT + plot_h

    def py(v: float) -> float:
        return _CASE_MT + plot_h - (v / y_top) * plot_h

    label_style = ft.TextStyle(size=12, color=ft.Colors.GREY_500)
    shapes: list[cv.Shape] = [
        cv.Line(_CASE_ML, _CASE_MT, _CASE_ML, baseline, _stroke(ft.Colors.GREY_500)),
        cv.Line(_CASE_ML, baseline, width - _CASE_MR, baseline, _stroke(ft.Colors.GREY_500)),
    ]
    tick = 0.0
    while tick <= y_top + 1e-9:
        y = py(tick)
        shapes.append(cv.Line(_CASE_ML, y, width - _CASE_MR, y, _stroke(ft.Colors.GREY_800)))
        shapes.append(cv.Text(4, y - 6, f"{tick:g}", label_style))
        tick += step
    for i, (cid, vals) in enumerate(data):
        x = _CASE_ML + i * pitch + _CASE_GAP / 2
        cum = 0.0
        for (_, color, _), v in zip(series, vals, strict=True):
            if v > 0:
                y_high, y_low = py(cum + v), py(cum)
                shapes.append(cv.Rect(x, y_high, _CASE_BAR_W, y_low - y_high, paint=_fill(color)))
                cum += v
        shapes.append(
            cv.Text(
                x + _CASE_BAR_W / 2,
                baseline + 6,
                _trunc(cid, 22),
                label_style,
                rotate=_CASE_LABEL_ANGLE,
                alignment=ft.Alignment(1, -1),
            )
        )
    chart = cv.Canvas(shapes=shapes, width=width, height=_CASE_CHART_H)
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
    return ft.Row(
        [
            ft.Row([chart], scroll=ft.ScrollMode.AUTO, expand=True),
            ft.Container(content=legend, height=_CASE_CHART_H, padding=ft.Padding.only(left=8)),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
        spacing=8,
    )


class DeepAnalysisTab:
    """Per-run deep-analysis views (bar-of-means, histogram, correlation)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.rail: DashboardRail | None = None
        self.body: ft.Column | None = None
        self.hist_dropdown: ft.Dropdown | None = None
        self.hist_container: ft.Container | None = None
        self._runs: list[dict[str, Any]] = []
        self._cases: list[dict[str, Any]] = []
        self._hist_metric: str | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        self.rail = DashboardRail(self.app, self.coordinator, on_change=self._render_body)
        rail_ctl = self.rail.build()
        self.body = ft.Column(
            [ft.Text("Run an evaluation, or press Refresh to analyze a run.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        return ft.Row(
            [
                rail_ctl,
                ft.Column(
                    [view_header("Deep Analysis"), self.body],
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
        """Reload the rail (runs + selection) then render the selected run."""
        if self.rail is not None:
            self.rail.refresh()
        self._render_body()

    def _render_body(self) -> None:
        """Render the deep-analysis body for the coordinator's selected run —
        the rail owns the selectors + recipe panel; this owns the body."""
        if self.body is None:
            return
        run_id = self.coordinator.selected_run_id
        if run_id is None:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.app.page.update()
            return
        ledger = self._ledger()
        run = ledger.get_run(run_id)
        if run is None:
            self.body.controls = [ft.Text("Run not found — press Refresh.", italic=True)]
            self.app.page.update()
            return
        self._runs = ledger.list_runs()
        self._cases = ledger.get_run_cases(run_id)
        self.body.controls = [
            *self._balance_sections(run),
            section_divider(),
            self._distribution_section(),
            section_divider(),
            self._correlation_section(),
            section_divider(),
            self._latency_section(),
            section_divider(),
            self._token_usage_section(),
        ]
        self.app.page.update()

    # ---- metric balance (bar-of-means) ------------------------------------

    def _n_present(self, key: str) -> int:
        """How many cases have a numeric value for `key` — the n behind that
        metric's mean (a no-gold case scores None and isn't counted). Shown on
        each bar so a mean over few cases can't read like a full one."""
        return sum(1 for c in self._cases if isinstance(c.get(key), (int, float)))

    def _balance_sections(self, run: dict[str, Any]) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        rows_by_group: dict[str, list[ft.Control]] = {}
        lower_better: list[tuple[str, float, int]] = []
        for group in ("retrieval", "chunk", "llm", "kg"):
            rows: list[ft.Control] = []
            for m in (m for m in METRICS if m.group == group and m.summary_avg_key):
                v = run.get(m.summary_avg_key)
                if not isinstance(v, (int, float)):
                    continue
                # Lower-is-better metrics (e.g. hallucination) get their own card,
                # band-coloured direction-aware — not mixed into the group bars.
                if not m.higher_is_better:
                    lower_better.append(
                        (labels.get(m.summary_avg_key, m.label), float(v), self._n_present(m.key))
                    )
                    continue
                rows.append(
                    ft.Row(
                        [
                            ft.Text(labels.get(m.summary_avg_key, m.label), width=120, size=12),
                            _hbar(float(v), 1.0, _score_color(v)),
                            ft.Text(f"{v:.2f} (n={self._n_present(m.key)})", size=12),
                        ],
                        spacing=8,
                    )
                )
            rows_by_group[group] = rows

        # Card order fills the 2-column grid: Source + Chunk on row 1, Judge +
        # Hallucination on row 2, then Knowledge Graph (KA extra) when present.
        # The three core groups always show a card (a 'no data' note if the group
        # wasn't evaluated), so the layout matches the reference.
        cards: list[ft.Control] = []
        for group in ("retrieval", "chunk", "llm"):
            body = rows_by_group[group] or [
                ft.Text(
                    "No data for this run — this metric group wasn't evaluated.",
                    italic=True,
                    size=12,
                    color=ft.Colors.GREY_500,
                )
            ]
            cards.append(self._group_card(group, body))
        cards.extend(self._lower_is_better_card(lbl, v, n) for lbl, v, n in lower_better)
        if rows_by_group.get("kg"):
            cards.append(self._group_card("kg", rows_by_group["kg"]))

        grid = ft.Column(
            [
                ft.Row(cards[i : i + 2], spacing=40, vertical_alignment=ft.CrossAxisAlignment.START)
                for i in range(0, len(cards), 2)
            ],
            spacing=28,
        )
        return [dashboard_section_header("Metric Balance"), grid]

    @staticmethod
    def _group_card(group: str, body: list[ft.Control]) -> ft.Control:
        """A fixed-width Metric-Balance card: group title + description + body
        (the metric bars, or a 'no data' note)."""
        return ft.Container(
            width=_GROUP_WIDTH,
            content=ft.Column(
                [
                    ft.Text(_GROUP_HEADERS[group], weight=ft.FontWeight.BOLD),
                    ft.Text(_GROUP_DESCRIPTIONS[group], size=12, color=ft.Colors.GREY_400),
                    *body,
                ],
                spacing=4,
            ),
        )

    @staticmethod
    def _lower_is_better_card(label: str, value: float, n: int) -> ft.Control:
        """A lower-is-better metric (e.g. hallucination) as its own grid card,
        band-coloured direction-aware so a low value reads green."""
        return ft.Container(
            width=_GROUP_WIDTH,
            content=ft.Column(
                [
                    ft.Text(label, weight=ft.FontWeight.BOLD),
                    ft.Text("Lower is better.", size=12, color=ft.Colors.GREY_400),
                    ft.Row(
                        [
                            _hbar(value, 1.0, _score_color(1.0 - value)),
                            ft.Text(f"{value:.2f} (n={n})", size=12),
                        ],
                        spacing=8,
                    ),
                ],
                spacing=4,
            ),
        )

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
            text_size=FIELD_LABEL_SIZE,
        )
        self.hist_dropdown.on_change = self._on_hist_metric_change
        self.hist_container = ft.Container(content=self._histogram(self._hist_metric))
        return ft.Column(
            [
                dashboard_section_header("Score Distribution"),
                ft.Text(
                    "Select a metric to see how scores spread across cases.",
                    size=12,
                    color=ft.Colors.GREY_400,
                ),
                self.hist_dropdown,
                self.hist_container,
            ],
            spacing=6,
        )

    def _histogram(self, key: str) -> ft.Control:
        """A vertical column chart (histogram) of a metric's per-case values on
        a native canvas — x = score bins 0..1, y = case count."""
        values = [float(c[key]) for c in self._cases if isinstance(c.get(key), (int, float))]
        if not values:
            return ft.Text("No values.", italic=True)
        bins = [0] * 10
        for v in values:
            bins[min(9, max(0, int(v * 10)))] += 1  # 0..1 -> bin 0..9 (1.0 -> 9)
        y_top, step = _nice_axis(max(bins) or 1, integer=True)

        plot_w = _HIST_W - _HIST_ML - _HIST_MR
        plot_h = _HIST_H - _HIST_MT - _HIST_MB
        baseline = _HIST_MT + plot_h
        band = plot_w / 10
        bar_w = band * 0.8

        def py(count: float) -> float:
            return _HIST_MT + plot_h - (count / y_top) * plot_h

        label_style = ft.TextStyle(size=12, color=ft.Colors.GREY_500)
        shapes: list[cv.Shape] = [
            cv.Line(_HIST_ML, _HIST_MT, _HIST_ML, baseline, _stroke(ft.Colors.GREY_500)),
            cv.Line(_HIST_ML, baseline, _HIST_ML + plot_w, baseline, _stroke(ft.Colors.GREY_500)),
        ]
        tick = 0.0
        while tick <= y_top + 1e-9:
            y = py(tick)
            shapes.append(cv.Line(_HIST_ML, y, _HIST_ML + plot_w, y, _stroke(ft.Colors.GREY_800)))
            shapes.append(cv.Text(4, y - 6, f"{tick:g}", label_style))
            tick += step
        for i, count in enumerate(bins):
            x = _HIST_ML + i * band + (band - bar_w) / 2
            shapes.append(
                cv.Rect(x, py(count), bar_w, baseline - py(count), paint=_fill(ft.Colors.BLUE))
            )
        for edge in range(0, 11, 2):  # x labels at bin edges 0, 0.2, … 1.0
            x = _HIST_ML + (edge / 10) * plot_w
            shapes.append(cv.Text(x - 6, baseline + 4, f"{edge / 10:g}", label_style))
        return cv.Canvas(shapes=shapes, width=_HIST_W, height=_HIST_H)

    def _on_hist_metric_change(self, _e: ft.Event) -> None:
        if self.hist_dropdown and self.hist_container and self.hist_dropdown.value:
            self._hist_metric = self.hist_dropdown.value
            self.hist_container.content = self._histogram(self._hist_metric)
            self.app.page.update()

    # ---- latency + token usage by case (column charts) --------------------

    def _latency_section(self) -> ft.Control:
        return ft.Column(
            [
                dashboard_section_header("Latency by Case"),
                ft.Text(
                    "Total latency per case (seconds). KA measures total latency only "
                    "— no retrieval/LLM split.",
                    size=12,
                    color=ft.Colors.GREY_400,
                ),
                _stacked_by_case_chart(
                    self._cases, [("Latency (s)", ft.Colors.ORANGE_400, "latency_seconds")]
                ),
            ],
            spacing=6,
        )

    def _token_usage_section(self) -> ft.Control:
        return ft.Column(
            [
                dashboard_section_header("Token Usage by Case"),
                ft.Text(
                    "Agent input and output tokens per case.", size=12, color=ft.Colors.GREY_400
                ),
                _stacked_by_case_chart(
                    self._cases,
                    [
                        ("Input", ft.Colors.INDIGO_400, "agent_input_tokens"),
                        ("Output", ft.Colors.TEAL_400, "agent_output_tokens"),
                    ],
                ),
            ],
            spacing=6,
        )

    # ---- correlation heatmap ----------------------------------------------

    def _correlation_section(self) -> ft.Control:
        # Score metrics present, EXCLUDING lower-is-better ones (e.g.
        # hallucination) whose sign would invert — the reference excludes them.
        metrics = self._corr_metrics()
        if len(metrics) < 2:
            return ft.Text("Correlation needs at least 2 comparable metrics.", italic=True)
        return ft.Column(
            [
                dashboard_section_header("Metric Correlations (Pearson)"),
                ft.Text(
                    "Pearson correlation. Hallucination is excluded "
                    "(lower = better, would invert the sign).",
                    size=12,
                    color=ft.Colors.GREY_400,
                ),
                ft.Row([self._corr_heatmap(metrics)], scroll=ft.ScrollMode.AUTO),
                ft.Text(
                    "= X means the metric scored X on every case (zero variance) — "
                    "correlation cannot be calculated.",
                    size=12,
                    color=ft.Colors.GREY_500,
                ),
            ],
            spacing=6,
        )

    def _corr_metrics(self) -> list[tuple[str, str]]:
        """Score metrics present in this run, excluding lower-is-better ones."""
        from knowledge_agent.evaluation.registry import METRICS, metric_labels

        labels = metric_labels()
        return [
            (m.key, labels.get(m.key, m.label))
            for m in METRICS
            if m.group in _SCORE_GROUPS
            and m.sql_column
            and m.higher_is_better
            and any(c.get(m.key) is not None for c in self._cases)
        ]

    def _corr_heatmap(self, metrics: list[tuple[str, str]]) -> ft.Control:
        """A diverging red↔blue Pearson heatmap on a native canvas — colour-filled
        cells (red = +1, blue = −1), the value in each cell, angled column labels,
        and a colourbar. Constant metrics show '= X' (can't be correlated)."""
        keys = [k for k, _ in metrics]
        vals: dict[str, list[float]] = {
            k: [float(c[k]) for c in self._cases if isinstance(c.get(k), (int, float))]
            for k in keys
        }
        const: dict[str, float | None] = {
            k: (v[0] if v and len(set(v)) <= 1 else None) for k, v in vals.items()
        }
        n = len(metrics)
        cell_w, cell_h = 76, 32
        label_w, top_pad, bottom_h, bar_gap = 140, 8, 108, 16
        width = label_w + n * cell_w + bar_gap + 60
        height = top_pad + n * cell_h + bottom_h
        text_style = ft.TextStyle(size=12, color=ft.Colors.GREY_300)

        shapes: list[cv.Shape] = []
        for i, (_, lbl) in enumerate(metrics):
            shapes.append(
                cv.Text(
                    label_w - 8,
                    top_pad + i * cell_h + cell_h / 2,
                    _trunc(lbl, 18),
                    text_style,
                    alignment=ft.Alignment(1, 0),
                )
            )
        for j, (_, lbl) in enumerate(metrics):
            shapes.append(
                cv.Text(
                    label_w + j * cell_w + cell_w / 2,
                    top_pad + n * cell_h + 6,
                    _trunc(lbl, 16),
                    text_style,
                    rotate=-0.62,
                    alignment=ft.Alignment(1, -1),
                )
            )
        for i, kr in enumerate(keys):
            for j, kc in enumerate(keys):
                x, y = label_w + j * cell_w, top_pad + i * cell_h
                fill, txt, txt_color = self._corr_cell_style(kr, kc, i == j, const)
                shapes.append(cv.Rect(x, y, cell_w - 2, cell_h - 2, paint=_fill(fill)))
                shapes.append(
                    cv.Text(
                        x + cell_w / 2,
                        y + cell_h / 2,
                        txt,
                        ft.TextStyle(size=12, color=txt_color),
                        alignment=ft.Alignment(0, 0),
                    )
                )
        # Colourbar (+1 red → transparent 0 → −1 blue) on the right — opacity
        # tracks |r| like the cells, so it's see-through at the midpoint. A thin
        # outline keeps the bar's extent readable where it's fully transparent.
        bar_x = label_w + n * cell_w + bar_gap
        bar_h = n * cell_h
        segs = 24
        for s in range(segs):
            r_val = 1.0 - 2.0 * s / (segs - 1)
            shapes.append(
                cv.Rect(
                    bar_x,
                    top_pad + s * bar_h / segs,
                    18,
                    bar_h / segs + 1,
                    paint=_fill(_corr_color(r_val)),
                )
            )
        shapes.append(cv.Rect(bar_x, top_pad, 18, bar_h, paint=_stroke(ft.Colors.GREY_700)))
        for frac, tick in ((0.0, "+1"), (0.5, "0"), (1.0, "−1")):
            shapes.append(cv.Text(bar_x + 24, top_pad + frac * bar_h, tick, text_style))
        return cv.Canvas(shapes=shapes, width=width, height=height)

    def _corr_cell_style(
        self, key_r: str, key_c: str, diagonal: bool, const: dict[str, float | None]
    ) -> tuple[str, str, str]:
        """(fill colour, cell text, text colour) for one heatmap cell. Fills are
        see-through — opacity tracks |r| (transparent at r≈0) — so text is always
        light, to read against the dark app or the saturated strong cells."""
        if diagonal:
            return _corr_color(1.0), "1.00", ft.Colors.WHITE
        if const[key_r] is not None or const[key_c] is not None:
            cval = const[key_r] if const[key_r] is not None else const[key_c]
            return ft.Colors.TRANSPARENT, f"= {cval:.2f}", ft.Colors.GREY_400
        xs, ys = [], []
        for c in self._cases:
            a, b = c.get(key_r), c.get(key_c)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                xs.append(float(a))
                ys.append(float(b))
        try:
            r = statistics.correlation(xs, ys) if len(xs) >= 2 else None
        except statistics.StatisticsError:
            r = None
        if r is None:
            return ft.Colors.TRANSPARENT, "—", ft.Colors.GREY_400
        return _corr_color(r), f"{r:.2f}", ft.Colors.WHITE if abs(r) > 0.55 else ft.Colors.GREY_300
