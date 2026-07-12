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
from flet import canvas as cv

from knowledge_agent.gui._styles import FIELD_LABEL_SIZE, PANEL_BG_RAISED, PANEL_RADIUS
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

# KPI card + section-header sizing, held within the tab's 12–16 text range:
# headers and metric values sit at the 16 ceiling (matching the app's
# panel_title), labels at the 12 floor. Named here so the overview row and the
# group sections stay in step.
_SECTION_HEADER_SIZE = 16  # section headers — the 16 ceiling (= panel_title)
_KPI_VALUE_SIZE = 16  # metric value — the 16 ceiling
_KPI_LABEL_SIZE = 12  # metric label; also the size of a "Not evaluated" value
_NOT_EVALUATED = "Not evaluated"  # sentinel value when a metric wasn't computed

# "Metric Scores by Case" grouped bar-chart geometry (px). Drawn on flet.canvas
# — the same native primitive the Trends tab uses (Flet 0.85 has no BarChart).
_CHART_H = 300
_CHART_ML, _CHART_MR, _CHART_MT, _CHART_MB = 46, 12, 18, 78  # tall bottom = angled labels
_CHART_BAR_W = 14  # a single metric bar
_CHART_BAR_INTRA = 2  # gap between bars within one case group
_CHART_GROUP_GAP = 22  # gap between case groups
_CHART_MIN_W = 460
_CHART_LABEL_ANGLE = -0.62  # radians (~ -35°) for the angled case labels

# Distinct bar/legend colours per metric (cycled if there are more metrics).
_METRIC_PALETTE: tuple[str, ...] = (
    ft.Colors.BLUE,
    ft.Colors.GREEN,
    ft.Colors.ORANGE,
    ft.Colors.PURPLE,
    ft.Colors.RED,
    ft.Colors.TEAL,
    ft.Colors.PINK,
    ft.Colors.BROWN,
    ft.Colors.CYAN,
    ft.Colors.LIME,
    ft.Colors.AMBER,
    ft.Colors.INDIGO,
    ft.Colors.DEEP_ORANGE,
    ft.Colors.LIGHT_GREEN,
    ft.Colors.DEEP_PURPLE,
    ft.Colors.BLUE_GREY,
)


def _score_color(value: Any) -> str:
    """Green/orange/red by the registry's score-band thresholds; grey when not
    evaluated. Thresholds come from the registry — never hardcoded here."""
    from knowledge_agent.evaluation.registry import (
        SCORE_BORDERLINE_THRESHOLD,
        SCORE_GOOD_THRESHOLD,
    )

    if not isinstance(value, (int, float)):
        return ft.Colors.GREY_500
    if value >= SCORE_GOOD_THRESHOLD:
        return ft.Colors.GREEN
    if value >= SCORE_BORDERLINE_THRESHOLD:
        return ft.Colors.ORANGE
    return ft.Colors.RED


def _fill_paint(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _stroke_paint(color: str, width: float = 1.0) -> ft.Paint:
    return ft.Paint(color=color, stroke_width=width, style=ft.PaintingStyle.STROKE)


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


class RunSummaryTab:
    """Per-run KPI + per-case-table sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.run_dropdown: ft.Dropdown | None = None
        self.metadata: ft.Column | None = None
        self.body: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []
        self._cases: list[dict[str, Any]] = []
        self._sort_key: str | None = None
        self._sort_asc: bool = True
        self._table_host: ft.Column | None = None
        # "Metric Scores by Case" chart state: multi-select metric/case filters
        # (None = all selected), the chart host, and the filter-title Text refs
        # whose live "selected/total" counts update on toggle.
        self._chart_metrics: set[str] | None = None
        self._chart_cases: set[str] | None = None
        self._chart_host: ft.Container | None = None
        self._metrics_title: ft.Text | None = None
        self._cases_title: ft.Text | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # Flet 0.85's Dropdown takes on_change as an attribute, not a ctor kwarg.
        self.run_dropdown = ft.Dropdown(
            label="Run", options=[], width=240, text_size=FIELD_LABEL_SIZE
        )
        self.run_dropdown.on_change = self._on_select_run
        self.metadata = ft.Column(controls=[], spacing=2)
        refresh_button = ft.TextButton("Refresh", on_click=self._on_refresh)
        rail = ft.Container(
            width=280,
            bgcolor=PANEL_BG_RAISED,
            padding=12,
            border_radius=PANEL_RADIUS,
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
        self._cases = cases
        # Previous run (next-lower run_id) drives the delta pills on the KPI
        # cards — mirroring the Streamlit dashboard's compare-to-previous view.
        prev_id = max((r["run_id"] for r in self._runs if r["run_id"] < run_id), default=None)
        prev_run = ledger.get_run(prev_id) if prev_id is not None else None
        self._render_metadata(run)
        # A newly loaded/refreshed run resets the chart filters to all-selected;
        # per-toggle changes (which don't re-enter refresh) persist within a run.
        self._chart_metrics = None
        self._chart_cases = None
        self.body.controls = self._build_body(run, prev_run, cases)
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

    def _build_body(
        self, run: dict[str, Any], prev_run: dict[str, Any] | None, cases: list[dict[str, Any]]
    ) -> list[ft.Control]:
        return [
            self._overview_row(run, prev_run),
            *self._kpi_sections(run, prev_run),
            self._case_table(cases),
            self._metric_scores_by_case(cases),
            self._answer_detail(cases),
        ]

    def _overview_row(self, run: dict[str, Any], prev_run: dict[str, Any] | None) -> ft.Control:
        """Top 'Average Performance' card row — a curated, cross-group set of
        headline metrics from the registry's OVERVIEW_ROW (each card's label,
        format, and delta direction all registry-sourced). These metrics are
        skipped in their own group sections so nothing is shown twice."""
        from knowledge_agent.evaluation.registry import (
            DEFAULT_FMT,
            OVERVIEW_LABEL,
            OVERVIEW_ROW,
            metric_directions,
            metric_fmts,
            metric_labels,
            summary_avg_pairs,
        )

        fmts = metric_fmts()
        labels = metric_labels()
        directions = metric_directions()
        # avg-key → source-key, so an overview card (keyed by its summary-avg key)
        # can find its per-metric n column (n_<source_key>). pass_rate has no
        # source key → no n line (its denominator is always the full case count).
        avg_to_source = dict(summary_avg_pairs())
        prev = prev_run or {}
        cards: list[ft.Control] = []
        for key in OVERVIEW_ROW:
            fmt = fmts.get(key, DEFAULT_FMT)
            d_text, d_good = self._delta(
                run.get(key), prev.get(key), fmt, lower_is_better=not directions.get(key, True)
            )
            src = avg_to_source.get(key)
            cards.append(
                self._kpi_card(
                    labels.get(key, key),
                    self._fmt(run.get(key), fmt),
                    d_text,
                    d_good,
                    self._n_text(run, src) if src else None,
                )
            )
        return ft.Column(
            [
                self._section_header(OVERVIEW_LABEL),
                ft.Row(cards, wrap=True, spacing=10),
            ],
            spacing=6,
        )

    def _kpi_sections(
        self, run: dict[str, Any], prev_run: dict[str, Any] | None
    ) -> list[ft.Control]:
        from knowledge_agent.evaluation.registry import (
            DEFAULT_FMT,
            GROUP_LABELS,
            METRICS,
            OVERVIEW_ROW,
            metric_fmts,
        )

        fmts = metric_fmts()
        prev = prev_run or {}
        sections: list[ft.Control] = []
        # Section order + labels both come from the registry's GROUP_LABELS.
        # Metrics already shown in the Average Performance overview row are
        # excluded here so they aren't duplicated; a group left with no cards
        # (e.g. summary, latency) is simply skipped.
        for group, header in GROUP_LABELS.items():
            metrics = [
                m
                for m in METRICS
                if m.group == group and m.summary_avg_key and m.summary_avg_key not in OVERVIEW_ROW
            ]
            if not metrics:
                continue
            cards: list[ft.Control] = []
            for m in metrics:
                key = m.summary_avg_key
                fmt = fmts.get(key, DEFAULT_FMT)
                # Green/red direction is the metric's own registry flag — no
                # hardcoded lower-is-better list in the GUI.
                d_text, d_good = self._delta(
                    run.get(key), prev.get(key), fmt, lower_is_better=not m.higher_is_better
                )
                cards.append(
                    self._kpi_card(
                        m.summary_avg_label or m.label,
                        self._fmt(run.get(key), fmt),
                        d_text,
                        d_good,
                        self._n_text(run, m.key),
                    )
                )
            sections.append(
                ft.Column(
                    [
                        self._section_header(header),
                        ft.Row(cards, wrap=True, spacing=10),
                    ],
                    spacing=6,
                )
            )
        return sections

    @staticmethod
    def _fmt(value: Any, fmt: str) -> str:
        if value is None:
            return _NOT_EVALUATED
        try:
            return format(value, fmt)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _delta(cur: Any, prev: Any, fmt: str, *, lower_is_better: bool) -> tuple[str | None, bool]:
        """Signed delta-vs-previous-run string + whether it's an improvement.

        Returns (None, False) when either value is missing (first run, or a
        not-evaluated metric) so the card renders without a pill. `good` is
        green-worthy: an increase for higher-is-better metrics, a decrease for
        lower-is-better ones (latency / tokens). The delta is formatted with
        the metric's own fmt so a percent value gets a percent delta, etc."""
        if not isinstance(cur, (int, float)) or not isinstance(prev, (int, float)):
            return None, False
        diff = cur - prev
        try:
            text = format(diff, f"+{fmt}")
        except (TypeError, ValueError):
            text = f"{diff:+.2f}"
        good = diff <= 0 if lower_is_better else diff >= 0
        return text, good

    @staticmethod
    def _n_text(run: dict[str, Any], source_key: str) -> str | None:
        """'n = X of Y' for a metric's run-level average — how many cases fed it
        (X = cases with a non-None value, Y = total cases). None when the run
        predates n-tracking (older ledger row) so the card just omits the line.
        Because a no-gold case is dropped from a mean rather than scored, X can
        be < Y, and this line keeps a mean over few cases from reading like a
        full one."""
        n = run.get(f"n_{source_key}")
        total = run.get("case_count")
        if n is None or total is None:
            return None
        return f"n = {n} of {total}"

    @staticmethod
    def _section_header(text: str) -> ft.Text:
        """A dashboard section header (Average Performance, Judge Scores, …).
        Larger than the app's section_title on purpose, matching the Streamlit
        reference's subheaders; size is the single `_SECTION_HEADER_SIZE`."""
        return ft.Text(text, size=_SECTION_HEADER_SIZE, weight=ft.FontWeight.BOLD)

    @staticmethod
    def _kpi_card(
        label: str,
        value: str,
        delta_text: str | None = None,
        delta_good: bool = False,
        n_text: str | None = None,
    ) -> ft.Control:
        """A Streamlit-style KPI card: white label, a large white value, an
        optional muted 'n = X of Y' line (how many cases fed the average, so a
        mean over few cases can't read like a full one), and an optional colored
        delta pill (green = improved vs the previous run, red = worse). No pill
        when `delta_text` is None (first run / not-evaluated)."""
        # A "Not evaluated" placeholder isn't a real value — show it at the label
        # size and slightly grayed, so it reads as absence rather than data.
        not_evaluated = value == _NOT_EVALUATED
        children: list[ft.Control] = [
            ft.Text(label, size=_KPI_LABEL_SIZE, color=ft.Colors.WHITE),
            ft.Text(
                value,
                size=_KPI_LABEL_SIZE if not_evaluated else _KPI_VALUE_SIZE,
                color=ft.Colors.with_opacity(0.7, ft.Colors.WHITE)
                if not_evaluated
                else ft.Colors.WHITE,
            ),
        ]
        # n line only for a real value (a not-evaluated card has n=0 — the
        # "Not evaluated" text already says that).
        if n_text is not None and not not_evaluated:
            children.append(ft.Text(n_text, size=_KPI_LABEL_SIZE, color=ft.Colors.GREY_500))
        if delta_text is not None:
            color = ft.Colors.GREEN if delta_good else ft.Colors.RED
            arrow = ft.Icons.ARROW_UPWARD if delta_text.startswith("+") else ft.Icons.ARROW_DOWNWARD
            children.append(
                ft.Container(
                    bgcolor=ft.Colors.with_opacity(0.18, color),
                    border_radius=4,
                    padding=ft.Padding.symmetric(horizontal=6, vertical=1),
                    content=ft.Row(
                        [
                            ft.Icon(arrow, size=12, color=color),
                            ft.Text(delta_text, size=12, color=color),
                        ],
                        spacing=2,
                        tight=True,
                    ),
                )
            )
        return ft.Container(
            width=150,
            padding=10,
            content=ft.Column(children, spacing=2),
        )

    def _case_table(self, cases: list[dict[str, Any]]) -> ft.Control:
        """Color-key legend + the per-case table: a frozen Case-ID column beside
        a separately horizontally-scrolling grid of Status + every registry
        metric, color-coded per cell. Every column header is click-to-sort
        (repeat-click flips asc/desc). Flet's DataTable can't freeze a column, so
        it's two height-aligned tables side by side."""
        if not cases:
            return ft.Text("No case-level data for this run.", italic=True)

        self._table_host = ft.Column(spacing=0)
        self._render_case_rows()
        return ft.Column(
            [
                ft.Text("Per-case results", weight=ft.FontWeight.BOLD, size=_SECTION_HEADER_SIZE),
                self._color_legend(),
                self._table_host,
            ],
            spacing=8,
        )

    @staticmethod
    def _color_legend() -> ft.Control:
        """The score-band key above the table, as a bulleted list matching the
        Streamlit reference — the color word is tinted; thresholds come from the
        registry (never hardcoded here)."""
        from knowledge_agent.evaluation.registry import (
            SCORE_BORDERLINE_THRESHOLD,
            SCORE_GOOD_THRESHOLD,
        )

        good, border = SCORE_GOOD_THRESHOLD, SCORE_BORDERLINE_THRESHOLD

        def key_row(color: str, word: str, desc: str) -> ft.Control:
            return ft.Text(
                size=12,
                spans=[
                    ft.TextSpan("•  "),
                    ft.TextSpan(word, style=ft.TextStyle(color=color, weight=ft.FontWeight.BOLD)),
                    ft.TextSpan(f" = {desc}"),
                ],
            )

        return ft.Column(
            spacing=2,
            controls=[
                ft.Text("Color keys for the table below:", size=12, color=ft.Colors.GREY_400),
                key_row(ft.Colors.GREEN, "green", f"good (score ≥ {good:g})"),
                key_row(
                    ft.Colors.ORANGE, "orange", f"borderline (≥ {border:g} but below {good:g})"
                ),
                key_row(ft.Colors.RED, "red", f"failing (< {border:g})"),
                key_row(ft.Colors.GREY_500, "gray", "not evaluated"),
            ],
        )

    def _sort_by(self, key: str) -> None:
        """Header click: sort by `key`; a repeat click on the same column flips
        the direction. Re-sorts the shared case list + rebuilds both tables."""
        if self._sort_key == key:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_key, self._sort_asc = key, True
        self._render_case_rows()
        self.app.page.update()

    def _sorted_cases(self) -> list[dict[str, Any]]:
        """`self._cases` ordered by the active sort column. Cases with no value
        for that column (not evaluated) always sort last, regardless of
        direction; no active sort returns insertion order."""
        if not self._sort_key:
            return self._cases
        key = self._sort_key
        present = [c for c in self._cases if c.get(key) is not None]
        absent = [c for c in self._cases if c.get(key) is None]
        present.sort(key=lambda c: c.get(key), reverse=not self._sort_asc)
        return present + absent

    def _render_case_rows(self) -> None:
        if self._table_host is None:
            return
        self._table_host.controls = [self._frozen_scroll_table(self._sorted_cases())]

    def _frozen_scroll_table(self, cases: list[dict[str, Any]]) -> ft.Control:
        from knowledge_agent.evaluation import registry as R

        if not cases:
            return ft.Text("No cases.", italic=True)

        fmts = R.metric_fmts()
        labels = R.metric_labels()
        directions = R.metric_directions()
        metric_cols = [col for col, _ in R.case_sql_columns()]
        # Only 0-1 score columns get the green/orange/red band; token counts,
        # latency (seconds) and integer counts (fmt "d") show plain — the same
        # choice the reference makes. Derived from the registry, not hardcoded.
        banded = {
            m.sql_column
            for m in R.METRICS
            if m.sql_column and m.group not in ("tokens", "latency") and m.fmt != "d"
        }
        row_h = 38

        # Only the Case ID is frozen; Status leads the scrolling metric grid.
        # Every column header is a click-to-sort control (repeat-click flips the
        # direction); the active column shows an ↑/↓ arrow.
        left = ft.DataTable(
            data_row_min_height=row_h,
            data_row_max_height=row_h,
            columns=[self._sort_col("Case", "case_id")],
            rows=[
                ft.DataRow(cells=[ft.DataCell(ft.Text(c.get("case_id") or "", size=12))])
                for c in cases
            ],
        )
        right = ft.DataTable(
            data_row_min_height=row_h,
            data_row_max_height=row_h,
            columns=[self._sort_col("Status", "status")]
            + [self._sort_col(labels.get(col, col), col) for col in metric_cols],
            rows=[
                ft.DataRow(
                    cells=[
                        ft.DataCell(
                            ft.Text(
                                c.get("status") or "",
                                size=12,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.GREEN
                                if c.get("status") == "PASS"
                                else ft.Colors.RED,
                            )
                        ),
                        *[
                            self._metric_cell(c.get(col), col, fmts, banded, directions)
                            for col in metric_cols
                        ],
                    ]
                )
                for c in cases
            ],
        )
        return ft.Row(
            [left, ft.Row([right], scroll=ft.ScrollMode.AUTO, expand=True)],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=0,
        )

    def _sort_col(self, header: str, key: str) -> ft.DataColumn:
        """A click-to-sort column header — a clickable label wired to `_sort_by`;
        the active sort column shows an ↑/↓ arrow. Uses a clickable label (not
        DataColumn.on_sort) so it doesn't depend on Flet's DataTable sort wiring."""
        arrow = ("  ↑" if self._sort_asc else "  ↓") if self._sort_key == key else ""
        return ft.DataColumn(
            ft.Container(
                content=ft.Text(f"{header}{arrow}", size=12, weight=ft.FontWeight.BOLD),
                on_click=lambda _e: self._sort_by(key),
            )
        )

    def _metric_cell(
        self,
        val: Any,
        col: str,
        fmts: dict[str, str],
        banded: set[str],
        directions: dict[str, bool],
    ) -> ft.DataCell:
        text = self._fmt(val, fmts.get(col, ".2f"))
        if not isinstance(val, (int, float)):
            color = ft.Colors.GREY_500
        elif col in banded:
            # Direction-aware: for lower-is-better metrics (e.g. hallucination)
            # the band inverts, so a low value reads green rather than red.
            color = _score_color(val if directions.get(col, True) else 1.0 - val)
        else:
            color = ft.Colors.WHITE
        return ft.DataCell(ft.Text(text, color=color, size=12))

    # ---- metric-scores-by-case chart --------------------------------------

    def _chart_metric_cols(self, cases: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """The 0–1 score columns that have data in this run — the chartable
        metrics. Same band derivation as the per-case table (registry-driven:
        score groups only, no tokens/latency/count columns), then filtered to
        those with at least one numeric value so the dropdown never offers an
        all-empty metric."""
        from knowledge_agent.evaluation import registry as R

        labels = R.metric_labels()
        banded = {
            m.sql_column
            for m in R.METRICS
            if m.sql_column and m.group not in ("tokens", "latency") and m.fmt != "d"
        }
        cols: list[tuple[str, str]] = []
        for col, _ in R.case_sql_columns():
            if col in banded and any(isinstance(c.get(col), (int, float)) for c in cases):
                cols.append((col, labels.get(col, col)))
        return cols

    def _metric_scores_by_case(self, cases: list[dict[str, Any]]) -> ft.Control:
        """'Metric Scores by Case' — the reference's grouped bar chart: per case,
        one bar per selected metric (coloured by metric, with a legend), on a
        native canvas. Two multi-select filters (Metrics / Cases) are collapsible
        checklists — Flet has no native multiselect dropdown — both defaulting to
        all selected. The metric set is registry-derived (no hardcodes)."""
        if not cases:
            return ft.Text("No case-level data for this run.", italic=True)
        cols = self._chart_metric_cols(cases)
        if not cols:
            return ft.Text("No 0–1 score metrics to chart for this run.", italic=True)

        metric_keys = [c for c, _ in cols]
        case_keys = [c.get("case_id") or "" for c in cases]
        if self._chart_metrics is None:
            self._chart_metrics = set(metric_keys)
        if self._chart_cases is None:
            self._chart_cases = set(case_keys)

        self._metrics_title = ft.Text(size=13, weight=ft.FontWeight.BOLD)
        self._cases_title = ft.Text(size=13, weight=ft.FontWeight.BOLD)
        self._sync_filter_titles(cases)
        metrics_filter = self._filter_expander(
            self._metrics_title, cols, self._chart_metrics, self._on_metric_toggle
        )
        cases_filter = self._filter_expander(
            self._cases_title,
            [(cid, cid) for cid in case_keys],
            self._chart_cases,
            self._on_case_toggle,
        )

        self._chart_host = ft.Container(content=self._draw_metric_chart(cases))
        return ft.Column(
            [
                self._section_header("Metric Scores by Case"),
                ft.Row(
                    [metrics_filter, cases_filter],
                    spacing=16,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                self._chart_host,
            ],
            spacing=8,
        )

    @staticmethod
    def _filter_expander(
        title: ft.Text,
        items: list[tuple[str, str]],
        selected: set[str],
        on_toggle: Callable[[str, bool], None],
    ) -> ft.Control:
        """A collapsible multi-select: an ExpansionTile titled with a live
        selected/total count, opening to a scrollable checkbox list. Native
        (ExpansionTile + Checkbox) — the substitute for Flet's absent multiselect
        dropdown; toggling a box redraws the chart without collapsing the tile."""
        checks = [
            ft.Checkbox(
                label=text,
                value=key in selected,
                on_change=lambda e, k=key: on_toggle(k, bool(e.control.value)),
            )
            for key, text in items
        ]
        return ft.Container(
            width=300,
            content=ft.ExpansionTile(
                title=title,
                controls=[
                    ft.Container(
                        height=min(220, 24 + 34 * len(checks)),
                        content=ft.Column(checks, scroll=ft.ScrollMode.AUTO, spacing=0),
                    )
                ],
            ),
        )

    def _draw_metric_chart(self, cases: list[dict[str, Any]]) -> ft.Control:
        """Draw the selected metrics' bars grouped by case on a canvas sized to
        the group count (horizontally scrollable). Each metric keeps one colour
        (legend below); case labels are angled so long ids don't collide. A
        metric/case with no value simply draws no bar."""
        cols = self._chart_metric_cols(cases)
        if not cols:
            return ft.Text("No 0–1 score metrics to chart for this run.", italic=True)
        color_of = {
            col: _METRIC_PALETTE[i % len(_METRIC_PALETTE)] for i, (col, _) in enumerate(cols)
        }
        label_of = dict(cols)
        selected_metrics = self._chart_metrics or set()
        selected_cases = self._chart_cases or set()
        metrics = [col for col, _ in cols if col in selected_metrics]
        chosen = [c for c in cases if (c.get("case_id") or "") in selected_cases]
        if not metrics or not chosen:
            return ft.Text("Select at least one metric and one case to chart.", italic=True)

        m = len(metrics)
        group_w = m * _CHART_BAR_W + (m - 1) * _CHART_BAR_INTRA
        pitch = group_w + _CHART_GROUP_GAP
        width = max(_CHART_MIN_W, _CHART_ML + _CHART_MR + len(chosen) * pitch)
        plot_h = _CHART_H - _CHART_MT - _CHART_MB
        baseline = _CHART_MT + plot_h

        def py(v: float) -> float:
            return _CHART_MT + plot_h - max(0.0, min(1.0, v)) * plot_h

        tick_style = ft.TextStyle(size=12, color=ft.Colors.GREY_500)
        shapes: list[cv.Shape] = [
            cv.Line(_CHART_ML, _CHART_MT, _CHART_ML, baseline, _stroke_paint(ft.Colors.GREY_500)),
            cv.Line(
                _CHART_ML, baseline, width - _CHART_MR, baseline, _stroke_paint(ft.Colors.GREY_500)
            ),
        ]
        for frac in (0.0, 0.5, 1.0):
            y = py(frac)
            shapes.append(
                cv.Line(_CHART_ML, y, width - _CHART_MR, y, _stroke_paint(ft.Colors.GREY_800))
            )
            shapes.append(cv.Text(2, y - 6, f"{frac:.1f}", tick_style))

        for g, c in enumerate(chosen):
            gx = _CHART_ML + g * pitch + _CHART_GROUP_GAP / 2
            for j, col in enumerate(metrics):
                v = c.get(col)
                if not isinstance(v, (int, float)):
                    continue
                x = gx + j * (_CHART_BAR_W + _CHART_BAR_INTRA)
                top = py(float(v))
                shapes.append(
                    cv.Rect(x, top, _CHART_BAR_W, baseline - top, paint=_fill_paint(color_of[col]))
                )
            shapes.append(
                cv.Text(
                    gx + group_w / 2,
                    baseline + 6,
                    _truncate(c.get("case_id") or "", 16),
                    tick_style,
                    rotate=_CHART_LABEL_ANGLE,
                    alignment=ft.Alignment(1, -1),
                )
            )

        chart = cv.Canvas(shapes=shapes, width=width, height=_CHART_H)
        # Legend on the RIGHT of the chart (matching the reference): a vertical
        # swatch+label list, top-aligned beside the horizontally-scrolling chart
        # and height-capped to the chart so a long legend scrolls, not overflows.
        legend = ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(width=12, height=12, bgcolor=color_of[col], border_radius=2),
                        ft.Text(label_of.get(col, col), size=12),
                    ],
                    spacing=6,
                )
                for col in metrics
            ],
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
        )
        return ft.Row(
            [
                ft.Row([chart], scroll=ft.ScrollMode.AUTO, expand=True),
                ft.Container(content=legend, height=_CHART_H, padding=ft.Padding.only(left=8)),
            ],
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=8,
        )

    def _sync_filter_titles(self, cases: list[dict[str, Any]]) -> None:
        """Refresh the two filter titles' live 'selected/total' counts."""
        total_m = len(self._chart_metric_cols(cases))
        if self._metrics_title is not None:
            sel_m = len(self._chart_metrics or set())
            self._metrics_title.value = f"Metrics ({sel_m}/{total_m})"
        if self._cases_title is not None:
            sel_c = len(self._chart_cases or set())
            self._cases_title.value = f"Cases ({sel_c}/{len(cases)})"

    def _on_metric_toggle(self, col: str, checked: bool) -> None:
        if self._chart_metrics is None:
            return
        (self._chart_metrics.add if checked else self._chart_metrics.discard)(col)
        self._sync_filter_titles(self._cases)
        self._redraw_chart()

    def _on_case_toggle(self, cid: str, checked: bool) -> None:
        if self._chart_cases is None:
            return
        (self._chart_cases.add if checked else self._chart_cases.discard)(cid)
        self._sync_filter_titles(self._cases)
        self._redraw_chart()

    def _redraw_chart(self) -> None:
        if self._chart_host is not None:
            self._chart_host.content = self._draw_metric_chart(self._cases)
            self.app.page.update()

    def _answer_detail(self, cases: list[dict[str, Any]]) -> ft.Control:
        """Per-case expandable panels: the question, the expected answer (the
        run's joined `expected_answer_points`), the agent's answer, and quick
        stats (judge score, latency, Hit@k). Expected + agent answers render as
        markdown so an embedded Evidence section (headings, lists) formats."""
        from knowledge_agent.evaluation.registry import metric_fmts

        if not cases:
            return ft.Text("No case-level data for this run.", italic=True)

        fmts = metric_fmts()
        tiles: list[ft.Control] = []
        for c in cases:
            status = c.get("status") or ""
            icon = "✅" if status == "PASS" else "⚠️"
            score = self._fmt(c.get("avg_judge_score"), fmts.get("avg_judge_score", ".2f"))
            tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        f"{icon}  {c.get('case_id') or ''} — {status}  ({score})", size=13
                    ),
                    controls=[
                        ft.Container(
                            padding=ft.Padding.symmetric(horizontal=12, vertical=4),
                            content=ft.Column(
                                spacing=8,
                                controls=[
                                    self._detail_block("Question", c.get("question")),
                                    self._detail_block(
                                        "Expected answer", c.get("expected_output"), markdown=True
                                    ),
                                    self._detail_block(
                                        "Agent answer", c.get("answer"), markdown=True
                                    ),
                                    self._labeled_box(
                                        "Quick Stats",
                                        ft.Row(
                                            [
                                                self._stat("Score", score),
                                                self._stat(
                                                    "Latency",
                                                    self._fmt(
                                                        c.get("latency_seconds"),
                                                        fmts.get("latency_seconds", ".2f"),
                                                    ),
                                                ),
                                                self._stat(
                                                    "Hit@k", self._hit_display(c.get("hit_at_k"))
                                                ),
                                            ],
                                            spacing=28,
                                        ),
                                    ),
                                ],
                            ),
                        )
                    ],
                )
            )
        return ft.Column([self._section_header("Answer Detail"), *tiles], spacing=4)

    @staticmethod
    def _labeled_box(label: str, body: ft.Control) -> ft.Control:
        """A bold label above a bordered box wrapping `body` — the shared shape
        for every Answer-Detail block (Question / Expected / Agent answer / Quick
        Stats) so they line up identically."""
        return ft.Column(
            spacing=2,
            controls=[
                ft.Text(label, size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
                ft.Container(
                    padding=8,
                    border=ft.Border.all(1, ft.Colors.GREY_700),
                    border_radius=4,
                    content=body,
                ),
            ],
        )

    @staticmethod
    def _detail_block(label: str, text: Any, *, markdown: bool = False) -> ft.Control:
        if not (text and str(text).strip()):
            body: ft.Control = ft.Text("—", size=12)
        elif markdown:
            # Render as markdown so an Evidence section / lists / bold format,
            # matching the Streamlit reference (Metrics Guide uses the same set).
            body = ft.Markdown(
                str(text),
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            )
        else:
            body = ft.Text(str(text), size=12, selectable=True)
        return RunSummaryTab._labeled_box(label, body)

    @staticmethod
    def _stat(label: str, value: str) -> ft.Control:
        return ft.Column(
            spacing=0,
            controls=[
                ft.Text(label, size=12, color=ft.Colors.GREY_400),
                ft.Text(value, size=14, weight=ft.FontWeight.BOLD),
            ],
        )

    @staticmethod
    def _hit_display(hit: Any) -> str:
        if not isinstance(hit, (int, float)):
            return "Not evaluated"
        return "Yes" if hit == 1.0 else "No"

    # ---- handlers ---------------------------------------------------------

    def _on_select_run(self, _e: ft.Event) -> None:
        if self.run_dropdown and self.run_dropdown.value:
            self.coordinator.selected_run_id = int(self.run_dropdown.value)
            self.refresh()

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()
