"""Evaluation → Compare Datasets sub-tab — several datasets' runs side by side.

The picker lives in the **shared `DashboardRail`** (the "Compare Datasets"
section: a dataset add-list with a run dropdown each), so every dashboard tab
shows the identical left column. This tab's BODY reads that picker's state off
the coordinator (`compare_selected`) and renders the comparison: a metric × run
table (the first dataset is the baseline, the others show a ± delta vs it, the
best run per metric in bold) plus grouped bars per score group.

A comparability banner flags when the selected runs used different recipes.
(Step-4 rework will re-scope this tab to the members of one suite-run — same
facts, swept knobs — which is the properly comparable case.)

Run-level `avg_*` columns are read straight from each run row — no in-GUI
aggregation. `RunSummaryTab._fmt` / `._delta` are reused from Run Summary
(single source, no second impl).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import dashboard_section_header, section_divider
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_METRIC_COL_W = 150  # width of one run's column in the compare table
_LABEL_COL_W = 190  # width of the metric-label column
_BAR_W = 120  # grouped-bar track width


class CompareDatasetsTab:
    """Compare several datasets' runs (one kind at a time) side by side. The
    picker is the shared rail's Compare Datasets section; this owns the body."""

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
            [
                ft.Text(
                    "Add at least 2 datasets to compare — use the Compare Datasets "
                    "section in the left column.",
                    italic=True,
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=16,
        )
        return ft.Row(
            [
                rail_ctl,
                ft.Column([view_header("Compare Datasets"), self.body], expand=True, spacing=8),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload the rail (runs + the Compare picker) then render the body."""
        if self.rail is not None:
            self.rail.refresh()
        self._render_body()

    def _render_body(self) -> None:
        if self.body is None:
            return
        chosen = [s for s in self.coordinator.compare_selected if s["run_id"] is not None]
        if len(chosen) < 2:
            self.body.controls = [
                ft.Text(
                    "Add at least 2 datasets (each with a run) to compare — use the "
                    "Compare Datasets section in the left column.",
                    italic=True,
                )
            ]
            self.app.page.update()
            return
        by_id = {r["run_id"]: r for r in self._ledger().list_runs()}
        cols = [(s["dataset"], by_id[s["run_id"]]) for s in chosen if s["run_id"] in by_id]
        if len(cols) < 2:
            self.body.controls = [ft.Text("Selected runs not found — press Refresh.", italic=True)]
            self.app.page.update()
            return
        controls: list[ft.Control] = []
        banner = self._comparability_banner(cols)
        if banner is not None:
            controls.append(banner)
        controls.append(self._compare_table(cols))
        controls.append(section_divider())
        controls.extend(self._compare_bars(cols))
        self.body.controls = controls
        self.app.page.update()

    # ---- comparison rendering ---------------------------------------------

    def _comparability_banner(self, cols: list[tuple[str, dict[str, Any]]]) -> ft.Control | None:
        """Warn when the selected runs used different recipes — same kind, but a
        different hash means different thresholds / judges / enabled groups, so
        some metrics may not line up. None when every hash matches."""
        hashes = {(run.get("recipe_hash") or "—") for _, run in cols}
        if len(hashes) <= 1:
            return None
        return ft.Container(
            bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.AMBER),
            border_radius=6,
            padding=8,
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.WARNING_AMBER, size=16, color=ft.Colors.AMBER),
                    ft.Text(
                        "These runs used different recipes (hashes differ) — metrics may "
                        "not be directly comparable; only shared ones are shown.",
                        size=12,
                        expand=True,
                    ),
                ],
                spacing=6,
            ),
        )

    def _compare_table(self, cols: list[tuple[str, dict[str, Any]]]) -> ft.Control:
        """Metric × run grid: baseline (first column) shows the raw value; the
        others add a ± delta vs baseline (green better / red worse, direction-
        aware); the best run per metric is bold. Metrics with no value in any
        selected run are skipped, so it degrades gracefully across recipes."""
        from knowledge_agent.evaluation.registry import (
            GROUP_LABELS,
            METRICS,
            OVERVIEW_LABEL,
            OVERVIEW_ROW,
            metric_directions,
            metric_fmts,
        )

        fmts = metric_fmts()
        directions = metric_directions()
        run_rows = [run for _, run in cols]

        header = self._grid_row(
            [_cell(ft.Text("Metric", size=12, weight=ft.FontWeight.BOLD), _LABEL_COL_W)]
            + [
                _cell(
                    ft.Column(
                        [
                            ft.Text(ds, size=12, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"Run {run['run_id']}"
                                + (" · baseline" if i == 0 else "")
                                + (
                                    f" · n={run.get('case_count')}" if run.get("case_count") else ""
                                ),
                                size=11,
                                color=ft.Colors.GREY_400,
                            ),
                        ],
                        spacing=0,
                    ),
                    _METRIC_COL_W,
                )
                for i, (ds, run) in enumerate(cols)
            ]
        )

        rows: list[ft.Control] = [header]

        def metric_row(key: str, label: str) -> ft.Control | None:
            values = [run.get(key) for run in run_rows]
            if not any(isinstance(v, (int, float)) for v in values):
                return None  # shared-only: skip a metric no selected run has
            fmt = fmts.get(key, ".2f")
            higher = directions.get(key, True)
            best = _best_index(values, higher_is_better=higher)
            cells = [_cell(ft.Text(label, size=12), _LABEL_COL_W)]
            base = values[0]
            for i, v in enumerate(values):
                cells.append(
                    _cell(
                        self._value_cell(v, base if i else None, fmt, higher, bold=i == best),
                        _METRIC_COL_W,
                    )
                )
            return self._grid_row(cells)

        # Overview row first, then each group (skipping the overview keys so a
        # metric isn't listed twice) — mirrors Run Summary's section order.
        rows.append(self._group_header(OVERVIEW_LABEL))
        for key in OVERVIEW_ROW:
            r = metric_row(key, _overview_label(key))
            if r is not None:
                rows.append(r)
        for group, glabel in GROUP_LABELS.items():
            if group == "summary":
                continue
            group_rows = []
            for m in METRICS:
                if m.group != group or not m.summary_avg_key or m.summary_avg_key in OVERVIEW_ROW:
                    continue
                r = metric_row(m.summary_avg_key, m.summary_avg_label or m.label)
                if r is not None:
                    group_rows.append(r)
            if group_rows:
                rows.append(self._group_header(glabel))
                rows.extend(group_rows)

        grid = ft.Column(rows, spacing=0)
        # Horizontal scroll when many runs push past the viewport width.
        return ft.Row([grid], scroll=ft.ScrollMode.AUTO)

    def _value_cell(
        self, value: Any, baseline: Any, fmt: str, higher: bool, *, bold: bool
    ) -> ft.Control:
        """One metric cell: the formatted value (bold if it's the best run) plus,
        for non-baseline columns, a signed delta vs baseline coloured by
        improvement."""
        text = RunSummaryTab._fmt(value, fmt)
        children: list[ft.Control] = [
            ft.Text(
                text,
                size=12,
                weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
                color=ft.Colors.WHITE if bold else ft.Colors.GREY_300,
            )
        ]
        if baseline is not None:
            d_text, d_good = RunSummaryTab._delta(value, baseline, fmt, lower_is_better=not higher)
            if d_text is not None:
                color = ft.Colors.GREEN if d_good else ft.Colors.RED
                children.append(ft.Text(d_text, size=11, color=color))
        return ft.Row(children, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _compare_bars(self, cols: list[tuple[str, dict[str, Any]]]) -> list[ft.Control]:
        """Grouped bars per 0–1 score group — one bar per selected run per metric,
        so the score groups read visually as well as in the table. Non-score
        groups (tokens/latency) are table-only (they aren't 0–1)."""
        from knowledge_agent.evaluation.registry import GROUP_LABELS, METRICS, metric_labels

        labels = metric_labels()
        score_groups = ("llm", "retrieval", "chunk", "kg", "citation")
        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=12, height=12, bgcolor=_bar_color(i), border_radius=2),
                        ft.Text(ds, size=12),
                    ],
                    spacing=4,
                )
                for i, (ds, _) in enumerate(cols)
            ],
            wrap=True,
            spacing=12,
        )
        out: list[ft.Control] = [dashboard_section_header("Score comparison"), legend]
        for group in score_groups:
            metrics = [
                m
                for m in METRICS
                if m.group == group
                and m.summary_avg_key
                and any(isinstance(run.get(m.summary_avg_key), (int, float)) for _, run in cols)
            ]
            if not metrics:
                continue
            metric_rows: list[ft.Control] = [
                ft.Text(
                    GROUP_LABELS[group],
                    size=12,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.GREY_400,
                )
            ]
            for m in metrics:
                key = m.summary_avg_key
                bars = [
                    _labeled_bar(run.get(key), _bar_color(i)) for i, (_, run) in enumerate(cols)
                ]
                metric_rows.append(
                    ft.Row(
                        [ft.Text(labels.get(key, m.label), size=12, width=_LABEL_COL_W), *bars],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    )
                )
            out.append(ft.Column(metric_rows, spacing=6))
        return out

    @staticmethod
    def _grid_row(cells: list[ft.Control]) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.symmetric(vertical=4),
            content=ft.Row(cells, spacing=0, vertical_alignment=ft.CrossAxisAlignment.START),
        )

    @staticmethod
    def _group_header(text: str) -> ft.Control:
        return ft.Container(
            padding=ft.Padding.only(top=10, bottom=2),
            content=ft.Text(text, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300),
        )


def _cell(content: ft.Control, width: int) -> ft.Control:
    return ft.Container(width=width, content=content)


def _overview_label(key: str) -> str:
    from knowledge_agent.evaluation.registry import metric_labels

    return metric_labels().get(key, key)


def _best_index(values: list[Any], *, higher_is_better: bool) -> int:
    """Index of the best numeric value (max when higher-is-better, else min);
    -1 when fewer than 2 numeric values differ (nothing to crown)."""
    nums = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
    if len(nums) < 2 or len({v for _, v in nums}) == 1:
        return -1
    return (max if higher_is_better else min)(nums, key=lambda p: p[1])[0]


def _bar_color(i: int) -> str:
    palette = (
        ft.Colors.BLUE,
        ft.Colors.GREEN,
        ft.Colors.ORANGE,
        ft.Colors.PURPLE,
        ft.Colors.TEAL,
        ft.Colors.PINK,
        ft.Colors.BROWN,
        ft.Colors.CYAN,
    )
    return palette[i % len(palette)]


def _labeled_bar(value: Any, color: str) -> ft.Control:
    """A left-filled 0–1 proportional bar + its value, for the grouped-bar view.
    A non-numeric value renders an empty track with '—'."""
    frac = value if isinstance(value, (int, float)) else 0.0
    frac = max(0.0, min(1.0, float(frac)))
    text = f"{value:.2f}" if isinstance(value, (int, float)) else "—"
    return ft.Row(
        [
            ft.Container(
                width=_BAR_W,
                height=12,
                bgcolor=ft.Colors.GREY_800,
                border_radius=3,
                alignment=ft.Alignment(-1, 0),
                content=ft.Container(
                    width=max(2.0, frac * _BAR_W), height=12, bgcolor=color, border_radius=3
                ),
            ),
            ft.Text(text, size=11, color=ft.Colors.GREY_300),
        ],
        spacing=4,
    )
