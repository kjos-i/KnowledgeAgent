"""Evaluation → Compare Datasets sub-tab — one suite-run's members side by side.

The picker lives in the **shared `DashboardRail`**, so every dashboard tab shows
the identical left column. This tab's BODY reads the coordinator's selected run,
finds its `suite_run_id`, and renders every member of that suite execution (the
same facts under swept retrieval knobs) as the comparison: a metric × run table
(the first member is the baseline, the others show a ± delta vs it, the best run
per metric in bold) plus grouped bars per score group.

A comparability banner flags when the compared members used different recipes
(normally they don't — a suite runs all its members under one recipe).

Run-level `avg_*` columns come from each run row; under an active origin filter
they are recomputed over the kept cases (`build_summary`). `RunSummaryTab._fmt`
/ `._delta` are reused from Run Summary (single source, no second impl).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import dashboard_section_header, section_divider
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail, dataset_of
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_METRIC_COL_W = 150  # width of one run's column in the compare table
_LABEL_COL_W = 190  # width of the metric-label column
_BAR_W = 120  # grouped-bar track width


class CompareDatasetsTab:
    """Compare the members of ONE suite execution side by side. The shared rail's
    cascade selects a run; this tab shows every run sharing that run's
    `suite_run_id` (the same facts under swept knobs) as a metric × run table +
    grouped bars. Body-only — the rail owns the selectors."""

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
                    "Select a run from a suite execution to compare its members. "
                    "Run a suite from the Run tab using the Whole suite scope.",
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
                ft.Column(
                    [
                        view_header(
                            "Compare Datasets",
                            trailing=info(self.app, "eval_compare.overview"),
                        ),
                        self.body,
                    ],
                    expand=True,
                    spacing=8,
                ),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload the rail (runs + cascade selection) then render the body."""
        if self.rail is not None:
            self.rail.refresh()
        self._render_body()

    def _render_body(self) -> None:
        if self.body is None:
            return
        ledger = self._ledger()
        run_id = self.coordinator.selected_run_id
        selected = ledger.get_run(run_id) if run_id is not None else None
        suite_run_id = selected.get("suite_run_id") if selected else None
        cols: list[tuple[str, dict[str, Any]]] = []
        if suite_run_id is not None:
            # Every run stamped with this suite_run_id = the members of one
            # execution. Recompute each member over the shared origin filter (all
            # checked = the stored member row, unchanged).
            from knowledge_agent.gui.evaluation._common import filtered_run

            origins = self.coordinator.selected_origins
            cols = [
                (dataset_of(r), filtered_run(self.app, r["run_id"], origins)[0])
                for r in ledger.get_suite_run(suite_run_id)
            ]
            # Drop a member with no cases left after the origin filter (it would
            # render an all-"Not evaluated" column with no explanation).
            cols = [(ds, run) for ds, run in cols if run is not None and run.get("case_count")]
        # Always render the comparison structure: with no suite-run to compare (a
        # single ordinary run, or none selected), or a filter that empties it, the
        # table + bars still draw — a single "No run selected" placeholder column,
        # every metric "Not evaluated" — beneath a guidance line (the always-show
        # rule), never collapsing to one message.
        controls: list[ft.Control] = []
        notice = self._skeleton_notice(suite_run_id, cols)
        if notice:
            controls.append(ft.Text(notice, italic=True, color=ft.Colors.GREY_500))
        banner = self._comparability_banner(cols)
        if banner is not None:
            controls.append(banner)
        controls.append(self._compare_table(cols))
        controls.append(section_divider())
        controls.extend(self._compare_bars(cols))
        self.body.controls = controls
        self.app.page.update()

    @staticmethod
    def _skeleton_notice(suite_run_id: Any, cols: list[tuple[str, dict[str, Any]]]) -> str | None:
        """The guidance line above an empty comparison skeleton, or None for a real
        2+ member comparison (nothing to explain then)."""
        if suite_run_id is None:
            return (
                "Select a run from a suite execution to compare its members (same "
                "facts, swept knobs). Run a suite from the Run tab using the Whole "
                "suite scope. The comparison structure is shown empty below."
            )
        if not cols:
            return (
                "No cases match the selected origin filter for this suite-run. The "
                "comparison structure is shown empty below."
            )
        if len(cols) < 2:
            return (
                "This suite-run has only one member with cases under the current "
                "filter, so there is nothing to compare against yet."
            )
        return None

    # ---- comparison rendering ---------------------------------------------

    def _comparability_banner(self, cols: list[tuple[str, dict[str, Any]]]) -> ft.Control | None:
        """Warn when the compared runs used different recipes — a different
        recipe_hash means different thresholds / judges / enabled groups, so some
        metrics may not line up. None when every hash matches (the normal case:
        a suite runs all its members under one recipe)."""
        hashes = {(run.get("recipe_hash") or "none") for _, run in cols}
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
                        "These runs used different recipes (hashes differ), so some "
                        "metrics may not line up; unmatched ones show as Not evaluated.",
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
        aware); the best run per metric is bold. EVERY registry metric row renders;
        a metric no member measured shows "Not evaluated" rather than being hidden.
        With no members (no suite-run selected) a single "No run selected"
        placeholder column carries the empty skeleton."""
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
        # One placeholder column when there are no members, so the metric-row
        # skeleton still has a value column (each cell "Not evaluated").
        display_cols = cols or [("No run selected", {})]

        def _col_sub(i: int, run: dict[str, Any]) -> str:
            if not run:
                return "Not evaluated"
            sub = f"Run {run['run_id']}"
            if i == 0:
                sub += " · baseline"
            if run.get("case_count"):
                sub += f" · n={run.get('case_count')}"
            return sub

        header = self._grid_row(
            [_cell(ft.Text("Metric", size=12, weight=ft.FontWeight.BOLD), _LABEL_COL_W)]
            + [
                _cell(
                    ft.Column(
                        [
                            ft.Text(ds, size=12, weight=ft.FontWeight.BOLD),
                            ft.Text(_col_sub(i, run), size=11, color=ft.Colors.GREY_400),
                        ],
                        spacing=0,
                    ),
                    _METRIC_COL_W,
                )
                for i, (ds, run) in enumerate(display_cols)
            ]
        )

        rows: list[ft.Control] = [header]

        def metric_row(key: str, label: str) -> ft.Control:
            values = [run.get(key) for run in run_rows]
            fmt = fmts.get(key, ".2f")
            higher = directions.get(key, True)
            best = _best_index(values, higher_is_better=higher)  # -1 when < 2 numeric
            cells = [_cell(ft.Text(label, size=12), _LABEL_COL_W)]
            base = values[0] if values else None
            # A placeholder [None] cell when there are no members, so every row
            # renders even with nothing to compare.
            display_values = values or [None]
            for i, v in enumerate(display_values):
                cells.append(
                    _cell(
                        self._value_cell(
                            v, base if (values and i) else None, fmt, higher, bold=i == best
                        ),
                        _METRIC_COL_W,
                    )
                )
            return self._grid_row(cells)

        # Overview row first, then each group (skipping the overview keys so a
        # metric isn't listed twice) — mirrors Run Summary's section order. A group
        # with no metrics of its own is skipped (structural), but a group whose
        # metrics simply have no data still shows, each cell "Not evaluated".
        rows.append(self._group_header(OVERVIEW_LABEL))
        for key in OVERVIEW_ROW:
            rows.append(metric_row(key, _overview_label(key)))
        for group, glabel in GROUP_LABELS.items():
            if group == "summary":
                continue
            group_metrics = [
                m
                for m in METRICS
                if m.group == group and m.summary_avg_key and m.summary_avg_key not in OVERVIEW_ROW
            ]
            if not group_metrics:
                continue
            rows.append(self._group_header(glabel))
            for m in group_metrics:
                rows.append(metric_row(m.summary_avg_key, m.summary_avg_label or m.label))

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
        """Grouped bars, one bar per run per 0-1 score metric, so the score groups
        read visually as well as in the table. EVERY 0-1 score metric in each group
        renders a bar; a value no member measured shows an empty track + "n/a"
        rather than being hidden. Count metrics (fmt "d", e.g. disallowed keyword
        hits) and the tokens/latency/summary groups stay table-only. With no members
        a single "No run selected" placeholder bar carries the empty skeleton."""
        from knowledge_agent.evaluation.registry import GROUP_LABELS, METRICS, metric_labels

        labels = metric_labels()
        score_groups = ("llm", "retrieval", "chunk", "kg", "citation", "keyword")
        display_cols = cols or [("No run selected", {})]
        legend = ft.Row(
            [
                ft.Row(
                    [
                        ft.Container(width=12, height=12, bgcolor=_bar_color(i), border_radius=2),
                        ft.Text(ds, size=12),
                    ],
                    spacing=4,
                )
                for i, (ds, _) in enumerate(display_cols)
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
                # 0-1 score metrics only: a count (fmt "d", e.g. disallowed keyword
                # hits) is not a 0-1 value and would mis-scale on a proportional bar.
                and m.fmt != "d"
            ]
            if not metrics:
                continue  # structural: the group has no 0-1 score metric
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
                    _labeled_bar(run.get(key), _bar_color(i))
                    for i, (_, run) in enumerate(display_cols)
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
    """A left-filled 0-1 proportional bar + its value, for the grouped-bar view.
    A non-numeric value renders an empty track with 'n/a'."""
    frac = value if isinstance(value, (int, float)) else 0.0
    frac = max(0.0, min(1.0, float(frac)))
    text = f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"
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
