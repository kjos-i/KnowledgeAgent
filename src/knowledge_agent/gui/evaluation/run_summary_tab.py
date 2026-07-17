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

import json
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._markdown import themed_markdown
from knowledge_agent.gui._styles import dashboard_section_header, section_divider, thin_rule
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
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


class RunSummaryTab:
    """Per-run KPI + per-case-table sub-tab."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.rail: DashboardRail | None = None
        self.body: ft.Column | None = None
        self._runs: list[dict[str, Any]] = []
        self._cases: list[dict[str, Any]] = []
        self._sort_key: str | None = None
        self._sort_asc: bool = True
        self._table_host: ft.Column | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        self.rail = DashboardRail(self.app, self.coordinator, on_change=self._render_body)
        rail_ctl = self.rail.build()
        self.body = ft.Column(
            controls=[
                ft.Text("Run an evaluation, or press Refresh to load past runs.", italic=True)
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=14,
        )
        return ft.Row(
            [
                rail_ctl,
                ft.Column(
                    [view_header("Run Summary"), self.body],
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
        """Reload the rail (runs + selection) then render the selected run.
        Safe before any run exists (shows an empty state)."""
        if self.rail is not None:
            self.rail.refresh()
        self._render_body()

    def _render_body(self) -> None:
        """Render the KPI + per-case body for the coordinator's selected run —
        the rail owns the selectors + recipe panel; this owns the body."""
        if self.body is None:
            return
        run_id = self.coordinator.selected_run_id
        if run_id is None:
            self.body.controls = [ft.Text("No evaluation runs recorded yet.", italic=True)]
            self.app.page.update()
            return
        from knowledge_agent.gui.evaluation._common import filtered_run

        # The shared origin filter re-scopes the KPIs + case list to the checked
        # provenances (all checked = the stored run, unchanged).
        origins = self.coordinator.selected_origins
        run, cases = filtered_run(self.app, run_id, origins)
        if run is None:
            self.body.controls = [ft.Text("Run not found — press Refresh.", italic=True)]
            self.app.page.update()
            return
        self._cases = cases
        self._runs = self._ledger().list_runs()
        # Previous run (next-lower run_id) drives the delta pills on the KPI
        # cards — mirroring the Streamlit dashboard's compare-to-previous view.
        # Filter it too, so the delta is like-for-like under the active filter.
        prev_id = max((r["run_id"] for r in self._runs if r["run_id"] < run_id), default=None)
        prev_run = filtered_run(self.app, prev_id, origins)[0] if prev_id is not None else None
        self.body.controls = self._build_body(run, prev_run, cases)
        self.app.page.update()

    # ---- body: KPIs + table ----------------------------------------------

    def _build_body(
        self, run: dict[str, Any], prev_run: dict[str, Any] | None, cases: list[dict[str, Any]]
    ) -> list[ft.Control]:
        # Metric sections (Average Performance + each group row) separated by a
        # thin sub-section rule; the heavier section_divider() below marks the
        # break from the metrics down to the case table / detail sections.
        metric_sections = [self._overview_row(run, prev_run), *self._kpi_sections(run, prev_run)]
        metrics: list[ft.Control] = []
        for i, section in enumerate(metric_sections):
            if i:
                metrics.append(thin_rule())
            metrics.append(section)
        return [
            *metrics,
            section_divider(),
            self._case_table(cases),
            section_divider(),
            self._case_details(cases),
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
    def _section_header(text: str) -> ft.Control:
        """A dashboard body section header (Average Performance, Judge Scores, …)
        — the shared non-bold style all four view tabs use (`section_divider()`
        separates the sections)."""
        return dashboard_section_header(text)

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
                self._section_header("Per-case results"),
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

    def _case_details(self, cases: list[dict[str, Any]]) -> ft.Control:
        """Per-case expandable panels: the question, the expected answer (the
        run's joined `expected_answer_points`), the agent's answer, quick stats
        (judge score, latency, Hit@k), and the Case Settings the case ran under
        (its retrieval knobs). Expected + agent answers render as markdown so an
        embedded Evidence section (headings, lists) formats."""
        from knowledge_agent.evaluation.registry import metric_fmts

        if not cases:
            return ft.Text("No case-level data for this run.", italic=True)

        fmts = metric_fmts()
        tiles: list[ft.Control] = []
        for c in cases:
            status = c.get("status") or ""
            icon = "✅" if status == "PASS" else "⚠️"
            score = self._fmt(c.get("avg_judge_score"), fmts.get("avg_judge_score", ".2f"))
            detail_controls: list[ft.Control] = [
                self._detail_block("Question", c.get("question")),
                self._detail_block("Expected answer", c.get("expected_output"), markdown=True),
                self._detail_block("Agent answer", c.get("answer"), markdown=True),
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
                            self._stat("Hit@k", self._hit_display(c.get("hit_at_k"))),
                        ],
                        spacing=28,
                    ),
                ),
            ]
            # The per-case retrieval settings this case ran under (NULL for runs
            # predating the ledger column → block omitted).
            settings = self._case_settings_block(c.get("case_settings"))
            if settings is not None:
                detail_controls.append(settings)
            # For origin="chat" cases only: show the conversation that produced
            # the question (provenance, snapshotted into the ledger at run time).
            conversation = self._conversation_block(c.get("source_conversation"))
            if conversation is not None:
                detail_controls.append(conversation)
            tiles.append(
                ft.ExpansionTile(
                    title=ft.Text(
                        f"{icon}  {c.get('case_id') or ''} — {status}  ({score})", size=13
                    ),
                    controls=[
                        ft.Container(
                            padding=ft.Padding.symmetric(horizontal=12, vertical=4),
                            content=ft.Column(spacing=8, controls=detail_controls),
                        )
                    ],
                )
            )
        return ft.Column([self._section_header("Case Details"), *tiles], spacing=4)

    @staticmethod
    def _case_settings_block(raw: Any) -> ft.Control | None:
        """The per-case retrieval settings this case ran under, or None when the
        run predates the ledger's `case_settings` column (old rows store NULL).
        `raw` is the ledger's JSON string (a `RetrievalSettings` dump); each knob
        is shown as a label/value cell. A nullable tuning knob left unpinned
        (None) fell back to the live global setting, shown as 'global'."""
        if not raw:
            return None
        try:
            settings = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            return None
        if not isinstance(settings, dict) or not settings:
            return None
        cells = [
            RunSummaryTab._stat(
                key.replace("_", " ").capitalize(),
                "global" if value is None else str(value),
            )
            for key, value in settings.items()
        ]
        return RunSummaryTab._labeled_box(
            "Case Settings", ft.Row(cells, wrap=True, spacing=28, run_spacing=10)
        )

    @staticmethod
    def _conversation_block(raw: Any) -> ft.Control | None:
        """The chat that produced an origin='chat' case, or None when the case has
        no stored conversation. `raw` is the ledger's JSON string (a list of
        {role, content} turns); rendered as markdown so the roles read clearly."""
        if not raw:
            return None
        try:
            turns = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            return None
        if not isinstance(turns, list):
            return None
        lines: list[str] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = (t.get("role") or "?").strip()
            content = (t.get("content") or "").strip()
            if content:
                lines.append(f"**{role}:** {content}")
        if not lines:
            return None
        return RunSummaryTab._detail_block("Conversation (chat)", "\n\n".join(lines), markdown=True)

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
            body = themed_markdown(str(text))
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
