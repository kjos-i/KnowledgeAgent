"""Evaluation → Ledger sub-tab — browse and prune the per-corpus eval ledger.

A management/editing tab, deliberately DISCONNECTED from the shared dashboard
cascade (`DashboardRail` / coordinator selection): it keeps its own local run
selection so browsing/deleting here never moves the analysis tabs. The tab sits
last (rightmost) and is flagged with an edit icon + a distinct label colour.

Left column (the familiar gray rail shell): Refresh + info on top, then run
FILTERS (suite / dataset / date), then a DELETE section ("Delete run" /
"Delete suite"). Right column is master-detail: a bounded-height, click-to-select
RUNS table on top and the selected run's CASES appended below (placeholder until
a run is picked); the whole right column scrolls for overflow.

Deletes go through `EvalLedger.delete_run` / `delete_suite` (confirm dialog
first). The ledger is one-per-corpus, so everything here is scoped to the active
corpus by construction. A delete removes the run from the shared ledger the other
dashboard tabs read, so they re-derive on their next refresh.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.gui._styles import (
    FIELD_LABEL_SIZE,
    PANEL_BG_RAISED,
    PANEL_RADIUS,
    dashboard_section_header,
    section_divider,
    sub_section_header,
)
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_RAIL_WIDTH = 280
_RUNS_HEIGHT = 360  # bounded height of the runs window (keeps cases reachable)
_ALL = "__all__"  # dropdown sentinel for "no filter"

# Runs table columns: (header, width, key-or-formatter). Easy to change later.
_RUN_COLS: tuple[tuple[str, int], ...] = (
    ("Run", 55),
    ("Date", 150),
    ("Dataset", 170),
    ("Suite", 120),
    ("Pass rate", 80),
    ("Cases", 60),
)
_CASE_COLS: tuple[tuple[str, int], ...] = (
    ("Case", 110),
    ("Status", 90),
    ("Question", 420),
)


class LedgerTab:
    """Browse + delete rows of the active corpus's eval ledger. Disconnected
    from the shared dashboard selection (its own local `_selected_run_id`)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator  # kept for signature parity; not read
        self._selected_run_id: int | None = None
        self._runs: list[dict[str, Any]] = []  # filtered runs currently shown
        # controls
        self.suite_dd: ft.Dropdown | None = None
        self.dataset_dd: ft.Dropdown | None = None
        self.date_from_field: ft.TextField | None = None  # keep runs on/after
        self.date_to_field: ft.TextField | None = None  # keep runs on/before
        self.delete_run_btn: ft.Button | None = None
        self.delete_suite_btn: ft.Button | None = None
        self.runs_body: ft.Column | None = None
        self.cases_body: ft.Column | None = None
        self.status: ft.Text | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        # No border label; the field name shows as the inside hint (empty =
        # unfiltered), and an explicit "All …" option resets a chosen filter.
        self.suite_dd = ft.Dropdown(
            hint_text="Suite",
            options=[ft.DropdownOption(key=_ALL, text="All suites")],
            width=_RAIL_WIDTH - 24,
            text_size=FIELD_LABEL_SIZE,
        )
        self.suite_dd.on_select = self._on_filter_change
        self.dataset_dd = ft.Dropdown(
            hint_text="Dataset",
            options=[ft.DropdownOption(key=_ALL, text="All datasets")],
            width=_RAIL_WIDTH - 24,
            text_size=FIELD_LABEL_SIZE,
        )
        self.dataset_dd.on_select = self._on_filter_change
        # Date range: From (on/after) and To (on/before), side by side. Either
        # blank = no bound on that side. Compared on the run's local date.
        _date_w = (_RAIL_WIDTH - 32) / 2
        self.date_from_field = ft.TextField(
            hint_text="From date",
            tooltip="Keep runs on or after this date (DD.MM.YYYY).",
            width=_date_w,
            text_size=FIELD_LABEL_SIZE,
            on_blur=self._on_filter_change,
        )
        self.date_to_field = ft.TextField(
            hint_text="To date",
            tooltip="Keep runs on or before this date (DD.MM.YYYY).",
            width=_date_w,
            text_size=FIELD_LABEL_SIZE,
            on_blur=self._on_filter_change,
        )

        self.delete_run_btn = ft.Button(
            content="Delete run",
            icon=ft.Icons.DELETE_OUTLINE,
            color=ft.Colors.RED_200,
            disabled=True,
            on_click=self._on_delete_run,
        )
        self.delete_suite_btn = ft.Button(
            content="Delete suite",
            icon=ft.Icons.DELETE_SWEEP_OUTLINED,
            color=ft.Colors.RED_200,
            disabled=True,
            on_click=self._on_delete_suite,
        )
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        rail = ft.Container(
            width=_RAIL_WIDTH,
            bgcolor=PANEL_BG_RAISED,
            padding=12,
            border_radius=PANEL_RADIUS,
            content=ft.Column(
                controls=[
                    ft.Row(
                        [
                            ft.TextButton("Refresh", on_click=lambda _e: self.refresh()),
                            info(self.app, "eval_ledger.overview"),
                        ],
                        spacing=6,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    sub_section_header("Filters"),
                    self.suite_dd,
                    self.dataset_dd,
                    ft.Row([self.date_from_field, self.date_to_field], spacing=8),
                    # No explicit divider here — sub_section_header("Delete")
                    # already draws its own thin rule above the section.
                    sub_section_header("Delete"),
                    self.delete_run_btn,
                    self.delete_suite_btn,
                    self.status,
                ],
                spacing=8,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        self.runs_body = ft.Column(spacing=0)
        self.cases_body = ft.Column(
            [ft.Text("Select a run to see its cases.", italic=True, color=ft.Colors.GREY_500)],
            spacing=0,
        )
        right = ft.Column(
            [
                view_header("Ledger", trailing=info(self.app, "eval_ledger.overview")),
                # Bounded-height runs window: vertical scroll inside, horizontal
                # scroll outside (sideways for extra columns).
                ft.Row(
                    [ft.Container(height=_RUNS_HEIGHT, content=self.runs_body)],
                    scroll=ft.ScrollMode.AUTO,
                ),
                section_divider(),
                dashboard_section_header("Cases"),
                ft.Row([self.cases_body], scroll=ft.ScrollMode.AUTO),
            ],
            expand=True,
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )
        return ft.Row(
            [rail, right],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    # ---- data / refresh ---------------------------------------------------

    def _ledger(self):
        from knowledge_agent.gui.evaluation._common import active_eval_ledger

        return active_eval_ledger(self.app)

    def refresh(self) -> None:
        """Reload runs from the active corpus's ledger, re-apply filters, and
        re-render. Never crashes on a missing corpus/ledger — shows a message."""
        if self.runs_body is None:
            return
        try:
            all_runs = self._ledger().list_runs()
        except Exception as exc:  # no active corpus / unreadable ledger
            self._runs = []
            self._selected_run_id = None
            self._sync_delete_buttons()
            self.runs_body.controls = [
                ft.Text(f"No ledger to show: {exc}", italic=True, color=ft.Colors.GREY_500)
            ]
            self._render_cases()
            self.app.page.update()
            return
        self._populate_filter_options(all_runs)
        self._runs = [r for r in all_runs if self._passes_filters(r)]
        # Drop a stale selection (the run may have been deleted or filtered out).
        if self._selected_run_id not in {r["run_id"] for r in self._runs}:
            self._selected_run_id = None
        self._render_runs()
        self._render_cases()
        self._sync_delete_buttons()
        self.app.page.update()

    def _populate_filter_options(self, runs: list[dict[str, Any]]) -> None:
        """Fill the suite/dataset dropdowns from the values present, preserving
        the current selection when it still exists."""
        suites = sorted({r.get("suite") for r in runs if r.get("suite")})
        datasets = sorted({_dataset_of(r) for r in runs if _dataset_of(r)})
        if self.suite_dd is not None:
            self.suite_dd.options = [ft.DropdownOption(key=_ALL, text="All suites")] + [
                ft.DropdownOption(key=s, text=s) for s in suites
            ]
            # Preserve None (the empty "Suite" hint = unfiltered) and any value
            # still present; only a stale specific value resets to _ALL.
            if self.suite_dd.value not in {None, _ALL, *suites}:
                self.suite_dd.value = _ALL
        if self.dataset_dd is not None:
            self.dataset_dd.options = [ft.DropdownOption(key=_ALL, text="All datasets")] + [
                ft.DropdownOption(key=d, text=d) for d in datasets
            ]
            if self.dataset_dd.value not in {None, _ALL, *datasets}:
                self.dataset_dd.value = _ALL

    def _passes_filters(self, run: dict[str, Any]) -> bool:
        if (
            self.suite_dd is not None
            and self.suite_dd.value not in (None, _ALL)
            and run.get("suite") != self.suite_dd.value
        ):
            return False
        if (
            self.dataset_dd is not None
            and self.dataset_dd.value not in (None, _ALL)
            and _dataset_of(run) != self.dataset_dd.value
        ):
            return False
        # Date range (DD.MM.YYYY). Either bound optional; compared on the run's
        # LOCAL date (the stored stamp is tz-aware UTC, converted below), so it
        # matches the calendar date the user typed. A run with an unparseable
        # date is excluded whenever any bound is set.
        run_date = _run_local_date(run.get("run_timestamp"))
        d_from = _parse_date(self.date_from_field.value if self.date_from_field else None)
        if d_from is not None and (run_date is None or run_date < d_from):
            return False
        d_to = _parse_date(self.date_to_field.value if self.date_to_field else None)
        return d_to is None or (run_date is not None and run_date <= d_to)

    def _on_filter_change(self, _e: ft.Event) -> None:
        self.refresh()

    # ---- rendering --------------------------------------------------------

    def _render_runs(self) -> None:
        if self.runs_body is None:
            return
        rows: list[ft.Control] = [_grid_header(_RUN_COLS)]
        if not self._runs:
            rows.append(
                ft.Text(
                    "No runs match the current filters.",
                    italic=True,
                    color=ft.Colors.GREY_500,
                )
            )
        for run in self._runs:
            rows.append(self._run_row(run))
        # Vertical scroll lives on the bounded-height column.
        self.runs_body.controls = rows
        self.runs_body.scroll = ft.ScrollMode.AUTO

    def _run_row(self, run: dict[str, Any]) -> ft.Control:
        rid = run["run_id"]
        selected = rid == self._selected_run_id
        cells = [
            _cell(str(rid), _RUN_COLS[0][1]),
            _cell(_fmt_ts(run.get("run_timestamp")), _RUN_COLS[1][1]),
            _cell(_dataset_of(run) or "—", _RUN_COLS[2][1]),
            _cell(run.get("suite") or "—", _RUN_COLS[3][1]),
            _cell(_fmt_rate(run.get("pass_rate")), _RUN_COLS[4][1]),
            _cell(
                str(run.get("case_count") if run.get("case_count") is not None else "—"),
                _RUN_COLS[5][1],
            ),
        ]
        return ft.Container(
            on_click=lambda _e, r=rid: self._select_run(r),
            bgcolor=ft.Colors.with_opacity(0.16, ft.Colors.BLUE) if selected else None,
            border_radius=4,
            padding=ft.Padding.symmetric(vertical=4, horizontal=2),
            content=ft.Row(cells, spacing=0),
        )

    def _select_run(self, run_id: int) -> None:
        self._selected_run_id = run_id
        self._render_runs()  # re-highlight
        self._render_cases()
        self._sync_delete_buttons()
        self.app.page.update()

    def _render_cases(self) -> None:
        if self.cases_body is None:
            return
        if self._selected_run_id is None:
            self.cases_body.controls = [
                ft.Text("Select a run to see its cases.", italic=True, color=ft.Colors.GREY_500)
            ]
            return
        try:
            cases = self._ledger().get_run_cases(self._selected_run_id)
        except Exception as exc:
            self.cases_body.controls = [
                ft.Text(f"Could not read cases: {exc}", italic=True, color=ft.Colors.GREY_500)
            ]
            return
        rows: list[ft.Control] = [_grid_header(_CASE_COLS)]
        if not cases:
            rows.append(ft.Text("This run has no cases.", italic=True, color=ft.Colors.GREY_500))
        for c in cases:
            rows.append(
                ft.Container(
                    padding=ft.Padding.symmetric(vertical=3, horizontal=2),
                    content=ft.Row(
                        [
                            _cell(str(c.get("case_id") or "—"), _CASE_COLS[0][1]),
                            _cell(str(c.get("status") or "—"), _CASE_COLS[1][1]),
                            _cell(_truncate(c.get("question"), 90), _CASE_COLS[2][1]),
                        ],
                        spacing=0,
                    ),
                )
            )
        self.cases_body.controls = rows

    def _sync_delete_buttons(self) -> None:
        run = self._current_run()
        has_sel = run is not None
        has_suite = bool(run and run.get("suite_run_id"))
        if self.delete_run_btn is not None:
            self.delete_run_btn.disabled = not has_sel
        if self.delete_suite_btn is not None:
            # Suite delete only when the selected run belongs to a suite.
            self.delete_suite_btn.disabled = not has_suite

    def _current_run(self) -> dict[str, Any] | None:
        if self._selected_run_id is None:
            return None
        return next((r for r in self._runs if r["run_id"] == self._selected_run_id), None)

    # ---- delete -----------------------------------------------------------

    def _on_delete_run(self, _e: ft.Event) -> None:
        run = self._current_run()
        if run is None:
            return
        rid = run["run_id"]
        self._confirm(
            "Delete run",
            f"Delete run {rid} ({_dataset_of(run) or 'dataset'}) and its "
            f"{run.get('case_count') or 0} cases? This cannot be undone, and it "
            "also removes the run from the other dashboard tabs.",
            lambda: self._do_delete_run(rid),
        )

    def _do_delete_run(self, run_id: int) -> None:
        try:
            n = self._ledger().delete_run(run_id)
        except Exception as exc:
            self._set_status(f"Delete failed: {exc}")
            return
        self._selected_run_id = None
        self._set_status(f"Deleted run {run_id}." if n else f"Run {run_id} was already gone.")
        self.refresh()

    def _on_delete_suite(self, _e: ft.Event) -> None:
        run = self._current_run()
        if run is None or not run.get("suite_run_id"):
            return
        suite_run_id = run["suite_run_id"]
        try:
            members = self._ledger().get_suite_run(suite_run_id)
        except Exception:
            members = []
        n = len(members)
        name = run.get("suite") or suite_run_id
        self._confirm(
            "Delete suite",
            f"Delete all {n} run(s) in suite '{name}' and their cases? This cannot "
            "be undone, and it also removes them from the other dashboard tabs.",
            lambda: self._do_delete_suite(suite_run_id),
        )

    def _do_delete_suite(self, suite_run_id: str) -> None:
        try:
            n = self._ledger().delete_suite(suite_run_id)
        except Exception as exc:
            self._set_status(f"Delete failed: {exc}")
            return
        self._selected_run_id = None
        self._set_status(f"Deleted {n} run(s) from the suite.")
        self.refresh()

    def _confirm(self, title: str, body: str, on_confirm) -> None:
        def _cancel(_e: ft.Event) -> None:
            self.app.page.pop_dialog()

        def _go(_e: ft.Event) -> None:
            self.app.page.pop_dialog()
            on_confirm()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.BOLD),
            content=ft.Text(body),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content="Delete", color=ft.Colors.RED_200, on_click=_go),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    def _set_status(self, msg: str) -> None:
        if self.status is not None:
            self.status.value = msg
            self.app.page.update()


# ---- module helpers -------------------------------------------------------


def _dataset_of(run: dict[str, Any]) -> str:
    return run.get("dataset_name") or run.get("dataset_path") or ""


def _grid_header(cols: tuple[tuple[str, int], ...]) -> ft.Control:
    return ft.Container(
        padding=ft.Padding.only(bottom=4),
        content=ft.Row(
            [
                _cell_ctl(
                    ft.Text(h, size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_300), w
                )
                for h, w in cols
            ],
            spacing=0,
        ),
    )


def _cell(text: str, width: int) -> ft.Control:
    return _cell_ctl(ft.Text(text, size=12, color=ft.Colors.GREY_300, no_wrap=True), width)


def _cell_ctl(content: ft.Control, width: int) -> ft.Control:
    return ft.Container(width=width, content=content)


def _fmt_ts(ts: Any) -> str:
    dt = _parse_ts(ts)
    # Stored UTC → show in local time for the Date column.
    return dt.astimezone().strftime("%Y-%m-%d %H:%M") if dt else (str(ts) if ts else "—")


def _parse_ts(ts: Any) -> datetime | None:
    """Parse a stored run_timestamp to a tz-AWARE datetime (or None).

    Production stamps are tz-aware UTC (runner.py); a naive/legacy stamp is
    treated as UTC so the date-filter comparison is always aware-vs-aware
    (avoids `TypeError: can't compare offset-naive and offset-aware`).
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _run_local_date(ts: Any) -> date | None:
    """The run's LOCAL calendar date (the stored stamp is tz-aware UTC), so a
    date-range filter matches the calendar date the user sees/types."""
    dt = _parse_ts(ts)
    return dt.astimezone().date() if dt else None


def _parse_date(text: Any) -> date | None:
    """Parse a DD.MM.YYYY filter field to a date; None when blank or invalid
    (an invalid entry simply applies no bound rather than erroring)."""
    s = str(text or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except ValueError:
        return None


def _fmt_rate(rate: Any) -> str:
    return f"{rate * 100:.0f}%" if isinstance(rate, (int, float)) else "—"


def _truncate(text: Any, n: int) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"
