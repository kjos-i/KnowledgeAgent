"""Tests for the Evaluation → Ledger sub-tab.

Seeds a tmp ledger, patches `active_eval_ledger` to it, builds the tab, and
checks browsing + the run/suite deletes. Catches Flet API errors in build/render
and verifies the delete wiring against the real EvalLedger methods.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation.ledger_tab import LedgerTab

_LEDGER = "knowledge_agent.gui.evaluation._common.active_eval_ledger"


def _coord() -> SimpleNamespace:
    return SimpleNamespace(selected_run_id=None)


def _report(
    dataset: str,
    *,
    suite_run_id: str | None = None,
    suite: str | None = None,
    run_timestamp: str = "2026-07-05T10:00:00",
) -> dict:
    return {
        "run_timestamp": run_timestamp,
        "dataset_path": f"{dataset}.json",
        "dataset_name": dataset,
        "suite": suite,
        "suite_run_id": suite_run_id,
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "summary": {"case_count": 1, "pass_count": 1, "pass_rate": 1.0},
        "results": [{"id": "c1", "category": "", "question": "q?", "status": "PASS", "errors": []}],
    }


def _seed(tmp_path) -> EvalLedger:
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("vec", suite_run_id="s1", suite="mysuite"))  # run 1
    led.save_run(_report("graph", suite_run_id="s1", suite="mysuite"))  # run 2 (same suite)
    led.save_run(_report("solo"))  # run 3 (no suite)
    return led


def test_ledger_tab_builds(fake_app):
    assert LedgerTab(fake_app, coordinator=_coord()).build() is not None


def test_ledger_tab_lists_runs(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    # header + 3 run rows
    assert len(tab._runs) == 3
    assert len(tab.runs_body.controls) == 4


def test_ledger_tab_no_ledger_shows_message_not_crash(fake_app):
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, side_effect=RuntimeError("no corpus")):
        tab.refresh()  # must not raise
    assert tab._runs == []
    assert len(tab.runs_body.controls) == 1  # the message


def test_ledger_tab_select_populates_cases_and_enables_delete(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        tab._select_run(1)  # the vec run (has a suite)
    assert tab._selected_run_id == 1
    assert not tab.delete_run_btn.disabled
    assert not tab.delete_suite_btn.disabled  # run 1 belongs to suite s1
    assert len(tab.cases_body.controls) >= 2  # header + >=1 case


def test_ledger_tab_delete_suite_button_gated_when_no_suite(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        tab._select_run(3)  # the solo run (no suite)
    assert not tab.delete_run_btn.disabled
    assert tab.delete_suite_btn.disabled  # no suite -> disabled


def test_ledger_tab_delete_run(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        tab._select_run(3)
        tab._do_delete_run(3)
    assert led.get_run(3) is None
    assert {r["run_id"] for r in tab._runs} == {1, 2}
    assert tab._selected_run_id is None  # selection cleared after delete


def test_ledger_tab_delete_suite(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        tab._select_run(1)
        tab._do_delete_suite("s1")
    assert led.get_run(1) is None
    assert led.get_run(2) is None
    assert led.get_run(3) is not None  # the solo run survives
    assert {r["run_id"] for r in tab._runs} == {3}


def test_ledger_tab_dataset_filter(fake_app, tmp_path):
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    tab.dataset_dd.value = "vec"
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    assert {_r["dataset_name"] for _r in tab._runs} == {"vec"}


def test_ledger_tab_date_from_filter_handles_tz_aware_timestamps(fake_app, tmp_path):
    """Production run_timestamp is tz-aware UTC; the From date bound must compare
    cleanly (regression: naive cutoff vs aware stamp raised TypeError)."""
    led = EvalLedger(tmp_path / "l.db")
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()  # aware UTC
    old = (datetime.now(UTC) - timedelta(days=60)).isoformat()  # aware UTC
    led.save_run(_report("recent", run_timestamp=recent))  # run 1
    led.save_run(_report("old", run_timestamp=old))  # run 2
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    tab.date_from_field.value = (datetime.now() - timedelta(days=7)).strftime("%d.%m.%Y")
    with patch(_LEDGER, return_value=led):
        tab.refresh()  # must NOT raise on the aware-timestamp comparison
    assert {_r["dataset_name"] for _r in tab._runs} == {"recent"}  # old one excluded


def test_ledger_tab_date_filter_handles_naive_legacy_timestamp(fake_app, tmp_path):
    """A legacy naive run_timestamp (no tz) is treated as UTC, so a date bound
    still compares cleanly (no aware-vs-naive crash)."""
    led = EvalLedger(tmp_path / "l.db")
    recent_naive = (datetime.now() - timedelta(days=1)).replace(microsecond=0).isoformat()
    led.save_run(_report("legacy", run_timestamp=recent_naive))  # naive stamp
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    tab.date_from_field.value = (datetime.now() - timedelta(days=7)).strftime("%d.%m.%Y")
    with patch(_LEDGER, return_value=led):
        tab.refresh()  # must not raise on the naive stamp
    assert {_r["dataset_name"] for _r in tab._runs} == {"legacy"}


def test_ledger_tab_fresh_tab_stays_unfiltered_after_refresh(fake_app, tmp_path):
    """A fresh tab's Suite/Dataset dropdowns keep their empty (None) hint state
    after refresh — they must not snap to the '_ALL' value."""
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    assert tab.suite_dd.value is None
    assert tab.dataset_dd.value is None
    assert len(tab._runs) == 3  # None hint = unfiltered


def test_ledger_tab_date_to_filter(fake_app, tmp_path):
    """The To bound keeps runs on/before that date; From+To make a window."""
    led = EvalLedger(tmp_path / "l.db")
    recent = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    old = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    led.save_run(_report("recent", run_timestamp=recent))
    led.save_run(_report("old", run_timestamp=old))
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    tab.date_to_field.value = (datetime.now() - timedelta(days=30)).strftime("%d.%m.%Y")
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    assert {_r["dataset_name"] for _r in tab._runs} == {"old"}  # recent is after the To bound


def test_ledger_tab_refresh_drops_stale_selection(fake_app, tmp_path):
    """A run that a filter removes must not stay selected (would mis-enable the
    delete buttons and show cases for an off-list run)."""
    led = _seed(tmp_path)
    tab = LedgerTab(fake_app, coordinator=_coord())
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        tab._select_run(3)  # the 'solo' run
        assert tab._selected_run_id == 3
        tab.dataset_dd.value = "vec"  # filter excludes the solo run
        tab.refresh()
    assert tab._selected_run_id is None
    assert tab.delete_run_btn.disabled  # re-synced off
    assert tab.delete_suite_btn.disabled
