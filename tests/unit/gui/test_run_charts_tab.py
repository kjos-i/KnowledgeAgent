"""Tests for the Evaluation Run Charts sub-tab.

Seeds a run with varying per-case metrics and refreshes — exercising the
bar-of-means, histogram (+ metric switch), and the stdlib-correlation grid.
`_ledger` is patched to a tmp DB.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation._common import ALL_ORIGINS
from knowledge_agent.gui.evaluation.run_charts_tab import RunChartsTab


def _report_multi() -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": "d.json",
        "git_commit": None,
        "prompts_snapshot": {},
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "summary": {
            "case_count": 3,
            "pass_count": 2,
            "pass_rate": 0.67,
            "avg_hit_at_k": 0.67,
            "avg_mrr": 0.5,
        },
        "results": [
            {
                "id": "c1",
                "category": "",
                "status": "PASS",
                "errors": [],
                "hit_at_k": 1.0,
                "mrr": 1.0,
            },
            {
                "id": "c2",
                "category": "",
                "status": "REVIEW",
                "errors": [],
                "hit_at_k": 0.0,
                "mrr": 0.0,
            },
            {
                "id": "c3",
                "category": "",
                "status": "PASS",
                "errors": [],
                "hit_at_k": 1.0,
                "mrr": 0.5,
            },
        ],
    }


def test_run_charts_builds(fake_app):
    assert (
        RunChartsTab(
            fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
        ).build()
        is not None
    )


def test_refresh_empty_ledger(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    tab = RunChartsTab(
        fake_app, coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert "No evaluation runs" in tab.body.controls[0].value


def test_refresh_renders_analysis_and_histogram_switch(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report_multi())
    coordinator = MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    coordinator.selected_run_id = None
    tab = RunChartsTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert coordinator.selected_run_id == 1
    assert len(tab.body.controls) > 1  # header + balance + distribution + correlation
    # switching the histogram metric rebuilds its container without error
    tab.hist_dropdown.value = "mrr"
    tab._on_hist_metric_change(MagicMock())
    assert tab.hist_container.content is not None


def test_n_present_counts_cases_with_a_value():
    """`_n_present` — how many cases fed a metric's mean (a None case is dropped);
    shown as (n=X) on each Metric Balance bar."""
    tab = RunChartsTab(
        MagicMock(), coordinator=MagicMock(selected_suite=None, selected_origins=set(ALL_ORIGINS))
    )
    tab._cases = [{"mrr": 1.0}, {"mrr": None}, {"mrr": 0.5}]
    assert tab._n_present("mrr") == 2
    assert tab._n_present("hit_at_k") == 0  # no case has this metric
