"""Tests for the Evaluation Deep Analysis sub-tab.

Seeds a run with varying per-case metrics and refreshes — exercising the
bar-of-means, histogram (+ metric switch), and the stdlib-correlation grid.
`_ledger` is patched to a tmp DB.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation.deep_analysis_tab import DeepAnalysisTab


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


def test_deep_analysis_builds(fake_app):
    assert DeepAnalysisTab(fake_app, coordinator=MagicMock()).build() is not None


def test_refresh_empty_ledger(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    tab = DeepAnalysisTab(fake_app, coordinator=MagicMock())
    tab.build()
    with patch.object(tab, "_ledger", return_value=led):
        tab.refresh()
    assert "No evaluation runs" in tab.body.controls[0].value


def test_refresh_renders_analysis_and_histogram_switch(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report_multi())
    coordinator = MagicMock()
    coordinator.selected_run_id = None
    tab = DeepAnalysisTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch.object(tab, "_ledger", return_value=led):
        tab.refresh()
    assert coordinator.selected_run_id == 1
    assert len(tab.body.controls) > 1  # header + balance + distribution + correlation
    # switching the histogram metric rebuilds its container without error
    tab.hist_dropdown.value = "mrr"
    tab._on_hist_metric_change(MagicMock())
    assert tab.hist_container.content is not None
