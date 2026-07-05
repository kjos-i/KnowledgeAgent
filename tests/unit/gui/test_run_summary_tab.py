"""Tests for the Evaluation Run Summary sub-tab.

`refresh()` renders a run's KPIs + per-case table from a seeded ledger;
an empty ledger shows the empty state. `_ledger` is patched to a tmp DB so
no real eval_output is touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation.run_summary_tab import RunSummaryTab


def _report() -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": "escrt_bootstrap.json",
        "git_commit": "abc12345",
        "prompts_snapshot": {},
        "enabled_groups": ["source", "chunk"],
        "gate_thresholds": {"required_keyword_threshold": 0.5},
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": 0.5},
        "results": [
            {"id": "c1", "category": "Factual", "status": "PASS", "errors": [], "hit_at_k": 1.0},
            {"id": "c2", "category": "", "status": "REVIEW", "errors": [], "hit_at_k": 0.0},
        ],
    }


def test_run_summary_builds(fake_app: MagicMock):
    assert RunSummaryTab(fake_app, coordinator=MagicMock()).build() is not None


def test_refresh_empty_ledger_shows_empty_state(fake_app: MagicMock, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    tab = RunSummaryTab(fake_app, coordinator=MagicMock())
    tab.build()
    with patch.object(tab, "_ledger", return_value=led):
        tab.refresh()
    assert "No evaluation runs" in tab.body.controls[0].value


def test_refresh_renders_selected_run(fake_app: MagicMock, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())
    coordinator = MagicMock()
    coordinator.selected_run_id = None
    tab = RunSummaryTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch.object(tab, "_ledger", return_value=led):
        tab.refresh()
    assert coordinator.selected_run_id == 1  # newest auto-selected
    assert tab.run_dropdown.options  # run listed in the selector
    assert len(tab.body.controls) > 1  # headline + KPI sections + table (not the empty state)
