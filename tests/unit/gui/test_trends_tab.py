"""Tests for the Evaluation Trends sub-tab.

Seeds 2+ runs sharing one recipe hash and refreshes — this exercises the
flet.canvas line-chart construction (catches any canvas API error) + the
recipe-hash scoping + the <2-runs guard. `_ledger` is patched to a tmp DB.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import flet as ft

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation.trends_tab import TrendsTab


def _report(dataset: str, hit: float, recipe_hash: str = "recipe-abc") -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": dataset,
        "git_commit": None,
        "prompts_snapshot": {},
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "recipe_hash": recipe_hash,
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": hit},
        "results": [
            {"id": "c1", "category": "", "status": "PASS", "errors": [], "hit_at_k": hit},
        ],
    }


def test_trends_builds(fake_app):
    assert TrendsTab(fake_app, coordinator=MagicMock()).build() is not None


def test_trends_needs_two_runs(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("escrt_bootstrap.json", 1.0))
    coordinator = MagicMock()
    coordinator.selected_run_id = None
    tab = TrendsTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert "Need at least 2 runs" in tab.body.controls[0].value


def test_trends_renders_charts_for_two_runs(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("escrt_bootstrap.json", 0.5))
    led.save_run(_report("escrt_bootstrap.json", 1.0))
    coordinator = MagicMock()
    coordinator.selected_run_id = None
    tab = TrendsTab(fake_app, coordinator=coordinator)
    tab.build()
    with patch("knowledge_agent.gui.evaluation._common.active_eval_ledger", return_value=led):
        tab.refresh()
    assert tab.rail.dataset_dd.options  # dataset filter populated
    # Charts rendered (not a guard message): the summary chart is a Column.
    assert isinstance(tab.body.controls[0], ft.Column)
