"""Tests for the Evaluation Compare Datasets sub-tab.

Compare shows the members of ONE suite execution — every run sharing the
selected run's `suite_run_id`. These seed a tmp ledger with a suite-run (two
runs sharing a suite_run_id), point the coordinator at a member run, and check
the body renders the table + grouped bars (catches any Flet API error).
`_ledger` is patched to a tmp DB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import flet as ft

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation._common import ALL_ORIGINS
from knowledge_agent.gui.evaluation.compare_tab import CompareDatasetsTab

_LEDGER = "knowledge_agent.gui.evaluation._common.active_eval_ledger"


def _coord() -> SimpleNamespace:
    return SimpleNamespace(
        selected_suite=None,
        selected_dataset=None,
        selected_run_id=None,
        selected_origins=set(ALL_ORIGINS),
    )


def _report(dataset: str, hit: float, *, suite_run_id: str | None = None) -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": f"{dataset}.json",
        "dataset_name": dataset,
        "facts_hash": "fh",
        "suite_run_id": suite_run_id,
        "recipe_hash": "r1",
        "git_commit": None,
        "prompts_snapshot": {},
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": hit},
        "results": [{"id": "c1", "category": "", "status": "PASS", "errors": [], "hit_at_k": hit}],
    }


def test_compare_tab_builds(fake_app):
    assert CompareDatasetsTab(fake_app, coordinator=_coord()).build() is not None


def test_compare_renders_members_of_suite_run(fake_app, tmp_path):
    """Pointing at a member run renders the whole suite-run (its members share a
    suite_run_id) as a table + grouped bars."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("vec", 0.5, suite_run_id="s1"))  # run 1
    led.save_run(_report("graph", 1.0, suite_run_id="s1"))  # run 2 — same suite-run
    coord = _coord()
    coord.selected_run_id = 1  # a member run
    tab = CompareDatasetsTab(fake_app, coordinator=coord)
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    # Table + divider + bars — not the single italic guard message.
    assert len(tab.body.controls) >= 2
    assert not isinstance(tab.body.controls[0], ft.Text)


def test_compare_needs_a_suite_run(fake_app, tmp_path):
    """A single-file run (no suite_run_id) can't be compared — shows guidance."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("solo", 0.5))  # suite_run_id=None
    coord = _coord()
    coord.selected_run_id = 1
    tab = CompareDatasetsTab(fake_app, coordinator=coord)
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
    assert "suite execution" in tab.body.controls[0].value
