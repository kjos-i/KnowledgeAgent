"""Tests for the Evaluation Compare Datasets sub-tab.

The picker now lives in the shared `DashboardRail` (Compare section), keyed off
coordinator state — so these drive the rail's compare controls and check the
tab's body renders the table + grouped bars (catches any Flet API error).
`_ledger` is patched to a tmp DB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import flet as ft

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation.compare_tab import CompareDatasetsTab

_LEDGER = "knowledge_agent.gui.evaluation._common.active_eval_ledger"


def _coord() -> SimpleNamespace:
    return SimpleNamespace(selected_run_id=None, selected_dataset=None, compare_selected=[])


def _report(dataset: str, hit: float, recipe_hash: str = "r1") -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": dataset,
        "recipe_hash": recipe_hash,
        "git_commit": None,
        "prompts_snapshot": {},
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": hit},
        "results": [{"id": "c1", "category": "", "status": "PASS", "errors": [], "hit_at_k": hit}],
    }


def _add(tab: CompareDatasetsTab, dataset: str) -> None:
    """Add a dataset via the rail's Compare picker (as a click would)."""
    tab.rail.compare_dataset_dd.value = dataset
    tab.rail._on_compare_add(MagicMock())


def test_compare_tab_builds(fake_app):
    assert CompareDatasetsTab(fake_app, coordinator=_coord()).build() is not None


def test_compare_renders_table_for_two_datasets(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha.json", 0.5))
    led.save_run(_report("beta.json", 1.0))
    coord = _coord()
    tab = CompareDatasetsTab(fake_app, coordinator=coord)
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        _add(tab, "alpha")
        _add(tab, "beta")
    assert len(coord.compare_selected) == 2
    # Table + divider + bars — not the single italic guard message.
    assert len(tab.body.controls) >= 2
    assert not isinstance(tab.body.controls[0], ft.Text)


def test_compare_needs_two_datasets(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha.json", 0.5))
    coord = _coord()
    tab = CompareDatasetsTab(fake_app, coordinator=coord)
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        _add(tab, "alpha")
    assert "at least 2" in tab.body.controls[0].value


def test_compare_offers_all_corpus_datasets(fake_app, tmp_path):
    """The picker offers every dataset in the corpus (no fact/knob scoping) and
    adding two of them selects both."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha.json", 0.5))
    led.save_run(_report("beta.json", 0.5))
    coord = _coord()
    tab = CompareDatasetsTab(fake_app, coordinator=coord)
    tab.build()
    with patch(_LEDGER, return_value=led):
        tab.refresh()
        assert sorted(o.key for o in tab.rail.compare_dataset_dd.options) == ["alpha", "beta"]
        _add(tab, "alpha")
        _add(tab, "beta")
    assert {s["dataset"] for s in coord.compare_selected} == {"alpha", "beta"}
