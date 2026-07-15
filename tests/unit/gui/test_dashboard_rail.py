"""Tests for the shared dashboard rail (Suite→Dataset→Run cascade + read-only
run context). Seeds a tmp ledger and patches `active_eval_ledger`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail

_LEDGER = "knowledge_agent.gui.evaluation._common.active_eval_ledger"


def _report(dataset: str, run_ts: str, facts: str = "sharedfacts") -> dict:
    return {
        "run_timestamp": run_ts,
        "dataset_path": f"{dataset}.json",
        "dataset_name": dataset,
        "facts_hash": facts,  # the suite key (same facts across a suite's members)
        "prompts_snapshot": {},
        "enabled_groups": ["source", "chunk"],
        "gate_thresholds": {"judge_threshold": 0.5},
        "judge_models": ["m1"],
        "recipe_hash": "abcdef1234567890",  # pragma: allowlist secret (fake test hash)
        "llm_provider": "anthropic",
        "synthesizer_model": "claude-sonnet-5",
        "summary": {"case_count": 3, "pass_count": 2, "pass_rate": 0.66},
        "results": [{"id": "c1", "category": "", "status": "PASS", "errors": []}],
    }


def _rail(fake_app, led, **coord_kw):
    state = {
        "selected_suite": None,
        "selected_run_id": None,
        "selected_dataset": None,
        **coord_kw,
    }
    coord = SimpleNamespace(**state)
    on_change = MagicMock()
    rail = DashboardRail(fake_app, coord, on_change=on_change)
    rail.build()
    with patch(_LEDGER, return_value=led):
        rail.refresh()
    return rail, coord, on_change


def test_rail_populates_and_autoselects_newest(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha", "2026-07-01T09:00:00"))
    led.save_run(_report("alpha", "2026-07-02T09:00:00"))
    rail, coord, _ = _rail(fake_app, led)
    assert [o.key for o in rail.dataset_dd.options] == ["alpha"]
    assert coord.selected_dataset == "alpha"
    assert coord.selected_run_id == 2  # newest auto-selected
    assert rail.run_dd.value == "2"
    assert len(rail.run_dd.options) == 2


def test_context_shows_recipe_from_run_row(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha", "2026-07-01T09:00:00"))
    rail, _, _ = _rail(fake_app, led)
    lines = [c.value for c in rail.context.controls if hasattr(c, "value")]
    joined = " | ".join(lines)
    assert "Recipe hash: abcdef12" in lines  # 8-char prefix, truncated from the full 16
    assert "chunk" in joined and "source" in joined  # enabled groups (ledger sorts them)
    assert "m1" in joined  # judge panel
    assert "claude-sonnet-5" in joined  # model


def test_dataset_change_narrows_runs(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha", "2026-07-01T09:00:00"))  # run 1
    led.save_run(_report("beta", "2026-07-02T09:00:00"))  # run 2 (newest)
    rail, coord, on_change = _rail(fake_app, led)
    assert coord.selected_dataset == "beta"  # newest run's dataset
    rail.dataset_dd.value = "alpha"
    with patch(_LEDGER, return_value=led):
        rail._on_dataset_change(MagicMock())
    assert coord.selected_dataset == "alpha"
    assert {o.key for o in rail.run_dd.options} == {"1"}  # only alpha's run
    assert on_change.called


def test_selected_run_wins_dataset(fake_app, tmp_path):
    """A pre-selected run (e.g. just finished) picks its own dataset even if the
    filter pointed elsewhere — so a fresh run always shows."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha", "2026-07-01T09:00:00"))  # run 1
    led.save_run(_report("beta", "2026-07-02T09:00:00"))  # run 2
    _, coord, _ = _rail(fake_app, led, selected_run_id=1, selected_dataset="beta")
    assert coord.selected_dataset == "alpha"  # run 1's dataset wins
    assert coord.selected_run_id == 1


def test_dropdowns_wired_to_on_select(fake_app, tmp_path):
    """Flet 0.85 Dropdowns fire on_select (NOT on_change) — the handlers must be
    on `on_select` or a picked run/dataset is silently lost and refresh snaps
    back to the newest run. (Direct-call tests can't catch this; assert wiring.)"""
    led = EvalLedger(tmp_path / "l.db")
    rail, _, _ = _rail(fake_app, led)
    # == (not is): a bound method is a fresh wrapper each access, but compares
    # equal by (__self__, __func__).
    assert rail.run_dd.on_select == rail._on_run_change
    assert rail.dataset_dd.on_select == rail._on_dataset_change
    assert rail.suite_dd.on_select == rail._on_suite_change


def test_suite_scopes_datasets(fake_app, tmp_path):
    """The Suite dropdown groups runs by facts_hash; selecting a suite scopes the
    Dataset list to that suite's members."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("vec", "2026-07-01T09:00:00", facts="factsA"))
    led.save_run(_report("graph", "2026-07-02T09:00:00", facts="factsA"))
    led.save_run(_report("other", "2026-07-03T09:00:00", facts="factsB"))  # newest
    rail, coord, _ = _rail(fake_app, led)
    # newest run (other / factsB) auto-selected → suite factsB, its one member
    assert coord.selected_suite == "factsB"
    assert {o.key for o in rail.dataset_dd.options} == {"other"}
    assert {o.key for o in rail.suite_dd.options} == {"factsA", "factsB"}
    # switch to suite factsA → its two members appear
    rail.suite_dd.value = "factsA"
    with patch(_LEDGER, return_value=led):
        rail._on_suite_change(MagicMock())
    assert coord.selected_suite == "factsA"
    assert {o.key for o in rail.dataset_dd.options} == {"vec", "graph"}


def test_named_suite_groups_by_name(fake_app, tmp_path):
    """Runs stamped with a `suite` name group under that name (not facts_hash),
    and the dropdown is labelled by the name."""
    led = EvalLedger(tmp_path / "l.db")
    r1 = _report("vec", "2026-07-01T09:00:00")
    r1["suite"] = "mode-sweep"
    r2 = _report("graph", "2026-07-02T09:00:00")
    r2["suite"] = "mode-sweep"
    led.save_run(r1)
    led.save_run(r2)
    rail, coord, _ = _rail(fake_app, led)
    assert coord.selected_suite == "mode-sweep"  # keyed by the suite NAME
    assert [(o.key, o.text) for o in rail.suite_dd.options] == [("mode-sweep", "mode-sweep")]
    assert {o.key for o in rail.dataset_dd.options} == {"vec", "graph"}


def test_selected_run_survives_refresh(fake_app, tmp_path):
    """Picking an older run then refreshing keeps it — refresh must not snap to
    the newest run (the reported bug)."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report("alpha", "2026-07-01T09:00:00"))  # run 1
    led.save_run(_report("alpha", "2026-07-02T09:00:00"))  # run 2
    led.save_run(_report("alpha", "2026-07-03T09:00:00"))  # run 3 (newest)
    rail, coord, _ = _rail(fake_app, led)
    assert coord.selected_run_id == 3  # newest by default
    # user picks run 2 via the dropdown (its on_select handler)
    rail.run_dd.value = "2"
    with patch(_LEDGER, return_value=led):
        rail._on_run_change(MagicMock())
    assert coord.selected_run_id == 2
    # refresh (Refresh button / tab-select) must KEEP run 2, not revert to 3
    with patch(_LEDGER, return_value=led):
        rail.refresh()
    assert coord.selected_run_id == 2
    assert rail.run_dd.value == "2"


def test_empty_ledger_no_selection(fake_app, tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    rail, coord, _ = _rail(fake_app, led)
    assert rail.dataset_dd.options == []
    assert coord.selected_run_id is None
    assert "No run selected" in rail.context.controls[0].value
