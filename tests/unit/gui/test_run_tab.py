"""Tests for the Evaluation Run sub-tab — form → EvalConfig, judge panel,
and the async run wiring. `runner.run` is mocked, so no graph / LLM / DB is
touched; the form controls are built statically first.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.evaluation.run_tab import RunTab


def _run_tab(fake_app: MagicMock) -> tuple[RunTab, MagicMock]:
    fake_app.gui_config.llm_provider = "anthropic"
    fake_app.gui_config.active_corpus_name = "test_corpus"
    fake_app.gui_config.corpus_config_path = None
    coordinator = MagicMock()
    tab = RunTab(fake_app, coordinator=coordinator)
    tab.build()
    return tab, coordinator


def test_run_tab_builds(fake_app: MagicMock):
    assert RunTab(fake_app, coordinator=MagicMock()).build() is not None


def test_run_tab_shows_output_path(fake_app: MagicMock):
    """Read-only 'Results save to:' line renders (selectable, not an input).
    Building it must resolve the output dir without erroring."""
    tab, _ = _run_tab(fake_app)
    assert tab.output_line is not None
    assert tab.output_line.value.startswith("Results save to:")
    assert tab.output_line.selectable is True


def test_refresh_active_corpus_updates_output_echo(fake_app: MagicMock):
    """An app-wide corpus switch refreshes the read-only 'Results save to'
    output-path echo without a rebuild. (The corpus-name echo itself was
    dropped — it lives in the main-window header now.)"""
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.active_corpus_name = "other_corpus"
    tab.refresh_active_corpus()
    assert tab.output_line.value.startswith("Results save to:")


def test_refresh_active_corpus_none_does_not_crash(fake_app: MagicMock):
    """With no active corpus, the output echo still resolves (falls back to the
    CWD eval_output) rather than erroring."""
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.active_corpus_name = None
    tab.refresh_active_corpus()
    assert tab.output_line.value.startswith("Results save to:")


def test_pick_dataset_loads_recipe_frozen(fake_app: MagicMock, tmp_path):
    """Choosing a dataset loads its saved recipe into the (shared) recipe form
    and freezes it; Unfreeze re-enables the controls for a one-off deviation
    (the dataset file is never touched). The judge-panel + group-check widgets
    themselves are covered in test_recipe_form.py — they moved to RecipeForm."""
    from knowledge_agent.evaluation.models import (
        EvalCase,
        EvalDataset,
        EvalRecipe,
        save_dataset,
    )

    p = tmp_path / "gold.json"
    save_dataset(
        EvalDataset(
            recipe=EvalRecipe(dataset_kind="knob", judge_threshold=0.0),
            cases=[EvalCase(id="c", question="q?")],
        ),
        p,
    )
    tab, _ = _run_tab(fake_app)
    tab._load_recipe_from(p)
    rf = tab.recipe_form
    assert rf.profile_group.value == "knob"  # loaded from the saved recipe
    assert rf.threshold_fields["judge_threshold"].value == "0"
    assert rf._wrapper.disabled is True  # frozen on load
    assert tab.unfreeze_button.visible is True
    tab._on_unfreeze_clicked(MagicMock())
    assert rf._wrapper.disabled is False  # editable for the one-off
    assert tab.deviation_note.visible is True


def test_build_config_maps_recipe_and_thresholds(fake_app: MagicMock):
    """`_build_config` reads the recipe form — metric groups, judge panel, AND
    the three gate thresholds all flow into the EvalConfig (C2c)."""
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = "/tmp/eval/my_gold.json"  # user Browsed to a dataset
    rf = tab.recipe_form
    rf.group_checks["chunk"].value = False
    rf.group_checks["kg"].value = False
    rf.group_checks["judge"].value = True
    rf.judge_dropdown.value = "claude-haiku-4-5-20251001"
    rf._on_add_judge_clicked(MagicMock())
    rf.threshold_fields["judge_threshold"].value = "0.9"
    tab.max_cases_field.value = "3"

    cfg = tab._build_config()
    assert cfg.enabled_groups == frozenset({"source", "judge"})
    assert cfg.judge_models == ("claude-haiku-4-5-20251001",)
    assert cfg.judge_threshold == 0.9  # gate threshold flowed through
    assert cfg.max_cases == 3
    assert cfg.dataset_path.name == "my_gold.json"


def test_execute_run_invokes_runner_and_notifies_coordinator(fake_app: MagicMock):
    tab, coordinator = _run_tab(fake_app)
    fake_result = MagicMock(run_id=5, report={"summary": {"pass_count": 2, "case_count": 3}})
    with patch(
        "knowledge_agent.evaluation.runner.run",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as run_mock:
        asyncio.run(tab._execute_run())
    run_mock.assert_awaited_once()
    assert "on_progress" in run_mock.await_args.kwargs  # progress hook threaded through
    coordinator.on_run_complete.assert_called_once_with(5)


def test_run_click_blocks_when_no_group_selected(fake_app: MagicMock):
    tab, _ = _run_tab(fake_app)
    for cb in tab.recipe_form.group_checks.values():
        cb.value = False
    # Force the loop guard open so the *group* validation is what stops it.
    with (
        patch.object(tab, "_loop_running", return_value=True),
        patch.object(tab, "_spawn") as spawn,
    ):
        tab._on_run_clicked(MagicMock())
    spawn.assert_not_called()  # validation stops the run


def test_trace_toggle_reveals_warning_and_enables_project(fake_app: MagicMock):
    """The data-safety warning is hidden until the user opts into tracing — so
    it surfaces exactly at opt-in. The project field stays visible throughout,
    just disabled (greyed) until tracing is ticked."""
    tab, _ = _run_tab(fake_app)
    assert tab.trace_check.value is False  # off by default
    assert tab.trace_warning.visible is False
    assert tab._project_row.disabled is True
    tab.trace_check.value = True
    tab._on_trace_toggle(MagicMock())
    assert tab.trace_warning.visible is True
    assert tab._project_row.disabled is False


def test_trace_toggle_warns_when_no_langsmith_key(fake_app: MagicMock):
    """Ticking tracing with no LangSmith key set surfaces the hint right away
    — not only when the run is blocked later."""
    tab, _ = _run_tab(fake_app)
    tab.trace_check.value = True
    with patch("knowledge_agent.gui.evaluation.run_tab.get_api_key", return_value=None):
        tab._on_trace_toggle(MagicMock())
    assert tab.trace_key_hint.visible is True


def test_trace_toggle_no_hint_when_key_set(fake_app: MagicMock):
    """With a key set, ticking tracing shows no missing-key hint."""
    tab, _ = _run_tab(fake_app)
    tab.trace_check.value = True
    with patch("knowledge_agent.gui.evaluation.run_tab.get_api_key", return_value="sk-x"):
        tab._on_trace_toggle(MagicMock())
    assert tab.trace_key_hint.visible is False


def test_execute_run_passes_trace_and_project(fake_app: MagicMock):
    tab, _ = _run_tab(fake_app)
    tab.trace_check.value = True
    tab.project_field.value = "proj-x"
    fake_result = MagicMock(run_id=5, report={"summary": {"pass_count": 1, "case_count": 1}})
    with patch(
        "knowledge_agent.evaluation.runner.run",
        new_callable=AsyncMock,
        return_value=fake_result,
    ) as run_mock:
        asyncio.run(tab._execute_run())
    kwargs = run_mock.await_args.kwargs
    assert kwargs["trace"] is True
    assert kwargs["langsmith_project"] == "proj-x"


def test_run_click_blocks_when_trace_without_key(fake_app: MagicMock):
    """Checking Trace with no LangSmith key stored stops the run with a
    pointer to Settings → Keys — no silent no-op trace."""
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = "/tmp/gold.json"  # get past the "select a dataset" guard
    tab.trace_check.value = True
    with (
        patch.object(tab, "_loop_running", return_value=True),
        patch("knowledge_agent.gui.evaluation.run_tab.get_api_key", return_value=None),
        patch.object(tab, "_spawn") as spawn,
    ):
        tab._on_run_clicked(MagicMock())
    spawn.assert_not_called()
    assert "LangSmith API key" in tab.status.value
