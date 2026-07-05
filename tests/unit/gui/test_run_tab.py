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


def test_judge_panel_hidden_until_judge_group_on(fake_app: MagicMock):
    tab, _ = _run_tab(fake_app)
    assert tab.judge_panel.visible is False  # judge off by default
    tab.group_checks["judge"].value = True
    tab._on_judge_toggle(MagicMock())
    assert tab.judge_panel.visible is True
    assert tab.add_judge_button.visible is True


def test_add_and_remove_judge_rows(fake_app: MagicMock):
    tab, _ = _run_tab(fake_app)
    tab._on_add_judge_clicked(MagicMock())
    tab._on_add_judge_clicked(MagicMock())
    assert len(tab.judge_dropdowns) == 2
    assert len(tab.judge_panel.controls) == 2
    tab._remove_judge_row(tab.judge_dropdowns[0], tab.judge_panel.controls[0])
    assert len(tab.judge_dropdowns) == 1
    assert len(tab.judge_panel.controls) == 1


def test_build_config_maps_form(fake_app: MagicMock):
    tab, _ = _run_tab(fake_app)
    tab.group_checks["chunk"].value = False
    tab.group_checks["kg"].value = False
    tab.group_checks["judge"].value = True
    tab._add_judge_row()
    tab.judge_dropdowns[0].value = "claude-haiku-4-5-20251001"
    tab.max_cases_field.value = "3"

    cfg = tab._build_config()
    assert cfg.enabled_groups == frozenset({"source", "judge"})
    assert cfg.judge_models == ("claude-haiku-4-5-20251001",)
    assert cfg.max_cases == 3
    assert cfg.dataset_path.name == "escrt_bootstrap.json"


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
    for cb in tab.group_checks.values():
        cb.value = False
    # Force the loop guard open so the *group* validation is what stops it.
    with (
        patch.object(tab, "_loop_running", return_value=True),
        patch.object(tab, "_spawn") as spawn,
    ):
        tab._on_run_clicked(MagicMock())
    spawn.assert_not_called()  # validation stops the run
