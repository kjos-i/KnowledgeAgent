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


def _saved(tmp_path, name, **kw):
    """Write a one-case dataset with the given header kwargs; return its path."""
    from knowledge_agent.evaluation.models import EvalCase, EvalDataset, save_dataset

    p = tmp_path / name
    save_dataset(EvalDataset(cases=[EvalCase(id="c", question="q?")], **kw), p)
    return p


def test_load_state_editable_when_not_frozen(fake_app: MagicMock, tmp_path):
    """A non-frozen dataset loads with the recipe EDITABLE. Max cases + Tracing
    are editable, the freeze checkbox is enabled, and Unfreeze is present but
    greyed out."""
    from knowledge_agent.evaluation.models import EvalRecipe

    p = _saved(tmp_path, "final.json", status="final", recipe=EvalRecipe(enabled_groups=["source"]))
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = str(p)
    tab._load_dataset_state(p)
    assert tab.recipe_form.group_checks["source"].value is True  # recipe loaded
    assert tab.recipe_form._wrapper.disabled is False  # editable (not frozen)
    assert tab.max_cases_field.disabled is False  # editable (not frozen)
    assert tab.trace_check.disabled is False
    assert tab.freeze_check.disabled is False  # final + not frozen → can freeze
    assert tab.freeze_check.value is False
    assert tab.unfreeze_button.disabled is True  # present but greyed until frozen


def test_freeze_checkbox_disabled_for_draft(fake_app: MagicMock, tmp_path):
    """A draft dataset can't be frozen — the checkbox is disabled + hint shows."""
    p = _saved(tmp_path, "draft.json", status="draft")
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = str(p)
    tab._load_dataset_state(p)
    assert tab.freeze_check.disabled is True
    assert tab.freeze_hint.visible is True


def test_frozen_dataset_loads_readonly(fake_app: MagicMock, tmp_path):
    """A frozen (final) dataset locks the whole tab: recipe + Max cases +
    Tracing all disabled, freeze checkbox checked + disabled, Unfreeze enabled,
    badge shown."""
    from knowledge_agent.evaluation.models import EvalRecipe

    p = _saved(tmp_path, "frozen.json", status="final", frozen=True, recipe=EvalRecipe())
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = str(p)
    tab._load_dataset_state(p)
    assert tab.recipe_form._wrapper.disabled is True  # read-only
    assert tab.max_cases_field.disabled is True  # whole tab locks
    assert tab.trace_check.disabled is True
    assert tab._project_row.disabled is True
    assert tab.freeze_check.value is True and tab.freeze_check.disabled is True
    assert tab.unfreeze_button.disabled is False  # active when frozen
    assert tab.frozen_indicator.visible is True


def test_unfreeze_persists_frozen_false(fake_app: MagicMock, tmp_path):
    """_do_unfreeze clears the frozen flag on disk and re-enables the recipe."""
    from knowledge_agent.evaluation.models import EvalRecipe, load_dataset

    p = _saved(tmp_path, "frozen.json", status="final", frozen=True, recipe=EvalRecipe())
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = str(p)
    tab._load_dataset_state(p)
    tab._do_unfreeze()
    assert load_dataset(p).frozen is False  # persisted
    assert tab.recipe_form._wrapper.disabled is False  # editable again
    assert tab.max_cases_field.disabled is False  # whole tab editable again
    assert tab.unfreeze_button.disabled is True  # greyed again (not frozen)


def test_freeze_on_run_persists(fake_app: MagicMock, tmp_path):
    """Ticking 'Freeze run settings' + a successful run persists frozen=true."""
    from knowledge_agent.evaluation.models import EvalRecipe, load_dataset

    p = _saved(tmp_path, "final.json", status="final", recipe=EvalRecipe())
    tab, _ = _run_tab(fake_app)
    tab.dataset_field.value = str(p)
    tab._load_dataset_state(p)
    tab.freeze_check.value = True  # opt-in
    fake_result = MagicMock(run_id=7, report={"summary": {"pass_count": 1, "case_count": 1}})
    with patch(
        "knowledge_agent.evaluation.runner.run",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        asyncio.run(tab._execute_run())
    assert load_dataset(p).frozen is True  # freeze persisted after the run


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


# ---- run scope: single file vs whole suite (step 3) ----


def _knob_variant(path, cid, mode, *, question="q?"):
    """A dataset with one case: `question` + a pinned retrieval mode (a knob
    sweep member). All required knobs pinned so the file is runnable."""
    from knowledge_agent.evaluation.models import (
        EvalCase,
        EvalDataset,
        RetrievalSettings,
        save_dataset,
    )

    save_dataset(
        EvalDataset(
            cases=[
                EvalCase(
                    id=cid,
                    question=question,
                    expected_sources=["d1"],
                    retrieval=RetrievalSettings(
                        retrieval_mode=mode,
                        num_candidates=100,
                        rrf_rank_constant=60,
                        kg_max_rows=50,
                    ),
                )
            ]
        ),
        path,
    )
    return path


def test_facts_siblings_groups_same_facts(fake_app: MagicMock, tmp_path):
    """_facts_siblings (the fallback suite for untagged files) returns the corpus
    files sharing the picked file's facts (same questions+gold, swept knobs) —
    including the picked one — and just [picked] for a lone (unique-facts) file."""
    vec = _knob_variant(tmp_path / "facts_vector.json", "c__vector", "lancedb_only")
    graph = _knob_variant(tmp_path / "facts_graph.json", "c__graph", "neo4j_only")
    _knob_variant(tmp_path / "other.json", "x", "lancedb_only", question="DIFFERENT?")
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"  # dir = tmp_path

    assert {p.name for p in tab._facts_siblings(vec)} == {"facts_vector.json", "facts_graph.json"}
    assert {p.name for p in tab._facts_siblings(graph)} == {"facts_vector.json", "facts_graph.json"}
    assert [p.name for p in tab._facts_siblings(tmp_path / "other.json")] == ["other.json"]


def test_run_scope_reflects_suite_members(fake_app: MagicMock, tmp_path):
    """Loading a dataset with siblings enables 'Whole suite' + names them; a
    lone dataset disables it and snaps the mode back to single."""
    vec = _knob_variant(tmp_path / "facts_vector.json", "c__vector", "lancedb_only")
    _knob_variant(tmp_path / "facts_graph.json", "c__graph", "neo4j_only")
    lone = _knob_variant(tmp_path / "lone.json", "z", "lancedb_only", question="ONLY?")
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"

    def _suite_radio():
        return next(r for r in tab.suite_mode.content.controls if r.value == "suite")

    tab.dataset_field.value = str(vec)
    tab._load_dataset_state(vec)
    assert _suite_radio().disabled is False  # ≥2 members → enabled
    assert "Whole suite = 2" in tab.suite_hint.value

    tab.suite_mode.value = "suite"
    tab.dataset_field.value = str(lone)
    tab._load_dataset_state(lone)
    assert _suite_radio().disabled is True  # lone → disabled
    assert tab.suite_mode.value == "single"  # and snapped back


def test_execute_run_suite_branch_calls_run_suite_and_opens_compare(fake_app: MagicMock):
    """With 'Whole suite' selected + ≥2 members, _execute_run calls run_suite
    with one config per member and hands the outcome to on_suite_complete."""
    from pathlib import Path

    tab, coordinator = _run_tab(fake_app)
    tab._suite_paths = [Path("a.json"), Path("b.json")]
    tab.suite_mode.value = "suite"
    tab._build_config = MagicMock(side_effect=lambda p=None: f"cfg:{p}")
    fake_suite = MagicMock(results=[MagicMock(run_id=1), MagicMock(run_id=2)])
    with patch(
        "knowledge_agent.evaluation.runner.run_suite",
        new_callable=AsyncMock,
        return_value=fake_suite,
    ) as suite_mock:
        asyncio.run(tab._execute_run())
    suite_mock.assert_awaited_once()
    assert len(suite_mock.await_args.args[0]) == 2  # one config per member
    coordinator.on_suite_complete.assert_called_once_with(fake_suite)


def _suite_variant(path, cid, mode, suites, *, question="q?"):
    """A knob-variant dataset tagged into the given named `suites`."""
    from knowledge_agent.evaluation.models import (
        EvalCase,
        EvalDataset,
        RetrievalSettings,
        save_dataset,
    )

    save_dataset(
        EvalDataset(
            suites=suites,
            cases=[
                EvalCase(
                    id=cid,
                    question=question,
                    expected_sources=["d1"],
                    retrieval=RetrievalSettings(
                        retrieval_mode=mode,
                        num_candidates=100,
                        rrf_rank_constant=60,
                        kg_max_rows=50,
                    ),
                )
            ],
        ),
        path,
    )
    return path


def test_whole_suite_uses_explicit_tag(fake_app: MagicMock, tmp_path):
    """A file tagged into a named suite runs THAT suite's tagged members, and the
    suite name flows into run_suite."""
    v = _suite_variant(tmp_path / "vec.json", "c__v", "lancedb_only", ["mode-sweep"])
    _suite_variant(tmp_path / "graph.json", "c__g", "neo4j_only", ["mode-sweep"])
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"
    tab.dataset_field.value = str(v)
    tab._load_dataset_state(v)
    assert tab._suite_name == "mode-sweep"
    assert {p.stem for p in tab._suite_paths} == {"vec", "graph"}

    tab.suite_mode.value = "suite"
    tab._build_config = MagicMock(side_effect=lambda p=None: f"cfg:{p}")
    with patch(
        "knowledge_agent.evaluation.runner.run_suite",
        new_callable=AsyncMock,
        return_value=MagicMock(results=[MagicMock(run_id=1)]),
    ) as suite_mock:
        asyncio.run(tab._execute_run())
    assert suite_mock.await_args.kwargs["suite"] == "mode-sweep"  # named suite threaded through


def test_multi_suite_picker_rescopes_members(fake_app: MagicMock, tmp_path):
    """A file in several suites shows the Suite picker; switching it re-scopes the
    'whole suite' members to that tag."""
    m = _suite_variant(tmp_path / "m.json", "c__m", "lancedb_only", ["sweepA", "sweepB"])
    _suite_variant(tmp_path / "a2.json", "c__a", "neo4j_only", ["sweepA"])
    _suite_variant(tmp_path / "b2.json", "c__b", "parallel_fused", ["sweepB"])
    tab, _ = _run_tab(fake_app)
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"
    tab.dataset_field.value = str(m)
    tab._load_dataset_state(m)
    assert tab.suite_select.visible is True
    assert {o.key for o in tab.suite_select.options} == {"sweepA", "sweepB"}
    assert tab._suite_name == "sweepA"  # first tag by default
    assert {p.stem for p in tab._suite_paths} == {"m", "a2"}
    # switch to sweepB → m + b2
    tab.suite_select.value = "sweepB"
    tab._on_suite_select_change(MagicMock())
    assert tab._suite_name == "sweepB"
    assert {p.stem for p in tab._suite_paths} == {"m", "b2"}


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
