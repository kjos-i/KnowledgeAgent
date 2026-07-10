"""Tests for the LLM-judge track (Phase 3).

deepeval is a base dependency now, but these tests never touch it: the
deepeval import is lazy (only inside `_build_metrics`, which these tests
don't call), and the panel aggregation is tested by stubbing the
per-model scorer — so no real judge LLM (and no deepeval) is invoked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from knowledge_agent.evaluation import judge as J
from knowledge_agent.evaluation import registry as R
from knowledge_agent.evaluation.models import EvalCase


def test_judge_metric_keys_match_registry_llm_group():
    assert list(J.JUDGE_METRIC_KEYS) == R.keys_in_group("llm")


def test_resolve_judge_models_uses_the_user_list():
    assert J.resolve_judge_models(["a", "b"]) == ["a", "b"]


def test_resolve_judge_models_defaults_to_provider_model(monkeypatch):
    monkeypatch.setattr(
        "knowledge_agent.config.get_settings",
        lambda: SimpleNamespace(mode_classifier_model="claude-haiku"),
    )
    assert J.resolve_judge_models(()) == ["claude-haiku"]


def test_build_judge_input_includes_chunks_and_kg_rows():
    run = SimpleNamespace(
        answer="Oslo is the capital.",
        retrieved_texts=["a chunk about Oslo"],
        kg_hits=[{"name": "Oslo"}],
    )
    case = EvalCase(id="c", question="capital?", expected_answer_points=["Oslo"])
    data = J.build_judge_input(run, case)
    assert data["input"] == "capital?"
    assert data["actual_output"] == "Oslo is the capital."
    assert data["expected_output"] == "Oslo"
    assert "a chunk about Oslo" in data["retrieval_context"]
    assert any("Oslo" in ctx for ctx in data["retrieval_context"])  # kg row stringified in


def test_run_judge_panel_aggregates_mean_and_sums_tokens(monkeypatch):
    async def fake_score(model, data, threshold):
        if model == "m1":
            return {k: 1.0 for k in J.JUDGE_METRIC_KEYS}, 100, 40
        return {k: 0.0 for k in J.JUDGE_METRIC_KEYS}, 50, 20

    monkeypatch.setattr(J, "_score_one_model", fake_score)
    scores, in_tok, out_tok = asyncio.run(J.run_judge_panel({}, ["m1", "m2"], 0.5))
    assert scores["faithfulness"] == 0.5  # mean(1.0, 0.0) across the panel
    assert scores["correctness_g_eval"] == 0.5
    assert in_tok == 150
    assert out_tok == 60


def test_run_judge_panel_none_safe_when_a_model_scores_none(monkeypatch):
    async def fake_score(model, data, threshold):
        if model == "m1":
            return {k: 0.8 for k in J.JUDGE_METRIC_KEYS}, 10, 5
        return dict.fromkeys(J.JUDGE_METRIC_KEYS, None), 10, 5  # this judge failed

    monkeypatch.setattr(J, "_score_one_model", fake_score)
    scores, _, _ = asyncio.run(J.run_judge_panel({}, ["m1", "m2"], 0.5))
    assert scores["faithfulness"] == 0.8  # None skipped → mean of the one real score
