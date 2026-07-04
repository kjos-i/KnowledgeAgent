"""Tests for the engine — per-case metric assembly, pass/review gating,
and concurrent fan-out. The adapter (`run_case`) is stubbed so no graph,
LLM, or DB is touched.
"""

from __future__ import annotations

import asyncio

from knowledge_agent.evaluation import engine as E
from knowledge_agent.evaluation.adapter import CaseRun
from knowledge_agent.evaluation.config import EvalConfig
from knowledge_agent.evaluation.models import EvalCase


def _run(**kw) -> CaseRun:
    base = {"question": "q?", "answer": ""}
    base.update(kw)
    return CaseRun(**base)


def _patch_run(monkeypatch, run: CaseRun) -> None:
    async def fake_run_case(case, corpus_config):  # matches engine's call
        return run

    monkeypatch.setattr(E, "run_case", fake_run_case)


def test_evaluate_case_pass(monkeypatch):
    case = EvalCase(
        id="c1",
        question="capital of Norway?",
        expected_sources=["d1"],
        required_keywords=["Oslo"],
    )
    _patch_run(monkeypatch, _run(answer="The capital is Oslo.", retrieved_doc_ids=["d1", "d2"]))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["id"] == "c1"
    assert result["hit_at_k"] == 1.0
    assert result["required_keyword_hit_rate"] == 1.0
    assert result["status"] == "PASS"


def test_evaluate_case_review_on_retrieval_miss(monkeypatch):
    case = EvalCase(id="c2", question="q?", expected_sources=["d1"])
    _patch_run(monkeypatch, _run(answer="x", retrieved_doc_ids=["dX"]))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["hit_at_k"] == 0.0
    assert result["status"] == "REVIEW"


def test_evaluate_case_review_on_disallowed_keyword(monkeypatch):
    case = EvalCase(id="c3", question="q?", disallowed_keywords=["lorem"])
    _patch_run(monkeypatch, _run(answer="contains lorem ipsum"))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["disallowed_keyword_hits"] == 1
    assert result["status"] == "REVIEW"


def test_evaluate_case_review_on_error(monkeypatch):
    case = EvalCase(id="c4", question="q?")
    _patch_run(monkeypatch, _run(answer="", error="RuntimeError: boom"))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["status"] == "REVIEW"
    assert result["errors"] == ["RuntimeError: boom"]


def test_disabled_group_yields_none_columns(monkeypatch):
    case = EvalCase(id="c5", question="q?", expected_sources=["d1"], expected_chunks=["snippet"])
    _patch_run(monkeypatch, _run(retrieved_doc_ids=["d1"], retrieved_texts=["snippet here"]))
    cfg = EvalConfig(enabled_groups=frozenset({"source"}))  # chunk group OFF
    result = asyncio.run(E.evaluate_case(case, None, cfg))
    assert result["hit_at_k"] == 1.0  # source ran
    assert result["chunk_hit_at_k"] is None  # chunk disabled → NULL column


def test_evaluate_cases_runs_all_in_order(monkeypatch):
    cases = [EvalCase(id=f"c{i}", question="q?") for i in range(4)]
    _patch_run(monkeypatch, _run(answer="a"))
    results = asyncio.run(E.evaluate_cases(cases, None, EvalConfig(concurrency=2)))
    assert [r["id"] for r in results] == ["c0", "c1", "c2", "c3"]


# ---- KG group (Phase 2) ----


def test_kg_metrics_computed_for_neo4j_case(monkeypatch):
    case = EvalCase(
        id="k1",
        question="q?",
        retrieval={"retrieval_mode": "neo4j_only"},
        expected_entities=["ESCRT-III"],
    )
    _patch_run(
        monkeypatch,
        _run(
            answer="a",
            cypher_query="MATCH (n) RETURN n",
            cypher_read_only=True,
            kg_hits=[{"name": "ESCRT-III"}],
            cited_kg_indices=[0],
        ),
    )
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["cypher_validity"] == 1.0
    assert result["cypher_nonempty"] == 1.0
    assert result["kg_hit_at_k"] == 1.0
    assert result["kg_entity_recall"] == 1.0
    assert result["kg_source_grounding"] == 1.0


def test_kg_metrics_none_for_lancedb_case(monkeypatch):
    case = EvalCase(id="k2", question="q?")  # lancedb_only, no cypher runs
    _patch_run(monkeypatch, _run(answer="a"))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["cypher_validity"] is None
    assert result["cypher_nonempty"] is None
    assert result["kg_hit_at_k"] is None  # no gold entities either


def test_mode_routing_metric_for_auto_case(monkeypatch):
    case = EvalCase(
        id="k3", question="q?", retrieval={"retrieval_mode": "auto"}, expected_mode="neo4j_only"
    )
    _patch_run(monkeypatch, _run(answer="a", routed_mode="neo4j_only"))
    result = asyncio.run(E.evaluate_case(case, None, EvalConfig()))
    assert result["mode_routing_correctness"] == 1.0


def test_kg_group_disabled_yields_none(monkeypatch):
    case = EvalCase(
        id="k4",
        question="q?",
        retrieval={"retrieval_mode": "neo4j_only"},
        expected_entities=["ESCRT-III"],
    )
    _patch_run(
        monkeypatch, _run(cypher_query="MATCH (n) RETURN n", kg_hits=[{"name": "ESCRT-III"}])
    )
    cfg = EvalConfig(enabled_groups=frozenset({"source"}))  # kg OFF
    result = asyncio.run(E.evaluate_case(case, None, cfg))
    assert result["kg_hit_at_k"] is None
    assert result["cypher_validity"] is None


# ---- judge track (Phase 3) — judge module stubbed, no deepeval ----


def test_judge_track_wired(monkeypatch):
    from knowledge_agent.evaluation import judge as J

    async def fake_panel(data, models, threshold):
        return {k: 0.9 for k in J.JUDGE_METRIC_KEYS}, 200, 80

    monkeypatch.setattr(J, "run_judge_panel", fake_panel)
    monkeypatch.setattr(J, "resolve_judge_models", lambda m: ["x"])
    monkeypatch.setattr(J, "build_judge_input", lambda run, case: {})
    case = EvalCase(id="j1", question="q?", expected_answer_points=["fact"])
    _patch_run(monkeypatch, _run(answer="a"))
    cfg = EvalConfig(enabled_groups=frozenset({"judge"}))
    result = asyncio.run(E.evaluate_case(case, None, cfg))
    assert result["faithfulness"] == 0.9
    assert result["avg_judge_score"] == 0.9
    assert result["judge_total_tokens"] == 280
    assert result["status"] == "PASS"  # faithfulness + answer_relevancy >= threshold


def test_judge_group_disabled_yields_none(monkeypatch):
    case = EvalCase(id="j2", question="q?")
    _patch_run(monkeypatch, _run(answer="a"))
    cfg = EvalConfig(enabled_groups=frozenset({"source"}))  # judge OFF
    result = asyncio.run(E.evaluate_case(case, None, cfg))
    assert result["faithfulness"] is None
    assert result["avg_judge_score"] is None
    assert result["judge_total_tokens"] is None
