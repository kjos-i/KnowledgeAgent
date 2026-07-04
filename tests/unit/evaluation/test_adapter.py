"""Tests for the KA graph adapter.

`build_case_run` reads a (typed) final state into a flat CaseRun with no
graph invocation; `run_case` is driven with an injected fake graph so no
real LLM/DB is touched — the harness stays isolated (mocks only).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_agent.evaluation import adapter as A
from knowledge_agent.evaluation.models import EvalCase


def _final_state() -> dict:
    answer = SimpleNamespace(
        answer="Norway's capital is Oslo.",
        chunk_sources=[SimpleNamespace(chunk_id="c1"), SimpleNamespace(chunk_id="cX")],
        kg_sources=[SimpleNamespace(hit_index=0)],
    )
    chunks = [
        SimpleNamespace(doc_id="d1", chunk_id="c1", text="Oslo is the capital"),
        SimpleNamespace(doc_id="d2", chunk_id="c2", text="other"),
    ]
    return {
        "final_answer": answer,
        "retrieved_chunks": chunks,
        "kg_hits": [SimpleNamespace(data={"name": "Oslo"})],
        "cypher_query": "MATCH (n) RETURN n",
        "routed_mode": "neo4j_only",
        "search_query": "norway capital",
    }


def _cb(usage=None) -> SimpleNamespace:
    return SimpleNamespace(usage_metadata=usage or {})


def test_sum_tokens_sums_across_models_and_none_when_empty():
    cb = _cb(
        {
            "claude": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "gpt": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        }
    )
    assert A._sum_tokens(cb) == (12, 8, 20)
    assert A._sum_tokens(_cb({})) == (None, None, None)


def test_build_case_run_reads_typed_state():
    run = A.build_case_run("Q?", _final_state(), _cb(), latency_seconds=1.5, error=None)
    assert run.answer == "Norway's capital is Oslo."
    assert run.retrieved_doc_ids == ["d1", "d2"]
    assert run.retrieved_chunk_ids == ["c1", "c2"]
    assert run.retrieved_texts == ["Oslo is the capital", "other"]
    assert run.cited_chunk_ids == ["c1", "cX"]  # note cX (fabricated) preserved for grounding check
    assert run.kg_hits == [{"name": "Oslo"}]
    assert run.cited_kg_indices == [0]
    assert run.cypher_query == "MATCH (n) RETURN n"
    assert run.routed_mode == "neo4j_only"
    assert run.latency_seconds == 1.5
    assert run.error is None


def test_build_case_run_handles_missing_answer():
    run = A.build_case_run("Q?", {"retrieved_chunks": []}, _cb(), 0.1, None)
    assert run.answer == ""
    assert run.cited_chunk_ids == []
    assert run.retrieved_doc_ids == []


def test_run_case_invokes_graph_with_state_overrides():
    case = EvalCase(
        id="x", question="what?", retrieval={"retrieval_mode": "neo4j_only", "top_k": 3}
    )
    fake_graph = SimpleNamespace(ainvoke=AsyncMock(return_value=_final_state()))
    run = asyncio.run(A.run_case(case, corpus_config=None, graph=fake_graph))
    fake_graph.ainvoke.assert_awaited_once()
    state = fake_graph.ainvoke.call_args.args[0]
    assert state["query"] == "what?"
    assert state["retrieval_mode"] == "neo4j_only"
    assert state["top_k"] == 3
    assert "corpus_config" in state
    assert run.answer.startswith("Norway")


def test_run_case_captures_graph_error():
    case = EvalCase(id="x", question="q?")
    fake_graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("boom")))
    run = asyncio.run(A.run_case(case, corpus_config=None, graph=fake_graph))
    assert run.error is not None and "boom" in run.error
    assert run.answer == ""  # empty final_state → empty answer, no crash
