"""Integration tests for the agent graph (`graph.py` + `nodes.py`).

This is the heaviest integration tier file: each test invokes the
LangGraph agent end-to-end through LanceDB + Neo4j + Anthropic
Claude. All 6 retrieval modes are covered:

  - `lancedb_only`        — query_builder (Haiku) + lancedb_retriever
                            + synthesizer (Sonnet)
  - `neo4j_only`          — cypher_builder (Sonnet) + neo4j_retriever
                            + synthesizer
  - `lancedb_then_neo4j`  — both retriever legs in sequence
  - `neo4j_then_lancedb`  — kg first, lance scoped to kg's doc_ids
  - `parallel_fused`      — both legs in parallel, synthesizer fuses
  - `auto`                — mode_classifier (Haiku) picks one of the 5

Cost guard: ~$0.05-0.10 per test for the kg-touching modes (Sonnet
Cypher call + Sonnet synthesizer), ~$0.01-0.02 for lancedb-only.
Total: ~$0.40 per full integration run of this file.

Setup pattern: each test ingests a sample PDF first (`L1-L5` corpus
config — no L6/L7/L8 so the ingest itself stays cheap), then
invokes the agent against the resulting corpus. The kg-touching
modes pass the corpus_config in state so cypher_builder can scope
its schema description.

Manual interactive counterpart: `scripts/smoke_agent.py` (same six
modes + per-mode question variants for hand-tuning).

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from knowledge_agent.graph import graph
from knowledge_agent.ingestion.pipeline import ingest_document
from knowledge_agent.kg.corpus_config import CorpusConfig, LayerFlags
from knowledge_agent.models import AgentAnswer

pytestmark = pytest.mark.integration


def _minimal_corpus_config() -> CorpusConfig:
    """Cheap L1-L5 corpus — no LLM-per-chunk extraction."""
    return CorpusConfig(
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=False,
            triples=False,
            cross_doc=False,
            cross_doc_xrefs=False,
            xrefs="none",
        ),
        allowed_types=["Paper"],
    )


@pytest.fixture
def seeded_corpus(
    kg_client: Any, lance_client: Any, ensure_constraints: None,
    sample_pdf: Path,
) -> str:
    """Wipe both stores, ingest the session sample PDF, return its
    doc_id.

    Module-scope would be cheaper (1 ingest for many tests) but
    function-scope keeps each test isolated — agent behaviour
    should not depend on prior test state.
    """
    import asyncio

    async def _seed() -> str:
        with kg_client.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        await lance_client.drop_chunks_table()

        config = _minimal_corpus_config()
        result = await ingest_document(
            sample_pdf, config, "Document", "Paper",
        )
        assert result.lancedb_ok is True
        return result.doc_id

    return asyncio.run(_seed())


def test_agent_lancedb_only_returns_structured_answer(
    seeded_corpus: str,
) -> None:
    """A query in `lancedb_only` mode produces a final state with a
    populated `AgentAnswer` — the synthesizer's structured output."""
    state = asyncio.run(graph.ainvoke({
        "query": "What does the article say about chronic disease and nutrition?",
        "retrieval_mode": "lancedb_only",
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    assert answer.answer  # non-empty
    # In lancedb_only mode, chunk_sources should be populated;
    # kg_sources is expected empty (no Cypher leg ran).
    assert len(answer.kg_sources) == 0


def test_agent_lancedb_only_populates_search_query_and_retrieved_chunks(
    seeded_corpus: str,
) -> None:
    """The graph fills in the intermediate state fields:
    `search_query` (from query_builder) + `retrieved_chunks`
    (from lancedb_retriever)."""
    state = asyncio.run(graph.ainvoke({
        "query": "nutrition and chronic disease prevention",
        "retrieval_mode": "lancedb_only",
    }))

    assert state.get("search_query")  # non-empty
    assert state.get("retrieved_chunks") is not None
    assert len(state["retrieved_chunks"]) > 0


def test_agent_skip_query_builder_uses_raw_query(
    seeded_corpus: str,
) -> None:
    """When `skip_query_builder=True` is set via state, the raw user
    query is used as the search query (no Haiku call to rewrite it)."""
    raw = "nutrition chronic disease"
    state = asyncio.run(graph.ainvoke({
        "query": raw,
        "retrieval_mode": "lancedb_only",
        "skip_query_builder": True,
    }))

    assert state.get("search_query") == raw


def test_agent_direct_retrieval_skips_synthesizer(
    seeded_corpus: str,
) -> None:
    """When `direct_retrieval=True`, the synthesizer is skipped: the
    final state's `final_answer` has empty `answer` and the retrieved
    chunks are returned as sources for the caller to render."""
    state = asyncio.run(graph.ainvoke({
        "query": "nutrition chronic disease",
        "retrieval_mode": "lancedb_only",
        "direct_retrieval": True,
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    # Synthesizer skipped → answer string empty; chunk_sources carry
    # the retrieved chunks for the GUI to render.
    assert answer.answer == ""
    assert len(answer.chunk_sources) > 0


# ---------------------------------------------------------------------------
# Modes 2-6 — kg-touching modes.
#
# Each kg-touching mode requires `corpus_config` in the initial state
# so `cypher_builder` can scope its schema description. The minimal
# L1-L5 corpus matches what `seeded_corpus` ingested.
# ---------------------------------------------------------------------------


def test_agent_neo4j_only_returns_structured_answer_with_kg_sources(
    seeded_corpus: str,
) -> None:
    """`neo4j_only` mode: cypher_builder (Sonnet) → neo4j_retriever
    → synthesizer (Sonnet). chunk_sources should be empty (no lance
    leg), kg_sources may be empty or populated depending on the LLM-
    generated Cypher hitting the corpus."""
    state = asyncio.run(graph.ainvoke({
        "query": "Who authored papers about nutrition and chronic disease?",
        "retrieval_mode": "neo4j_only",
        "corpus_config": _minimal_corpus_config(),
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    assert answer.answer  # non-empty
    assert state.get("cypher_query")  # cypher_builder ran
    # kg_hits state is populated (may be empty list if Cypher returned
    # nothing, but the key must exist).
    assert state.get("kg_hits") is not None
    # No LanceDB leg ran in this mode.
    assert len(answer.chunk_sources) == 0


def test_agent_lancedb_then_neo4j_runs_both_legs_in_order(
    seeded_corpus: str,
) -> None:
    """`lancedb_then_neo4j` runs LanceDB first, then KG enriches via
    the Lance hits' doc_ids. Both `search_query` + `cypher_query` and
    both `retrieved_chunks` + `kg_hits` populated."""
    state = asyncio.run(graph.ainvoke({
        "query": "What is the role of nutrition in chronic disease?",
        "retrieval_mode": "lancedb_then_neo4j",
        "corpus_config": _minimal_corpus_config(),
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    # Both query types produced.
    assert state.get("search_query")
    assert state.get("cypher_query")
    # Both retriever outputs in state.
    assert state.get("retrieved_chunks") is not None
    assert len(state["retrieved_chunks"]) > 0
    assert state.get("kg_hits") is not None


def test_agent_neo4j_then_lancedb_runs_kg_first_then_lance(
    seeded_corpus: str,
) -> None:
    """`neo4j_then_lancedb`: KG first, then LanceDB scoped to KG-
    returned doc_ids. Same surface contract as `lancedb_then_neo4j`
    (both query types + both retriever outputs)."""
    state = asyncio.run(graph.ainvoke({
        "query": "List authors who wrote about chronic disease.",
        "retrieval_mode": "neo4j_then_lancedb",
        "corpus_config": _minimal_corpus_config(),
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    assert state.get("cypher_query")
    assert state.get("search_query")
    assert state.get("kg_hits") is not None
    assert state.get("retrieved_chunks") is not None


def test_agent_parallel_fused_runs_both_legs_in_parallel(
    seeded_corpus: str,
) -> None:
    """`parallel_fused`: both legs run in parallel, synthesizer
    fuses. Both state fields populated; the synthesizer's
    AgentAnswer may carry both source types."""
    state = asyncio.run(graph.ainvoke({
        "query": "Comprehensive overview of nutrition and chronic disease.",
        "retrieval_mode": "parallel_fused",
        "corpus_config": _minimal_corpus_config(),
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    assert state.get("search_query")
    assert state.get("cypher_query")
    assert state.get("retrieved_chunks") is not None
    assert state.get("kg_hits") is not None


def test_agent_auto_mode_routes_via_classifier(
    seeded_corpus: str,
) -> None:
    """`auto` mode runs `mode_classifier_node` (Haiku) first, which
    sets `routed_mode` to one of the 5 concrete modes. The graph
    then dispatches accordingly."""
    state = asyncio.run(graph.ainvoke({
        "query": "Which authors wrote about chronic disease?",
        "retrieval_mode": "auto",
        "corpus_config": _minimal_corpus_config(),
    }))

    answer = state.get("final_answer")
    assert answer is not None
    assert isinstance(answer, AgentAnswer)
    # The classifier resolved auto into a concrete mode.
    routed = state.get("routed_mode")
    assert routed in (
        "lancedb_only",
        "neo4j_only",
        "lancedb_then_neo4j",
        "neo4j_then_lancedb",
        "parallel_fused",
    )
