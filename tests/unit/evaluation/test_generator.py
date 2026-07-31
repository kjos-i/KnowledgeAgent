"""Unit tests for the LLM eval-case generator.

`generate_targeted` and `generate_gold_for_question` are tested with an
INJECTED fake LLM (a `RunnableLambda` returning a canned draft), so no real
provider/network is touched. `with_retry` is patched to identity module-wide
so the fake runs directly (no backoff, no `get_settings`). LanceDB + Neo4j
reads go through fake clients.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from knowledge_agent.evaluation.generator import (
    EvalGenerationConnectionError,
    GeneratedAdvancedCase,
    GeneratedGold,
    Passage,
    generate_gold_for_question,
    generate_targeted,
    generation_capabilities,
    passages_from_sources,
)
from knowledge_agent.evaluation.models import validate_case

# The generator reads retrieval knobs from `config.retrieval_defaults()` (the
# real fixed field defaults), so no fake retrieval config is needed. It only
# calls get_settings() for the model default, and only when no LLM is injected;
# a minimal fake keeps that fast + isolated from the user .env.
_FAKE_SETTINGS = SimpleNamespace(mode_classifier_model="fake-model")


@pytest.fixture(autouse=True)
def _identity_with_retry(monkeypatch):
    """Make `with_retry` a pass-through so the fake runnable runs directly (no
    backoff loop) and stub `get_settings` (used only for the model default) so
    these unit tests stay fast + isolated from the real Settings / user .env."""
    import knowledge_agent.evaluation.generator as gen_mod

    monkeypatch.setattr(gen_mod, "with_retry", lambda r: r)
    monkeypatch.setattr("knowledge_agent.config.get_settings", lambda: _FAKE_SETTINGS)


def _fake_llm(structured_result):
    """A fake chat model whose `with_structured_output(...)` returns a
    runnable yielding `structured_result` (a canned draft, or a callable
    to run for its side effect)."""
    fn = structured_result if callable(structured_result) else (lambda _m: structured_result)
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(fn)
    return llm


# ---- fake LanceDB client (shared by the generate_targeted tests) ----


def _fake_client(docs, chunks_by_doc):
    client = MagicMock()
    client.list_indexed_docs = AsyncMock(return_value=docs)
    client.get_chunks_by_doc_id = AsyncMock(side_effect=lambda doc_id: chunks_by_doc[doc_id])
    return client


# ---- generate_gold_for_question (the "Query from chat" + LLM path) ----


async def test_generate_gold_for_question_writes_gold_for_a_given_question():
    """The question is FIXED (the chat's distilled query); the LLM writes only
    the gold (answer points + keywords) for it, grounded in the passages."""
    llm = _fake_llm(GeneratedGold(answer_points=["the valve corroded"], keywords=["valve"]))
    gold = await generate_gold_for_question(
        "why did the valve fail?", [Passage("d1", "the valve corroded over time")], llm=llm
    )
    assert isinstance(gold, GeneratedGold)
    assert gold.answer_points == ["the valve corroded"]
    assert gold.keywords == ["valve"]


async def test_generate_gold_for_question_aborts_on_connection_error():
    def _conn_boom(_messages):
        raise ConnectionError("Connection error.")

    with pytest.raises(EvalGenerationConnectionError, match="network"):
        await generate_gold_for_question("q?", [Passage("d1", "x")], llm=_fake_llm(_conn_boom))


# ---- passages_from_sources (re-fetch cited chunk text for grounding) ----


def _chunk_client(chunks_by_doc):
    client = MagicMock()
    client.get_chunks_by_doc_id = AsyncMock(
        side_effect=lambda doc_id: chunks_by_doc.get(doc_id, [])
    )
    return client


async def test_passages_from_sources_refetches_cited_chunk_full_text():
    """chunk_sources carry only a short quote; the helper re-fetches the cited
    chunk's FULL text by chunk_id (not every chunk of the doc)."""
    client = _chunk_client(
        {
            "d1": [
                {"chunk_id": "d1-0", "text": "full text zero"},
                {"chunk_id": "d1-1", "text": "full text one"},
            ]
        }
    )
    sources = [SimpleNamespace(doc_id="d1", chunk_id="d1-1", quote="short quote")]
    passages = await passages_from_sources(sources, client=client)
    assert [(p.doc_id, p.text) for p in passages] == [("d1", "full text one")]


async def test_passages_from_sources_falls_back_to_quote_when_refetch_empty():
    client = _chunk_client({})  # nothing re-fetched (chunk gone / no table)
    sources = [SimpleNamespace(doc_id="d1", chunk_id="d1-1", quote="the anchoring quote")]
    passages = await passages_from_sources(sources, client=client)
    assert [(p.doc_id, p.text) for p in passages] == [("d1", "the anchoring quote")]


async def test_passages_from_sources_dedups_preserving_citation_order():
    client = _chunk_client(
        {
            "d1": [{"chunk_id": "d1-0", "text": "zero"}],
            "d2": [{"chunk_id": "d2-0", "text": "two"}],
        }
    )
    sources = [
        SimpleNamespace(doc_id="d1", chunk_id="d1-0", quote=None),
        SimpleNamespace(doc_id="d2", chunk_id="d2-0", quote=None),
        SimpleNamespace(doc_id="d1", chunk_id="d1-0", quote=None),  # dup → dropped
    ]
    passages = await passages_from_sources(sources, client=client)
    assert [(p.doc_id, p.text) for p in passages] == [("d1", "zero"), ("d2", "two")]


# ---- generate_targeted (mode-targeted generation) ----


def _fake_kg(*, entities_by_doc=None, triples=None, graph_docs=None, l9=None, l10=None):
    """A fake Neo4j client. `get_entities_by_chunk` returns a doc's entities;
    `read_query` dispatches on the Cypher: graph-eligible docs, L9 :RELATED_TO
    pairs, L10 :RELATED_BY_XREF pairs, or an entity's triples."""
    kg = MagicMock()
    kg.get_entities_by_chunk = AsyncMock(
        side_effect=lambda doc_id: (entities_by_doc or {}).get(doc_id, {})
    )

    async def _read_query(cypher, **params):
        if "RETURN DISTINCT d.doc_id" in cypher:  # graph-eligible docs
            return [{"doc_id": d} for d in (graph_docs or [])]
        if "RELATED_TO" in cypher:  # L9 pairs
            return list(l9 or [])
        if "RELATED_BY_XREF" in cypher:  # L10 pairs
            return list(l10 or [])
        return (triples or {}).get(params.get("key"), [])  # an entity's triples

    kg.read_query = AsyncMock(side_effect=_read_query)
    return kg


def _many_docs(n):
    """A fake LanceDB client with `n` docs, each a single usable chunk."""
    docs = [f"doc-{i}" for i in range(n)]
    return _fake_client([{"doc_id": d} for d in docs], {d: [{"text": "text " * 50}] for d in docs})


async def test_generate_targeted_text_mode_from_full_doc():
    """A lancedb_only target: a case grounded in the sampled doc, pinned to the
    text leg, runnable out of the box, expected_sources filled."""
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "chunk one " * 20}]})
    gen = GeneratedAdvancedCase(question="What is X?", answer_points=["x"], keywords=["x"])
    cases = await generate_targeted(
        1,
        modes=["lancedb_only"],
        lance_client=lance,
        kg_client=_fake_kg(),
        llm=_fake_llm(gen),
        rng=random.Random(0),
    )
    assert len(cases) == 1
    c = cases[0]
    assert c.origin == "llm"
    assert c.expected_sources == ["doc-a"]
    assert c.retrieval.retrieval_mode == "lancedb_only"
    assert c.id.startswith("gen-00-")
    assert validate_case(c) == []


async def test_generate_targeted_graph_mode_pins_neo4j():
    """A neo4j_only target draws from an entity-bearing doc, pins the graph leg,
    and carries the LLM's expected_entities."""
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "text " * 50}]})
    kg = _fake_kg(
        graph_docs=["doc-a"],
        entities_by_doc={"doc-a": {"c0": [("escrt-iii", "Protein")]}},
        triples={"escrt-iii": [{"rel": "BINDS", "target": "chmp4b"}]},
    )
    gen = GeneratedAdvancedCase(
        question="What does ESCRT-III bind?",
        answer_points=["CHMP4B"],
        keywords=["ESCRT"],
        expected_entities=["ESCRT-III", "CHMP4B"],
    )
    cases = await generate_targeted(
        1,
        modes=["neo4j_only"],
        lance_client=lance,
        kg_client=kg,
        llm=_fake_llm(gen),
        rng=random.Random(0),
    )
    assert len(cases) == 1
    c = cases[0]
    assert c.retrieval.retrieval_mode == "neo4j_only"
    assert c.expected_entities == ["ESCRT-III", "CHMP4B"]
    assert validate_case(c) == []


async def test_generate_targeted_graph_mode_empty_pool_yields_nothing():
    """A graph target with no entity-bearing docs produces no cases (its pool is
    empty); it does not fall back to text docs."""
    lance = _many_docs(3)
    kg = _fake_kg(graph_docs=[])
    cases = await generate_targeted(
        3,
        modes=["neo4j_only"],
        lance_client=lance,
        kg_client=kg,
        llm=_fake_llm(GeneratedAdvancedCase(question="Q?")),
        rng=random.Random(0),
    )
    assert cases == []


async def test_generate_targeted_cross_doc_related_spans_two_docs():
    """A cross-doc (L9) target deals an edge pair: the case spans both docs and
    pins parallel_fused."""
    lance = _fake_client(
        [{"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
        {"doc-a": [{"text": "text " * 50}], "doc-b": [{"text": "text " * 50}]},
    )
    kg = _fake_kg(l9=[{"a": "doc-a", "b": "doc-b", "shared": ["escrt"]}])
    gen = GeneratedAdvancedCase(question="How do A and B relate?")
    cases = await generate_targeted(
        1,
        cross_doc=["related"],
        lance_client=lance,
        kg_client=kg,
        llm=_fake_llm(gen),
        rng=random.Random(0),
    )
    assert len(cases) == 1
    c = cases[0]
    assert c.expected_sources == ["doc-a", "doc-b"]
    assert c.retrieval.retrieval_mode == "parallel_fused"


async def test_generate_targeted_round_robin_splits_across_modes():
    """N cases split evenly, round-robin, across the ticked modes."""
    lance = _many_docs(6)
    cases = await generate_targeted(
        4,
        modes=["lancedb_only", "auto"],
        lance_client=lance,
        kg_client=_fake_kg(),
        llm=_fake_llm(GeneratedAdvancedCase(question="Q?")),
        rng=random.Random(0),
    )
    assert len(cases) == 4
    modes = [c.retrieval.retrieval_mode for c in cases]
    assert modes.count("lancedb_only") == 2
    assert modes.count("auto") == 2


async def test_generate_targeted_rolls_over_when_a_target_runs_dry():
    """A target whose pool is smaller than its share rolls the remainder onto the
    others (coverage-first): one graph doc -> 1 KG case, the rest text."""
    lance = _many_docs(6)
    kg = _fake_kg(graph_docs=["doc-0"], entities_by_doc={"doc-0": {"c": [("e", "T")]}})
    gen = GeneratedAdvancedCase(question="Q?", expected_entities=["e"])
    cases = await generate_targeted(
        4,
        modes=["neo4j_only", "lancedb_only"],
        lance_client=lance,
        kg_client=kg,
        llm=_fake_llm(gen),
        rng=random.Random(0),
    )
    assert len(cases) == 4
    modes = [c.retrieval.retrieval_mode for c in cases]
    assert modes.count("neo4j_only") == 1
    assert modes.count("lancedb_only") == 3


async def test_generate_targeted_unique_doc_per_mode():
    """Within one mode target, no document is used twice (dealt without
    replacement)."""
    lance = _many_docs(5)
    cases = await generate_targeted(
        5,
        modes=["lancedb_only"],
        lance_client=lance,
        kg_client=_fake_kg(),
        llm=_fake_llm(GeneratedAdvancedCase(question="Q?")),
        rng=random.Random(0),
    )
    sources = [tuple(c.expected_sources) for c in cases]
    assert len(sources) == len(set(sources)) == 5


async def test_generate_targeted_no_targets_returns_empty():
    assert await generate_targeted(5, modes=[], cross_doc=[]) == []


async def test_generate_targeted_aborts_on_connection_error():
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "t " * 50}]})

    def _conn_boom(_messages):
        raise ConnectionError("Connection error.")

    with pytest.raises(EvalGenerationConnectionError, match="network"):
        await generate_targeted(
            1,
            modes=["lancedb_only"],
            lance_client=lance,
            kg_client=_fake_kg(),
            llm=_fake_llm(_conn_boom),
            rng=random.Random(0),
        )


async def test_generate_targeted_graph_mode_without_kg_yields_nothing(monkeypatch):
    """A graph target with no reachable KG produces nothing (empty pool), no crash."""

    def _no_kg():
        raise RuntimeError("no graph")

    monkeypatch.setattr("knowledge_agent.kg.client.get_kg_client", _no_kg)
    lance = _many_docs(2)
    cases = await generate_targeted(
        2,
        modes=["neo4j_only"],
        lance_client=lance,
        llm=_fake_llm(GeneratedAdvancedCase(question="Q?")),
        rng=random.Random(0),
    )
    assert cases == []


# ---- generation_capabilities (GUI grey-out probe) ----


async def test_generation_capabilities_reflects_graph_counts():
    kg = MagicMock()
    kg.count_mentions = AsyncMock(return_value=5)
    kg.count_related_to_edges = AsyncMock(return_value=2)
    kg.count_related_by_xref_edges = AsyncMock(return_value=0)
    caps = await generation_capabilities(kg_client=kg)
    assert caps == {"text": True, "graph": True, "cross_related": True, "cross_xref": False}


async def test_generation_capabilities_text_only_without_graph(monkeypatch):
    def _no_kg():
        raise RuntimeError("no graph")

    monkeypatch.setattr("knowledge_agent.kg.client.get_kg_client", _no_kg)
    caps = await generation_capabilities()
    assert caps == {"text": True, "graph": False, "cross_related": False, "cross_xref": False}
