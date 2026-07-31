"""Unit tests for the LLM eval-case generator.

`generate_advanced` and `generate_gold_for_question` are tested with an
INJECTED fake LLM (a `RunnableLambda` returning a canned draft), so no real
provider/network is touched. `with_retry` is patched to identity module-wide
so the fake runs directly (no backoff, no `get_settings`). LanceDB + Neo4j
reads go through fake clients.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from knowledge_agent.evaluation.generator import (
    EvalGenerationConnectionError,
    GeneratedAdvancedCase,
    GeneratedGold,
    Passage,
    generate_advanced,
    generate_gold_for_question,
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


# ---- fake LanceDB client (shared by the generate_advanced tests) ----


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


# ---- generate_advanced (602): full-doc + KG grounding ----


def _fake_kg(*, entities_by_doc=None, triples=None, pairs=None):
    """A fake Neo4j client: `get_entities_by_chunk` returns a doc's entities, and
    `read_query` answers the cross-doc-pair aggregation OR an entity's triples by
    inspecting the Cypher."""
    kg = MagicMock()
    kg.get_entities_by_chunk = AsyncMock(
        side_effect=lambda doc_id: (entities_by_doc or {}).get(doc_id, {})
    )

    async def _read_query(cypher, **params):
        if "collect(DISTINCT d.doc_id)" in cypher:  # the cross-doc pair query
            return pairs or []
        return (triples or {}).get(params.get("key"), [])  # an entity's triples

    kg.read_query = AsyncMock(side_effect=_read_query)
    return kg


async def test_generate_advanced_hybrid_from_full_doc():
    """No graph facts → a hybrid case grounded in the doc, pinned to the text
    leg, runnable out of the box, expected_sources filled (it knows the doc)."""
    lance = _fake_client(
        [{"doc_id": "doc-a"}],
        {"doc-a": [{"text": "chunk one " * 20}, {"text": "chunk two " * 20}]},
    )
    gen = GeneratedAdvancedCase(
        case_type="hybrid", question="What is X?", answer_points=["x"], keywords=["x"]
    )
    cases = await generate_advanced(1, lance_client=lance, kg_client=_fake_kg(), llm=_fake_llm(gen))
    assert len(cases) == 1
    c = cases[0]
    assert c.origin == "llm"
    assert c.expected_sources == ["doc-a"]  # grounded in the sampled doc
    assert c.expected_entities == []
    assert c.retrieval.retrieval_mode == "lancedb_only"  # text leg
    assert c.id.startswith("adv-00-")
    assert validate_case(c) == []  # runnable


async def test_generate_advanced_kg_from_graph_facts():
    """A 'kg' draft (with entities) → a KG case pinned to the graph leg, with
    expected_entities filled."""
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "text " * 50}]})
    kg = _fake_kg(
        entities_by_doc={"doc-a": {"c0": [("escrt-iii", "Protein")]}},
        triples={"escrt-iii": [{"rel": "BINDS", "target": "chmp4b"}]},
    )
    gen = GeneratedAdvancedCase(
        case_type="kg",
        question="What does ESCRT-III bind?",
        answer_points=["CHMP4B"],
        keywords=["ESCRT"],
        expected_entities=["ESCRT-III", "CHMP4B"],
    )
    cases = await generate_advanced(1, lance_client=lance, kg_client=kg, llm=_fake_llm(gen))
    assert len(cases) == 1
    c = cases[0]
    assert c.retrieval.retrieval_mode == "neo4j_only"  # graph leg
    assert c.expected_entities == ["ESCRT-III", "CHMP4B"]
    assert validate_case(c) == []


async def test_generate_advanced_adds_cross_doc_case():
    """When entities are shared across docs, part of the batch is cross-document
    (expected_sources spans both docs)."""
    lance = _fake_client(
        [{"doc_id": d} for d in ("doc-a", "doc-b", "doc-c", "doc-d")],
        {d: [{"text": "text " * 50}] for d in ("doc-a", "doc-b", "doc-c", "doc-d")},
    )
    kg = _fake_kg(pairs=[{"key": "shared", "a": "doc-a", "b": "doc-b"}])
    gen = GeneratedAdvancedCase(case_type="hybrid", question="Q?", answer_points=[], keywords=[])
    cases = await generate_advanced(4, lance_client=lance, kg_client=kg, llm=_fake_llm(gen))
    cross = [c for c in cases if len(c.expected_sources) == 2]
    assert cross and cross[0].expected_sources == ["doc-a", "doc-b"]


async def test_generate_advanced_degrades_without_graph(monkeypatch):
    """No KG client available → hybrid-only, still produces full-doc cases."""

    def _no_kg():
        raise RuntimeError("no graph")

    monkeypatch.setattr("knowledge_agent.kg.client.get_kg_client", _no_kg)
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "t " * 50}]})
    gen = GeneratedAdvancedCase(case_type="hybrid", question="Q?", answer_points=[], keywords=[])
    cases = await generate_advanced(1, lance_client=lance, llm=_fake_llm(gen))  # kg_client=None
    assert len(cases) == 1
    assert cases[0].expected_sources == ["doc-a"]
    assert cases[0].retrieval.retrieval_mode == "lancedb_only"


async def test_generate_advanced_aborts_on_connection_error():
    lance = _fake_client([{"doc_id": "doc-a"}], {"doc-a": [{"text": "t " * 50}]})

    def _conn_boom(_messages):
        raise ConnectionError("Connection error.")

    with pytest.raises(EvalGenerationConnectionError, match="network"):
        await generate_advanced(
            1, lance_client=lance, kg_client=_fake_kg(), llm=_fake_llm(_conn_boom)
        )
