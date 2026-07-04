"""Unit tests for `knowledge_agent.models`.

The models module hosts two families of Pydantic schemas:

  - LLM-output schemas (`SearchQueryRewrite`, `CypherQueryRewrite`,
    `ModeChoice`, `AgentAnswer`, `ChunkSource`, `KGSource`) — the
    structured-output enforcement target for `with_structured_output`.
  - Retrieval schemas (`RetrievedChunk`, `KGHit`) — the typed
    return shape of search methods on `LanceClient` and the Neo4j
    retriever.

These tests pin: required-field construction, missing-required
raises, default values for optional fields, and serialization
round-trips (`model_dump` -> `model_validate` recovers the same
state). Catching breakage here protects every downstream caller
that relies on field names + defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from knowledge_agent.models import (
    AgentAnswer,
    ChunkSource,
    CypherQueryRewrite,
    KGHit,
    KGSource,
    ModeChoice,
    RetrievedChunk,
    SearchQueryRewrite,
)

# ---------------------------------------------------------------------------
# SearchQueryRewrite
# ---------------------------------------------------------------------------


def test_search_query_rewrite_minimal() -> None:
    m = SearchQueryRewrite(search_query="machine learning")
    assert m.search_query == "machine learning"


def test_search_query_rewrite_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        SearchQueryRewrite()  # type: ignore[call-arg]


def test_search_query_rewrite_roundtrip() -> None:
    original = SearchQueryRewrite(search_query="hybrid search")
    restored = SearchQueryRewrite.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# CypherQueryRewrite
# ---------------------------------------------------------------------------


def test_cypher_query_rewrite_minimal() -> None:
    m = CypherQueryRewrite(cypher_query="MATCH (d:Document) RETURN d LIMIT 5")
    assert m.cypher_query.startswith("MATCH")


def test_cypher_query_rewrite_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        CypherQueryRewrite()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ModeChoice
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "lancedb_only",
        "neo4j_only",
        "lancedb_then_neo4j",
        "neo4j_then_lancedb",
        "parallel_fused",
    ],
)
def test_mode_choice_accepts_all_concrete_modes(mode: str) -> None:
    m = ModeChoice(mode=mode)  # type: ignore[arg-type]
    assert m.mode == mode


def test_mode_choice_rejects_auto() -> None:
    """`auto` is what the classifier resolves — it must NOT itself be
    a valid choice for the classifier's output."""
    with pytest.raises(ValidationError):
        ModeChoice(mode="auto")  # type: ignore[arg-type]


def test_mode_choice_rejects_unknown_string() -> None:
    with pytest.raises(ValidationError):
        ModeChoice(mode="invented_mode")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RetrievedChunk
# ---------------------------------------------------------------------------


def test_retrieved_chunk_minimal() -> None:
    """Identity + text are required; every display field defaults to None."""
    c = RetrievedChunk(chunk_id="d#0", doc_id="d", text="hello")
    assert c.chunk_id == "d#0"
    assert c.doc_id == "d"
    assert c.text == "hello"
    assert c.score is None
    assert c.title is None
    assert c.year is None
    assert c.authors_display is None
    assert c.doi is None
    assert c.openalex_id is None
    assert c.venue is None
    assert c.source_url is None
    assert c.section is None
    assert c.page is None


def test_retrieved_chunk_with_full_metadata() -> None:
    c = RetrievedChunk(
        chunk_id="d#0",
        doc_id="d",
        text="hello",
        score=0.87,
        title="A paper",
        year=2024,
        authors_display="Smith et al.",
        doi="10.1/abc",
        openalex_id="W123",
        venue="Nature",
        source_url="https://example.com/x.pdf",
        section="Introduction",
        page=1,
    )
    assert c.score == pytest.approx(0.87)
    assert c.year == 2024
    assert c.page == 1


def test_retrieved_chunk_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        RetrievedChunk(chunk_id="d#0", doc_id="d")  # type: ignore[call-arg]


def test_retrieved_chunk_roundtrip() -> None:
    original = RetrievedChunk(chunk_id="d#0", doc_id="d", text="text", score=0.5, year=2023)
    restored = RetrievedChunk.model_validate(original.model_dump())
    assert restored == original


# ---------------------------------------------------------------------------
# KGHit
# ---------------------------------------------------------------------------


def test_kg_hit_with_arbitrary_dict() -> None:
    """`data` is a flexible dict because Cypher row shape depends on
    the LLM-generated query — no fixed field set possible."""
    h = KGHit(data={"author": "Smith", "year": 2024, "count": 42})
    assert h.data["author"] == "Smith"
    assert h.data["year"] == 2024


def test_kg_hit_empty_dict_ok() -> None:
    h = KGHit(data={})
    assert h.data == {}


def test_kg_hit_missing_data_raises() -> None:
    with pytest.raises(ValidationError):
        KGHit()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ChunkSource
# ---------------------------------------------------------------------------


def test_chunk_source_minimal() -> None:
    s = ChunkSource(chunk_id="d#0", doc_id="d")
    assert s.chunk_id == "d#0"
    assert s.quote is None


def test_chunk_source_with_quote() -> None:
    s = ChunkSource(chunk_id="d#0", doc_id="d", quote="key finding")
    assert s.quote == "key finding"


def test_chunk_source_missing_chunk_id_raises() -> None:
    with pytest.raises(ValidationError):
        ChunkSource(doc_id="d")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# KGSource
# ---------------------------------------------------------------------------


def test_kg_source_minimal() -> None:
    s = KGSource(hit_index=0)
    assert s.hit_index == 0
    assert s.quote is None


def test_kg_source_with_quote() -> None:
    s = KGSource(hit_index=3, quote="this row")
    assert s.hit_index == 3
    assert s.quote == "this row"


def test_kg_source_missing_hit_index_raises() -> None:
    with pytest.raises(ValidationError):
        KGSource()  # type: ignore[call-arg]


def test_kg_source_hit_index_must_be_int() -> None:
    """Pydantic should coerce numeric strings, but reject genuinely
    non-int values."""
    with pytest.raises(ValidationError):
        KGSource(hit_index="not a number")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AgentAnswer
# ---------------------------------------------------------------------------


def test_agent_answer_minimal() -> None:
    """Answer text required; both source lists default to empty."""
    a = AgentAnswer(answer="The answer is 42.")
    assert a.answer == "The answer is 42."
    assert a.chunk_sources == []
    assert a.kg_sources == []


def test_agent_answer_with_both_source_types() -> None:
    a = AgentAnswer(
        answer="See [1] and [2].",
        chunk_sources=[ChunkSource(chunk_id="d#0", doc_id="d")],
        kg_sources=[KGSource(hit_index=0)],
    )
    assert len(a.chunk_sources) == 1
    assert len(a.kg_sources) == 1
    assert a.chunk_sources[0].chunk_id == "d#0"
    assert a.kg_sources[0].hit_index == 0


def test_agent_answer_missing_answer_raises() -> None:
    with pytest.raises(ValidationError):
        AgentAnswer()  # type: ignore[call-arg]


def test_agent_answer_roundtrip_with_sources() -> None:
    """A populated AgentAnswer must survive json round-trip — this is
    what the synthesizer's structured output uses on the wire."""
    original = AgentAnswer(
        answer="Hybrid retrieval [1] outperforms vector-only [2].",
        chunk_sources=[
            ChunkSource(chunk_id="d#0", doc_id="d", quote="hybrid wins"),
            ChunkSource(chunk_id="d#5", doc_id="d"),
        ],
        kg_sources=[KGSource(hit_index=0, quote="row evidence")],
    )
    serialized = original.model_dump_json()
    restored = AgentAnswer.model_validate_json(serialized)
    assert restored == original


# ---------------------------------------------------------------------------
# Cross-model: default factories don't share state.
#
# A subtle Pydantic-default trap is using a mutable default (`[]`)
# instead of `default_factory=list`. AgentAnswer uses `default_factory`
# — these tests pin that two fresh instances have independent lists,
# so a mutation on one doesn't bleed into another.
# ---------------------------------------------------------------------------


def test_agent_answer_default_lists_are_independent_per_instance() -> None:
    a1 = AgentAnswer(answer="x")
    a2 = AgentAnswer(answer="y")
    # Mutating one's list must not affect the other.
    a1.chunk_sources.append(ChunkSource(chunk_id="d#0", doc_id="d"))
    assert a2.chunk_sources == []
