"""Integration tests for `kg/cross_doc_writes` (L9).

Exercises the L9 cross-document `:RELATED_TO` recompute against a
real Neo4j test instance. Pure-Cypher pass; no LLM involved.

`recompute_cross_doc_edges(doc_id, threshold)` walks
`(focal)<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(:Entity)
<-[:MENTIONS]-(:Chunk)-[:PART_OF]->(other)` and writes one
undirected `:RELATED_TO` edge per (this, other) pair whose distinct
shared-entity count meets `threshold`. Edge properties carry the
shared keys + count + timestamp.

Manual interactive counterpart: `scripts/smoke_kg_l9_cross_doc.py`.

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.ingestion.ids import make_chunk_id

DOC_ALPHA = "integ-doc-l9-alpha"
DOC_BETA = "integ-doc-l9-beta"
DOC_GAMMA = "integ-doc-l9-gamma"

pytestmark = pytest.mark.integration


def _seed_doc(client: Any, doc_id: str, entities: list[tuple[str, str]]) -> str:
    """Write a synthetic :Document + one :Chunk + :MENTIONS edges.
    Returns chunk_id."""
    chunk_id = make_chunk_id(doc_id, 0)
    with client.driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.in_corpus = true, d:Paper "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.section = 'Synthetic', c.page = 1, c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
        for key, etype in entities:
            session.run(
                "MERGE (e:Entity {key: $key, entity_type: $type}) "
                "WITH e MATCH (c:Chunk {chunk_id: $chunk_id}) "
                "MERGE (c)-[m:MENTIONS]->(e) "
                "ON CREATE SET m.offset = 0",
                key=key,
                type=etype,
                chunk_id=chunk_id,
            )
    return chunk_id


SHARED_ENTITIES = [
    ("metformin", "drug"),
    ("ampk", "protein"),
    ("type 2 diabetes mellitus", "disease"),
]
ALPHA_UNIQUE = [("glut4", "protein")]
BETA_UNIQUE = [("hba1c", "biomarker")]


async def test_recompute_writes_related_to_edge_when_threshold_met(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Two docs sharing 3 distinct entities meet `threshold=2`; the
    recompute writes one :RELATED_TO edge between them, carrying
    shared_count=3 and the shared_keys list."""
    _seed_doc(kg_client, DOC_ALPHA, SHARED_ENTITIES + ALPHA_UNIQUE)
    _seed_doc(kg_client, DOC_BETA, SHARED_ENTITIES + BETA_UNIQUE)

    n = await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 2)
    assert n == 1

    with kg_client.driver.session() as session:
        row = session.run(
            "MATCH (a:Document {doc_id: $a})-[r:RELATED_TO]-"
            "(b:Document {doc_id: $b}) "
            "RETURN r.shared_count AS n, r.shared_entities AS keys",
            a=DOC_ALPHA,
            b=DOC_BETA,
        ).single()
    assert row is not None
    assert row["n"] == 3
    assert set(row["keys"]) == {k for k, _ in SHARED_ENTITIES}


async def test_recompute_writes_no_edge_below_threshold(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Two docs sharing 3 entities + threshold=5 → no edge."""
    _seed_doc(kg_client, DOC_ALPHA, SHARED_ENTITIES + ALPHA_UNIQUE)
    _seed_doc(kg_client, DOC_BETA, SHARED_ENTITIES + BETA_UNIQUE)

    n = await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 5)
    assert n == 0


async def test_recompute_wipes_previous_edges_incident_to_doc(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Recompute is wipe-and-rewrite per doc — a second call with the
    same threshold produces the same count, not double."""
    _seed_doc(kg_client, DOC_ALPHA, SHARED_ENTITIES + ALPHA_UNIQUE)
    _seed_doc(kg_client, DOC_BETA, SHARED_ENTITIES + BETA_UNIQUE)

    await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 2)
    n_second = await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 2)
    assert n_second == 1

    with kg_client.driver.session() as session:
        n_edges = session.run(
            "MATCH (:Document {doc_id: $a})-[r:RELATED_TO]-(:Document) RETURN count(r) AS n",
            a=DOC_ALPHA,
        ).single()["n"]
    assert n_edges == 1


async def test_recompute_handles_multi_doc_neighbourhood(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """When alpha shares enough entities with TWO other docs (beta
    and gamma), recompute returns 2 and writes 2 :RELATED_TO edges
    incident to alpha."""
    _seed_doc(kg_client, DOC_ALPHA, SHARED_ENTITIES)
    _seed_doc(kg_client, DOC_BETA, SHARED_ENTITIES)
    _seed_doc(kg_client, DOC_GAMMA, SHARED_ENTITIES)

    n = await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 2)
    assert n == 2

    with kg_client.driver.session() as session:
        neighbours = list(
            session.run(
                "MATCH (a:Document {doc_id: $a})-[:RELATED_TO]-(other:Document) "
                "RETURN other.doc_id AS doc_id ORDER BY other.doc_id",
                a=DOC_ALPHA,
            )
        )
    assert {r["doc_id"] for r in neighbours} == {DOC_BETA, DOC_GAMMA}


async def test_recompute_no_edge_for_single_doc(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """With only one doc present, recompute returns 0 — there's
    nothing to relate to."""
    _seed_doc(kg_client, DOC_ALPHA, SHARED_ENTITIES + ALPHA_UNIQUE)

    n = await kg_client.recompute_cross_doc_edges(DOC_ALPHA, 2)
    assert n == 0
