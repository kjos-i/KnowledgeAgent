"""Integration tests for `kg/triples_writes` (L8).

Exercises the typed-relation triples write + delete primitives
against a real Neo4j test instance. Uses synthetic ExtractedTriple
objects fed directly to `write_triples` — the LLM-based extractor is
not invoked here (that's `tests/unit/ingestion/test_triples_extractor.py`).

L8 writes one edge per (subject, predicate, object) triple using
one of the 15 predicate edge types in `schema.TRIPLE_PREDICATE_RELS`.
Out-of-vocabulary predicates are dropped per row, not raised.

Manual interactive counterpart: `scripts/smoke_kg_l8_triples.py` —
seeds the same synthetic graph + runs the real extractor.

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.ingestion.ids import make_chunk_id
from knowledge_agent.kg.schema import TRIPLE_PREDICATE_RELS
from knowledge_agent.kg.triples_writes import ExtractedTriple

SYNTHETIC_DOC_ID = "integ-doc-l8-001"

pytestmark = pytest.mark.integration


def _seed_chunk_and_entities(
    client: Any, doc_id: str, entities: list[tuple[str, str]]
) -> str:
    """Create focal :Document + :Chunk + :Entity nodes (with
    :MENTIONS edges from the chunk). Returns chunk_id."""
    chunk_id = make_chunk_id(doc_id, 0)
    with client.driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.in_corpus = true, d:Paper "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.section = 'Results', c.page = 1, c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
        for key, etype in entities:
            session.run(
                "MERGE (e:Entity {key: $key, entity_type: $type}) "
                "WITH e "
                "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                "MERGE (c)-[m:MENTIONS]->(e) "
                "ON CREATE SET m.offset = 0",
                key=key, type=etype, chunk_id=chunk_id,
            )
    return chunk_id


SYNTHETIC_ENTITIES = [
    ("metformin", "drug"),
    ("ampk", "protein"),
    ("glut4", "protein"),
    ("type 2 diabetes mellitus", "disease"),
]


# Pick the first three predicates from the vocabulary — any are fine.
_PREDS = list(TRIPLE_PREDICATE_RELS)[:3]

SYNTHETIC_TRIPLES = [
    ExtractedTriple(
        subject_key="metformin",
        subject_entity_type="drug",
        predicate=_PREDS[0],
        object_key="ampk",
        object_entity_type="protein",
        evidence_span="metformin activates AMPK signalling",
    ),
    ExtractedTriple(
        subject_key="ampk",
        subject_entity_type="protein",
        predicate=_PREDS[1],
        object_key="glut4",
        object_entity_type="protein",
        evidence_span="AMPK activates the GLUT4 transporter",
    ),
    ExtractedTriple(
        subject_key="metformin",
        subject_entity_type="drug",
        predicate=_PREDS[2],
        object_key="type 2 diabetes mellitus",
        object_entity_type="disease",
        evidence_span="metformin is associated with reduced T2DM incidence",
    ),
]


async def test_write_triples_creates_one_edge_per_triple(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """write_triples produces one typed edge per ExtractedTriple,
    using the matching predicate from TRIPLE_PREDICATE_RELS as the
    edge type. Each edge carries doc_id + chunk_id properties."""
    chunk_id = _seed_chunk_and_entities(
        kg_client, SYNTHETIC_DOC_ID, SYNTHETIC_ENTITIES
    )
    ok = await kg_client.write_triples(
        SYNTHETIC_DOC_ID, [(chunk_id, SYNTHETIC_TRIPLES)]
    )
    assert ok is None  # success: returns None (typed-errors contract)

    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    with kg_client.driver.session() as session:
        rows = list(session.run(
            f"MATCH (s:Entity)-[r:{rel_union}]->(o:Entity) "
            f"WHERE r.doc_id = $doc_id "
            f"RETURN type(r) AS predicate, s.key AS subj, o.key AS obj, "
            f"r.chunk_id AS chunk_id, r.evidence_span AS evidence "
            f"ORDER BY predicate",
            doc_id=SYNTHETIC_DOC_ID,
        ))
    assert len(rows) == 3
    predicates = {r["predicate"] for r in rows}
    assert predicates == set(_PREDS)
    for r in rows:
        assert r["chunk_id"] == chunk_id
        assert r["evidence"]  # non-empty evidence_span per triple


async def test_write_triples_out_of_vocabulary_predicate_is_skipped(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """A triple with a predicate not in TRIPLE_PREDICATE_RELS is
    dropped silently — does NOT raise, does NOT block other triples
    in the same batch."""
    chunk_id = _seed_chunk_and_entities(
        kg_client, SYNTHETIC_DOC_ID, SYNTHETIC_ENTITIES
    )
    mixed = [
        SYNTHETIC_TRIPLES[0],  # valid
        ExtractedTriple(
            subject_key="metformin",
            subject_entity_type="drug",
            predicate="NOT_A_REAL_PREDICATE",
            object_key="ampk",
            object_entity_type="protein",
            evidence_span="synthetic bad-predicate row",
        ),
        SYNTHETIC_TRIPLES[1],  # valid
    ]
    ok = await kg_client.write_triples(SYNTHETIC_DOC_ID, [(chunk_id, mixed)])
    assert ok is None  # success: returns None (typed-errors contract)

    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    with kg_client.driver.session() as session:
        n_real = session.run(
            f"MATCH ()-[r:{rel_union}]->() "
            f"WHERE r.doc_id = $doc_id RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
    assert n_real == 2  # only the two valid triples landed


async def test_delete_triples_removes_only_target_doc_edges(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """delete_triples_by_doc_id removes ALL 15-predicate-union edges
    sourced by the target doc, AND preserves edges sourced by other
    docs."""
    other_doc = "integ-doc-l8-other"
    chunk_a = _seed_chunk_and_entities(
        kg_client, SYNTHETIC_DOC_ID, SYNTHETIC_ENTITIES
    )
    chunk_b = _seed_chunk_and_entities(
        kg_client, other_doc, SYNTHETIC_ENTITIES
    )
    await kg_client.write_triples(SYNTHETIC_DOC_ID, [(chunk_a, SYNTHETIC_TRIPLES)])
    await kg_client.write_triples(other_doc, [(chunk_b, SYNTHETIC_TRIPLES)])

    ok = await kg_client.delete_triples_by_doc_id(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    with kg_client.driver.session() as session:
        n_target = session.run(
            f"MATCH ()-[r:{rel_union}]->() "
            f"WHERE r.doc_id = $doc_id RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        n_other = session.run(
            f"MATCH ()-[r:{rel_union}]->() "
            f"WHERE r.doc_id = $doc_id RETURN count(r) AS n",
            doc_id=other_doc,
        ).single()["n"]
    assert n_target == 0
    assert n_other == 3


async def test_get_entities_by_chunk_returns_l6a_vocabulary(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """The L8 backfill helper reads back this doc's L6a vocabulary
    grouped by chunk_id. Useful so backfill_triples doesn't need to
    re-run L6a — it consults the existing :Entity nodes."""
    chunk_id = _seed_chunk_and_entities(
        kg_client, SYNTHETIC_DOC_ID, SYNTHETIC_ENTITIES
    )

    by_chunk = await kg_client.get_entities_by_chunk(SYNTHETIC_DOC_ID)
    assert chunk_id in by_chunk
    vocab = {(k, t) for k, t in by_chunk[chunk_id]}
    assert vocab == set(SYNTHETIC_ENTITIES)
