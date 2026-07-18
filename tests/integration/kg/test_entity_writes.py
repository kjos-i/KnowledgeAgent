"""Integration tests for `kg/entity_writes` (L6).

Exercises the entity-mention write + delete primitives against a
real Neo4j test instance:

  - `write_entities(doc_id, chunk_mentions)`: MERGEs :Entity nodes
    (composite key on `(key, type)`) + :MENTIONS edges with offset
    properties.
  - `delete_entities_by_doc_id(doc_id)`: drops :MENTIONS from this
    doc's chunks + orphan-GCs :Entity with no remaining inbound
    :MENTIONS.

Uses synthetic Mention objects fed directly to `write_entities` —
no extractor adapter is invoked here. The extractors themselves are
unit-tested in `tests/unit/entity_extractors/`; this file pins the
KG-side contract.

Manual interactive counterpart: `scripts/smoke_kg_l6_entities.py`
— runs each extractor against synthetic text, then writes via
write_entities. Use the smoke when you want to see what a particular
adapter actually emits; use this file for the KG-write contract.

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.ingestion.ids import make_chunk_id

SYNTHETIC_DOC_ID = "integ-doc-l6-001"

pytestmark = pytest.mark.integration


def _seed_chunk(client: Any, doc_id: str, chunk_index: int = 0) -> str:
    """Create the focal :Document + one :Chunk so write_entities has a
    target. Returns chunk_id."""
    chunk_id = make_chunk_id(doc_id, chunk_index)
    with client.driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.in_corpus = true, d:Paper "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = $idx, "
            "c.section = 'Introduction', c.page = 1, c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id,
            chunk_id=chunk_id,
            idx=chunk_index,
        )
    return chunk_id


# Synthetic mentions covering 3 entity types + one duplicate-key case
# (two "aspirin" mentions: the :MENTIONS edge MERGEs on (chunk, entity)
# so they collapse to ONE edge from this chunk to the aspirin :Entity).
SYNTHETIC_MENTIONS = [
    Mention(raw_text="Aspirin", entity_type="drug", offset=0),
    Mention(raw_text="diabetes", entity_type="disease", offset=20),
    Mention(raw_text="BRCA1", entity_type="gene", offset=40),
    Mention(raw_text="aspirin", entity_type="drug", offset=60),
]


async def test_write_entities_creates_entity_nodes_and_mentions_edges(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """write_entities MERGEs distinct (key, type) :Entity nodes + one
    :MENTIONS edge per distinct (chunk, entity) pair.

    The two "Aspirin"/"aspirin" mentions share a lowercased key,
    AND fire from the same chunk — so they collapse to ONE :Entity
    AND ONE :MENTIONS edge (the MERGE on (chunk, entity) deduplicates).
    """
    chunk_id = _seed_chunk(kg_client, SYNTHETIC_DOC_ID)
    ok = await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, SYNTHETIC_MENTIONS)])
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        rows = list(
            session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id})-[m:MENTIONS]->(e:Entity) "
                "RETURN e.key AS key, e.entity_type AS type, m.offset AS offset "
                "ORDER BY m.offset",
                chunk_id=chunk_id,
            )
        )
    # 3 distinct entities + 3 :MENTIONS edges (aspirin dedupes).
    assert len(rows) == 3
    distinct_entities = {(r["key"], r["type"]) for r in rows}
    assert distinct_entities == {
        ("aspirin", "drug"),
        ("diabetes", "disease"),
        ("brca1", "gene"),
    }
    # offset is set ON CREATE only — first mention wins. The first
    # aspirin Mention (offset=0) is what the edge records.
    aspirin_offset = next(r["offset"] for r in rows if r["key"] == "aspirin")
    assert aspirin_offset == 0


async def test_write_entities_lowercases_keys(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Entity keys are stored lowercased — different casings of the
    same word collapse to one :Entity. Catches casing-leak regressions.

    All three mentions are from the same chunk → the :MENTIONS MERGE
    dedupes to a single edge."""
    chunk_id = _seed_chunk(kg_client, SYNTHETIC_DOC_ID)
    mentions = [
        Mention(raw_text="EGFR", entity_type="gene", offset=0),
        Mention(raw_text="egfr", entity_type="gene", offset=10),
        Mention(raw_text="Egfr", entity_type="gene", offset=20),
    ]
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, mentions)])

    with kg_client.driver.session() as session:
        n_entities = session.run(
            "MATCH (e:Entity {entity_type: 'gene'}) RETURN count(e) AS n"
        ).single()["n"]
        n_mentions = session.run(
            "MATCH (c:Chunk {chunk_id: $chunk_id})-[m:MENTIONS]->(:Entity) RETURN count(m) AS n",
            chunk_id=chunk_id,
        ).single()["n"]
    assert n_entities == 1
    assert n_mentions == 1  # MERGE dedupes within the chunk


async def test_write_entities_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Re-running write_entities on the same chunk_mentions does NOT
    duplicate :Entity nodes or :MENTIONS edges."""
    chunk_id = _seed_chunk(kg_client, SYNTHETIC_DOC_ID)
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, SYNTHETIC_MENTIONS)])
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, SYNTHETIC_MENTIONS)])

    with kg_client.driver.session() as session:
        n_entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        n_mentions = session.run(
            "MATCH (c:Chunk {chunk_id: $chunk_id})-[m:MENTIONS]->(:Entity) RETURN count(m) AS n",
            chunk_id=chunk_id,
        ).single()["n"]
    assert n_entities == 3
    # 4 mentions → 3 distinct (chunk, entity) pairs since "Aspirin" +
    # "aspirin" share a chunk + key and the :MENTIONS MERGE dedupes.
    assert n_mentions == 3


async def test_delete_entities_by_doc_id_orphan_gcs_entities(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """delete_entities_by_doc_id drops :MENTIONS from this doc's
    chunks AND garbage-collects :Entity nodes that have no remaining
    inbound :MENTIONS edges from any chunk."""
    chunk_id = _seed_chunk(kg_client, SYNTHETIC_DOC_ID)
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, SYNTHETIC_MENTIONS)])

    ok = await kg_client.delete_entities_by_doc_id(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        n_entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        n_mentions = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id})-[m:MENTIONS]->() RETURN count(m) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
    # All 3 entities had only this doc's mentions → all gone.
    assert n_entities == 0
    assert n_mentions == 0


async def test_delete_entities_gcs_orphan_with_outgoing_canonical_to_edge(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """An orphan :Entity that still carries an OUTGOING :CANONICAL_TO edge
    (written by the L7 ontology-linking layer) must STILL be GC'd.

    The orphan check only looks at INCOMING :MENTIONS, so once L7 has
    canonicalised an entity, a plain DELETE would raise "Cannot delete node,
    because it still has relationships" and abort the whole re-ingest/delete.
    DETACH DELETE removes the entity + its :CANONICAL_TO edge while leaving the
    shared :OntologyTerm at the far end intact. Regression test for the
    plain-DELETE bug.
    """
    term_id = "MESH:D001943"
    chunk_id = _seed_chunk(kg_client, SYNTHETIC_DOC_ID)
    await kg_client.write_entities(
        SYNTHETIC_DOC_ID,
        [(chunk_id, [Mention(raw_text="BRCA1", entity_type="gene", offset=0)])],
    )
    # Simulate the L7 canonical link: (:Entity)-[:CANONICAL_TO]->(:OntologyTerm).
    with kg_client.driver.session() as session:
        session.run(
            "MATCH (e:Entity {key: 'brca1', entity_type: 'gene'}) "
            "MERGE (t:OntologyTerm {id: $term_id}) "
            "SET e.canonicalised = true "
            "MERGE (e)-[:CANONICAL_TO]->(t)",
            term_id=term_id,
        )

    # Deleting the doc's mentions orphans BRCA1 (no incoming :MENTIONS) while it
    # still points at the term. Plain DELETE errors here; DETACH DELETE must GC
    # the entity and keep the term.
    ok = await kg_client.delete_entities_by_doc_id(SYNTHETIC_DOC_ID)
    assert ok is None

    with kg_client.driver.session() as session:
        n_entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
        n_terms = session.run(
            "MATCH (t:OntologyTerm {id: $term_id}) RETURN count(t) AS n", term_id=term_id
        ).single()["n"]
    assert n_entities == 0  # orphan GC'd despite its outgoing :CANONICAL_TO edge
    assert n_terms == 1  # the shared ontology term survives (DETACH deletes only the edge)


async def test_delete_entities_preserves_entities_shared_with_other_docs(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """When :Entity has remaining inbound :MENTIONS from OTHER docs'
    chunks, deletion of this doc's mentions must NOT GC the entity."""
    other_doc = "integ-doc-l6-other"
    chunk_a = _seed_chunk(kg_client, SYNTHETIC_DOC_ID, 0)
    chunk_b = _seed_chunk(kg_client, other_doc, 0)

    shared = [
        Mention(raw_text="aspirin", entity_type="drug", offset=0),
    ]
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_a, shared)])
    await kg_client.write_entities(other_doc, [(chunk_b, shared)])

    # Delete only SYNTHETIC_DOC_ID's mentions.
    await kg_client.delete_entities_by_doc_id(SYNTHETIC_DOC_ID)

    with kg_client.driver.session() as session:
        n_entities = session.run("MATCH (e:Entity {key: 'aspirin'}) RETURN count(e) AS n").single()[
            "n"
        ]
        n_mentions_other = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id})-[m:MENTIONS]->(:Entity) RETURN count(m) AS n",
            doc_id=other_doc,
        ).single()["n"]
    assert n_entities == 1  # survived because other_doc still mentions it
    assert n_mentions_other == 1


async def test_write_entities_dedupes_same_entity_across_docs(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """CROSS-DOC DEDUP: the same (key, type) mentioned from TWO docs collapses
    to ONE shared :Entity node (MERGE on the composite key), with a :MENTIONS
    edge from EACH doc's chunk — the property the L9 cross-doc pass relies on.
    """
    other_doc = "integ-doc-l6-other"
    chunk_a = _seed_chunk(kg_client, SYNTHETIC_DOC_ID, 0)
    chunk_b = _seed_chunk(kg_client, other_doc, 0)
    shared = [Mention(raw_text="Metformin", entity_type="drug", offset=0)]
    await kg_client.write_entities(SYNTHETIC_DOC_ID, [(chunk_a, shared)])
    await kg_client.write_entities(other_doc, [(chunk_b, shared)])

    with kg_client.driver.session() as session:
        n_nodes = session.run(
            "MATCH (e:Entity {key: 'metformin', entity_type: 'drug'}) RETURN count(e) AS n"
        ).single()["n"]
        mentioning_docs = sorted(
            r["doc_id"]
            for r in session.run(
                "MATCH (c:Chunk)-[:MENTIONS]->(:Entity {key: 'metformin'}) "
                "RETURN DISTINCT c.doc_id AS doc_id"
            )
        )
    assert n_nodes == 1  # ONE shared node, not one per doc
    assert mentioning_docs == sorted([SYNTHETIC_DOC_ID, other_doc])  # both docs cite it
