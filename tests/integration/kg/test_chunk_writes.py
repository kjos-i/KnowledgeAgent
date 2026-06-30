"""Integration tests for `kg/chunk_writes` (L5).

Exercises the per-document :Chunk write + delete primitives against
a real Neo4j test instance:

  - `write_chunks(doc_id, chunks, main_label, sub_label)`: MERGEs the
    focal node + creates one :Chunk per `chunks` entry + :PART_OF
    edges to the focal.
  - `delete_chunks_by_doc_id(doc_id)`: wipes every :Chunk for the doc.

Identity-only writes — :Chunk nodes carry chunk_id, doc_id,
chunk_index, section, page, content_type. Text + embeddings stay in
LanceDB.

Manual interactive counterpart: `scripts/smoke_kg_l1_l5.py` (covers
the L5 step alongside L1-L4) and `scripts/smoke_lancedb.py` (covers
the LanceDB side of chunk identity).

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from knowledge_agent.ingestion.ids import make_chunk_id

SYNTHETIC_DOC_ID = "integ-doc-chunks-001"

pytestmark = pytest.mark.integration


@dataclass
class _SyntheticChunk:
    """Duck-typed stand-in for `ingestion.parse.ParsedChunk`."""

    chunk_index: int
    section: str | None
    page: int | None
    content_type: str = "text"


SYNTHETIC_CHUNKS = [
    _SyntheticChunk(0, "Introduction", 1, "text"),
    _SyntheticChunk(1, "Methods", 2, "text"),
    _SyntheticChunk(2, None, 3, "figure"),
]


async def test_write_chunks_creates_focal_and_chunks_with_part_of_edges(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """write_chunks MERGEs the focal :Document:Paper + N :Chunk nodes
    + N :PART_OF edges to the focal."""
    ok = await kg_client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        # Focal :Document:Paper with in_corpus flag.
        focal = session.run(
            "MATCH (d:Document:Paper {doc_id: $doc_id}) "
            "RETURN d.in_corpus AS flag",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        assert focal is not None
        assert focal["flag"] is True

        # 3 :Chunk nodes with the expected chunk_id pattern + properties.
        chunks = list(session.run(
            "MATCH (c:Chunk)-[:PART_OF]->(d:Document {doc_id: $doc_id}) "
            "RETURN c.chunk_id AS chunk_id, c.chunk_index AS idx, "
            "c.section AS section, c.page AS page, "
            "c.content_type AS ctype ORDER BY c.chunk_index",
            doc_id=SYNTHETIC_DOC_ID,
        ))
    assert len(chunks) == 3
    assert chunks[0]["chunk_id"] == make_chunk_id(SYNTHETIC_DOC_ID, 0)
    assert chunks[0]["section"] == "Introduction"
    assert chunks[0]["page"] == 1
    assert chunks[1]["chunk_id"] == make_chunk_id(SYNTHETIC_DOC_ID, 1)
    assert chunks[1]["section"] == "Methods"
    assert chunks[2]["section"] is None
    assert chunks[2]["ctype"] == "figure"


async def test_write_chunks_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Re-writing the same chunks doesn't duplicate nodes — MERGE on
    chunk_id protects this."""
    await kg_client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )
    await kg_client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )

    with kg_client.driver.session() as session:
        n_chunks = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        n_part_of = session.run(
            "MATCH (:Chunk {doc_id: $doc_id})-[r:PART_OF]->(:Document) "
            "RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
    assert n_chunks == 3
    assert n_part_of == 3


async def test_write_chunks_with_empty_list_is_noop(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """An empty chunks list is a documented no-op: returns True, no
    focal created, no chunks created. The caller already wrote (or
    will write) the focal via a separate path."""
    ok = await kg_client.write_chunks(SYNTHETIC_DOC_ID, [], "Document", "Paper")
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        focal = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        n_chunks = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]

    assert focal is None
    assert n_chunks == 0


async def test_delete_chunks_wipes_only_target_doc(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """delete_chunks_by_doc_id removes :Chunk nodes ONLY for the
    requested doc. Other docs' chunks survive."""
    other_doc = "integ-doc-chunks-other"
    await kg_client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )
    await kg_client.write_chunks(
        other_doc, SYNTHETIC_CHUNKS, "Document", "Paper"
    )

    ok = await kg_client.delete_chunks_by_doc_id(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        n_target = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        n_other = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN count(c) AS n",
            doc_id=other_doc,
        ).single()["n"]
    assert n_target == 0
    assert n_other == 3


async def test_write_chunks_preserves_existing_focal_labels(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """When a previous write (e.g. write_citations) created the focal
    with :Document:Paper, a later write_chunks call must NOT strip
    the :Paper sub-label — ON CREATE clauses fire only on creation."""
    # First: create focal via write_citations (sets :Document:Paper).
    work: dict[str, Any] = {
        "id": "https://openalex.org/W9990000001",
        "doi": "https://doi.org/10.9990/x",
        "referenced_works": [],
    }
    await kg_client.write_citations(SYNTHETIC_DOC_ID, work)
    # Then: write_chunks should not clobber the :Paper label.
    await kg_client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )

    with kg_client.driver.session() as session:
        labels = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN labels(d) AS lbls",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["lbls"]
    assert "Document" in labels
    assert "Paper" in labels
