"""Integration tests for `kg/openalex_writes` (L1-L4).

Exercises the four OpenAlex-derived KG layers against a real Neo4j
test instance:

  - L1 `write_citations`: focal :Document:Paper + cited :Document:Paper
    shadows + :CITES edges.
  - L2 `write_authorships`: :Author nodes + :AUTHORED edges (with
    position + is_corresponding properties).
  - L3 `write_venue`: :Venue node + :PUBLISHED_IN edge.
  - L4 `write_topics`: :Topic nodes + :ABOUT_TOPIC edges (with score
    property).

Plus the surgical cleanup primitives:
  - `delete_doc`: full L1-L4 wipe with orphan GC.
  - `delete_doc_l1_l4_edges`: edges-only variant for partial re-resolve.

Tests assert post-write graph state via Cypher reads — counts,
property values, and relationship topology.

Manual interactive counterpart: `scripts/smoke_kg_l1_l5.py` —
same scenarios, but pauses for inspection in Neo4j Desktop and
prints state instead of asserting. Use the smoke when you want to
eyeball what landed; use this file for regression catching.

Requires the test Neo4j instance from `.env.test` per
[[test-instance-setup]]. Skipped by default; opt in via
`pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

# Synthetic OpenAlex-shaped work covering L1-L4. IDs use 999 prefixes
# so they can't collide with any real OpenAlex entity.
SYNTHETIC_DOC_ID = "integ-doc-openalex-001"

SYNTHETIC_WORK: dict[str, Any] = {
    "id": "https://openalex.org/W9990000001",
    "doi": "https://doi.org/10.9990/integration-test-1",
    "referenced_works": [
        "https://openalex.org/W9990000100",
        "https://openalex.org/W9990000101",
        "https://openalex.org/W9990000102",
    ],
    "authorships": [
        {
            "author_position": "first",
            "author": {
                "id": "https://openalex.org/A9990000001",
                "display_name": "Jane Integration",
            },
            "is_corresponding": True,
        },
        {
            "author_position": "middle",
            "author": {
                "id": "https://openalex.org/A9990000002",
                "display_name": "Bob Synthetic",
            },
            "is_corresponding": False,
        },
        {
            "author_position": "last",
            "author": {
                "id": "https://openalex.org/A9990000003",
                "display_name": "Carol Stub",
            },
            "is_corresponding": False,
        },
    ],
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S9990000001",
            "display_name": "Journal of Integration Tests",
            "type": "journal",
            "issn_l": "9999-0001",
        },
    },
    "topics": [
        {
            "id": "https://openalex.org/T9990000001",
            "display_name": "Integration Topic Alpha",
            "score": 0.95,
        },
        {
            "id": "https://openalex.org/T9990000002",
            "display_name": "Integration Topic Beta",
            "score": 0.71,
        },
    ],
}

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# L1: write_citations
# ---------------------------------------------------------------------------


def test_write_citations_creates_focal_and_cited_documents(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L1 produces the focal :Document:Paper + N cited shadow
    :Document:Paper nodes + N :CITES edges."""
    ok = kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        # Focal document.
        focal = session.run(
            "MATCH (d:Document:Paper {doc_id: $doc_id}) "
            "RETURN d.openalex_id AS oa_id, d.doi AS doi",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        assert focal is not None
        assert focal["oa_id"] == "W9990000001"

        # Cited shadow documents — one per referenced_work.
        cited = list(session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[:CITES]->(c:Document) "
            "RETURN c.openalex_id AS oa_id ORDER BY oa_id",
            doc_id=SYNTHETIC_DOC_ID,
        ))
        assert len(cited) == 3
        assert {r["oa_id"] for r in cited} == {
            "W9990000100", "W9990000101", "W9990000102",
        }


def test_write_citations_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Re-running write_citations on the same doc must NOT duplicate
    nodes or edges — the MERGE pattern protects this."""
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    with kg_client.driver.session() as session:
        n_cites = session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:CITES]->(:Document) "
            "RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        assert n_cites == 3  # not 6 — MERGE prevented duplication


# ---------------------------------------------------------------------------
# L2: write_authorships
# ---------------------------------------------------------------------------


def test_write_authorships_creates_authors_with_position_and_corresponding(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L2 produces 3 :Author nodes + 3 :AUTHORED edges with
    `position` and `is_corresponding` properties matching the work."""
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        rows = list(session.run(
            "MATCH (a:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) "
            "RETURN a.display_name AS name, r.position AS pos, "
            "r.is_corresponding AS corresp ORDER BY r.position",
            doc_id=SYNTHETIC_DOC_ID,
        ))

    assert len(rows) == 3
    by_position = {r["pos"]: r for r in rows}
    assert by_position["first"]["name"] == "Jane Integration"
    assert by_position["first"]["corresp"] is True
    assert by_position["middle"]["corresp"] is False
    assert by_position["last"]["corresp"] is False


def test_write_authorships_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    with kg_client.driver.session() as session:
        n_authored = session.run(
            "MATCH (:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) "
            "RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        assert n_authored == 3


# ---------------------------------------------------------------------------
# L3: write_venue
# ---------------------------------------------------------------------------


def test_write_venue_creates_venue_and_published_in_edge(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L3 produces 1 :Venue + 1 :PUBLISHED_IN edge."""
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        row = session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[:PUBLISHED_IN]->(v:Venue) "
            "RETURN v.name AS name, v.openalex_id AS oa_id",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
    assert row is not None
    assert row["name"] == "Journal of Integration Tests"
    assert row["oa_id"] == "S9990000001"


# ---------------------------------------------------------------------------
# L4: write_topics
# ---------------------------------------------------------------------------


def test_write_topics_creates_topics_with_score(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L4 produces 2 :Topic + 2 :ABOUT_TOPIC edges, each with a
    `score` property matching the work."""
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = kg_client.write_topics(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        rows = list(session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:ABOUT_TOPIC]->(t:Topic) "
            "RETURN t.display_name AS name, r.score AS score "
            "ORDER BY r.score DESC",
            doc_id=SYNTHETIC_DOC_ID,
        ))

    assert len(rows) == 2
    assert rows[0]["name"] == "Integration Topic Alpha"
    assert rows[0]["score"] == pytest.approx(0.95)
    assert rows[1]["name"] == "Integration Topic Beta"
    assert rows[1]["score"] == pytest.approx(0.71)


# ---------------------------------------------------------------------------
# delete_doc + delete_doc_l1_l4_edges
# ---------------------------------------------------------------------------


def test_delete_doc_wipes_full_l1_l4_with_orphan_gc(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """delete_doc removes the focal :Document AND garbage-collects
    :Author / :Venue / :Topic that no longer have inbound edges."""
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_topics(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    ok = kg_client.delete_doc(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        # Focal gone.
        focal = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        assert focal is None

        # Orphan-GC: authors / venue / topics with no remaining inbound
        # edges should be gone. (Cited shadow :Document nodes are also
        # GC'd by the delete_doc contract.)
        n_authors = session.run(
            "MATCH (a:Author) WHERE a.openalex_id STARTS WITH 'A999' "
            "RETURN count(a) AS n"
        ).single()["n"]
        n_venues = session.run(
            "MATCH (v:Venue) WHERE v.openalex_id STARTS WITH 'S999' "
            "RETURN count(v) AS n"
        ).single()["n"]
        n_topics = session.run(
            "MATCH (t:Topic) WHERE t.openalex_id STARTS WITH 'T999' "
            "RETURN count(t) AS n"
        ).single()["n"]
        assert n_authors == 0
        assert n_venues == 0
        assert n_topics == 0


def test_delete_doc_l1_l4_edges_preserves_focal_and_chunks(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """The surgical edges-only delete drops L1-L4 edges + L1-L4
    shadow nodes (via orphan GC) but PRESERVES the focal :Document
    and any :Chunk nodes hanging off it via :PART_OF.

    This is what `resolve_openalex` uses to re-write metadata without
    orphaning chunks.
    """
    # Seed L1-L4.
    kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    # Seed a chunk so the test verifies it survives.
    with kg_client.driver.session() as session:
        session.run(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=SYNTHETIC_DOC_ID,
            chunk_id=f"{SYNTHETIC_DOC_ID}#0",
        )

    ok = kg_client.delete_doc_l1_l4_edges(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    with kg_client.driver.session() as session:
        # Focal survives.
        focal = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        assert focal is not None
        # Chunk survives.
        chunk = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()
        assert chunk is not None
        # L1 edges gone.
        n_cites = session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:CITES]->() "
            "RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        assert n_cites == 0
        # L2 edges gone.
        n_authored = session.run(
            "MATCH (:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) "
            "RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        ).single()["n"]
        assert n_authored == 0
