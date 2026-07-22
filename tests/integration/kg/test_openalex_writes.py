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

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via
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


async def test_write_citations_creates_focal_and_cited_documents(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L1 produces the focal :Document:Paper + N cited shadow
    :Document:Paper nodes + N :CITES edges."""
    ok = await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        # Focal document.
        result = await session.run(
            "MATCH (d:Document:Paper {doc_id: $doc_id}) "
            "RETURN d.openalex_id AS oa_id, d.doi AS doi",
            doc_id=SYNTHETIC_DOC_ID,
        )
        focal = await result.single()
        assert focal is not None
        assert focal["oa_id"] == "W9990000001"

        # Cited shadow documents — one per referenced_work.
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[:CITES]->(c:Document) "
            "RETURN c.openalex_id AS oa_id ORDER BY oa_id",
            doc_id=SYNTHETIC_DOC_ID,
        )
        cited = [r async for r in result]
        assert len(cited) == 3
        assert {r["oa_id"] for r in cited} == {
            "W9990000100",
            "W9990000101",
            "W9990000102",
        }


async def test_write_citations_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Re-running write_citations on the same doc must NOT duplicate
    nodes or edges — the MERGE pattern protects this."""
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:CITES]->(:Document) RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        )
        n_cites = (await result.single())["n"]
        assert n_cites == 3  # not 6 — MERGE prevented duplication


# ---------------------------------------------------------------------------
# L1 (cont.): focal identity — shadow upgrade / duplicate file /
# resolve-after-pending. These are the fix-#3 scenarios: the focal is
# keyed by doc_id, so two files never collapse or clobber, and the
# openalex_id (UNIQUE) is claimed only when free.
# ---------------------------------------------------------------------------


async def test_write_citations_ingesting_a_cited_work_absorbs_its_shadow(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Cited-then-ingested upgrade: a paper cites work W (creating a shadow);
    later W's OWN file is ingested. The focal MERGE-by-doc_id must ABSORB the
    shadow (re-point its incoming :CITES onto the focal, delete it) — ONE node
    for W, not two, and the citation survives pointing at the in-corpus focal.
    """
    citing_doc_id = "integ-citing-doc"
    target_openalex = "W9990000500"
    target_doc_id = "integ-target-doc"

    # 1. A citing paper references target_openalex -> creates a SHADOW.
    await kg_client.write_citations(
        citing_doc_id,
        {
            "id": "https://openalex.org/W9990000499",
            "referenced_works": [f"https://openalex.org/{target_openalex}"],
        },
    )
    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (s:Document {openalex_id: $oa}) "
            "RETURN s.in_corpus AS in_corpus, s.doc_id AS doc_id",
            oa=target_openalex,
        )
        shadow = await result.single()
    assert shadow is not None
    assert shadow["in_corpus"] is False
    assert shadow["doc_id"] is None

    # 2. Now ingest target_openalex's OWN file (the focal).
    await kg_client.write_citations(
        target_doc_id,
        {"id": f"https://openalex.org/{target_openalex}", "doi": "10.9990/target"},
    )

    async with kg_client.driver.session() as session:
        # Exactly ONE node holds target_openalex — the shadow was absorbed.
        result = await session.run(
            "MATCH (d:Document {openalex_id: $oa}) RETURN count(d) AS n",
            oa=target_openalex,
        )
        assert (await result.single())["n"] == 1
        # ...and it is the upgraded corpus doc (doc_id set, in_corpus flipped).
        result = await session.run(
            "MATCH (d:Document {openalex_id: $oa}) "
            "RETURN d.doc_id AS doc_id, d.in_corpus AS in_corpus",
            oa=target_openalex,
        )
        node = await result.single()
        assert node["doc_id"] == target_doc_id
        assert node["in_corpus"] is True
        # The citing paper's :CITES now points at the single focal.
        result = await session.run(
            "MATCH (:Document {doc_id: $citing})-[:CITES]->(:Document {doc_id: $target}) "
            "RETURN count(*) AS n",
            citing=citing_doc_id,
            target=target_doc_id,
        )
        assert (await result.single())["n"] == 1


async def test_write_citations_two_files_same_work_keeps_both_no_clobber(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Two DIFFERENT files (different doc_id) resolving to the SAME OpenAlex
    work must NOT collapse onto one node or clobber a doc_id (the old bug).
    Both nodes survive with their own doc_id; the UNIQUE openalex_id stays on
    the first (option A) and the second simply forgoes it — no crash."""
    work = {"id": "https://openalex.org/W9990000600", "doi": None}
    first_doc_id = "integ-dup-file-a"
    second_doc_id = "integ-dup-file-b"

    await kg_client.write_citations(first_doc_id, work)
    # Second ingest of the same work as a different file must NOT raise.
    await kg_client.write_citations(second_doc_id, work)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document) WHERE d.doc_id IN [$a, $b] "
            "RETURN d.doc_id AS doc_id, d.openalex_id AS oa ORDER BY d.doc_id",
            a=first_doc_id,
            b=second_doc_id,
        )
        rows = [r async for r in result]
    # Both files survive as their own nodes (no clobber, no collapse).
    assert [r["doc_id"] for r in rows] == [first_doc_id, second_doc_id]
    # openalex_id (UNIQUE) lives on the first; the second forgoes it.
    by_doc = {r["doc_id"]: r["oa"] for r in rows}
    assert by_doc[first_doc_id] == "W9990000600"
    assert by_doc[second_doc_id] is None


async def test_write_citations_resolve_after_pending_no_constraint_violation(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """A doc first ingested WITHOUT openalex (keyed by doc_id) and later
    re-resolved to a work that ALSO exists as a cited shadow must upgrade in
    place — claim the openalex_id + absorb the shadow — with NO doc_id-
    uniqueness violation (the old MERGE-by-openalex + SET doc_id crashed here).
    """
    pending_doc_id = "integ-pending-doc"
    citer_doc_id = "integ-citer-doc"
    target_openalex = "W9990000700"

    # 1. Ingest as non-scholarly: node keyed by doc_id, no openalex_id.
    await kg_client.write_citations(pending_doc_id, {"doi": "10.9990/pending"})
    # 2. A citing paper creates a shadow for the SAME work.
    await kg_client.write_citations(
        citer_doc_id,
        {
            "id": "https://openalex.org/W9990000699",
            "referenced_works": [f"https://openalex.org/{target_openalex}"],
        },
    )
    # 3. Re-resolve the pending doc to that work. Must NOT raise.
    await kg_client.write_citations(
        pending_doc_id, {"id": f"https://openalex.org/{target_openalex}"}
    )

    async with kg_client.driver.session() as session:
        # The pending doc now carries the openalex_id + stays in_corpus.
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc}) "
            "RETURN d.openalex_id AS oa, d.in_corpus AS in_corpus",
            doc=pending_doc_id,
        )
        node = await result.single()
        assert node["oa"] == target_openalex
        assert node["in_corpus"] is True
        # Exactly one node holds it (shadow absorbed, not duplicated).
        result = await session.run(
            "MATCH (d:Document {openalex_id: $oa}) RETURN count(d) AS n",
            oa=target_openalex,
        )
        assert (await result.single())["n"] == 1
        # The citer's citation now points at the upgraded pending doc.
        result = await session.run(
            "MATCH (:Document {doc_id: $citer})-[:CITES]->(:Document {doc_id: $doc}) "
            "RETURN count(*) AS n",
            citer=citer_doc_id,
            doc=pending_doc_id,
        )
        assert (await result.single())["n"] == 1


# ---------------------------------------------------------------------------
# L2: write_authorships
# ---------------------------------------------------------------------------


async def test_write_authorships_creates_authors_with_position_and_corresponding(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L2 produces 3 :Author nodes + 3 :AUTHORED edges with
    `position` and `is_corresponding` properties matching the work."""
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = await kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (a:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) "
            "RETURN a.display_name AS name, r.position AS pos, "
            "r.is_corresponding AS corresp ORDER BY r.position",
            doc_id=SYNTHETIC_DOC_ID,
        )
        rows = [r async for r in result]

    assert len(rows) == 3
    by_position = {r["pos"]: r for r in rows}
    assert by_position["first"]["name"] == "Jane Integration"
    assert by_position["first"]["corresp"] is True
    assert by_position["middle"]["corresp"] is False
    assert by_position["last"]["corresp"] is False


async def test_write_authorships_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        )
        n_authored = (await result.single())["n"]
        assert n_authored == 3


# ---------------------------------------------------------------------------
# L3: write_venue
# ---------------------------------------------------------------------------


async def test_write_venue_creates_venue_and_published_in_edge(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L3 produces 1 :Venue + 1 :PUBLISHED_IN edge."""
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = await kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[:PUBLISHED_IN]->(v:Venue) "
            "RETURN v.name AS name, v.openalex_id AS oa_id",
            doc_id=SYNTHETIC_DOC_ID,
        )
        row = await result.single()
    assert row is not None
    assert row["name"] == "Journal of Integration Tests"
    assert row["oa_id"] == "S9990000001"


# ---------------------------------------------------------------------------
# L4: write_topics
# ---------------------------------------------------------------------------


async def test_write_topics_creates_topics_with_score(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """L4 produces 2 :Topic + 2 :ABOUT_TOPIC edges, each with a
    `score` property matching the work."""
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    ok = await kg_client.write_topics(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:ABOUT_TOPIC]->(t:Topic) "
            "RETURN t.display_name AS name, r.score AS score "
            "ORDER BY r.score DESC",
            doc_id=SYNTHETIC_DOC_ID,
        )
        rows = [r async for r in result]

    assert len(rows) == 2
    assert rows[0]["name"] == "Integration Topic Alpha"
    assert rows[0]["score"] == pytest.approx(0.95)
    assert rows[1]["name"] == "Integration Topic Beta"
    assert rows[1]["score"] == pytest.approx(0.71)


# ---------------------------------------------------------------------------
# delete_doc + delete_doc_l1_l4_edges
# ---------------------------------------------------------------------------


async def test_delete_doc_wipes_full_l1_l4_with_orphan_gc(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """delete_doc removes the focal :Document AND garbage-collects
    :Author / :Venue / :Topic that no longer have inbound edges."""
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_topics(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)

    ok = await kg_client.delete_doc(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        # Focal gone.
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
            doc_id=SYNTHETIC_DOC_ID,
        )
        focal = await result.single()
        assert focal is None

        # Orphan-GC: authors / venue / topics with no remaining inbound
        # edges should be gone. (Cited shadow :Document nodes are also
        # GC'd by the delete_doc contract.)
        result = await session.run(
            "MATCH (a:Author) WHERE a.openalex_id STARTS WITH 'A999' RETURN count(a) AS n"
        )
        n_authors = (await result.single())["n"]
        result = await session.run(
            "MATCH (v:Venue) WHERE v.openalex_id STARTS WITH 'S999' RETURN count(v) AS n"
        )
        n_venues = (await result.single())["n"]
        result = await session.run(
            "MATCH (t:Topic) WHERE t.openalex_id STARTS WITH 'T999' RETURN count(t) AS n"
        )
        n_topics = (await result.single())["n"]
        assert n_authors == 0
        assert n_venues == 0
        assert n_topics == 0


async def test_delete_doc_l1_l4_edges_preserves_focal_and_chunks(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """The surgical edges-only delete drops L1-L4 edges + L1-L4
    shadow nodes (via orphan GC) but PRESERVES the focal :Document
    and any :Chunk nodes hanging off it via :PART_OF.

    This is what `resolve_openalex` uses to re-write metadata without
    orphaning chunks.
    """
    # Seed L1-L4.
    await kg_client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    await kg_client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    # Seed a chunk so the test verifies it survives.
    async with kg_client.driver.session() as session:
        await session.run(
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=SYNTHETIC_DOC_ID,
            chunk_id=f"{SYNTHETIC_DOC_ID}#0",
        )

    ok = await kg_client.delete_doc_l1_l4_edges(SYNTHETIC_DOC_ID)
    assert ok is None  # success: returns None (typed-errors contract)

    async with kg_client.driver.session() as session:
        # Focal survives.
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN d",
            doc_id=SYNTHETIC_DOC_ID,
        )
        focal = await result.single()
        assert focal is not None
        # Chunk survives.
        result = await session.run(
            "MATCH (c:Chunk {doc_id: $doc_id}) RETURN c",
            doc_id=SYNTHETIC_DOC_ID,
        )
        chunk = await result.single()
        assert chunk is not None
        # L1 edges gone.
        result = await session.run(
            "MATCH (d:Document {doc_id: $doc_id})-[r:CITES]->() RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        )
        n_cites = (await result.single())["n"]
        assert n_cites == 0
        # L2 edges gone.
        result = await session.run(
            "MATCH (:Author)-[r:AUTHORED]->(d:Document {doc_id: $doc_id}) RETURN count(r) AS n",
            doc_id=SYNTHETIC_DOC_ID,
        )
        n_authored = (await result.single())["n"]
        assert n_authored == 0
