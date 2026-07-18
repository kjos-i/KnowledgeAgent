"""KG writes derived from OpenAlex `work` payloads.

Each function here populates one layer of the knowledge graph from a
single OpenAlex work JSON. They are module-level functions (not methods)
so the per-layer code stays focused and `kg/client.py` stays small as
more layers land. Future layer-specific writes (chunks for L5, entities
for L6) follow the same module-per-concern pattern.

`Neo4jClient` in `kg/client.py` exposes each function as a method via a
1-line wrapper - callers still do `client.write_citations(...)` and the
public API is unchanged.

All functions follow the typed-errors contract:
- Raise `ValueError` when called with no `doc_id`.
- Propagate the original Cypher / driver exception on failure; the
  orchestrator boundary catches and stores on the matching
  `_error: ErrorDetail | None` result field.
- Return `None` on success, including the no-data-to-write case
  (no references, no authorships, no venue, no topics).

All functions also assume the focal `:Document` already exists. The
ingestion pipeline calls `write_citations` first, which creates the focal,
then the others can MATCH it by `doc_id`.
"""

import logging
from typing import Any

from knowledge_agent.kg.schema import (
    ABOUT_TOPIC_REL,
    AUTHOR_LABEL,
    AUTHORED_REL,
    CITES_REL,
    DOCUMENT_LABEL,
    PAPER_LABEL,
    PUBLISHED_IN_REL,
    TOPIC_LABEL,
    VENUE_LABEL,
)

logger = logging.getLogger(__name__)


# ---- citations (L1) ----


async def write_citations(client, doc_id: str, work: dict[str, Any]) -> None:
    """Write the focal :Document + its referenced works + :CITES edges.

    From a locally-generated `doc_id` (SHA-256 of the file bytes) and one
    OpenAlex work JSON:

      1. MERGE the focal :Document by `doc_id` (the per-file identity, so
         two files that resolve to the same work never collapse onto one
         node or clobber each other's `doc_id`). When `openalex_id` is
         known, additionally, in the same atomic write:
           (a) absorb any pre-existing *shadow* for that `openalex_id` -
               re-point its incoming :CITES onto the focal and delete it
               (the cited-then-now-ingested upgrade), and
           (b) claim the `openalex_id` on the focal, but ONLY when no
               OTHER :Document already holds it. Both `doc_id` and
               `openalex_id` are UNIQUE; a rival holder is a duplicate
               corpus file, so the focal keeps all its data and simply
               forgoes the id rather than clobber or violate a constraint.
         The focal ends up with `doc_id` + `in_corpus = true` (+ `doi`
         when known, + `openalex_id` when known and free).
      2. UNWIND-MERGE shadow :Document nodes for each referenced work,
         keyed on `openalex_id` (all we know for citations). ON CREATE
         only - never downgrade an existing corpus doc to shadow.
      3. UNWIND-MERGE :CITES edges from focal (matched by doc_id) ->
         each referenced doc (matched by openalex_id).

    Three Cypher round-trips total when there are references (focal +
    shadow-MERGE + CITES-edges); one when there are none (focal only).

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: write_citations called with no doc_id")
    openalex_id = _extract_openalex_id(work.get("id"))
    doi = _clean_doi_for_storage(work.get("doi"))
    references: list[str] = []
    for ref in work.get("referenced_works", []) or []:
        ref_id = _extract_openalex_id(ref)
        if ref_id:
            references.append(ref_id)

    async with client.driver.session() as session:
        # 1. Focal document, always keyed by doc_id (the per-file
        #    identity - never clobbered, never collapses two files).
        #    - openalex_id known: MERGE by doc_id, then in the SAME atomic
        #      write (a) absorb any same-openalex_id shadow (re-point its
        #      :CITES onto the focal, delete it) and (b) claim openalex_id
        #      only if no other :Document holds it (both ids are UNIQUE; a
        #      rival holder is a duplicate corpus file).
        #    - openalex_id unknown (non-scholarly doc): MERGE by doc_id.
        #    Always SET in_corpus = true; SET doi when known.
        if openalex_id:
            params: dict[str, Any] = {
                "openalex_id": openalex_id,
                "doc_id": doc_id,
            }
            focal_set = [
                f"d:{PAPER_LABEL}",
                "d.in_corpus = true",
            ]
            if doi:
                focal_set.append("d.doi = $doi")
                params["doi"] = doi
            await session.run(
                f"MERGE (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
                f"SET {', '.join(focal_set)} "
                f"WITH d "
                # Absorb a same-openalex_id shadow (cited-then-ingested):
                # move its incoming :CITES onto the focal, delete it, so
                # the focal can hold the UNIQUE openalex_id. Unit CALL
                # subquery - a no-shadow MATCH just leaves the focal as-is.
                f"CALL {{ WITH d "
                f"MATCH (shadow:{DOCUMENT_LABEL} {{openalex_id: $openalex_id}}) "
                f"WHERE shadow.in_corpus = false AND elementId(shadow) <> elementId(d) "
                f"OPTIONAL MATCH (citer)-[:{CITES_REL}]->(shadow) "
                f"WITH d, shadow, collect(citer) AS citers "
                f"FOREACH (c IN citers | MERGE (c)-[:{CITES_REL}]->(d)) "
                f"DETACH DELETE shadow }} "
                f"WITH d "
                # Claim openalex_id only when no OTHER node holds it
                # (shadows already absorbed; a rival is a duplicate file).
                # Bare importing `WITH d`, THEN a filtering `WITH d WHERE`
                # - an importing WITH cannot itself carry a WHERE.
                f"CALL {{ WITH d "
                f"WITH d WHERE NOT EXISTS {{ "
                f"MATCH (other:{DOCUMENT_LABEL} {{openalex_id: $openalex_id}}) "
                f"WHERE elementId(other) <> elementId(d) }} "
                f"SET d.openalex_id = $openalex_id }}",
                **params,
            )
        else:
            params = {"doc_id": doc_id}
            set_clauses = [
                f"d:{PAPER_LABEL}",
                "d.in_corpus = true",
            ]
            if doi:
                set_clauses.append("d.doi = $doi")
                params["doi"] = doi
            await session.run(
                f"MERGE (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) SET {', '.join(set_clauses)}",
                **params,
            )

        if not references:
            logger.info("KG: wrote document %s with 0 citations", doc_id)
            return

        # 2. Shadow documents. ON CREATE only - don't downgrade an
        #    existing corpus document to shadow when it's cited from
        #    this paper. Shadows come from referenced_works, so they
        #    get the :Paper subtype label too.
        await session.run(
            f"UNWIND $cited_ids AS cid "
            f"MERGE (c:{DOCUMENT_LABEL} {{openalex_id: cid}}) "
            f"ON CREATE SET c:{PAPER_LABEL}, c.in_corpus = false",
            cited_ids=references,
        )

        # 3. Citation edges. Focal matched by doc_id (just SET above),
        #    shadows by openalex_id.
        await session.run(
            f"MATCH (p:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"UNWIND $cited_ids AS cid "
            f"MATCH (c:{DOCUMENT_LABEL} {{openalex_id: cid}}) "
            f"MERGE (p)-[:{CITES_REL}]->(c)",
            doc_id=doc_id,
            cited_ids=references,
        )
    logger.info("KG: wrote document %s with %d citations", doc_id, len(references))


# ---- authorships (L2) ----


async def write_authorships(client, doc_id: str, work: dict[str, Any]) -> None:
    """Write :Author nodes + :AUTHORED edges from one OpenAlex work.

    From `work["authorships"]`:

      1. UNWIND-MERGE :Author nodes keyed by OpenAlex author ID, with
         unconditional `SET display_name` so a name correction from
         OpenAlex propagates on next ingest.
      2. MATCH the focal :Document by `doc_id`, UNWIND-MATCH each
         :Author, MERGE the edge :AUTHORED with `position` and
         `is_corresponding` as edge properties.

    Two round trips total, regardless of author count. Skips authorships
    with no `author` or no `author.id` (defensive). No-op success when
    no usable authorships are found.

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.

    Assumes the focal :Document already exists - `write_citations` writes
    it first; calling this method without that ordering would create
    zero edges (MATCH on a missing doc yields nothing).
    """
    if not doc_id:
        raise ValueError("KG: write_authorships called with no doc_id")

    authorships: list[dict[str, Any]] = []
    for au in work.get("authorships", []) or []:
        author = au.get("author") or {}
        author_id = _extract_openalex_id(author.get("id"))
        if not author_id:
            continue
        authorships.append(
            {
                "openalex_id": author_id,
                "display_name": author.get("display_name"),
                "position": au.get("author_position"),
                "is_corresponding": bool(au.get("is_corresponding")),
            }
        )

    if not authorships:
        logger.info("KG: write_authorships found no authorships for %s", doc_id)
        return

    async with client.driver.session() as session:
        # 1. Author nodes - MERGE + SET display_name so a later
        #    OpenAlex correction (e.g. expanded full name) wins on
        #    subsequent ingests.
        await session.run(
            f"UNWIND $authorships AS a "
            f"MERGE (au:{AUTHOR_LABEL} {{openalex_id: a.openalex_id}}) "
            f"SET au.display_name = a.display_name",
            authorships=authorships,
        )
        # 2. AUTHORED edges. The focal :Document must already exist
        #    (write_citations runs before this); MATCH on a missing
        #    doc would silently produce zero edges.
        await session.run(
            f"MATCH (p:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"UNWIND $authorships AS a "
            f"MATCH (au:{AUTHOR_LABEL} {{openalex_id: a.openalex_id}}) "
            f"MERGE (au)-[r:{AUTHORED_REL}]->(p) "
            f"SET r.position = a.position, "
            f"r.is_corresponding = a.is_corresponding",
            doc_id=doc_id,
            authorships=authorships,
        )
    logger.info(
        "KG: wrote authorships for %s (%d authors)",
        doc_id,
        len(authorships),
    )


# ---- venue (L3) ----


async def write_venue(client, doc_id: str, work: dict[str, Any]) -> None:
    """Write the :Venue node + :PUBLISHED_IN edge from one OpenAlex work.

    From `work["primary_location"]["source"]`:

      1. MERGE the :Venue node keyed by `openalex_id`, SET name + type
         + issn_l (the canonical ISSN).
      2. MERGE the :PUBLISHED_IN edge from the focal :Document to the
         venue. No edge properties (the venue itself carries name + type).

    One Cypher round trip. No-op success when the work has no usable
    venue (some documents have no `primary_location` or no `source`).

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.

    Assumes the focal :Document already exists - `write_citations` writes
    it first; calling this method without that ordering would yield no
    edge (MATCH on a missing focal returns nothing).
    """
    if not doc_id:
        raise ValueError("KG: write_venue called with no doc_id")

    source = (work.get("primary_location") or {}).get("source") or {}
    venue_openalex_id = _extract_openalex_id(source.get("id"))
    if not venue_openalex_id:
        logger.info("KG: write_venue found no venue for %s", doc_id)
        return

    params: dict[str, Any] = {
        "doc_id": doc_id,
        "openalex_id": venue_openalex_id,
        "name": source.get("display_name"),
        "venue_type": source.get("type"),
        "issn": source.get("issn_l"),
    }
    async with client.driver.session() as session:
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"MERGE (v:{VENUE_LABEL} {{openalex_id: $openalex_id}}) "
            f"SET v.name = $name, v.type = $venue_type, "
            f"v.issn = $issn "
            f"MERGE (d)-[:{PUBLISHED_IN_REL}]->(v)",
            **params,
        )
    logger.info("KG: wrote venue %r for doc %s", venue_openalex_id, doc_id)


# ---- topics (L4) ----


async def write_topics(client, doc_id: str, work: dict[str, Any]) -> None:
    """Write :Topic nodes + :ABOUT_TOPIC edges from one OpenAlex work.

    From `work["topics"]`:

      1. UNWIND-MERGE :Topic nodes keyed by `openalex_id`, with an
         unconditional `SET display_name` so an OpenAlex relabel of a
         topic propagates on the next ingest.
      2. MATCH the focal :Document by `doc_id`, UNWIND-MATCH each :Topic,
         MERGE `:ABOUT_TOPIC` edges with `score` (0-1) as edge property.

    Two round trips total (or zero when there are no topics). Skips
    topics without a usable `openalex_id` (defensive). No-op success
    when no usable topics are found.

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.

    Assumes the focal :Document already exists - `write_citations` writes
    it first.
    """
    if not doc_id:
        raise ValueError("KG: write_topics called with no doc_id")

    topics: list[dict[str, Any]] = []
    for t in work.get("topics", []) or []:
        topic_id = _extract_openalex_id(t.get("id"))
        if not topic_id:
            continue
        topics.append(
            {
                "openalex_id": topic_id,
                "display_name": t.get("display_name"),
                "score": t.get("score"),
            }
        )

    if not topics:
        logger.info("KG: write_topics found no topics for %s", doc_id)
        return

    async with client.driver.session() as session:
        # 1. Topic nodes - MERGE + SET display_name so an OpenAlex
        #    relabel wins on the next ingest.
        await session.run(
            f"UNWIND $topics AS t "
            f"MERGE (n:{TOPIC_LABEL} {{openalex_id: t.openalex_id}}) "
            f"SET n.display_name = t.display_name",
            topics=topics,
        )
        # 2. ABOUT_TOPIC edges with score (per-paper relevance).
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"UNWIND $topics AS t "
            f"MATCH (n:{TOPIC_LABEL} {{openalex_id: t.openalex_id}}) "
            f"MERGE (d)-[r:{ABOUT_TOPIC_REL}]->(n) "
            f"SET r.score = t.score",
            doc_id=doc_id,
            topics=topics,
        )
    logger.info("KG: wrote topics for %s (%d topics)", doc_id, len(topics))


# ---- delete (per-doc wipe + GC orphans across L1-L4) ----


async def delete_doc(client, doc_id: str) -> None:
    """Wipe a paper's L1-L4 KG data: focal :Document + edges + GC orphans.

    Steps:
      1. DETACH DELETE the focal :Document for `doc_id`. This removes the
         node AND all its edges (:CITES outgoing, :AUTHORED incoming,
         :PUBLISHED_IN outgoing, :ABOUT_TOPIC outgoing).
      2. GC orphan :Author / :Topic / :Venue / shadow :Document nodes -
         entities the deleted paper's edges were the LAST connection to.
         Shared entities (authors of other papers, etc.) remain because
         they still have edges from those other papers.

    Called by the ingestion pipeline before the L1-L4 writes, so re-ingest
    of the same doc produces a clean state - no orphan edges, no orphan
    nodes. Mirrors the LanceDB `delete_chunks_by_doc_id` step in spirit.

    Fires unconditionally during ingestion (not gated by metadata
    resolution): if a previous ingest of this doc resolved OpenAlex but
    the current one didn't, we still want to wipe the stale L1-L4 data.

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: delete_doc called with no doc_id")

    async with client.driver.session() as session:
        # 1. Wipe focal + its edges in one shot.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DETACH DELETE d",
            doc_id=doc_id,
        )
        # 2. GC orphans. Each query targets one label and uses the
        # WHERE-NOT pattern to find disconnected nodes. Plain DELETE
        # (not DETACH DELETE) because the WHERE clause already
        # guarantees no edges - if a new edge type is added later
        # without updating the WHERE, DELETE will error so the bug
        # surfaces instead of being silently masked.
        await session.run(f"MATCH (a:{AUTHOR_LABEL}) WHERE NOT (a)-[:{AUTHORED_REL}]->() DELETE a")
        await session.run(
            f"MATCH (t:{TOPIC_LABEL}) WHERE NOT ()-[:{ABOUT_TOPIC_REL}]->(t) DELETE t"
        )
        await session.run(
            f"MATCH (v:{VENUE_LABEL}) WHERE NOT ()-[:{PUBLISHED_IN_REL}]->(v) DELETE v"
        )
        # Shadow documents: in_corpus=false AND nothing cites them.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL}) "
            f"WHERE d.in_corpus = false AND NOT ()-[:{CITES_REL}]->(d) "
            f"DELETE d"
        )
    logger.info("KG: delete_doc cleared %s + GC'd orphans", doc_id)


async def delete_doc_l1_l4_edges(client, doc_id: str) -> None:
    """Wipe L1-L4 edges for `doc_id` while keeping the focal `:Document`
    and other layers intact.

    Unlike `delete_doc` (full focal DETACH DELETE, which collateral-damages
    `:PART_OF` from `:Chunk`s and orphans the chunks from the doc), this
    surgical wipe is for partial ops like `pipeline.resolve_openalex`
    that need to refresh L1-L4 metadata WITHOUT touching L5 chunks.

    Removes four edge types incident to the focal `:Document`:
      - outgoing `:CITES`           (L1)
      - incoming `:AUTHORED`        (L2)
      - outgoing `:PUBLISHED_IN`    (L3)
      - outgoing `:ABOUT_TOPIC`     (L4)

    Then GC orphan `:Author` / `:Topic` / `:Venue` / shadow `:Document`
    nodes - same final-state guarantee as `delete_doc` for those labels.

    Preserved:
      - The focal `:Document` node itself (properties unchanged; writes
        downstream MERGE-update them).
      - `:PART_OF` edges from `:Chunk`s to the focal.
      - `:Chunk` and `:Entity` nodes plus their `:MENTIONS` / `:CANONICAL_TO`
        edges (untouched).

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: delete_doc_l1_l4_edges called with no doc_id")

    async with client.driver.session() as session:
        # 1. Outgoing :CITES from this doc.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}})-[r:{CITES_REL}]->() DELETE r",
            doc_id=doc_id,
        )
        # 2. Incoming :AUTHORED to this doc.
        await session.run(
            f"MATCH ()-[r:{AUTHORED_REL}]->(d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DELETE r",
            doc_id=doc_id,
        )
        # 3. Outgoing :PUBLISHED_IN from this doc.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}})-[r:{PUBLISHED_IN_REL}]->() DELETE r",
            doc_id=doc_id,
        )
        # 4. Outgoing :ABOUT_TOPIC from this doc.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}})-[r:{ABOUT_TOPIC_REL}]->() DELETE r",
            doc_id=doc_id,
        )
        # 5. GC orphans across the L1-L4 reference vocab. Same WHERE-NOT
        # pattern as `delete_doc` so the post-state guarantee matches.
        await session.run(f"MATCH (a:{AUTHOR_LABEL}) WHERE NOT (a)-[:{AUTHORED_REL}]->() DELETE a")
        await session.run(
            f"MATCH (t:{TOPIC_LABEL}) WHERE NOT ()-[:{ABOUT_TOPIC_REL}]->(t) DELETE t"
        )
        await session.run(
            f"MATCH (v:{VENUE_LABEL}) WHERE NOT ()-[:{PUBLISHED_IN_REL}]->(v) DELETE v"
        )
        # Shadow docs: in_corpus=false AND nothing cites them.
        # The :CITES edges from this doc were removed in step 1, so any
        # shadow doc that was ONLY cited from here is now collectable.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL}) "
            f"WHERE d.in_corpus = false AND NOT ()-[:{CITES_REL}]->(d) "
            f"DELETE d"
        )
    logger.info("KG: delete_doc_l1_l4_edges wiped %s + GC'd orphans", doc_id)


# ---- OpenAlex payload helpers ----


def _extract_openalex_id(value: str | None) -> str | None:
    """Bare OpenAlex work ID from a URL or already-bare ID.

    OpenAlex returns `id: "https://openalex.org/W1234"`; we store the bare
    `W1234` form so it's compact and joins cross-store cleanly.
    """
    if not value:
        return None
    value = value.strip()
    return value.rsplit("/", 1)[-1] if "/" in value else value


def _clean_doi_for_storage(doi_url_or_bare: str | None) -> str | None:
    """Normalize a DOI for KG storage (strip URL prefix, lowercase).

    The same DOI must be stored identically on both sides - the join field
    has to compare equal byte-for-byte across LanceDB and Neo4j.
    """
    if not doi_url_or_bare:
        return None
    cleaned = doi_url_or_bare.replace("https://doi.org/", "").strip()
    return cleaned.rstrip(".,;)").lower()
