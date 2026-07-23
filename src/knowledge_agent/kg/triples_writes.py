"""KG writes for L8 triples - typed entity-to-entity relations.

Each L8 edge encodes one LLM-extracted assertion: subject `:Entity` →
predicate (one of the 15 fixed types in `schema.TRIPLE_PREDICATE_RELS`)
→ object `:Entity`, with provenance properties pointing back at the
chunk it came from.

Per-predicate edge types (the predicate IS the edge label):

    (:Entity)-[:INHIBITS    {chunk_id, doc_id, evidence_span}]->(:Entity)
    (:Entity)-[:ACTIVATES   {chunk_id, doc_id, evidence_span}]->(:Entity)
    ...

Idiomatic Neo4j: typed queries become one-hop index-backed traversals
("what inhibits corrosion?" = `MATCH (a)-[:INHIBITS]->(:Entity
{key:"corrosion"})`). Adding a new predicate means appending one constant
to `schema.TRIPLE_PREDICATE_RELS`; the writes here iterate that tuple.

Provenance + edge-collapse policy:
  - ONE edge per chunk assertion. If the LLM extracts the same triple
    from 5 chunks, 5 edges exist (each with its own `chunk_id` +
    `evidence_span`).
  - `evidence_span` stores a short verbatim snippet of chunk text that
    supports the assertion - lets the agent cite WHY it believes the
    triple, not just THAT it exists.

Subject + object must already exist as `:Entity` nodes (the LLM is
prompted with the chunk's L6 entity vocabulary). If a triple references
a missing entity (extractor bug, L6/L8 race), the inner MATCH yields
no rows and the CREATE silently no-ops for that row - logged via the
mismatch count.

Re-ingest discipline:
  - `delete_triples_by_doc_id(doc_id)` wipes all L8 edges with this
    `doc_id`, across all 15 predicate types in one Cypher call.
  - Fires unconditionally inside the ingestion pipeline so re-running
    a doc never leaves stale triples.

`Neo4jClient` exposes these as 1-line delegate methods in
`kg/client.py`, matching the pattern used by `entity_writes` and
`chunk_writes`.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from knowledge_agent.kg.schema import (
    CHUNK_LABEL,
    ENTITY_LABEL,
    MENTIONS_REL,
    TRIPLE_PREDICATE_RELS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedTriple:
    """One (subject, predicate, object) assertion from chunk text.

    Subject and object are `(key, entity_type)` composite identities
    matching `:Entity` NODE KEY. Both `_key` fields MUST already be
    lowercased (matches how L6 stores entity keys); the extractor is
    expected to lowercase before constructing.

    `predicate` must be one of `schema.TRIPLE_PREDICATE_RELS`. Out-of-
    vocabulary predicates are logged + skipped at write time rather
    than raised - lets a flaky LLM run degrade gracefully instead of
    aborting the whole pipeline.

    `evidence_span` is a short verbatim snippet from the chunk that
    supports the assertion. Used for citations + diagnostic display
    (the agent can show "asserted at: 'the subsidy accelerated solar
    adoption...'" when answering).
    """

    subject_key: str
    subject_entity_type: str
    predicate: str
    object_key: str
    object_entity_type: str
    evidence_span: str


async def get_entities_by_chunk(client, doc_id: str) -> dict[str, list[tuple[str, str]]]:
    """Return `{chunk_id: [(entity_key, entity_type), ...], ...}` for a doc.

    Used by `pipeline.backfill_triples` to rebuild each chunk's L6
    entity vocabulary from the existing graph state (rather than
    re-running L6). Each entity contributes one (key, entity_type)
    tuple per chunk that mentions it.

    Returns an empty dict when the doc has no
    `:Chunk-[:MENTIONS]->:Entity` rows (legitimate empty-state).

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: get_entities_by_chunk called with no doc_id")
    out: dict[str, list[tuple[str, str]]] = defaultdict(list)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}})"
            f"-[:{MENTIONS_REL}]->(e:{ENTITY_LABEL}) "
            f"RETURN c.chunk_id AS chunk_id, "
            f"       e.key AS key, "
            f"       e.entity_type AS entity_type",
            doc_id=doc_id,
        )
        records = await result.data()
    for row in records:
        out[row["chunk_id"]].append((row["key"], row["entity_type"]))
    return dict(out)


async def delete_triples_by_doc_id(client, doc_id: str) -> None:
    """Drop L8 edges whose `doc_id` property matches.

    One Cypher MATCH/DELETE that unions all 15 predicate types via the
    `:R1|R2|...` pipe syntax - hits the doc's triples wherever they
    live in the graph, regardless of which predicates the extractor
    happened to produce.

    Idempotent: when no triples exist for the doc, the MATCH yields
    nothing and DELETE is a no-op.

    Called by the ingestion pipeline before `write_triples` so a re-
    extract leaves no stale L8 edges. Also part of `pipeline.delete_doc`.

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: delete_triples_by_doc_id called with no doc_id")
    # Pipe-join all 15 predicate types. The closed list comes from
    # schema.TRIPLE_PREDICATE_RELS so a new predicate added there is
    # picked up here automatically.
    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    async with client.driver.session() as session:
        await session.run(
            f"MATCH ()-[r:{rel_union}]->() WHERE r.doc_id = $doc_id DELETE r",
            doc_id=doc_id,
        )
    logger.info("KG: deleted L8 triples for doc %s", doc_id)


async def write_triples(
    client,
    doc_id: str,
    chunk_triples: list[tuple[str, list[ExtractedTriple]]],
) -> None:
    """Write L8 typed-relation edges for one document.

    Args:
      client:        Neo4jClient (passed through by the delegate method).
      doc_id:        identity stamped on every edge for `delete_*` to
                     find them later. Each row also carries the
                     `chunk_id` it came from.
      chunk_triples: list of (chunk_id, list[ExtractedTriple]) per chunk.
                     Empty inner lists fine (chunk yielded no triples);
                     empty outer list fine too (whole-doc no-op).

    Cypher pattern:
      - Group rows by predicate (one Cypher per predicate that has rows,
        max 15). Necessary because Neo4j cannot set a relationship type
        from a row variable - the type has to be literal in the query.
      - Per predicate: UNWIND rows; MATCH subject + object `:Entity` by
        composite (key, entity_type); CREATE the typed edge with
        chunk_id + doc_id + evidence_span properties.
      - CREATE (not MERGE): the delete pass above wipes all stale L8
        edges first, so the per-chunk uniqueness is naturally enforced
        by the empty-target invariant. CREATE is also faster and
        produces clean "one edge per assertion" semantics.

    If the subject or object entity doesn't exist (L6/L8 race, or LLM
    referenced an out-of-vocabulary key), the inner MATCH yields no
    rows and the CREATE silently no-ops for that row. Counted via the
    final session.run result_summary (Neo4j reports the actual created
    count; mismatch with input row count flags the issue).

    Out-of-vocabulary predicates are dropped + logged before the
    Cypher loop runs - keeps the loop iterating only valid predicates.

    Success no-op for empty `chunk_triples` and all-out-of-vocab cases.

    Raises `ValueError` for an empty `doc_id`; Cypher / driver failures
    propagate.
    """
    if not doc_id:
        raise ValueError("KG: write_triples called with no doc_id")
    if not chunk_triples:
        logger.info("KG: write_triples got no chunk_triples for %s", doc_id)
        return

    # Bucket rows by predicate so we can issue one Cypher per predicate
    # type. predicate->list[row]. Out-of-vocab predicates are dropped
    # here (logged once per unique bad predicate to avoid spam).
    rows_by_pred: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_bad: set[str] = set()
    for chunk_id, triples in chunk_triples:
        for t in triples:
            if t.predicate not in TRIPLE_PREDICATE_RELS:
                if t.predicate not in seen_bad:
                    logger.warning(
                        "KG: write_triples dropping triple with unknown "
                        "predicate %r (subject=%r object=%r)",
                        t.predicate,
                        t.subject_key,
                        t.object_key,
                    )
                    seen_bad.add(t.predicate)
                continue
            rows_by_pred[t.predicate].append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "subject_key": t.subject_key,
                    "subject_entity_type": t.subject_entity_type,
                    "object_key": t.object_key,
                    "object_entity_type": t.object_entity_type,
                    "evidence_span": t.evidence_span,
                }
            )

    if not rows_by_pred:
        logger.info(
            "KG: write_triples found no valid triples across %d chunks for %s",
            len(chunk_triples),
            doc_id,
        )
        return

    total = sum(len(v) for v in rows_by_pred.values())
    async with client.driver.session() as session:
        for predicate, rows in rows_by_pred.items():
            # Predicate is interpolated as a literal here ONLY
            # because it's been validated against the closed
            # TRIPLE_PREDICATE_RELS set above. No user input flows
            # into this f-string.
            cypher = (
                f"UNWIND $rows AS row "
                f"MATCH (s:{ENTITY_LABEL} "
                f"  {{key: row.subject_key, "
                f"   entity_type: row.subject_entity_type}}) "
                f"MATCH (o:{ENTITY_LABEL} "
                f"  {{key: row.object_key, "
                f"   entity_type: row.object_entity_type}}) "
                f"CREATE (s)-[r:{predicate} "
                f"  {{chunk_id: row.chunk_id, doc_id: row.doc_id, "
                f"   evidence_span: row.evidence_span}}]->(o)"
            )
            await session.run(cypher, rows=rows)
    logger.info(
        "KG: wrote %d L8 triples (%d predicates) for doc %s",
        total,
        len(rows_by_pred),
        doc_id,
    )
