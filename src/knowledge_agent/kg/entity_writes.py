"""KG writes for entities (L6).

Each `:Entity` node carries the merge key + type for one extracted concept:
`key` (lowercased text, the MERGE target), `entity_type` (LLM- or NER-
chosen category), and `canonicalised` (always False at L6; flips True
when the L7 ontology-linking layer matches this entity to a
canonical ID).

Each `:MENTIONS` edge from a `:Chunk` to an `:Entity` records that the
chunk's text contained a mention. The edge optionally carries `offset`
(char position in the chunk text - NER adapters populate; LLM leaves
null) and `confidence` (model's score - NER may populate; LLM leaves
null).

Identity model:
  - `:Entity` node is identified by composite `(key, entity_type)` via
    NODE KEY constraint in `schema.py`. Same word with different types
    ("Apple" as GENE vs as COMPANY) stays correctly distinct.
  - `key` is the LOWERCASED form of the extractor's `raw_text`. The
    write path does the lowercasing - extractors return original
    spelling so NER offsets stay valid against chunk text.

Re-ingest discipline:
  - `delete_entities_by_doc_id(doc_id)` removes `:MENTIONS` edges
    sourced from this doc's chunks and GC's any orphan `:Entity` nodes.
  - Fires unconditionally inside the ingestion pipeline so re-running
    the same doc leaves no stale entity data.

`Neo4jClient` exposes these as 1-line delegate methods in `kg/client.py`,
matching the pattern used for `openalex_writes` and `chunk_writes`.
"""

import logging
from typing import Any

from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.kg.schema import (
    CHUNK_LABEL,
    ENTITY_LABEL,
    MENTIONS_REL,
)

logger = logging.getLogger(__name__)


async def delete_entities_by_doc_id(client, doc_id: str) -> None:
    """Drop `:MENTIONS` edges from this doc's chunks + GC orphan `:Entity`.

    Steps:
      1. Delete every `:MENTIONS` edge whose source `:Chunk` has
         doc_id = $doc_id. Idempotent: works whether or not the
         chunks themselves have already been deleted (in which case the
         match returns no rows and the DELETE is a no-op).
      2. GC orphan `:Entity` nodes - any `:Entity` with no incoming
         `:MENTIONS` edge from any chunk. Shared entities (mentioned in
         other docs' chunks) keep their other edges and stay alive.

    The order matters: step 1 runs BEFORE step 2 so the entities this
    doc was the last reference to are eligible for GC in the same call.

    Called by the ingestion pipeline before `write_entities` so a re-
    extract leaves no stale `:MENTIONS` edges or orphan entities.

    Raises:
      - `ValueError` for an empty `doc_id`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: delete_entities_by_doc_id called with no doc_id")
    async with client.driver.session() as session:
        # 1. Drop MENTIONS edges sourced from this doc's chunks.
        # DELETE m removes only the relationship, not the chunk or
        # entity at either end - chunks may still belong to this
        # doc (if write_chunks hasn't been called yet on re-ingest)
        # and entities may still be referenced by other docs.
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}})-[m:{MENTIONS_REL}]->() DELETE m",
            doc_id=doc_id,
        )
        # 2. GC orphan entities. WHERE-NOT pattern matches the
        # existing L1-L4 orphan GC in openalex_writes.delete_doc.
        # Plain DELETE (not DETACH DELETE) because the WHERE
        # guarantees no incoming MENTIONS - if a future edge type
        # is added without updating this WHERE, DELETE will error
        # so the bug surfaces rather than silently masks.
        await session.run(f"MATCH (e:{ENTITY_LABEL}) WHERE NOT ()-[:{MENTIONS_REL}]->(e) DELETE e")
    logger.info("KG: deleted entity mentions for doc %s + GC'd orphans", doc_id)


async def write_entities(
    client,
    doc_id: str,
    chunk_mentions: list[tuple[str, list[Mention]]],
) -> None:
    """Write `:Entity` nodes + `:MENTIONS` edges for one document.

    Args:
      client:         Neo4jClient (passed through by the delegate method).
      doc_id:         identity for logging only - the actual chunk_id on
                      each row is what binds the MENTIONS edge to the
                      right chunk. Kept in the signature for parity with
                      `delete_entities_by_doc_id` and the other write
                      functions.
      chunk_mentions: list of (chunk_id, list[Mention]) tuples produced
                      by the pipeline's per-chunk extraction loop.
                      Empty inner lists are fine (a chunk yielded no
                      entities); empty outer list is fine too (whole-
                      doc no-op).

    Each Mention's `raw_text` is lowercased to compute the MERGE key.
    The original `raw_text` is NOT stored on the node - it's recoverable
    from chunk text via offset for NER mentions, and for LLM mentions
    the lowercased form is what we keep. Entity `canonicalised` flag is
    set to false on creation (flips True at L6b+ canonical-linking).

    Cypher pattern:
      - One UNWIND batch over all (chunk_id, mention) rows for the doc.
      - MERGE on `:Entity` by composite (key, entity_type) - matches the
        NODE KEY constraint. Existing entities are reused; new ones get
        `canonicalised=false` ON CREATE.
      - MATCH the chunk (it MUST exist because write_chunks ran first;
        if it doesn't, MATCH yields no rows for that mention and the
        edge silently isn't written - we log a count mismatch).
      - MERGE the :MENTIONS edge so repeated mentions of the same
        entity in the same chunk dedup to one edge. offset + confidence
        are set ON CREATE only - first mention wins for offset.

    Empty `chunk_mentions` AND all-empty inner lists both return as
    success no-ops (no writes dispatched).

    Raises:
      - `ValueError` for an empty `doc_id`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: write_entities called with no doc_id")
    if not chunk_mentions:
        logger.info("KG: write_entities got no chunk_mentions for %s", doc_id)
        return

    rows: list[dict[str, Any]] = []
    for chunk_id, mentions in chunk_mentions:
        for m in mentions:
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "key": m.raw_text.lower(),
                    "entity_type": m.entity_type,
                    "offset": m.offset,
                    "confidence": m.confidence,
                    # Provenance: which extractor(s) found this span in the
                    # priority-ordered union. `getattr` keeps this duck-typed
                    # for callers/tests passing plain mention stand-ins.
                    "sources": list(getattr(m, "sources", None) or []),
                }
            )

    if not rows:
        # Every chunk yielded zero mentions - nothing to write but not
        # an error. Success no-op so the pipeline marks the layer as OK.
        logger.info(
            "KG: write_entities found no mentions across %d chunks for %s",
            len(chunk_mentions),
            doc_id,
        )
        return

    async with client.driver.session() as session:
        await session.run(
            f"UNWIND $rows AS row "
            f"MERGE (e:{ENTITY_LABEL} "
            f"  {{key: row.key, entity_type: row.entity_type}}) "
            f"  ON CREATE SET e.canonicalised = false "
            f"WITH e, row "
            f"MATCH (c:{CHUNK_LABEL} {{chunk_id: row.chunk_id}}) "
            f"MERGE (c)-[m:{MENTIONS_REL}]->(e) "
            f"  ON CREATE SET m.offset = row.offset, "
            f"                m.confidence = row.confidence, "
            f"                m.sources = row.sources",
            rows=rows,
        )
    logger.info(
        "KG: wrote %d entity mentions across %d chunks for doc %s",
        len(rows),
        len(chunk_mentions),
        doc_id,
    )
