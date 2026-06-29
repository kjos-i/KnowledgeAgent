"""KG writes for chunks (L5).

Each `:Chunk` node tracks the identity + structural metadata of one chunk:
chunk_id (join key with LanceDB), doc_id (parent), chunk_index, section,
page, content_type. Text and embedding NEVER appear here - those stay in
LanceDB. The KG side is identity + structure only.

Idempotency pattern: delete-then-write per doc_id, mirroring how the
LanceDB chunk writes work in `ingestion/pipeline.py`. On re-ingest of
the same doc, the old chunks are wiped and the fresh set is written.

Unlike L1-L4 entities, chunks are NOT shared across documents - each
chunk belongs to exactly one doc - so `delete_chunks_by_doc_id` is a
clean per-doc wipe with no orphan-GC concern.

`Neo4jClient` exposes these as 1-line delegate methods in `kg/client.py`,
matching the pattern used for `openalex_writes`.
"""

import logging
from typing import Any

from knowledge_agent.ingestion.ids import make_chunk_id
from knowledge_agent.kg.schema import (
    CHUNK_LABEL,
    MAIN_LABELS,
    PART_OF_REL,
    SUB_LABEL_TO_MAIN,
)

logger = logging.getLogger(__name__)


def delete_chunks_by_doc_id(client, doc_id: str) -> None:
    """Wipe every :Chunk node for `doc_id`. Idempotent.

    Used by the ingestion pipeline before `write_chunks` so a re-chunk
    leaves no orphans. `DETACH DELETE` handles any edges chunks may have
    accumulated (e.g., future L6 `:MENTIONS` edges to entities).

    Raises:
      - `ValueError` for an empty `doc_id`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: delete_chunks_by_doc_id called with no doc_id")
    with client.driver.session() as session:
        session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}}) "
            f"DETACH DELETE c",
            doc_id=doc_id,
        )
    logger.info("KG: deleted chunks for doc %s", doc_id)


def write_chunks(
    client,
    doc_id: str,
    chunks: list[Any],
    main_label: str,
    sub_label: str | None = None,
) -> None:
    """Write :Chunk nodes + :PART_OF edges for one document.

    `chunks` is a list of objects with these attributes (typically
    `ParsedChunk` from `ingestion/parse.py`, but duck-typed):
      - chunk_index   (int)
      - section       (str | None)
      - page          (int | None)
      - content_type  (str)

    `main_label` is `"Document"` or `"Artifact"` - the top-level label
    applied to the focal node. `sub_label` is an optional sub-label
    (e.g. `"Paper"`, `"Note"`) co-applied on creation via `ON CREATE`,
    so if a previous write (e.g. `write_citations`) already produced
    the focal with different labels they're preserved.

    Each chunk gets:
      - `:Chunk` node MERGEd by `chunk_id = make_chunk_id(doc_id, chunk_index)`.
      - `:PART_OF` edge from `:Chunk` to the focal node.

    One Cypher round trip total (UNWIND batches all chunks).
    Empty `chunks` list is a no-op success (returns without writing).

    Raises:
      - `ValueError` for an empty `doc_id`, an unknown `main_label`, or
        a `sub_label` that doesn't belong under `main_label`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: write_chunks called with no doc_id")
    if main_label not in MAIN_LABELS:
        raise ValueError(
            f"KG: write_chunks called with invalid main_label={main_label!r}"
        )
    if sub_label is not None and SUB_LABEL_TO_MAIN.get(sub_label) != main_label:
        raise ValueError(
            f"KG: write_chunks sub_label={sub_label!r} doesn't belong "
            f"under :{main_label}"
        )
    if not chunks:
        logger.info("KG: write_chunks found no chunks for %s", doc_id)
        return

    chunk_payload = [
        {
            "chunk_id": make_chunk_id(doc_id, c.chunk_index),
            "chunk_index": c.chunk_index,
            "section": c.section,
            "page": c.page,
            "content_type": c.content_type,
        }
        for c in chunks
    ]

    # Compose the focal-node MERGE: the sub-label, when present, is added
    # only ON CREATE so an existing focal (e.g. left by an earlier
    # `write_citations` call) keeps its labels.
    on_create_clauses = ["d.in_corpus = true"]
    if sub_label:
        on_create_clauses.append(f"d:{sub_label}")
    focal_merge = (
        f"MERGE (d:{main_label} {{doc_id: $doc_id}}) "
        f"ON CREATE SET {', '.join(on_create_clauses)} "
    )

    with client.driver.session() as session:
        session.run(
            focal_merge
            + "WITH d "
            + "UNWIND $chunks AS ch "
            + f"MERGE (c:{CHUNK_LABEL} {{chunk_id: ch.chunk_id}}) "
            + "SET c.doc_id = $doc_id, "
            + "c.chunk_index = ch.chunk_index, "
            + "c.section = ch.section, "
            + "c.page = ch.page, "
            + "c.content_type = ch.content_type "
            + f"MERGE (c)-[:{PART_OF_REL}]->(d)",
            doc_id=doc_id,
            chunks=chunk_payload,
        )
    logger.info(
        "KG: wrote %d chunks + PART_OF edges for doc %s",
        len(chunks), doc_id,
    )
