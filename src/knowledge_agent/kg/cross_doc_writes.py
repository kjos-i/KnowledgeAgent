"""KG writes for L9 cross-document synthesis - materialised
`:RELATED_TO` edges between documents that share at least N L6a
entities.

Design (see [[researchliteratureagent-l8-l9-specs]] for the locked
rationale):

  - **Edge shape**: undirected (one edge per pair, written via MERGE
    on the undirected pattern). Properties: `shared_entities` (list
    of entity keys), `shared_count` (size), `computed_at` (timestamp).
  - **Link signal**: shared L6a `:Entity` nodes, NOT canonicals. Two
    docs are related iff at least `threshold` distinct `:Entity` nodes
    are mentioned by chunks of both. This catches personal notes-side
    terminology that L7 wouldn't anchor, and removes the dependency
    on ontology import / link state.
  - **Threshold**: default 2. Single shared entity = noise floor (a
    common generic term like "patient"); two distinct shared entities
    = meaningful signal.
  - **Scope**: covers BOTH `:Document` and `:Artifact` focal nodes.
    Cross-type linking ("this dataset is related to that paper") is
    real, and the query topology is identical for both.

Per-doc recompute semantics:

  - `recompute_cross_doc_edges(doc_id, threshold)` is the only write
    primitive. It wipes any existing `:RELATED_TO` edges incident to
    this doc, then runs the shared-entity query to materialise fresh
    edges between this doc and every other doc that meets the
    threshold.
  - The MERGE in the rewrite step is undirected. Whichever side of
    the pair fires the MERGE first creates the edge; later calls
    from the other doc's perspective match the existing edge in
    either direction.
  - `delete_doc` does NOT need an explicit L9 cleanup step: the focal
    node's DETACH DELETE in `openalex_writes.delete_doc` removes the
    `:RELATED_TO` edges incident to it as part of standard Neo4j
    cascade. L8 needed an explicit delete because triples sit between
    `:Entity` nodes (not anchored to the focal); L9 edges ARE incident
    to the focal, so DETACH DELETE handles them.
  - No "global rebuild" hook on ontology import/delete: since L9 uses
    raw L6a entities (not canonicals), canonical churn doesn't
    invalidate L9 edges.

`Neo4jClient` exposes the recompute as a 1-line delegate method in
`kg/client.py`, matching the pattern used by L8.
"""

import logging

from knowledge_agent.kg.schema import (
    ARTIFACT_LABEL,
    CHUNK_LABEL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
    MENTIONS_REL,
    PART_OF_REL,
    RELATED_TO_REL,
)

logger = logging.getLogger(__name__)


# Default threshold: minimum number of distinct shared :Entity nodes
# required to create a :RELATED_TO edge between two docs. Tuned to
# filter the noise floor of single-shared generic terms ("patient",
# "model", "study") while keeping meaningful pairings (two shared
# biomedical entities = real signal).
DEFAULT_SHARED_COUNT_THRESHOLD = 2


def recompute_cross_doc_edges(
    client,
    doc_id: str,
    threshold: int = DEFAULT_SHARED_COUNT_THRESHOLD,
) -> int:
    """Wipe + rewrite L9 `:RELATED_TO` edges for one doc.

    Steps:
      1. DELETE every `:RELATED_TO` edge incident to the node with
         this `doc_id`. Idempotent: no-op when the doc has no L9
         edges yet.
      2. Re-run the shared-entity overlap query: for each other doc
         that shares at least `threshold` distinct `:Entity` keys
         with this one, MERGE a `:RELATED_TO` edge between them and
         SET `shared_entities`, `shared_count`, `computed_at`.

    The focal MATCH uses `:Document|Artifact` (Neo4j 5 label
    disjunction) so artifacts (datasets, code, audio, etc.) participate
    on equal footing with documents. Both sides of the pair use the
    same disjunction.

    `WHERE other.doc_id <> $doc_id` filters self-edges (the same
    chunks reach the same focal via the symmetric MENTIONS pattern).

    `WITH ... collect(DISTINCT e.key) AS shared` groups per (this,
    other) pair across the join, giving one row per other doc.
    `size(shared) >= $threshold` is the cutoff.

    `MERGE (d)-[r:RELATED_TO]-(other)` is undirected: it matches an
    existing edge in either direction OR creates a new one with
    Neo4j's default left-to-right storage direction. The undirected
    MATCH side means subsequent rebuilds from the other doc's
    perspective find the same edge regardless of who originally
    wrote it.

    Returns the count of edges (re)written. 0 means "no other doc met
    the threshold" — a clean outcome.

    Raises:
      - `ValueError` for an empty `doc_id` or `threshold < 1`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: recompute_cross_doc_edges called with no doc_id")
    if threshold < 1:
        raise ValueError(
            f"KG: recompute_cross_doc_edges threshold must be >= 1, "
            f"got {threshold}"
        )
    with client.driver.session() as session:
        # Step 1: wipe existing edges incident to this doc. The
        # undirected `[r:RELATED_TO]-` pattern catches edges
        # written in either direction.
        session.run(
            f"MATCH (d:{DOCUMENT_LABEL}|{ARTIFACT_LABEL} "
            f"  {{doc_id: $doc_id}})-[r:{RELATED_TO_REL}]-() "
            f"DELETE r",
            doc_id=doc_id,
        )
        # Step 2: recompute. The double-hop pattern walks
        # focal -> chunk -> entity <- chunk -> other_focal,
        # collects distinct entity keys per (this, other), filters
        # by threshold, and writes one edge per surviving pair.
        result = session.run(
            f"MATCH (d:{DOCUMENT_LABEL}|{ARTIFACT_LABEL} "
            f"  {{doc_id: $doc_id}})"
            f"<-[:{PART_OF_REL}]-(:{CHUNK_LABEL})"
            f"-[:{MENTIONS_REL}]->(e:{ENTITY_LABEL})"
            f"<-[:{MENTIONS_REL}]-(:{CHUNK_LABEL})"
            f"-[:{PART_OF_REL}]->(other:{DOCUMENT_LABEL}|{ARTIFACT_LABEL}) "
            f"WHERE other.doc_id <> $doc_id "
            f"WITH d, other, collect(DISTINCT e.key) AS shared "
            f"WHERE size(shared) >= $threshold "
            f"MERGE (d)-[r:{RELATED_TO_REL}]-(other) "
            f"SET r.shared_entities = shared, "
            f"    r.shared_count = size(shared), "
            f"    r.computed_at = datetime() "
            f"RETURN count(r) AS n",
            doc_id=doc_id,
            threshold=threshold,
        )
        row = result.single()
        # Cypher RETURN count(r) always emits exactly one row;
        # `row is None` here implies driver-level corruption.
        n = int(row["n"]) if row else 0
    logger.info(
        "KG: recomputed L9 :RELATED_TO edges for doc %s -> %d edges "
        "(threshold=%d)",
        doc_id, n, threshold,
    )
    return n
