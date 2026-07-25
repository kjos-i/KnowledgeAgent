"""KG writes for L10 cross-document synthesis via the xref graph —
materialised `:RELATED_BY_XREF` edges between documents that share at
least N canonical concepts equivalent via L7 xref edges.

Design:

  - **Edge shape**: undirected (one edge per pair, MERGE on the
    undirected pattern). Properties: `shared_concepts` (list of
    OntologyTerm IDs), `shared_count` (size), `computed_at` (timestamp).
  - **Link signal**: shared canonical concepts via xref equivalence,
    NOT raw `:Entity` shape. Two docs are concept-related iff at least
    `threshold` distinct OntologyTerm concepts on this doc's side have
    an equivalent concept on the other doc's side. "Equivalent" means
    either (a) the same `:OntologyTerm` node (both canonical to the
    same concept) OR (b) a `:<X>_XREF` edge between them. The xref
    graph is what L7's `"use"` mode materialises.
  - **Parallel to L9, NOT a replacement**: L9 (`:RELATED_TO`) uses raw
    shared entity strings — robust against unimported ontologies,
    catches terminology that hasn't been canonicalised. L10
    (`:RELATED_BY_XREF`) uses canonical concepts — invariant under
    label variation, surfaces cross-ontology equivalence (the same
    drug under MeSH and ChEBI). Both edges can coexist; their
    semantics are distinct.
  - **Threshold**: default 2. Single shared concept = noise floor
    (a common high-level term like "Disease"); two distinct shared
    concepts = meaningful signal.
  - **Scope**: covers BOTH `:Document` and `:Artifact` focal nodes,
    matching L9.

Per-doc recompute:

  - `recompute_cross_doc_xrefs_edges(doc_id, threshold)` is the
    per-doc primitive. Wipes any existing `:RELATED_BY_XREF` edges
    incident to this doc, then re-runs the shared-concept-via-xref
    query to materialise fresh edges between this doc and every other
    doc meeting the threshold.
  - Called from `pipeline.py` after L9 step, gated on the
    `cross_doc_xrefs` layer flag (which itself requires `entities=true`
    AND `xrefs="use"`).

Graph-wide recompute:

  - `recompute_cross_doc_xrefs_global(threshold)` wipes every
    `:RELATED_BY_XREF` edge in the graph then rebuilds from scratch
    over all `(doc, other)` pairs. Used by the `backfill_xrefs`
    bulk_op when the user's already-imported corpus catches up on
    newly-resolved xref edges (e.g. after backfill turned dangling
    strings into real edges, the L10 view needs a refresh).

Idempotency:

  - The per-doc rebuild's wipe + write covers re-runs on the same
    doc. The global rebuild's wipe + write is naturally idempotent.
  - `delete_doc` does NOT need an explicit L10 cleanup: the focal
    node's DETACH DELETE removes incident `:RELATED_BY_XREF` edges
    via Neo4j cascade, same as L9.
  - Ontology delete (via `clear_xref_edges_for_ontology` or
    `delete_ontology_terms`) invalidates concept equivalence; the
    bulk_op `recompute_cross_doc_xrefs` is the user-facing way to
    refresh after that kind of change.

`Neo4jClient` exposes the per-doc recompute as a 1-line delegate in
`kg/client.py` (same pattern as `recompute_cross_doc_edges` for L9).

Note: the equivalence WHERE clause uses a pipe-union of all 18 xref
edge types (computed once at import time from `ONTOLOGY_XREF_RELS`)
rather than `type(r) ENDS WITH '_XREF'`. The pipe form uses Neo4j's
typed-relationship lookup and is dramatically faster on large graphs;
the string-suffix form would force a full edge scan.
"""

import logging

from knowledge_agent.kg.schema import (
    ARTIFACT_LABEL,
    CANONICAL_TO_REL,
    CHUNK_LABEL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
    MENTIONS_REL,
    ONTOLOGY_TERM_LABEL,
    ONTOLOGY_XREF_RELS,
    PART_OF_REL,
    RELATED_BY_XREF_REL,
)

logger = logging.getLogger(__name__)


# Default threshold: minimum number of distinct shared canonical
# concepts (counted on the focal doc's side) required to materialise
# a :RELATED_BY_XREF edge. Tuned to filter the noise floor of a
# single high-level shared term ("Disease" via MeSH-MONDO xref) while
# keeping meaningful pairings (two distinct shared concepts = real
# topical overlap). Override via `[cross_doc_xrefs].threshold` in
# `corpus.toml`.
DEFAULT_SHARED_COUNT_THRESHOLD = 2


# Pre-built pipe union of the 18 xref edge types. Used inside the
# equivalence WHERE clause to drive the typed-relationship lookup.
# Built ONCE at module import; the constant is purely derived from
# `ONTOLOGY_XREF_RELS`, so it stays in sync as new ontologies ship.
_XREF_REL_PIPE = "|".join(ONTOLOGY_XREF_RELS)


async def recompute_cross_doc_xrefs_edges(
    client,
    doc_id: str,
    threshold: int = DEFAULT_SHARED_COUNT_THRESHOLD,
) -> int:
    """Wipe + rewrite L10 `:RELATED_BY_XREF` edges for one doc.

    Steps:
      1. DELETE every `:RELATED_BY_XREF` edge incident to the node
         with this `doc_id`. Idempotent: no-op when the doc has no
         L10 edges yet.
      2. Re-run the shared-concept-via-xref query: for each other
         doc that shares at least `threshold` distinct canonical
         concepts via xref equivalence with this one, MERGE a
         `:RELATED_BY_XREF` edge and SET `shared_concepts`,
         `shared_count`, `computed_at`.

    Equivalence semantics: a concept on this doc's side is "shared"
    with another doc when EITHER (a) both docs canonicalise some
    entity to the same `:OntologyTerm` node, OR (b) the two
    canonicals are connected by a `:<X>_XREF` edge. The
    `WHERE t1 = t2 OR (t1)-[<pipe>]-(t2)` clause expresses both.

    `WHERE other.doc_id <> $doc_id` filters self-edges.

    The aggregation `collect(DISTINCT t1.id)` enumerates which
    canonical concepts on this doc's side back the relationship.
    The list is stored on the edge for downstream "why is this
    related?" inspection.

    Returns the count of edges (re)written. 0 means "no other doc met
    the threshold" — a clean outcome.

    Raises:
      - `ValueError` for an empty `doc_id` or `threshold < 1`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if not doc_id:
        raise ValueError("KG: recompute_cross_doc_xrefs_edges called with no doc_id")
    if threshold < 1:
        raise ValueError(
            f"KG: recompute_cross_doc_xrefs_edges threshold must be >= 1, got {threshold}"
        )
    async with client.driver.session() as session:
        # Step 1: wipe existing L10 edges incident to this doc.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL}|{ARTIFACT_LABEL} "
            f"  {{doc_id: $doc_id}})-[r:{RELATED_BY_XREF_REL}]-() "
            f"DELETE r",
            doc_id=doc_id,
        )
        # Step 2: recompute. The path walks
        # this -> chunk -> entity -> canonical -> ontology_term
        # on both sides; equivalence joins the two terms either
        # via identity (t1 = t2) or via an xref edge.
        result = await session.run(
            f"MATCH (this:{DOCUMENT_LABEL}|{ARTIFACT_LABEL} "
            f"  {{doc_id: $doc_id}})"
            f"<-[:{PART_OF_REL}]-(:{CHUNK_LABEL})"
            f"-[:{MENTIONS_REL}]->(:{ENTITY_LABEL})"
            f"-[:{CANONICAL_TO_REL}]->(t1:{ONTOLOGY_TERM_LABEL}) "
            f"MATCH (other:{DOCUMENT_LABEL}|{ARTIFACT_LABEL})"
            f"<-[:{PART_OF_REL}]-(:{CHUNK_LABEL})"
            f"-[:{MENTIONS_REL}]->(:{ENTITY_LABEL})"
            f"-[:{CANONICAL_TO_REL}]->(t2:{ONTOLOGY_TERM_LABEL}) "
            f"WHERE other.doc_id <> $doc_id "
            f"  AND (t1 = t2 OR EXISTS {{ "
            f"    MATCH (t1)-[:{_XREF_REL_PIPE}]-(t2) "
            f"  }}) "
            f"WITH this, other, "
            f"     collect(DISTINCT t1.id) AS shared_concepts "
            f"WHERE size(shared_concepts) >= $threshold "
            f"MERGE (this)-[r:{RELATED_BY_XREF_REL}]-(other) "
            f"SET r.shared_concepts = shared_concepts, "
            f"    r.shared_count = size(shared_concepts), "
            f"    r.computed_at = datetime() "
            f"RETURN count(r) AS n",
            doc_id=doc_id,
            threshold=threshold,
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "KG: recomputed L10 :RELATED_BY_XREF edges for doc %s -> %d edges (threshold=%d)",
        doc_id,
        n,
        threshold,
    )
    return n


async def recompute_cross_doc_xrefs_global(
    client,
    threshold: int = DEFAULT_SHARED_COUNT_THRESHOLD,
) -> int:
    """Graph-wide rebuild of `:RELATED_BY_XREF` edges.

    Steps:
      1. DELETE every `:RELATED_BY_XREF` edge in the graph. Cheap
         when the layer was previously off (no edges) and necessary
         when the xref surface has changed (e.g. a new ontology
         import resolved previously-dangling targets).
      2. Re-run the shared-concept-via-xref query over ALL
         `(this, other)` doc pairs and MERGE new edges where the
         threshold is met.

    `WHERE id(this) < id(other)` deduplicates the unordered pair
    (without it, each pair would be matched twice — once from
    each direction — and the `MERGE` would be idempotent at the
    edge level but the aggregation would double-count). Internal
    node IDs are used because they're guaranteed unique even
    across the `:Document` / `:Artifact` label split.

    Used by the `backfill_xrefs` bulk_op after dangling-xref
    resolution, and by the dedicated `recompute_cross_doc_xrefs`
    bulk_op for manual refresh.

    Returns the count of edges (re)written. Heavy query — expect
    minutes on a corpus with thousands of docs and tens of thousands
    of canonical concepts; runs in one transaction.

    Raises:
      - `ValueError` for `threshold < 1`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if threshold < 1:
        raise ValueError(
            f"KG: recompute_cross_doc_xrefs_global threshold must be >= 1, got {threshold}"
        )
    async with client.driver.session() as session:
        # Step 1: wipe all L10 edges. One MATCH across both
        # focal labels via the type predicate.
        await session.run(f"MATCH ()-[r:{RELATED_BY_XREF_REL}]-() DELETE r")
        # Step 2: recompute over all doc pairs.
        result = await session.run(
            f"MATCH (this:{DOCUMENT_LABEL}|{ARTIFACT_LABEL})"
            f"<-[:{PART_OF_REL}]-(:{CHUNK_LABEL})"
            f"-[:{MENTIONS_REL}]->(:{ENTITY_LABEL})"
            f"-[:{CANONICAL_TO_REL}]->(t1:{ONTOLOGY_TERM_LABEL}) "
            f"MATCH (other:{DOCUMENT_LABEL}|{ARTIFACT_LABEL})"
            f"<-[:{PART_OF_REL}]-(:{CHUNK_LABEL})"
            f"-[:{MENTIONS_REL}]->(:{ENTITY_LABEL})"
            f"-[:{CANONICAL_TO_REL}]->(t2:{ONTOLOGY_TERM_LABEL}) "
            f"WHERE id(this) < id(other) "
            f"  AND (t1 = t2 OR EXISTS {{ "
            f"    MATCH (t1)-[:{_XREF_REL_PIPE}]-(t2) "
            f"  }}) "
            f"WITH this, other, "
            f"     collect(DISTINCT t1.id) AS shared_concepts "
            f"WHERE size(shared_concepts) >= $threshold "
            f"MERGE (this)-[r:{RELATED_BY_XREF_REL}]-(other) "
            f"SET r.shared_concepts = shared_concepts, "
            f"    r.shared_count = size(shared_concepts), "
            f"    r.computed_at = datetime() "
            f"RETURN count(r) AS n",
            threshold=threshold,
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "KG: recomputed L10 :RELATED_BY_XREF graph-wide -> %d edges (threshold=%d)",
        n,
        threshold,
    )
    return n


async def count_related_by_xref_edges(client) -> int:
    """Count all `:RELATED_BY_XREF` edges in the graph (each stored once).

    A directed match counts every undirected edge exactly once - an
    undirected `()-[r]-()` count would double every edge. The L10 recompute
    plan and the ontology install plan use this so the "existing edges"
    figure they show matches the true count the rebuild reports.
    """
    async with client.driver.session() as session:
        result = await session.run(f"MATCH ()-[r:{RELATED_BY_XREF_REL}]->() RETURN count(r) AS n")
        row = await result.single()
        return int(row["n"]) if row else 0
