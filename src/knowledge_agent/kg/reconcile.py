"""Corpus-wide layer reconciliation: remove KG data no longer covered
by the current `CorpusConfig`.

Design principle: when a layer flag flips off or a sub-selection
narrows, the KG must lose the data that flag/selection no longer
covers. Stale nodes and edges accumulate otherwise (:MeSHTerm sitting
after MeSH is disabled, :RELATED_TO edges after cross_doc is disabled,
:Entity nodes of a removed type after `entity_types` shrinks).

Called at the top of `aingest_document` and every `bulk_backfill_*_execute`
so both the main ingest path and the layer-specific rebuild buttons
honor the "refresh from this stage with the new options" semantics.

**Fail-hard contract**: any Cypher failure inside a reconcile helper
propagates. Silently continuing on a delete failure would leave the
graph half-reconciled + hide the underlying problem. The caller
(ingest / bulk_op) is the orchestrator boundary that decides whether
to surface a user-facing error.

Scope split across 5 helpers, one per layer:

  - `reconcile_ontologies_to_config`  (L7 — per-ontology flag)
  - `reconcile_entities_to_config`    (L6 — layer flag + entity_types)
  - `reconcile_triples_to_config`     (L8 — layer flag)
  - `reconcile_cross_doc_to_config`   (L9 — layer flag)
  - `reconcile_cross_doc_xrefs_to_config` (L10 — layer flag)

Each is idempotent: after the first call in a session, a subsequent
call with the same config finds nothing to wipe. Cheap probes gate
the actual DELETEs so the no-op path is one Cypher round trip.

Downstream cascades are handled by DETACH DELETE on nodes (wiping
`:Entity` also removes `:MENTIONS`, `:CANONICAL_TO`, and any
`:INHIBITS`/`:ACTIVATES`/... triple edges incident to the entity)
and by direct DELETE on edges (L9, L10, individual triple predicates
when the layer is off but entities stay).
"""

from __future__ import annotations

import logging

from knowledge_agent.kg.ontology_linking import ONTOLOGY_REGISTRY
from knowledge_agent.kg.schema import (
    ENTITY_LABEL,
    RELATED_BY_XREF_REL,
    RELATED_TO_REL,
    TRIPLE_PREDICATE_RELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L7: per-ontology reconciliation.
# ---------------------------------------------------------------------------


async def reconcile_ontologies_to_config(client, config) -> list[str]:
    """Wipe every ontology whose flag is off in `config` but still
    present in the KG.

    Per-ontology granularity: each ontology has its own layer flag
    (`ontology_mesh`, `ontology_go`, etc.). Iterates the registry;
    for each disabled ontology, checks `is_imported_fn` (cheap
    MATCH-LIMIT-1) and, if imported, calls `delete_fn` — a DETACH
    DELETE on the ontology's term nodes that cascades to
    `:CANONICAL_TO` and `:<X>_XREF` edges automatically.

    Returns the list of ontology names actually wiped (for logging +
    result reporting). Empty when everything was already in sync.
    """
    wiped: list[str] = []
    enabled = set(config._enabled_ontology_layers())
    for name, entry in ONTOLOGY_REGISTRY.items():
        if name in enabled:
            continue
        if not await entry["is_imported_fn"](client):
            continue
        logger.info(
            "reconcile: wiping ontology %s (disabled in config but present in KG)",
            name,
        )
        await entry["delete_fn"](client)
        wiped.append(name)
    return wiped


# ---------------------------------------------------------------------------
# L6: entity reconciliation — layer flag OR entity_types narrowing.
# ---------------------------------------------------------------------------


async def reconcile_entities_to_config(client, config) -> dict[str, int]:
    """Wipe :Entity nodes not covered by the current config.

    Two independent triggers:

      1. `layers.entities = false` → wipe ALL :Entity nodes corpus-wide.
         DETACH DELETE cascades to :MENTIONS, :CANONICAL_TO, and any
         triple edges incident to the entity.

      2. `layers.entities = true` but `entity_types` narrowed → wipe
         :Entity nodes whose `entity_type` property is no longer in
         `config.entities.entity_types`. Empty `entity_types` = "any
         type accepted" (open-vocab LLM), so no per-type wipe fires
         in that case.

    Returns `{"n_wiped": int}` for result reporting. Zero when
    nothing needed wiping.
    """
    wiped_total = 0

    if not config.layers.entities:
        # Layer off — wipe all entities corpus-wide.
        async with client.driver.session() as session:
            # Collect first so the count is the TOTAL (one aggregated row), then
            # DETACH DELETE each via FOREACH. The old `WITH e, count(e)` grouped
            # per-node, so count(e) was always 1 and n_wiped never exceeded 1.
            result = await session.run(
                f"MATCH (e:{ENTITY_LABEL}) "
                f"WITH collect(e) AS ents "
                f"FOREACH (x IN ents | DETACH DELETE x) "
                f"RETURN size(ents) AS n"
            )
            row = await result.single()
            wiped_total = int(row["n"]) if row is not None else 0
        if wiped_total > 0:
            logger.info(
                "reconcile: wiped %d :Entity nodes (layers.entities=false)",
                wiped_total,
            )
        return {"n_wiped": wiped_total}

    # Layer on: check for entity_types narrowing.
    if config.entities is None or not config.entities.entity_types:
        # Open-vocab: no per-type restriction. Nothing to reconcile.
        return {"n_wiped": 0}

    kept_types = list(config.entities.entity_types)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (e:{ENTITY_LABEL}) WHERE NOT e.entity_type IN $kept "
            f"WITH collect(e) AS ents "
            f"FOREACH (x IN ents | DETACH DELETE x) "
            f"RETURN size(ents) AS n",
            kept=kept_types,
        )
        row = await result.single()
        wiped_total = int(row["n"]) if row is not None else 0
    if wiped_total > 0:
        logger.info(
            "reconcile: wiped %d :Entity nodes with types outside %r",
            wiped_total,
            kept_types,
        )
    return {"n_wiped": wiped_total}


# ---------------------------------------------------------------------------
# L8: triples reconciliation — layer flag.
# ---------------------------------------------------------------------------


async def reconcile_triples_to_config(client, config) -> dict[str, int]:
    """Wipe all triple edges corpus-wide when `layers.triples = false`.

    Triples are edges between :Entity nodes typed by one of the 15
    predicates in `TRIPLE_PREDICATE_RELS`. When the layer is off,
    every edge with any of those types should go.

    Returns `{"n_wiped": int}`. Zero when the layer is on OR no
    triple edges exist yet.
    """
    if config.layers.triples:
        return {"n_wiped": 0}

    rel_pattern = "|".join(TRIPLE_PREDICATE_RELS)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH ()-[r:{rel_pattern}]->() "
            f"WITH collect(r) AS rels "
            f"FOREACH (x IN rels | DELETE x) "
            f"RETURN size(rels) AS n"
        )
        row = await result.single()
        wiped = int(row["n"]) if row is not None else 0
    if wiped > 0:
        logger.info(
            "reconcile: wiped %d triple edges (layers.triples=false)",
            wiped,
        )
    return {"n_wiped": wiped}


# ---------------------------------------------------------------------------
# L9: cross-doc reconciliation — layer flag.
# ---------------------------------------------------------------------------


async def reconcile_cross_doc_to_config(client, config) -> dict[str, int]:
    """Wipe all `:RELATED_TO` edges corpus-wide when `layers.cross_doc = false`.

    Cross-doc edges connect focal document nodes (`:Document` /
    `:Artifact`). No cascade concerns — they don't anchor anything
    downstream except the L10 `:RELATED_BY_XREF` (a separate edge
    type covered by the next helper).

    Returns `{"n_wiped": int}`.
    """
    if config.layers.cross_doc:
        return {"n_wiped": 0}

    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH ()-[r:{RELATED_TO_REL}]->() "
            f"WITH collect(r) AS rels "
            f"FOREACH (x IN rels | DELETE x) "
            f"RETURN size(rels) AS n"
        )
        row = await result.single()
        wiped = int(row["n"]) if row is not None else 0
    if wiped > 0:
        logger.info(
            "reconcile: wiped %d :RELATED_TO edges (layers.cross_doc=false)",
            wiped,
        )
    return {"n_wiped": wiped}


# ---------------------------------------------------------------------------
# L10: cross-doc-xrefs reconciliation — layer flag.
# ---------------------------------------------------------------------------


async def reconcile_cross_doc_xrefs_to_config(
    client,
    config,
) -> dict[str, int]:
    """Wipe all `:RELATED_BY_XREF` edges when `layers.cross_doc_xrefs = false`.

    Independent from L9 above — these are cross-doc edges computed
    via canonical-xref paths, not via shared-entity thresholds. Same
    shape as `reconcile_cross_doc_to_config`.

    Returns `{"n_wiped": int}`.
    """
    if config.layers.cross_doc_xrefs:
        return {"n_wiped": 0}

    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH ()-[r:{RELATED_BY_XREF_REL}]->() "
            f"WITH collect(r) AS rels "
            f"FOREACH (x IN rels | DELETE x) "
            f"RETURN size(rels) AS n"
        )
        row = await result.single()
        wiped = int(row["n"]) if row is not None else 0
    if wiped > 0:
        logger.info(
            "reconcile: wiped %d :RELATED_BY_XREF edges (layers.cross_doc_xrefs=false)",
            wiped,
        )
    return {"n_wiped": wiped}
