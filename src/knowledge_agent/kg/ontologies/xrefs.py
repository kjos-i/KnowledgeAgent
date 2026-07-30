"""L7 cross-ontology xref primitives — four single-purpose ops.

This module is the write-side counterpart to `dangling_xrefs` (the
property set on `:<X>Term` nodes by `write_ontology_terms` when the
`xrefs` layer is in `"collect_only"` or `"use"` mode). Each primitive
touches exactly ONE artifact (edges OR the property), never both, so the
GUI can expose four buttons that each do one thing to one ontology:

  - `materialize_xref_edges_for_ontology(client, term_label)`: for each
    source term with a non-empty `dangling_xrefs` list, MERGE a
    `:<X>_XREF` edge to any `:OntologyTerm` whose `id` matches. Targets
    not yet imported are silently skipped. Does NOT strip the property —
    mirrors the ingest-time `use`-mode behaviour (create edges, leave the
    list alone). Idempotent via `MERGE`.

  - `strip_materialized_xrefs_for_ontology(client, term_label)`: remove
    from each term's `dangling_xrefs` list only the entries that already
    have a matching `:<X>_XREF` edge. Pure property tidy against the
    edges that exist NOW; creates no edges. This is the deliberate fix
    for the inflated `count_dangling_xrefs` reading left by a
    materialize (or an ingest), run when you want the list to hold only
    the genuinely-still-pending entries.

  - `remove_xref_edges_for_ontology(client, term_label)`: delete every
    OUTGOING `:<X>_XREF` edge from this ontology's terms. Leaves the
    property untouched and leaves INBOUND edges (from other ontologies)
    intact — deleting those would corrupt the other ontology's surface;
    wipe both directions by calling this for every ontology in turn.

  - `remove_dangling_xrefs_for_ontology(client, term_label)`: remove the
    entire `dangling_xrefs` property from this ontology's terms. Leaves
    edges untouched. After this the ontology cannot be auto-connected by
    a future `materialize` until it is re-imported with the xrefs layer
    on (the declared xref strings are gone).

Two diagnostics complement the primitives:
  - `count_dangling_xrefs(client, term_label)`: source nodes with a
    non-empty `dangling_xrefs` list. Used by install plans to estimate
    "X xrefs available to resolve" before the user enables a layer.
    NOTE: this counts entries regardless of whether they already have an
    edge, so it reads high after a materialize/ingest until
    `strip_materialized_xrefs_for_ontology` is run.
  - `count_xref_edges(client, term_label)`: live edge count for one
    ontology (or all 18 when `term_label=None`). Used by status views.

All four primitives + both diagnostics RAISE on an unknown `term_label`
and let `neo4j.exceptions.*` propagate to the orchestrator boundary
(the bulk_op execute), which maps that into a per-ontology fail-soft
outcome. None of them is called from `delete_ontology`: that flow's
`DETACH DELETE` on `:<X>Term` nodes already cascades through the xref
edges via Neo4j standard semantics.

Edge type derivation: the `:<X>_XREF` predicate is derivable from the
term sub-label via `_xref_rel_from_term_label("MeSHTerm") -> "MESH_XREF"`
(declared in `helpers`). Per-ontology modules don't need to register
their xref edge type explicitly — the helper computes it at query time.
Schema constants in `ONTOLOGY_XREF_RELS` are for prompt rendering /
dispatch documentation only.
"""

from __future__ import annotations

import logging

from knowledge_agent.kg.ontologies.helpers import (
    _xref_rel_from_term_label,
)
from knowledge_agent.kg.schema import (
    ONTOLOGY_SUB_LABELS,
    ONTOLOGY_TERM_LABEL,
    ONTOLOGY_XREF_RELS,
)

logger = logging.getLogger(__name__)


def _validate_term_label(term_label: str, fn_name: str) -> None:
    """Raise `ValueError` when `term_label` isn't a shipped ontology
    sub-label. Shared guard for every primitive so an out-of-range
    dispatch fails loudly at the boundary rather than running a no-op
    Cypher against a non-existent label."""
    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(
            f"{fn_name}: unknown term_label {term_label!r}; "
            f"expected one of {sorted(ONTOLOGY_SUB_LABELS)}"
        )


# ---------------------------------------------------------------------------
# Edge primitives — create / delete `:<X>_XREF` edges (property untouched).
# ---------------------------------------------------------------------------


async def materialize_xref_edges_for_ontology(client, term_label: str) -> int:
    """Resolve one ontology's dangling xrefs into `:<X>_XREF` edges.

    For each source term with a non-empty `dangling_xrefs` list, unwind
    it and MERGE a `:<xref_rel>` edge to any `:OntologyTerm` whose `id`
    matches. Targets that don't exist yet (an ontology not imported, or
    an LCSH/EuroVoc URI we don't ship) are silently skipped by the inner
    MATCH — their strings stay in `dangling_xrefs` for a later run once
    the missing ontology lands.

    Does NOT strip resolved entries (that is
    `strip_materialized_xrefs_for_ontology`); this mirrors ingest-time
    `use`-mode: create edges, leave the property as-is. So the dangling
    count reads high after this until the strip op runs.

    Returns the count of MERGE rows produced (idempotent — re-running on
    a resolved corpus counts matched-existing edges too). Raises
    `ValueError` for an unknown `term_label`; Cypher/driver errors
    propagate.
    """
    _validate_term_label(term_label, "materialize_xref_edges_for_ontology")
    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"  AND size(s.dangling_xrefs) > 0 "
            f"UNWIND s.dangling_xrefs AS xref_id "
            f"MATCH (t:{ONTOLOGY_TERM_LABEL} {{id: xref_id}}) "
            f"MERGE (s)-[r:{xref_rel}]->(t) "
            f"RETURN count(r) AS n"
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "materialize_xref_edges_for_ontology (%s): %d :%s edges resolved",
        term_label,
        n,
        xref_rel,
    )
    return n


async def remove_xref_edges_for_ontology(client, term_label: str) -> int:
    """Delete every OUTGOING `:<X>_XREF` edge from one ontology's terms.

    Leaves `dangling_xrefs` untouched (use
    `remove_dangling_xrefs_for_ontology` for the property) and leaves
    INBOUND edges from OTHER ontologies' terms pointing AT this ontology
    intact — removing them would corrupt the other ontology's xref
    surface without re-running its materialize. To wipe both directions,
    call this for every ontology in turn (the multi-select picker does
    exactly that when all are selected).

    Returns the count of edges deleted. Idempotent: no edges = 0-row
    no-op returning 0. Raises `ValueError` for an unknown `term_label`;
    Cypher/driver errors propagate.
    """
    _validate_term_label(term_label, "remove_xref_edges_for_ontology")
    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label})-[r:{xref_rel}]->() DELETE r RETURN count(r) AS n"
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "remove_xref_edges_for_ontology (%s): deleted %d :%s edges",
        term_label,
        n,
        xref_rel,
    )
    return n


# ---------------------------------------------------------------------------
# Property primitives — tidy / wipe `dangling_xrefs` (edges untouched).
# ---------------------------------------------------------------------------


async def strip_materialized_xrefs_for_ontology(client, term_label: str) -> int:
    """Drop already-materialized entries from one ontology's `dangling_xrefs`.

    For each source term with a `dangling_xrefs` list AND at least one
    outgoing `:<xref_rel>` edge, rewrite the list to exclude entries
    whose string matches an existing edge target's `id`. Source terms
    with no resolved edges are left untouched (their list stays in full).

    Creates NO edges: it only tidies the property against the edges that
    already exist. So run it AFTER a materialize (or any time you want
    the list to hold only genuinely-pending entries); run alone with no
    prior materialize and it simply prunes whatever is already
    edge-backed. This is the op that corrects the inflated
    `count_dangling_xrefs` reading.

    Returns the count of source nodes whose lists were rewritten.
    Idempotent — re-running on a tidy corpus is a no-op. Raises
    `ValueError` for an unknown `term_label`; Cypher/driver errors
    propagate.
    """
    _validate_term_label(term_label, "strip_materialized_xrefs_for_ontology")
    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"OPTIONAL MATCH (s)-[:{xref_rel}]->(t:{ONTOLOGY_TERM_LABEL}) "
            f"WITH s, collect(DISTINCT t.id) AS resolved "
            f"WHERE size(resolved) > 0 "
            f"SET s.dangling_xrefs = "
            f"  [x IN s.dangling_xrefs WHERE NOT x IN resolved] "
            f"RETURN count(s) AS n"
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "strip_materialized_xrefs_for_ontology (%s): tidied %d source nodes",
        term_label,
        n,
    )
    return n


async def remove_dangling_xrefs_for_ontology(client, term_label: str) -> int:
    """Remove the entire `dangling_xrefs` property from one ontology's terms.

    Wipes ALL dangling entries (resolved or not), leaving the `:<X>_XREF`
    edges untouched (use `remove_xref_edges_for_ontology` for those).
    After this the ontology can no longer be auto-connected by a future
    `materialize` run: the declared xref strings are gone, so a
    re-import (with the xrefs layer on) is needed to repopulate them.

    Returns the count of source nodes whose property was removed.
    Idempotent: no property = 0-row no-op returning 0. Raises
    `ValueError` for an unknown `term_label`; Cypher/driver errors
    propagate.
    """
    _validate_term_label(term_label, "remove_dangling_xrefs_for_ontology")
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"REMOVE s.dangling_xrefs "
            f"RETURN count(s) AS n"
        )
        row = await result.single()
        n = int(row["n"]) if row else 0
    logger.info(
        "remove_dangling_xrefs_for_ontology (%s): cleared property on %d source nodes",
        term_label,
        n,
    )
    return n


# ---------------------------------------------------------------------------
# Diagnostics — counts used by install plans + status views.
# ---------------------------------------------------------------------------


async def count_dangling_xrefs(
    client,
    term_label: str,
) -> int:
    """Source nodes with a non-empty `dangling_xrefs` list.

    Per-ontology. Used by the `install_xrefs_plan` to surface "X
    xrefs available to resolve" before the user enables the layer.

    NOTE: counts entries regardless of whether they already have an
    edge, so after a materialize (or an ingest) this reads high until
    `strip_materialized_xrefs_for_ontology` prunes the resolved ones.

    Raises `ValueError` for an unknown `term_label`; Cypher / driver
    failures propagate.
    """
    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(f"count_dangling_xrefs: unknown term_label {term_label!r}")
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"  AND size(s.dangling_xrefs) > 0 "
            f"RETURN count(s) AS n"
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def count_xref_edges(
    client,
    term_label: str | None = None,
) -> int:
    """Live xref-edge count.

    When `term_label` is None, counts edges across ALL 18 xref edge
    types (the union via `ONTOLOGY_XREF_RELS`) — useful for the global
    status view. When `term_label` is set, counts only edges of that
    ontology's derived xref type.

    Raises `ValueError` for an unknown `term_label`; Cypher / driver
    failures propagate.
    """
    if term_label is None:
        async with client.driver.session() as session:
            # Union the 18 typed-edge MATCHes. Pipe syntax inside
            # the relationship pattern keeps it to one query.
            pipe = "|".join(ONTOLOGY_XREF_RELS)
            result = await session.run(f"MATCH ()-[r:{pipe}]->() RETURN count(r) AS n")
            row = await result.single()
            return int(row["n"]) if row else 0

    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(f"count_xref_edges: unknown term_label {term_label!r}")
    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label})-[r:{xref_rel}]->() RETURN count(r) AS n"
        )
        row = await result.single()
        return int(row["n"]) if row else 0
