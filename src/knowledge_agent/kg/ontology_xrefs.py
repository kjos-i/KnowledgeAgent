"""L7 cross-ontology xref primitives — backfill + clear.

This module is the write-side counterpart to `dangling_xrefs` (the
property set on `:<X>Term` nodes by `write_ontology_terms` when the
`xrefs` layer is in `"collect_only"` or `"use"` mode). Two primitives:

  - `backfill_resolved_xrefs(client)`: walk every shipped ontology's
    term sub-label, write resolved `:<X>_XREF` edges for every dangling
    entry whose target already exists as an `:OntologyTerm` node, then
    strip resolved entries from the `dangling_xrefs` list. Idempotent
    via `MERGE`. Used by:
      - the `backfill_xrefs` bulk_op (user-facing batch run)
      - delayed-resolution when the user imports an ontology AFTER
        another ontology already declared xrefs at it (e.g. MONDO
        declares `MESH:D003920`, MeSH wasn't yet imported; after MeSH
        ships, backfill turns that string into a `:MONDO_XREF` edge)

  - `clear_xref_edges_for_ontology(client, term_label)`: delete every
    outgoing `:<X>_XREF` edge from one ontology's nodes AND wipe its
    `dangling_xrefs` properties. Used by:
      - the user-facing `clear_xref_edges` bulk_op (maintenance)
      - explicit "stop using xrefs from this ontology" flows
    NOT called from `delete_ontology`: that flow's `DETACH DELETE` on
    `:<X>Term` nodes already cascades through the xref edges via Neo4j
    standard semantics.

Two diagnostics complement the primitives:
  - `count_dangling_xrefs(client, term_label)`: source nodes with a
    non-empty `dangling_xrefs` list. Used by install plans to estimate
    "X xrefs available to resolve" before the user enables a layer.
  - `count_xref_edges(client, term_label)`: live edge count for one
    ontology (or all 18 when `term_label=None`). Used by status views.

All four return None on Neo4j error (logged); the bulk_op caller
maps that into a fail-soft outcome.

Edge type derivation: the `:<X>_XREF` predicate is derivable from the
term sub-label via `_xref_rel_from_term_label("MeSHTerm") -> "MESH_XREF"`
(declared in `ontology_helpers`). Per-ontology modules don't need to
register their xref edge type explicitly — the helper computes it at
query time. Schema constants in `ONTOLOGY_XREF_RELS` are for prompt
rendering / dispatch documentation only.

Cleanup semantics: after backfill, sources with `dangling_xrefs = []`
(empty list, no more strings to resolve) keep the property set. We
don't `REMOVE` the property entirely because a future ontology import
might add new xref targets that resolve them retroactively — leaving
the empty list makes the schema explicit ("this term DID export
xrefs"). Storage cost is negligible (one empty-list slot per term).
"""

from __future__ import annotations

import logging

from knowledge_agent.kg.ontology_helpers import (
    _xref_rel_from_term_label,
)
from knowledge_agent.kg.schema import (
    ONTOLOGY_SUB_LABELS,
    ONTOLOGY_TERM_LABEL,
    ONTOLOGY_XREF_RELS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# backfill_resolved_xrefs — write resolved xref edges, clean dangling list.
# ---------------------------------------------------------------------------


async def backfill_resolved_xrefs(client,
) -> dict[str, dict[str, int]]:
    """Walk every shipped ontology, resolve dangling xrefs into edges.

    Per ontology sub-label (`:MeSHTerm`, `:GOTerm`, ...) two Cypher
    passes:

      1. **Resolve + write**. For each source term with a non-empty
         `dangling_xrefs` list, unwind the list, attempt a MATCH on
         `:OntologyTerm` keyed by the xref string; on hit, MERGE the
         `:<X>_XREF` edge. Targets that don't exist yet (e.g. xref
         points at an ontology not imported, or at an LCSH/EuroVoc
         URI we don't ship) are silently skipped by the inner MATCH —
         their strings stay in `dangling_xrefs` for a future backfill
         run after the missing ontology lands.

      2. **Strip resolved entries**. Walk source nodes that have at
         least one outgoing `:<X>_XREF` edge AND a `dangling_xrefs`
         list; remove from the list every entry whose string matches
         an existing edge target's `id`. Idempotent — re-running the
         pass on a fully-resolved corpus is a no-op.

    Returns a per-ontology dict:
      `{"mesh": {"n_edges_attempted": N, "n_sources_cleaned": M}, ...}`

    `n_edges_attempted` counts the rows produced by the resolve pass —
    real "new edges written" can be less when MERGE is idempotent
    against prior runs, but the number is a useful "this many resolved
    matches were found" signal.

    `n_sources_cleaned` is the number of source nodes whose
    `dangling_xrefs` list was rewritten (entries removed).

    **Per-ontology resilience is preserved**: a per-ontology Cypher
    failure (e.g. one ontology's MATCH errors) is caught inside
    `_backfill_one_ontology` / `_strip_resolved_entries`, recorded as
    zeros in the dict, and the walk continues. This is a deliberate
    domain-aware choice — one bad ontology shouldn't kill resolution
    of the other 17.

    **Session-level failures propagate** (driver dead, auth failed):
    the orchestrator boundary catches them.
    """
    results: dict[str, dict[str, int]] = {}
    async with client.driver.session() as session:
        for term_label in ONTOLOGY_SUB_LABELS:
            xref_rel = _xref_rel_from_term_label(term_label)
            attempted = await _backfill_one_ontology(
                session, term_label=term_label, xref_rel=xref_rel,
            )
            cleaned = await _strip_resolved_entries(
                session, term_label=term_label, xref_rel=xref_rel,
            )
            if attempted is None or cleaned is None:
                # Per-ontology failure: record zeros, keep going.
                results[term_label] = {
                    "n_edges_attempted": 0,
                    "n_sources_cleaned": 0,
                }
                continue
            results[term_label] = {
                "n_edges_attempted": attempted,
                "n_sources_cleaned": cleaned,
            }

    total_attempted = sum(r["n_edges_attempted"] for r in results.values())
    total_cleaned = sum(r["n_sources_cleaned"] for r in results.values())
    logger.info(
        "backfill_resolved_xrefs: %d edges attempted across %d ontologies; "
        "stripped resolved entries from %d source nodes",
        total_attempted, len(results), total_cleaned,
    )
    return results


async def _backfill_one_ontology(
    session, *, term_label: str, xref_rel: str,
) -> int | None:
    """Resolve dangling xrefs for one ontology sub-label.

    Walks source terms with a non-empty `dangling_xrefs` list, unwinds
    the list, and MERGEs `:<xref_rel>` edges to any `:OntologyTerm`
    whose `id` matches. Returns the count of MERGE rows produced
    (includes both newly-created and matched-existing edges — MERGE
    is idempotent, so re-running on a resolved corpus is a no-op
    but still counts matched rows).

    Returns None on Cypher failure (logged).
    """
    try:
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
        return int(row["n"]) if row else 0
    except Exception as exc:
        logger.warning(
            "backfill_resolved_xrefs (%s): resolve pass failed: %r",
            term_label, exc,
        )
        return None


async def _strip_resolved_entries(
    session, *, term_label: str, xref_rel: str,
) -> int | None:
    """Remove already-resolved entries from `dangling_xrefs`.

    For each source term with a `dangling_xrefs` list AND at least one
    outgoing `:<xref_rel>` edge, rewrites the list to exclude entries
    whose string matches an existing edge target's `id`. Source terms
    with no resolved edges are left untouched (their dangling_xrefs
    remain in full).

    Returns the count of source nodes whose lists were rewritten.
    `None` on Cypher failure (logged).

    Why a second pass instead of inlining into pass 1: the resolve
    pass's `UNWIND` expansion makes per-source SET semantics awkward
    (an aggregate over UNWIND'd rows would need a separate WITH).
    Two passes are clearer and the cost is one extra cheap MATCH per
    ontology.
    """
    try:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"OPTIONAL MATCH (s)-[:{xref_rel}]->"
            f"(t:{ONTOLOGY_TERM_LABEL}) "
            f"WITH s, collect(DISTINCT t.id) AS resolved "
            f"WHERE size(resolved) > 0 "
            f"SET s.dangling_xrefs = "
            f"  [x IN s.dangling_xrefs WHERE NOT x IN resolved] "
            f"RETURN count(s) AS n"
        )
        row = await result.single()
        return int(row["n"]) if row else 0
    except Exception as exc:
        logger.warning(
            "backfill_resolved_xrefs (%s): strip pass failed: %r",
            term_label, exc,
        )
        return None


# ---------------------------------------------------------------------------
# clear_xref_edges_for_ontology — wipe one ontology's xref surface.
# ---------------------------------------------------------------------------


async def clear_xref_edges_for_ontology(client, term_label: str,
) -> int:
    """Delete every outgoing xref edge from one ontology + clear the
    `dangling_xrefs` property on its terms.

    Inbound xref edges (from OTHER ontologies' terms pointing AT this
    ontology, e.g. `:GO_XREF` edges with their target on a `:MeSHTerm`)
    are LEFT INTACT. Removing them would corrupt the other ontology's
    xref surface without re-running its backfill; that's not the
    semantic the user asked for. To wipe both directions, call this
    for every ontology in turn.

    Returns the total of (edges deleted + properties cleared) — useful
    for the bulk_op summary.

    Idempotent: when no xref data exists for this ontology, both passes
    are 0-row no-ops and the function returns 0.

    Raises:
      - `ValueError` for an unknown `term_label`.
      - The original `neo4j.exceptions.*` for Cypher / driver failures;
        the orchestrator boundary catches and stores on the matching
        `_error: ErrorDetail | None` result field.
    """
    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(
            f"clear_xref_edges_for_ontology: unknown term_label "
            f"{term_label!r}; expected one of "
            f"{sorted(ONTOLOGY_SUB_LABELS)}"
        )

    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        edge_result = await session.run(
            f"MATCH (s:{term_label})-[r:{xref_rel}]->() "
            f"DELETE r "
            f"RETURN count(r) AS n"
        )
        edge_row = await edge_result.single()
        n_edges = int(edge_row["n"]) if edge_row else 0

        prop_result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"REMOVE s.dangling_xrefs "
            f"RETURN count(s) AS n"
        )
        prop_row = await prop_result.single()
        n_props = int(prop_row["n"]) if prop_row else 0

    logger.info(
        "clear_xref_edges_for_ontology (%s): deleted %d :%s edges, "
        "cleared dangling_xrefs on %d source nodes",
        term_label, n_edges, xref_rel, n_props,
    )
    return n_edges + n_props


# ---------------------------------------------------------------------------
# Diagnostics — counts used by install plans + status views.
# ---------------------------------------------------------------------------


async def count_dangling_xrefs(client, term_label: str,
) -> int:
    """Source nodes with a non-empty `dangling_xrefs` list.

    Per-ontology. Used by the `install_xrefs_plan` to surface "X
    xrefs available to resolve" before the user enables the layer.

    Raises `ValueError` for an unknown `term_label`; Cypher / driver
    failures propagate.
    """
    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(
            f"count_dangling_xrefs: unknown term_label {term_label!r}"
        )
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label}) "
            f"WHERE s.dangling_xrefs IS NOT NULL "
            f"  AND size(s.dangling_xrefs) > 0 "
            f"RETURN count(s) AS n"
        )
        row = await result.single()
        return int(row["n"]) if row else 0


async def count_xref_edges(client, term_label: str | None = None,
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
            result = await session.run(
                f"MATCH ()-[r:{pipe}]->() "
                f"RETURN count(r) AS n"
            )
            row = await result.single()
            return int(row["n"]) if row else 0

    if term_label not in ONTOLOGY_SUB_LABELS:
        raise ValueError(
            f"count_xref_edges: unknown term_label {term_label!r}"
        )
    xref_rel = _xref_rel_from_term_label(term_label)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{term_label})-[r:{xref_rel}]->() "
            f"RETURN count(r) AS n"
        )
        row = await result.single()
        return int(row["n"]) if row else 0
