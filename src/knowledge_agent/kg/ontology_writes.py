"""Neo4j Cypher write helpers shared by all ontology importers.

One of three format families split out of the historical
`ontology_helpers.py` (2026-06-29):

  - `ontology_pronto.py` — OBO Foundry via `pronto` library
  - `ontology_rdf.py` — RDF/OWL/SKOS via `rdflib` library
  - `ontology_writes.py` (THIS FILE) — Neo4j Cypher writes
    (format-agnostic)

The write side is intentionally format-blind: all three readers
produce `OntologyTerm` records of the same shape, and these helpers
consume them via UNWIND batches. Per-ontology modules collapse to
thin wrappers of constants + delegating calls.

History: this family used to be prefixed `pronto_*` because only
OBO-via-pronto routed through them. Renamed 2026-06-25 to the
neutral `*_ontology_terms` family when the L7 xref / L10
cross_doc_xrefs ship made OWL + SKOS first-class alongside pronto.

Four public functions + two internal helpers:

  - `is_ontology_imported(client, term_label, ontology_name)` —
    boolean check used by `import_ontology_data` to short-circuit
    when the ontology is already present.
  - `delete_ontology_terms(client, term_label, ontology_name)` —
    `DETACH DELETE` all nodes carrying `term_label`.
  - `write_ontology_terms(client, terms, ...)` — the core write:
    MERGE nodes, MERGE hierarchy edges, optionally MERGE xref edges.
  - `import_ontology_data(client, url, cache_filename, ...)` — the
    high-level orchestrator: download via `ensure_cached`, delegate
    to the format-specific `read_and_extract` callback, write.
"""

from __future__ import annotations

import logging
from typing import Any

from knowledge_agent.kg.ontology_helpers import OntologyTerm, ensure_cached


logger = logging.getLogger(__name__)


def _xref_rel_from_term_label(term_label: str) -> str:
    """Derive the per-ontology xref edge type from the term sub-label.

    Examples: `"MeSHTerm" -> "MESH_XREF"`, `"ChEBITerm" -> "CHEBI_XREF"`,
    `"NCBITaxonTerm" -> "NCBITAXON_XREF"`.

    Strips the trailing "Term" suffix, uppercases, appends `"_XREF"`. The
    18 derived strings match the 18 `<X>_XREF_REL` constants declared in
    `kg/schema.py` by construction — per-ontology modules don't need to
    pass an explicit xref edge type through to this family of helpers.
    """
    base = term_label.removesuffix("Term").upper()
    return f"{base}_XREF"


def _validate_xrefs_mode(value: str) -> None:
    """Boundary check for `xrefs_mode`.

    The corpus-config layer enforces the same `Literal` at parse time,
    so this is a defense-in-depth check for direct callers (tests,
    smoke scripts). Raises `ValueError` on an unrecognised value.
    """
    if value not in ("none", "collect_only", "use"):
        raise ValueError(
            f"xrefs_mode must be 'none', 'collect_only', or 'use'; "
            f"got {value!r}."
        )


async def is_ontology_imported(client, *, term_label: str, ontology_name: str) -> bool:
    """True when at least one node with `term_label` exists in Neo4j.

    Generic for any per-ontology module routing through this family. The
    label is the only thing that differs across ontologies (`:GOTerm`,
    `:HPOTerm`, ...). `ontology_name` is used in log messages.

    Cypher failures propagate to the caller (typed-errors contract);
    `ensure_ontology_imported` is the orchestrator boundary that catches
    per-ontology so one bad lookup doesn't kill the walk.
    """
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (t:{term_label}) RETURN count(t) > 0 AS present"
        )
        row = await result.single()
        return bool(row and row["present"])


async def delete_ontology_terms(client, *, term_label: str, ontology_name: str,
) -> None:
    """DETACH DELETE every node carrying `term_label` plus its edges.

    Generic for any per-ontology module routing through this family.
    Idempotent — safe to call when the ontology was never imported.

    Cypher failures propagate to the caller (typed-errors contract).
    """
    async with client.driver.session() as session:
        await session.run(
            f"MATCH (t:{term_label}) DETACH DELETE t"
        )
    logger.info(
        "%s: deleted all :%s nodes + edges", ontology_name, term_label,
    )


async def write_ontology_terms(client,
    terms: list[OntologyTerm],
    *,
    term_label: str,
    hierarchy_rel: str,
    ontology_name: str,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:<term_label>` nodes + `:<hierarchy_rel>` edges,
    plus optional L7 cross-ontology xrefs.

    Generic for any per-ontology module routing through this family.
    Two mandatory passes:
      1. MERGE nodes by `id`; SET label / synonyms / definition.
      2. MERGE hierarchy edges between existing nodes (`hierarchy_rel`,
         which is per-ontology — `:GO_IS_A` / `:HPO_IS_A` / etc.).

    Two more conditional passes are added when `xrefs_mode != "none"`:
      3a. SET `dangling_xrefs` property on each term with at least one
          xref. The property stores the full list of cross-ontology xref
          strings as the source ontology declared them (verbatim, not
          normalised). Order-independent: re-importing the OTHER
          ontology later lets the backfill bulk_op resolve these into
          real edges without needing to revisit this ontology.
      3b. Only when `xrefs_mode == "use"`: also MERGE resolved
          `:<X>_XREF` edges (where `<X>` is the source ontology's
          uppercased label, derived via `_xref_rel_from_term_label`)
          for xrefs whose targets already exist as `:OntologyTerm`
          nodes. Targets that don't yet exist (e.g. an ontology not
          imported, or an LCSH/EuroVoc URI MeSH points at that we
          don't ship) are silently skipped by the `MATCH` — they
          stay recoverable via the `dangling_xrefs` property.

    Note: resolved xrefs are NOT removed from `dangling_xrefs` at write
    time. The backfill bulk_op handles cleanup (via REMOVE-list logic)
    so the import path stays simple.

    `xrefs_mode` accepts the three layer-flag states: `"none"` (no xref
    work at all), `"collect_only"` (store property, defer edges),
    `"use"` (store property AND write edges immediately).

    Empty input is a no-op (returns without any session). Raises
    `ValueError` for unknown `xrefs_mode`. Cypher failures propagate
    to the caller.
    """
    _validate_xrefs_mode(xrefs_mode)

    if not terms:
        logger.info("%s: no terms to write", ontology_name)
        return

    node_rows = [
        {
            "id": t.id,
            "label": t.label,
            "synonyms": list(t.synonyms),
            "definition": t.definition,
        }
        for t in terms
    ]
    hierarchy_rows = [
        {"child": t.id, "parent": p}
        for t in terms
        for p in t.parents
    ]
    # Only build xref payload when the layer actually wants it. Empty
    # `t.xrefs` is the common case; filtering keeps storage minimal
    # (no empty `dangling_xrefs = []` properties on terms that have
    # nothing to declare).
    xref_rows: list[dict[str, Any]] = []
    if xrefs_mode != "none":
        xref_rows = [
            {"source_id": t.id, "xrefs": list(t.xrefs)}
            for t in terms
            if t.xrefs
        ]
    xref_rel = _xref_rel_from_term_label(term_label)
    n_resolved_edges = 0

    async with client.driver.session() as session:
        await session.run(
            f"UNWIND $rows AS row "
            f"MERGE (t:OntologyTerm:{term_label} "
            f"  {{id: row.id}}) "
            f"SET t.label = row.label, "
            f"    t.synonyms = row.synonyms, "
            f"    t.definition = row.definition",
            rows=node_rows,
        )
        if hierarchy_rows:
            await session.run(
                f"UNWIND $rows AS row "
                f"MATCH (c:{term_label} {{id: row.child}}) "
                f"MATCH (p:{term_label} {{id: row.parent}}) "
                f"MERGE (c)-[:{hierarchy_rel}]->(p)",
                rows=hierarchy_rows,
            )
        if xref_rows:
            # Pass 3a: store the verbatim xref list on each source
            # term as `dangling_xrefs`. Always done when the layer
            # is active in any mode.
            await session.run(
                f"UNWIND $rows AS row "
                f"MATCH (s:{term_label} {{id: row.source_id}}) "
                f"SET s.dangling_xrefs = row.xrefs",
                rows=xref_rows,
            )
            if xrefs_mode == "use":
                # Pass 3b: write resolved edges to any xref targets
                # that already exist as :OntologyTerm nodes. Targets
                # that aren't present get silently skipped by the
                # inner MATCH — they stay in `dangling_xrefs` for
                # the backfill bulk_op to pick up later. The query
                # returns the count of edges actually written so we
                # can log it (resolved count differs from xref_rows
                # count whenever some xrefs don't resolve yet).
                result = await session.run(
                    f"UNWIND $rows AS row "
                    f"MATCH (s:{term_label} {{id: row.source_id}}) "
                    f"UNWIND row.xrefs AS xref_id "
                    f"MATCH (t:OntologyTerm {{id: xref_id}}) "
                    f"MERGE (s)-[r:{xref_rel}]->(t) "
                    f"RETURN count(r) AS n",
                    rows=xref_rows,
                )
                row = await result.single()
                n_resolved_edges = int(row["n"]) if row else 0

    logger.info(
        "%s: wrote %d :%s nodes + %d :%s edges",
        ontology_name, len(node_rows), term_label,
        len(hierarchy_rows), hierarchy_rel,
    )
    if xref_rows:
        logger.info(
            "%s: stored dangling_xrefs on %d terms (mode=%s); "
            "resolved %d :%s edges",
            ontology_name, len(xref_rows), xrefs_mode,
            n_resolved_edges, xref_rel,
        )


async def import_ontology_data(client,
    *,
    ontology_name: str,
    url: str,
    cache_filename: str,
    term_label: str,
    hierarchy_rel: str,
    read_and_extract,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Generic import: download + parse + write.

    Used by all per-ontology modules routing through this family.
    Idempotent — when `force=False` and the ontology is already
    imported, returns `False` (no-op). When the import actually runs
    to completion, returns `True`. When `force=True`, deletes existing
    data first and re-imports.

    Return contract under typed-errors:
      - `True`  — newly imported (or re-imported under force)
      - `False` — no-op because the ontology was already present
      - raises — download / parse / write failure; the orchestrator
        (`ensure_ontology_imported`'s per-ontology caller — the pipeline
        / backfill loop) is the boundary that catches per-ontology so
        one failure doesn't kill the walk of the others.

    `read_and_extract` is a callback `(Path) -> list[OntologyTerm]`
    supplied by the per-ontology module. Per-ontology modules keep
    their own `_read_and_extract` (typically a thin wrapper that reads
    the source file and extracts OntologyTerm records via the
    format-appropriate helper) so test fixtures that patch
    `<module>._read_and_extract` keep working.

    `xrefs_mode` controls L7 cross-ontology xref handling and is passed
    through to `write_ontology_terms`. Default `"none"` — no xref work.
    See `write_ontology_terms` for the three accepted values.

    `RuntimeError` is raised when the extraction yields zero terms,
    which almost always indicates an upstream format change (not a
    cache hit / disk error — those are distinct failure modes that
    propagate from `ensure_cached` / `read_and_extract`).
    """
    _validate_xrefs_mode(xrefs_mode)
    if not force and await is_ontology_imported(
        client, term_label=term_label, ontology_name=ontology_name,
    ):
        logger.info(
            "%s: already imported; use force=True to re-import",
            ontology_name,
        )
        return False

    if force:
        logger.info(
            "%s: force=True - dropping existing data before re-import",
            ontology_name,
        )
        await delete_ontology_terms(
            client, term_label=term_label, ontology_name=ontology_name,
        )

    logger.info("%s: downloading %s", ontology_name, url)
    # External boundary catches stay where they are domain-aware (per-
    # ontology resilience lives in the OUTER caller — see
    # `ensure_ontology_imported`); here we let download / parse failures
    # propagate so the caller can distinguish "network down" from
    # "Cypher problem" from "empty extraction" via the exception type.
    path = ensure_cached(url, cache_filename)
    terms = read_and_extract(path)

    if not terms:
        raise RuntimeError(
            f"{ontology_name}: extracted 0 terms - unexpected, aborting write"
        )

    await write_ontology_terms(
        client, terms,
        term_label=term_label,
        hierarchy_rel=hierarchy_rel,
        ontology_name=ontology_name,
        xrefs_mode=xrefs_mode,
    )
    return True
