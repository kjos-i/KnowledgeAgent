"""L7 linking pass - connect :Entity nodes to canonical ontology terms.

After an ontology has been imported (its `:MeSHTerm` / `:GOTerm` nodes
+ hierarchy edges live in Neo4j) AND L6 has written `:Entity` nodes
from chunk text, this module wires the two together. For each entity
it looks up its `key` against an in-memory label+synonym index built
from the imported ontology, and writes `(:Entity)-[:CANONICAL_TO]->
(:OntologyTerm)` edges where matches are found.

Two matching strategies are supported per ontology (configured via
`OntologyConfig.matching` in `corpus.toml`):

  - **exact** (default): the entity's `key` must match a term's
    primary label or one of its synonyms exactly (case-insensitive,
    since both sides are lowercased). Highest precision, lowest
    recall.
  - **fuzzy**: exact tried first; on miss, try simple morphological
    variants (plural/singular flip, hyphen<->space swap). Catches
    common variation at some risk of false positives. The edge
    records `strategy="fuzzy"` and a heuristic `confidence` so
    downstream callers can filter.

The match output is multi-hit by design: one entity can link to
multiple ontology terms (within an ontology when its key matches
several term synonyms; across ontologies when several enabled
ontologies cover the same concept). Each link is its own
`:CANONICAL_TO` edge.

Pipeline integration: `ensure_ontology_imported(client, name)` is
called by the ingestion pipeline for each enabled ontology; if the
ontology hasn't been imported yet, it triggers the heavy import then
runs a GLOBAL linking pass against all existing entities. On
subsequent ingests, the import is a no-op and only THIS doc's
entities are linked.

This module is a SHARED helper across ontologies (mirrors how
`entity_writes` is shared across extractors). Each per-ontology
module just registers itself in `ONTOLOGY_REGISTRY` below.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from knowledge_agent.kg.ontologies import (
    chebi_writes,
    cl_writes,
    dron_writes,
    eco_writes,
    efo_writes,
    envo_writes,
    fibo_writes,
    foodon_writes,
    go_writes,
    hpo_writes,
    mesh_writes,
    mondo_writes,
    ncbitaxon_writes,
    obi_writes,
    po_writes,
    pr_writes,
    so_writes,
    uberon_writes,
)
from knowledge_agent.kg.schema import (
    CANONICAL_TO_REL,
    CHEBI_TERM_LABEL,
    CHUNK_LABEL,
    CL_TERM_LABEL,
    DRON_TERM_LABEL,
    ECO_TERM_LABEL,
    EFO_TERM_LABEL,
    ENTITY_LABEL,
    ENVO_TERM_LABEL,
    FIBO_TERM_LABEL,
    FOODON_TERM_LABEL,
    GO_TERM_LABEL,
    HPO_TERM_LABEL,
    MENTIONS_REL,
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    NCBITAXON_TERM_LABEL,
    OBI_TERM_LABEL,
    PO_TERM_LABEL,
    PR_TERM_LABEL,
    SO_TERM_LABEL,
    UBERON_TERM_LABEL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry: maps ontology name -> per-ontology metadata used by the linking
# pass. Adding a new ontology (chunk 4 for MeSH, chunk 5 for GO, plus future
# ChEBI etc.) means appending one entry here.
# ---------------------------------------------------------------------------


ONTOLOGY_REGISTRY: dict[str, dict[str, Any]] = {
    "mesh": {
        "term_label": MESH_TERM_LABEL,
        "is_imported_fn": mesh_writes.is_imported,
        "import_fn": mesh_writes.import_mesh,
        "delete_fn": mesh_writes.delete_imported,
        "download_size_mb": mesh_writes.DOWNLOAD_SIZE_MB,
        "provenance": mesh_writes._MESH_PROVENANCE,
        "download_url": mesh_writes.MESH_DOWNLOAD_URL,
        "download_filename": mesh_writes.MESH_CACHE_FILENAME,
    },
    "go": {
        "term_label": GO_TERM_LABEL,
        "is_imported_fn": go_writes.is_imported,
        "import_fn": go_writes.import_go,
        "delete_fn": go_writes.delete_imported,
        "download_size_mb": go_writes.DOWNLOAD_SIZE_MB,
        "provenance": go_writes._GO_PROVENANCE,
        "download_url": go_writes.GO_DOWNLOAD_URL,
        "download_filename": go_writes.GO_CACHE_FILENAME,
    },
    "hpo": {
        "term_label": HPO_TERM_LABEL,
        "is_imported_fn": hpo_writes.is_imported,
        "import_fn": hpo_writes.import_hpo,
        "delete_fn": hpo_writes.delete_imported,
        "download_size_mb": hpo_writes.DOWNLOAD_SIZE_MB,
        "provenance": hpo_writes._HPO_PROVENANCE,
        "download_url": hpo_writes.HPO_DOWNLOAD_URL,
        "download_filename": hpo_writes.HPO_CACHE_FILENAME,
    },
    "uberon": {
        "term_label": UBERON_TERM_LABEL,
        "is_imported_fn": uberon_writes.is_imported,
        "import_fn": uberon_writes.import_uberon,
        "delete_fn": uberon_writes.delete_imported,
        "download_size_mb": uberon_writes.DOWNLOAD_SIZE_MB,
        "provenance": uberon_writes._UBERON_PROVENANCE,
        "download_url": uberon_writes.UBERON_DOWNLOAD_URL,
        "download_filename": uberon_writes.UBERON_CACHE_FILENAME,
    },
    "mondo": {
        "term_label": MONDO_TERM_LABEL,
        "is_imported_fn": mondo_writes.is_imported,
        "import_fn": mondo_writes.import_mondo,
        "delete_fn": mondo_writes.delete_imported,
        "download_size_mb": mondo_writes.DOWNLOAD_SIZE_MB,
        "provenance": mondo_writes._MONDO_PROVENANCE,
        "download_url": mondo_writes.MONDO_DOWNLOAD_URL,
        "download_filename": mondo_writes.MONDO_CACHE_FILENAME,
    },
    "chebi": {
        "term_label": CHEBI_TERM_LABEL,
        "is_imported_fn": chebi_writes.is_imported,
        "import_fn": chebi_writes.import_chebi,
        "delete_fn": chebi_writes.delete_imported,
        "download_size_mb": chebi_writes.DOWNLOAD_SIZE_MB,
        "provenance": chebi_writes._CHEBI_PROVENANCE,
        "download_url": chebi_writes.CHEBI_DOWNLOAD_URL,
        "download_filename": chebi_writes.CHEBI_CACHE_FILENAME,
    },
    "eco": {
        "term_label": ECO_TERM_LABEL,
        "is_imported_fn": eco_writes.is_imported,
        "import_fn": eco_writes.import_eco,
        "delete_fn": eco_writes.delete_imported,
        "download_size_mb": eco_writes.DOWNLOAD_SIZE_MB,
        "provenance": eco_writes._ECO_PROVENANCE,
        "download_url": eco_writes.ECO_DOWNLOAD_URL,
        "download_filename": eco_writes.ECO_CACHE_FILENAME,
    },
    "so": {
        "term_label": SO_TERM_LABEL,
        "is_imported_fn": so_writes.is_imported,
        "import_fn": so_writes.import_so,
        "delete_fn": so_writes.delete_imported,
        "download_size_mb": so_writes.DOWNLOAD_SIZE_MB,
        "provenance": so_writes._SO_PROVENANCE,
        "download_url": so_writes.SO_DOWNLOAD_URL,
        "download_filename": so_writes.SO_CACHE_FILENAME,
    },
    "pr": {
        "term_label": PR_TERM_LABEL,
        "is_imported_fn": pr_writes.is_imported,
        "import_fn": pr_writes.import_pr,
        "delete_fn": pr_writes.delete_imported,
        "download_size_mb": pr_writes.DOWNLOAD_SIZE_MB,
        "provenance": pr_writes._PR_PROVENANCE,
        "download_url": pr_writes.PR_DOWNLOAD_URL,
        "download_filename": pr_writes.PR_CACHE_FILENAME,
    },
    "cl": {
        "term_label": CL_TERM_LABEL,
        "is_imported_fn": cl_writes.is_imported,
        "import_fn": cl_writes.import_cl,
        "delete_fn": cl_writes.delete_imported,
        "download_size_mb": cl_writes.DOWNLOAD_SIZE_MB,
        "provenance": cl_writes._CL_PROVENANCE,
        "download_url": cl_writes.CL_DOWNLOAD_URL,
        "download_filename": cl_writes.CL_CACHE_FILENAME,
    },
    "po": {
        "term_label": PO_TERM_LABEL,
        "is_imported_fn": po_writes.is_imported,
        "import_fn": po_writes.import_po,
        "delete_fn": po_writes.delete_imported,
        "download_size_mb": po_writes.DOWNLOAD_SIZE_MB,
        "provenance": po_writes._PO_PROVENANCE,
        "download_url": po_writes.PO_DOWNLOAD_URL,
        "download_filename": po_writes.PO_CACHE_FILENAME,
    },
    "foodon": {
        "term_label": FOODON_TERM_LABEL,
        "is_imported_fn": foodon_writes.is_imported,
        "import_fn": foodon_writes.import_foodon,
        "delete_fn": foodon_writes.delete_imported,
        "download_size_mb": foodon_writes.DOWNLOAD_SIZE_MB,
        "provenance": foodon_writes._FOODON_PROVENANCE,
        "download_url": foodon_writes.FOODON_DOWNLOAD_URL,
        "download_filename": foodon_writes.FOODON_CACHE_FILENAME,
    },
    "envo": {
        "term_label": ENVO_TERM_LABEL,
        "is_imported_fn": envo_writes.is_imported,
        "import_fn": envo_writes.import_envo,
        "delete_fn": envo_writes.delete_imported,
        "download_size_mb": envo_writes.DOWNLOAD_SIZE_MB,
        "provenance": envo_writes._ENVO_PROVENANCE,
        "download_url": envo_writes.ENVO_DOWNLOAD_URL,
        "download_filename": envo_writes.ENVO_CACHE_FILENAME,
    },
    "ncbitaxon": {
        "term_label": NCBITAXON_TERM_LABEL,
        "is_imported_fn": ncbitaxon_writes.is_imported,
        "import_fn": ncbitaxon_writes.import_ncbitaxon,
        "delete_fn": ncbitaxon_writes.delete_imported,
        "download_size_mb": ncbitaxon_writes.DOWNLOAD_SIZE_MB,
        "provenance": ncbitaxon_writes._NCBITAXON_PROVENANCE,
        "download_url": ncbitaxon_writes.NCBITAXON_DOWNLOAD_URL,
        "download_filename": ncbitaxon_writes.NCBITAXON_CACHE_FILENAME,
    },
    "obi": {
        "term_label": OBI_TERM_LABEL,
        "is_imported_fn": obi_writes.is_imported,
        "import_fn": obi_writes.import_obi,
        "delete_fn": obi_writes.delete_imported,
        "download_size_mb": obi_writes.DOWNLOAD_SIZE_MB,
        "provenance": obi_writes._OBI_PROVENANCE,
        "download_url": obi_writes.OBI_DOWNLOAD_URL,
        "download_filename": obi_writes.OBI_CACHE_FILENAME,
    },
    "efo": {
        "term_label": EFO_TERM_LABEL,
        "is_imported_fn": efo_writes.is_imported,
        "import_fn": efo_writes.import_efo,
        "delete_fn": efo_writes.delete_imported,
        "download_size_mb": efo_writes.DOWNLOAD_SIZE_MB,
        "provenance": efo_writes._EFO_PROVENANCE,
        "download_url": efo_writes.EFO_DOWNLOAD_URL,
        "download_filename": efo_writes.EFO_CACHE_FILENAME,
    },
    "dron": {
        "term_label": DRON_TERM_LABEL,
        "is_imported_fn": dron_writes.is_imported,
        "import_fn": dron_writes.import_dron,
        "delete_fn": dron_writes.delete_imported,
        "download_size_mb": dron_writes.DOWNLOAD_SIZE_MB,
        "provenance": dron_writes._DRON_PROVENANCE,
        "download_url": dron_writes.DRON_DOWNLOAD_URL,
        "download_filename": dron_writes.DRON_CACHE_FILENAME,
    },
    "fibo": {
        "term_label": FIBO_TERM_LABEL,
        "is_imported_fn": fibo_writes.is_imported,
        "import_fn": fibo_writes.import_fibo,
        "delete_fn": fibo_writes.delete_imported,
        "download_size_mb": fibo_writes.DOWNLOAD_SIZE_MB,
        "provenance": fibo_writes._FIBO_PROVENANCE,
        # FIBO is multi-file: ~70 RDF files walked from a GitHub API
        # listing. Uses a custom download_fn instead of a single URL,
        # and a subdir instead of a single filename for the delete
        # side. `download_subdir` matches `FIBO_CACHE_SUBDIR`.
        "download_fn": fibo_writes._walk_and_cache_fibo,
        "download_subdir": fibo_writes.FIBO_CACHE_SUBDIR,
    },
}


# ---------------------------------------------------------------------------
# Import orchestration: idempotent ensure_imported called by the pipeline.
# ---------------------------------------------------------------------------


async def count_ontology_terms(client, ontology_name: str) -> int:
    """Count the term nodes of a given ontology (`:MeSHTerm` or `:GOTerm`).

    Used by `lifecycle.delete_ontology_plan` to show the user
    exactly how many term nodes will be wiped before they confirm. Also
    fine as a generic post-import inspection ("is MeSH actually in here?").

    Returns 0 when the ontology is known but no terms have been
    imported yet.

    Raises `ValueError` for an unknown `ontology_name`; Cypher / driver
    failures propagate.
    """
    entry = ONTOLOGY_REGISTRY.get(ontology_name)
    if entry is None:
        raise ValueError(f"count_ontology_terms: unknown ontology {ontology_name!r}")
    term_label = entry["term_label"]
    async with client.driver.session() as session:
        result = await session.run(f"MATCH (n:{term_label}) RETURN count(n) AS n")
        return int((await result.single())["n"])


async def count_canonical_links(client, ontology_name: str) -> int:
    """Count `:CANONICAL_TO` edges pointing at terms of a given ontology.

    Used by `lifecycle.delete_ontology_plan` so the user sees
    the canonical-linking impact of dropping the ontology (those edges
    die when the term nodes are wiped via DETACH DELETE).

    Raises `ValueError` for an unknown `ontology_name`; Cypher / driver
    failures propagate.
    """
    entry = ONTOLOGY_REGISTRY.get(ontology_name)
    if entry is None:
        raise ValueError(f"count_canonical_links: unknown ontology {ontology_name!r}")
    term_label = entry["term_label"]
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH ()-[r:{CANONICAL_TO_REL}]->(:{term_label}) RETURN count(r) AS n"
        )
        return int((await result.single())["n"])


async def ensure_ontology_imported(
    client,
    ontology_name: str,
    *,
    xrefs_mode: str = "none",
) -> bool:
    """Make sure the named ontology is imported into Neo4j.

    Returns `was_already_imported`: `True` when the ontology was
    already present in the KG (no-op short-circuit), `False` when the
    import actually ran to completion (newly imported / re-imported).
    The pipeline uses this to decide whether to run a global linking
    pass (first import = link everything that already exists) or just
    link the current doc's entities (subsequent ingests).

    Idempotent: when the ontology is already imported, no work happens
    and `True` returns. Note: the idempotent short-circuit does NOT
    re-apply xref logic to an already-imported ontology. If the user
    flipped `xrefs` from `"none"` to `"use"` after the ontology was
    already in the KG, run the `backfill_xrefs` bulk_op (which walks
    `dangling_xrefs` properties — none will exist for ontologies
    imported under the old `"none"` mode, so a force re-import is the
    right path in that case).

    `xrefs_mode` is forwarded to the per-ontology `import_fn`. Default
    `"none"` preserves the pre-L7-xrefs behaviour. See
    `write_ontology_terms` for the three accepted values and what each
    writes.

    Raises `ValueError` for an unknown ontology name. Import failures
    (network down, parse error, Cypher problem, empty extraction)
    propagate to the caller (typed-errors contract). The pipeline's
    per-ontology try/except is the orchestrator boundary that catches
    per-ontology so one bad ontology doesn't kill the walk of the
    others.
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown ontology name {ontology_name!r}. "
            f"Known ontologies: {sorted(ONTOLOGY_REGISTRY)}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    if await entry["is_imported_fn"](client):
        return True
    # import_fn returns True on newly imported, False on no-op. The
    # no-op branch is unreachable here since we already confirmed
    # not-imported above; either way the answer is "not already
    # imported" so the caller should run the global linking pass.
    await entry["import_fn"](client, xrefs_mode=xrefs_mode)
    return False


# ---------------------------------------------------------------------------
# Linking pass: top-level entry point + the matcher + the writes.
# ---------------------------------------------------------------------------


async def link_entities(
    client,
    ontology_name: str,
    matching_strategy: Literal["exact", "fuzzy"],
    *,
    doc_id: str | None = None,
) -> int:
    """Run the linking pass for one ontology.

    Args:
      ontology_name: name from `ONTOLOGY_REGISTRY` (e.g. "mesh", "go").
      matching_strategy: "exact" or "fuzzy". From the per-ontology
        OntologyConfig.matching setting.
      doc_id: when provided, only entities mentioned by chunks of THIS
        doc are linked. When None, the pass runs globally on all
        `:Entity` nodes not yet linked to this ontology. Use None on
        first-time import (after a new ontology lands), `doc_id` on
        subsequent ingests.

    Returns the count of `:CANONICAL_TO` edges written. Returns 0 when
    the ontology has no terms or no entities matched.

    Flips `e.canonicalised = true` on every linked `:Entity`. Existing
    `canonicalised` values are preserved (an entity already linked to
    another ontology keeps `canonicalised=true`).
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown ontology name {ontology_name!r}. Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    term_label = entry["term_label"]

    index = await _build_term_index(client, term_label)
    if not index:
        logger.info(
            "L7 linking (%s): no terms in Neo4j; skipping pass",
            ontology_name,
        )
        return 0

    entities = await _fetch_entities_to_link(client, term_label, doc_id)
    if not entities:
        logger.info(
            "L7 linking (%s): no entities to link%s",
            ontology_name,
            f" for doc {doc_id}" if doc_id else " (global)",
        )
        return 0

    edges: list[dict[str, Any]] = []
    for entity in entities:
        matches = _match_entity_key(entity["key"], index, matching_strategy)
        for term_id, confidence in matches:
            edges.append(
                {
                    "entity_key": entity["key"],
                    "entity_type": entity["entity_type"],
                    "term_id": term_id,
                    "strategy": matching_strategy,
                    "confidence": confidence,
                }
            )

    if not edges:
        logger.info(
            "L7 linking (%s): %d entities scanned, 0 matched",
            ontology_name,
            len(entities),
        )
        return 0

    written = await _write_canonical_links(client, edges, term_label)
    logger.info(
        "L7 linking (%s): wrote %d :CANONICAL_TO edges across %d entities",
        ontology_name,
        written,
        len(entities),
    )
    return written


# ---------------------------------------------------------------------------
# Matching helpers: build the index, match keys, generate fuzzy variants.
# ---------------------------------------------------------------------------


async def _build_term_index(client, term_label: str) -> dict[str, list[str]]:
    """Load all terms for one ontology into an in-memory lookup index.

    The index maps `lowercased_text -> list[term_id]`. Each term
    contributes one entry for its primary label and one for each
    synonym (already lowercased at import time). Multiple terms
    sharing the same label/synonym text produce a list of IDs - the
    caller writes one :CANONICAL_TO edge per ID.

    Returns an empty dict when the ontology has no terms loaded.
    Cypher / driver failures propagate to the orchestrator boundary.
    """
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (t:{term_label}) "
            f"RETURN t.id AS id, t.label AS label, "
            f"       t.synonyms AS synonyms"
        )
        records = await result.data()

    index: dict[str, list[str]] = {}
    for row in records:
        term_id = row["id"]
        # Per-term dedup: when a term's primary label and one of its
        # synonyms collapse to the same lowercased text, contribute
        # one entry, not two. Prevents the index from listing the same
        # term twice under the same lookup key.
        seen_for_term: set[str] = set()
        label = (row["label"] or "").lower()
        if label:
            seen_for_term.add(label)
            index.setdefault(label, []).append(term_id)
        for syn in row.get("synonyms") or []:
            if syn in seen_for_term:
                continue
            seen_for_term.add(syn)
            # Synonyms already lowercased at import time.
            index.setdefault(syn, []).append(term_id)
    return index


async def _fetch_entities_to_link(
    client, term_label: str, doc_id: str | None
) -> list[dict[str, str]]:
    """Return [{key, entity_type}, ...] for entities the pass should
    try to link.

    Two modes:
      - `doc_id` set: entities mentioned in THIS doc's chunks. Used
        on every ingest after the ontology is already imported.
      - `doc_id` None: entities NOT YET linked to this ontology
        anywhere in the graph. Used on first-time ontology import.

    Both queries return one row per unique (key, entity_type) pair.
    """
    if doc_id:
        cypher = (
            f"MATCH (e:{ENTITY_LABEL})<-[:{MENTIONS_REL}]-"
            f"(c:{CHUNK_LABEL} {{doc_id: $doc_id}}) "
            f"RETURN DISTINCT e.key AS key, e.entity_type AS entity_type"
        )
        params = {"doc_id": doc_id}
    else:
        cypher = (
            f"MATCH (e:{ENTITY_LABEL}) "
            f"WHERE NOT EXISTS {{ "
            f"  MATCH (e)-[:{CANONICAL_TO_REL}]->(:{term_label}) "
            f"}} "
            f"RETURN e.key AS key, e.entity_type AS entity_type"
        )
        params = {}
    async with client.driver.session() as session:
        result = await session.run(cypher, **params)
        return await result.data()


def _match_entity_key(
    key: str,
    index: dict[str, list[str]],
    strategy: Literal["exact", "fuzzy"],
) -> list[tuple[str, float | None]]:
    """Look up `key` in the index. Returns a list of (term_id, confidence).

    Exact hits have `confidence=None` (no scoring needed - the match
    is binary). Fuzzy hits have a heuristic confidence in [0, 1].

    Strategy A (exact): one dictionary lookup. Returns all term_ids
    sharing this exact lowercased text.

    Strategy B (fuzzy): tries exact first; on miss, generates simple
    morphological variants (plural/singular flip, hyphen/space swap)
    and looks each up. Returns the matches from the FIRST variant
    that hits (so the result respects variant ordering: more "likely"
    variants come first).
    """
    hits = index.get(key)
    if hits:
        return [(tid, None) for tid in hits]
    if strategy == "exact":
        return []

    for variant, confidence in _fuzzy_variants(key):
        hits = index.get(variant)
        if hits:
            return [(tid, confidence) for tid in hits]
    return []


def _fuzzy_variants(key: str) -> list[tuple[str, float]]:
    """Generate fuzzy variants of `key` with heuristic confidences.

    The variants are tried in order; the first match wins. Confidences
    reflect how "close" the variant is to the original:
      - Trailing 's' flip: 0.9 (handles plural/singular).
      - Hyphen/space swap: 0.85 (handles punctuation differences).
      - Multiple transformations combined: 0.75.

    Duplicates are removed while preserving order.
    """
    seen: set[str] = {key}
    variants: list[tuple[str, float]] = []

    def _add(text: str, score: float) -> None:
        if text and text != key and text not in seen:
            seen.add(text)
            variants.append((text, score))

    # Trailing 's' flip (singular/plural).
    if key.endswith("s") and len(key) > 1:
        _add(key[:-1], 0.9)
    else:
        _add(key + "s", 0.9)

    # Hyphen <-> space swap.
    if "-" in key:
        _add(key.replace("-", " "), 0.85)
    if " " in key:
        _add(key.replace(" ", "-"), 0.85)

    # Combined transformations (plural + hyphen swap, etc.).
    if "-" in key:
        no_hyphen = key.replace("-", " ")
        if no_hyphen.endswith("s") and len(no_hyphen) > 1:
            _add(no_hyphen[:-1], 0.75)
        else:
            _add(no_hyphen + "s", 0.75)

    return variants


# ---------------------------------------------------------------------------
# Cypher write for the :CANONICAL_TO edges.
# ---------------------------------------------------------------------------


async def _write_canonical_links(client, rows: list[dict[str, Any]], term_label: str) -> int:
    """Batch-write :CANONICAL_TO edges via UNWIND + MERGE.

    Each row carries the entity key + type (composite identity), the
    target term_id, the strategy used ("exact"/"fuzzy"), and an
    optional confidence (None for exact matches).

    MERGE on the edge means re-runs are idempotent: an already-linked
    edge stays put. ON CREATE SET only fires on new edges, so the
    strategy/confidence reflects the FIRST time the link was made.

    Sets `e.canonicalised = true` on every linked entity (even if the
    link already existed - cheap to be over-eager here).

    Returns the count of edge rows submitted (not necessarily the
    count of NEW edges created, since MERGE dedupes).
    """
    if not rows:
        return 0
    async with client.driver.session() as session:
        await session.run(
            f"UNWIND $rows AS row "
            f"MATCH (e:{ENTITY_LABEL} "
            f"  {{key: row.entity_key, entity_type: row.entity_type}}) "
            f"MATCH (t:{term_label} {{id: row.term_id}}) "
            f"MERGE (e)-[r:{CANONICAL_TO_REL}]->(t) "
            f"  ON CREATE SET r.strategy = row.strategy, "
            f"                r.confidence = row.confidence "
            f"SET e.canonicalised = true",
            rows=rows,
        )
    return len(rows)
