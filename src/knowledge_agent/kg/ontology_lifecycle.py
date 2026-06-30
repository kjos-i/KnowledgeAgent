"""GUI-facing ontology lifecycle ops (KG reference-data admin).

Same plan/execute UI contract as the document `bulk_ops` module, but
operates on KG reference data (ontology nodes + hierarchy + canonical
links), not on user documents. Lives in `kg/` because that's the
domain - alongside `ontology_*_writes.py` and the registry in
`ontology_linking.py`.

Three ops, mirroring the install/run/uninstall split used by
`extractor_lifecycle.py` and `parser_lifecycle.py`:

  - `import_ontology` — download + parse + write the ontology's term
    nodes and hierarchy edges. Does NOT touch `:Entity` nodes. Pure
    data import.

  - `link_ontology` — run the linking pass: scan `:Entity` nodes
    (globally or scoped to one doc), match against the imported
    ontology's labels + synonyms, write `:CANONICAL_TO` edges, flip
    `e.canonicalised=true` on matched entities. Requires the ontology
    to already be imported; no-op (with `link_ok=False`) otherwise.

  - `delete_ontology` — wipe term nodes + hierarchy + the
    `:CANONICAL_TO` edges that point at them. Frees KG storage when
    the user no longer wants this ontology in the corpus.

The split lets the GUI offer two separate buttons (Install + Link) so
users can defer linking ("install ChEBI tonight, link tomorrow"),
re-link after backfilling more entities without re-importing, or
install an ontology to inspect terms before committing to linking.

Auto-on-ingest still bundles import + link in
`pipeline.ingest_document` so freshly ingested docs are immediately
canonicalised - that orchestration lives in `pipeline.py`, not here.

All three ops dispatch generically via `ONTOLOGY_REGISTRY` (in
`ontology_linking.py`); adding a new ontology means one new entry in
that registry, no changes here.
"""

import logging
from dataclasses import dataclass

from knowledge_agent.errors import ErrorDetail
from knowledge_agent.kg.client import get_kg_client
from knowledge_agent.kg.ontology_linking import ONTOLOGY_REGISTRY
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.ontology_xrefs import (
    count_dangling_xrefs,
    count_xref_edges,
)
from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS

# Re-export so callers continue to do
# `from knowledge_agent.kg.ontology_lifecycle import OntologyProvenance`.
# The dataclass itself lives in the leaf `ontology_provenance` module so
# per-ontology `_<NAME>_PROVENANCE = OntologyProvenance(...)` constants
# can build at module load time without a circular import back through
# this file (which depends on ONTOLOGY_REGISTRY which depends on the
# per-ontology modules).
__all__ = ["OntologyProvenance"]

logger = logging.getLogger(__name__)


# ---- import_ontology ----


@dataclass(frozen=True)
class ImportOntologyPlan:
    """Plan for a manual ontology import.

    Fields the dialog uses:
      - `already_imported`: whether the ontology is already in the KG;
        the dialog can say "Already imported" rather than dangle a
        useless "Import" button.
      - `download_size_mb`: hardcoded per-ontology constant so the user
        sees the cost up front ("MeSH (~115 MB download)"). When
        `provenance` is present this is `provenance.download_size_mb`
        (single source of truth via the registry build); kept as a
        separate field for back-compat with callers that construct the
        plan without provenance (tests, transitional code).
      - `xrefs_mode`: forwarded to the underlying import. When set to
        `"collect_only"` or `"use"`, the summary picks up an extra
        line about `dangling_xrefs` / resolved-edge writes so the user
        knows the import will do more than just term-node writes.
      - `provenance`: full per-ontology structured metadata when
        available. Defaults `None` for back-compat — existing tests
        and any transitional code paths that construct
        `ImportOntologyPlan` directly continue to work. When set, the
        summary picks up the rich provenance render (publisher,
        license, file format, description, covers_labels) instead of
        the bare legacy "Import X (~N MB)" line.
    """

    ontology_name: str
    already_imported: bool
    download_size_mb: int
    xrefs_mode: str = "none"
    provenance: OntologyProvenance | None = None
    extractor_candidates: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Pre-computed cross-link surface: ((extractor_name, matching_labels), ...).

    Filled by `import_ontology_plan` via `get_extractor_candidates`.
    Empty tuple when no shipped extractor pairs with this ontology
    (covers_labels=() OR no overlap with any extractor's
    emitted_labels). Surfaced by the rich summary's "Pairs well with"
    line so the user knows what to install alongside."""

    @property
    def summary(self) -> str:
        if self.already_imported:
            return f"{self.ontology_name} is already imported."
        base = self._base_summary()
        if self.xrefs_mode == "collect_only":
            return (
                f"{base} L7 xrefs layer = \"collect_only\": this "
                f"ontology's cross-ontology xref strings will be "
                f"stored as `dangling_xrefs` properties on each "
                f"source term. No `:<X>_XREF` edges are written at "
                f"import time; run the `backfill_xrefs` op (or flip "
                f"the layer to \"use\") to materialise resolved edges."
            )
        if self.xrefs_mode == "use":
            return (
                f"{base} L7 xrefs layer = \"use\": this ontology's "
                f"cross-ontology xref strings will be stored as "
                f"`dangling_xrefs` AND any xref whose target already "
                f"exists in the graph will get a resolved `:<X>_XREF` "
                f"edge written immediately. Xrefs pointing at "
                f"not-yet-imported ontologies stay dangling for a "
                f"later backfill."
            )
        return base

    def _base_summary(self) -> str:
        """The pre-xrefs-clause portion of the summary.

        Renders the rich provenance block when `provenance` is set;
        falls back to the legacy "Import X (~N MB)." one-liner
        otherwise so callers without provenance (tests, transitional
        code) still get something meaningful.
        """
        if self.provenance is None:
            return (
                f"Import {self.ontology_name} (~{self.download_size_mb} "
                f"MB download)."
            )
        p = self.provenance
        lines = [
            f"Import {p.full_name} ({p.ontology_name}) "
            f"— {p.download_size_mb} MB {p.file_format}.",
            f"  Publisher: {p.publisher}",
            f"  License: {p.license}",
            f"  Source: {p.source_url}",
            f"  Estimated terms: ~{p.estimated_terms:,}",
        ]
        if p.domain_tags:
            lines.append(
                f"  Domain tags: {', '.join(p.domain_tags)}"
            )
        if p.covers_labels:
            lines.append(
                f"  Covers entity labels: {', '.join(p.covers_labels)}"
            )
        lines.append(f"  {p.description}")
        if p.heavy_warning:
            lines.append(f"  WARNING: {p.heavy_warning}")
        # Cross-link surface: which shipped extractors emit labels this
        # ontology will canonicalise. Rendered after the description /
        # warning so the user sees the install context just before the
        # final xrefs clause appears.
        if self.extractor_candidates:
            lines.append("  Pairs well with extractors:")
            for ext_name, labels in self.extractor_candidates:
                lines.append(
                    f"    - {ext_name}: {', '.join(labels)}"
                )
        elif p.covers_labels:
            # covers_labels declared but no shipped extractor emits
            # them — surface that explicitly so the user knows to
            # configure entity_types on the llm adapter.
            lines.append(
                "  No shipped NER extractor emits the labels this "
                "ontology canonicalises — use the `llm` adapter "
                "with custom `entity_types` to bridge."
            )
        else:
            # covers_labels=() (ECO / FOODON / ENVO / OBI / FIBO) —
            # the ontology has no extractor-pairable surface at all.
            lines.append(
                "  Note: this ontology has no extractor-pairable "
                "entity surface (covers_labels=()) — useful for its "
                "own queries / cross-doc xrefs, not for entity "
                "canonicalisation."
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class ImportOntologyResult:
    """Outcome of `import_ontology_execute`.

    `did_import` is False when the ontology was already imported (the
    op was a no-op). `import_ok` is False when the import was attempted
    but failed (network down, parse error, KG write failure).

    `import_error` carries the typed-error detail (message +
    exception_type) when `import_ok=False` so the UI dialog can
    surface the real cause instead of a generic "import failed". None
    on success and on the no-op path.
    """

    ontology_name: str
    did_import: bool
    import_ok: bool
    import_error: ErrorDetail | None = None


async def import_ontology_plan(
    ontology_name: str, *, xrefs_mode: str = "none",
) -> ImportOntologyPlan:
    """Build a plan for importing the named ontology.

    `xrefs_mode` defaults to `"none"` so callers that don't care about
    L7 xrefs get the same behaviour as before. Pass
    `"collect_only"` / `"use"` to surface xref handling in the plan
    summary AND have it propagated through `import_ontology_execute`
    to the per-ontology `import_fn`.

    Reads `provenance` from the registry entry when present; the
    resulting plan's summary renders the rich publisher / license /
    domain / covers_labels block. Registry entries without a
    `"provenance"` key (transitional state during the 18-ontology
    rollout) produce a plan with `provenance=None` — summary falls
    back to the legacy one-liner.

    Raises `ValueError` for an unknown ontology name (registry miss).
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"import_ontology_plan: unknown ontology {ontology_name!r}. "
            f"Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    kg_client = get_kg_client()
    already_imported = entry["is_imported_fn"](kg_client)
    # Pre-compute the cross-link surface at plan-build time so the
    # rendered summary stays deterministic (no surprise registry
    # reads between dialog show + execute).
    extractor_candidates = get_extractor_candidates(ontology_name)
    return ImportOntologyPlan(
        ontology_name=ontology_name,
        already_imported=already_imported,
        download_size_mb=entry["download_size_mb"],
        xrefs_mode=xrefs_mode,
        provenance=entry.get("provenance"),
        extractor_candidates=extractor_candidates,
    )


async def import_ontology_execute(
    plan: ImportOntologyPlan,
) -> ImportOntologyResult:
    """Perform the import promised by `plan`.

    Already-imported is treated as success with `did_import=False`
    (idempotent). On a real first import, the entry's `import_fn`
    runs (network download + parse + Cypher writes) with the plan's
    `xrefs_mode` forwarded. Under the typed-errors contract, the
    `import_fn` raises on failure — we catch here at the orchestrator
    boundary so the user-facing result still carries `did_import` /
    `import_ok` rather than the raw exception.
    """
    if plan.already_imported:
        return ImportOntologyResult(
            ontology_name=plan.ontology_name,
            did_import=False,
            import_ok=True,
        )

    entry = ONTOLOGY_REGISTRY[plan.ontology_name]
    kg_client = get_kg_client()
    import_error: ErrorDetail | None = None
    try:
        entry["import_fn"](kg_client, xrefs_mode=plan.xrefs_mode)
        import_ok = True
    except Exception as exc:
        logger.warning(
            "import_ontology_execute (%s): import_fn failed: %r",
            plan.ontology_name, exc,
        )
        import_ok = False
        import_error = ErrorDetail.from_exception(exc)
    return ImportOntologyResult(
        ontology_name=plan.ontology_name,
        did_import=True,
        import_ok=import_ok,
        import_error=import_error,
    )


# ---- link_ontology ----


@dataclass(frozen=True)
class LinkOntologyPlan:
    """Plan for running the L7 linking pass over an imported ontology.

    Fields the dialog uses:
      - `is_imported`: whether the ontology terms are in the KG. False
        means the dialog should say "Install the ontology first" and
        the user cannot proceed - linking against nothing produces
        nothing.
      - `scope`: "global" to scan every `:Entity` node not yet linked
        to this ontology; a `doc_id` string to scope to a single doc.
        Drives both the summary and the eventual `:CANONICAL_TO`
        write set.
      - `matching_strategy`: "exact" or "fuzzy" - the per-ontology
        config setting that the user picked in `corpus.toml`.
    """

    ontology_name: str
    is_imported: bool
    scope: str
    matching_strategy: str

    @property
    def summary(self) -> str:
        if not self.is_imported:
            return (
                f"{self.ontology_name} is not imported - cannot link "
                f"entities. Install the ontology first."
            )
        scope_phrase = (
            "globally across all unlinked entities"
            if self.scope == "global"
            else f"for entities in doc {self.scope}"
        )
        return (
            f"Link entities to {self.ontology_name} {scope_phrase} "
            f"(matching: {self.matching_strategy})."
        )


@dataclass(frozen=True)
class LinkOntologyResult:
    """Outcome of `link_ontology_execute`.

    `n_links_written` is the count of `:CANONICAL_TO` edges submitted
    to MERGE - already-existing edges are no-ops at the Cypher level
    but still counted here. `link_ok` is False when the ontology was
    not imported (the op short-circuited) or the underlying linking
    pass crashed.

    `link_error` carries the typed-error detail when `link_ok=False`
    due to a crash (not for the not-imported short-circuit, which is
    a user-state condition rather than a runtime failure).
    """

    ontology_name: str
    scope: str
    n_links_written: int
    link_ok: bool
    link_error: ErrorDetail | None = None


async def link_ontology_plan(
    ontology_name: str,
    matching_strategy: str = "exact",
    *,
    scope: str = "global",
) -> LinkOntologyPlan:
    """Build a plan for linking entities to the named ontology.

    `scope` is "global" (the default) or a `doc_id` string. Pre-flight
    check: `is_imported_fn` says whether the ontology terms exist; if
    not, the plan reports `is_imported=False` and the GUI should
    redirect the user to Install first.

    Raises `ValueError` for an unknown ontology name or an invalid
    `matching_strategy`.
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"link_ontology_plan: unknown ontology {ontology_name!r}. "
            f"Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    if matching_strategy not in ("exact", "fuzzy"):
        raise ValueError(
            f"link_ontology_plan: matching_strategy must be 'exact' or "
            f"'fuzzy', got {matching_strategy!r}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    kg_client = get_kg_client()
    is_imported = entry["is_imported_fn"](kg_client)
    return LinkOntologyPlan(
        ontology_name=ontology_name,
        is_imported=is_imported,
        scope=scope,
        matching_strategy=matching_strategy,
    )


async def link_ontology_execute(plan: LinkOntologyPlan) -> LinkOntologyResult:
    """Perform the linking pass promised by `plan`.

    Not-imported short-circuits to `link_ok=False, n_links_written=0`
    (linking against nothing is the user's mistake, not a runtime
    failure - the plan already flagged it). Otherwise dispatches to
    `kg_client.link_entities_to_ontology` with the plan's scope +
    matching strategy.

    Under the typed-errors contract, the underlying linking pass
    propagates Cypher / driver failures; this boundary catches them
    and reports `link_ok=False` + `link_error=ErrorDetail(...)` so the
    UI dialog can surface the cause. Zero with `link_ok=True` is a
    valid "scanned, nothing matched" outcome.
    """
    if not plan.is_imported:
        return LinkOntologyResult(
            ontology_name=plan.ontology_name,
            scope=plan.scope,
            n_links_written=0,
            link_ok=False,
        )
    kg_client = get_kg_client()
    doc_id = None if plan.scope == "global" else plan.scope
    try:
        n = await kg_client.link_entities_to_ontology(
            plan.ontology_name,
            plan.matching_strategy,
            doc_id=doc_id,
        )
        return LinkOntologyResult(
            ontology_name=plan.ontology_name,
            scope=plan.scope,
            n_links_written=int(n),
            link_ok=True,
        )
    except Exception as exc:
        logger.warning(
            "link_ontology_execute (%s): linking pass failed: %r",
            plan.ontology_name, exc,
        )
        return LinkOntologyResult(
            ontology_name=plan.ontology_name,
            scope=plan.scope,
            n_links_written=0,
            link_ok=False,
            link_error=ErrorDetail.from_exception(exc),
        )


# ---- delete_ontology ----


@dataclass(frozen=True)
class DeleteOntologyPlan:
    """Plan for wiping an ontology from the KG.

    Counts come from live KG queries (`count_ontology_terms`,
    `count_canonical_links`) - the dialog reports the exact impact
    ("Delete MeSH? 30,142 terms and 18 canonical links will be
    removed.") before the user confirms.

    If `is_imported` is False, the dialog should treat the op as a
    no-op and not even offer a delete button - nothing to remove.
    """

    ontology_name: str
    is_imported: bool
    n_terms: int
    n_canonical_links: int

    @property
    def summary(self) -> str:
        if not self.is_imported:
            return f"{self.ontology_name} is not imported - nothing to delete."
        return (
            f"Delete {self.ontology_name}: {self.n_terms} terms and "
            f"{self.n_canonical_links} canonical links will be removed."
        )


@dataclass(frozen=True)
class DeleteOntologyResult:
    """Outcome of `delete_ontology_execute`.

    `did_delete` is False when the ontology wasn't imported (op was a
    no-op). `delete_ok` is False when the delete was attempted but the
    underlying Cypher raised (KG error). `delete_error` carries the
    typed-error detail when `delete_ok=False`; None for the no-op
    path and on success.
    """

    ontology_name: str
    did_delete: bool
    delete_ok: bool
    delete_error: ErrorDetail | None = None


async def delete_ontology_plan(ontology_name: str) -> DeleteOntologyPlan:
    """Build a plan for deleting the named ontology.

    Three KG round-trips: `is_imported`, `count_ontology_terms`,
    `count_canonical_links`. Cheap; runs once when the user opens
    the confirm dialog.

    Counts default to 0 when the ontology isn't imported (the count
    queries are skipped). Under the typed-errors contract, Cypher
    failures in the count primitives propagate to the caller.
    Raises `ValueError` for an unknown ontology name.
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"delete_ontology_plan: unknown ontology {ontology_name!r}. "
            f"Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    kg_client = get_kg_client()
    is_imported = entry["is_imported_fn"](kg_client)
    if not is_imported:
        return DeleteOntologyPlan(
            ontology_name=ontology_name,
            is_imported=False,
            n_terms=0,
            n_canonical_links=0,
        )
    n_terms = await kg_client.count_ontology_terms(ontology_name)
    n_links = await kg_client.count_canonical_links(ontology_name)
    return DeleteOntologyPlan(
        ontology_name=ontology_name,
        is_imported=True,
        n_terms=n_terms,
        n_canonical_links=n_links,
    )


async def delete_ontology_execute(
    plan: DeleteOntologyPlan,
) -> DeleteOntologyResult:
    """Perform the delete promised by `plan`.

    Not-imported is treated as a no-op success. On an imported
    ontology, the entry's `delete_fn` runs (DETACH DELETE on the term
    nodes; `:CANONICAL_TO` edges die with them). Under the typed-errors
    contract, the `delete_fn` raises on failure — we catch here at the
    orchestrator boundary so the user-facing result carries
    `delete_ok=False` rather than the raw exception.
    """
    if not plan.is_imported:
        return DeleteOntologyResult(
            ontology_name=plan.ontology_name,
            did_delete=False,
            delete_ok=True,
        )

    entry = ONTOLOGY_REGISTRY[plan.ontology_name]
    kg_client = get_kg_client()
    delete_error: ErrorDetail | None = None
    try:
        entry["delete_fn"](kg_client)
        delete_ok = True
    except Exception as exc:
        logger.warning(
            "delete_ontology_execute (%s): delete_fn failed: %r",
            plan.ontology_name, exc,
        )
        delete_ok = False
        delete_error = ErrorDetail.from_exception(exc)
    return DeleteOntologyResult(
        ontology_name=plan.ontology_name,
        did_delete=True,
        delete_ok=delete_ok,
        delete_error=delete_error,
    )


# ---- install_xrefs (L7 layer flip planning) ----


@dataclass(frozen=True)
class InstallXrefsPlan:
    """Plan describing what flipping `layers.xrefs` to a new mode
    means for the corpus.

    Built from a live KG snapshot — counts how many ontologies are
    currently imported, how many source terms hold `dangling_xrefs`
    awaiting resolution, and how many resolved `:<X>_XREF` edges
    already exist. The summary turns those facts plus the target
    mode into a user-facing description.

    Read-only: no `execute` side. The actual flip is a `corpus.toml`
    edit by the user / GUI; the data materialised by the flip happens
    on the next ingest OR via the `backfill_xrefs` bulk_op.

    Fields:
      - `new_xrefs_mode`: the mode the user is about to switch to
        (`"none"` / `"collect_only"` / `"use"`). Validated at
        construction.
      - `n_imported_ontologies`: 0-18, how many of the shipped
        sub-labels currently have at least one term in the KG.
      - `n_dangling_xref_sources`: total source terms (summed across
        all imported ontologies) whose `dangling_xrefs` list is
        non-empty. Surfaces "X resolvable strings sitting on disk".
      - `n_existing_xref_edges`: total `:<X>_XREF` edges currently
        materialised (across all 18 typed edges). Lets the user see
        if the layer has done work in a prior session.
    """

    new_xrefs_mode: str
    n_imported_ontologies: int
    n_dangling_xref_sources: int
    n_existing_xref_edges: int

    def __post_init__(self) -> None:
        if self.new_xrefs_mode not in ("none", "collect_only", "use"):
            raise ValueError(
                "new_xrefs_mode must be 'none', 'collect_only', or "
                f"'use'; got {self.new_xrefs_mode!r}."
            )

    @property
    def summary(self) -> str:
        # Always-present state snapshot.
        state = (
            f"{self.n_imported_ontologies} of "
            f"{len(ONTOLOGY_SUB_LABELS)} shipped ontologies currently "
            f"imported; {self.n_dangling_xref_sources} terms hold "
            f"dangling xref strings; "
            f"{self.n_existing_xref_edges} resolved :<X>_XREF edges "
            f"in the graph."
        )

        if self.new_xrefs_mode == "none":
            return (
                f"Set L7 xrefs layer to \"none\". {state} Future "
                "imports will skip xref extraction entirely. Existing "
                "dangling_xrefs properties and resolved edges are NOT "
                "removed by this flip — run `clear_xref_edges` "
                "per-ontology if you want to wipe them."
            )
        if self.new_xrefs_mode == "collect_only":
            return (
                "Set L7 xrefs layer to \"collect_only\". "
                f"{state} Future imports will extract xrefs and store "
                "them as `dangling_xrefs` properties, but will NOT "
                "write `:<X>_XREF` edges. Already-imported ontologies "
                "stay unchanged — force-re-import them or run the "
                "future-targeted `backfill_xrefs` op to start "
                "collecting their xrefs."
            )
        # "use"
        return (
            f"Set L7 xrefs layer to \"use\". {state} Future imports "
            "will extract xrefs AND immediately write `:<X>_XREF` "
            "edges for every xref whose target already exists. Xrefs "
            "pointing at not-yet-imported ontologies stay in "
            "`dangling_xrefs` for a later backfill. Already-imported "
            "ontologies stay unchanged — force-re-import them or run "
            "`backfill_xrefs` to materialise their xrefs."
        )


async def install_xrefs_plan(client, new_xrefs_mode: str) -> InstallXrefsPlan:
    """Build the read-only plan summarising what flipping
    `layers.xrefs` to `new_xrefs_mode` means against the current
    KG state.

    Walks all 18 shipped term sub-labels and counts via the
    `kg.ontology_xrefs` diagnostics. Per-ontology fail-soft (a None
    return from a count primitive is treated as 0 — the plan still
    surfaces what we DID measure).

    Raises `ValueError` for an unknown `new_xrefs_mode`.
    """
    if new_xrefs_mode not in ("none", "collect_only", "use"):
        raise ValueError(
            "install_xrefs_plan: new_xrefs_mode must be 'none', "
            f"'collect_only', or 'use'; got {new_xrefs_mode!r}."
        )

    n_imported = 0
    n_dangling = 0
    for term_label in ONTOLOGY_SUB_LABELS:
        # Cheap per-ontology probe: count_dangling_xrefs returns the
        # count of source nodes with a non-empty list. If a label
        # has any terms at all (`>= 0`), `is_imported_fn` would have
        # returned True; we infer "imported" by counting any term
        # node via the same dangling probe (which returns 0 when
        # imported but no dangling, and None on Cypher error).
        dangling = count_dangling_xrefs(client, term_label)
        if dangling is None:
            # Cypher error for this sub-label — skip; the plan still
            # builds with the data we got from the other 17.
            continue
        n_dangling += dangling
        # Cheap second probe: any term at all? Reuse count_xref_edges
        # to ask "are there outgoing edges from this label?" — that's
        # not exactly "is imported" but is a strict subset (no edges
        # without nodes). For "imported" the truer probe is the
        # registry entry's `is_imported_fn`, so use that.
        entry = ONTOLOGY_REGISTRY.get(_term_label_to_registry_key(term_label))
        if entry is not None and entry["is_imported_fn"](client):
            n_imported += 1

    n_edges = count_xref_edges(client, None)
    n_edges_safe = n_edges if n_edges is not None else 0

    return InstallXrefsPlan(
        new_xrefs_mode=new_xrefs_mode,
        n_imported_ontologies=n_imported,
        n_dangling_xref_sources=n_dangling,
        n_existing_xref_edges=n_edges_safe,
    )


# Mapping from term sub-label (e.g. "MeSHTerm") to the registry key
# (e.g. "mesh") so `install_xrefs_plan` can route through the existing
# `is_imported_fn`. Built once at module load.
_TERM_LABEL_TO_REGISTRY_KEY: dict[str, str] = {
    entry["term_label"]: name
    for name, entry in ONTOLOGY_REGISTRY.items()
}


def _term_label_to_registry_key(term_label: str) -> str | None:
    return _TERM_LABEL_TO_REGISTRY_KEY.get(term_label)


# ---- install_cross_doc_xrefs (L10 layer flip planning) ----


@dataclass(frozen=True)
class InstallCrossDocXrefsPlan:
    """Plan describing what flipping `layers.cross_doc_xrefs` means
    for the corpus.

    Read-only: no `execute` side. Flipping the flag changes
    `corpus.toml`; the actual L10 edges materialise on the next
    ingest (per-doc trigger) OR via the `recompute_cross_doc_xrefs`
    bulk_op (graph-wide).

    Fields:
      - `enabling`: True for off->on, False for on->off. The summary
        uses this to choose between rebuild-prediction vs cleanup
        commentary.
      - `n_docs`: corpus size (count of `:Document` + `:Artifact`
        nodes). The L10 rebuild's cost scales with N^2 for the worst-
        case pair-overlap query, so this is the user-visible cost
        signal.
      - `n_existing_l10_edges`: live count of `:RELATED_BY_XREF`
        edges currently in the graph. Tells the user how much L10
        work has accumulated so far.
      - `entities_layer_on`: dependency check — L10 needs L6a
        entities. The `CorpusConfig` validator already enforces this
        at flip time, but the plan still reports the live state for
        transparency.
      - `xrefs_layer_use`: dependency check — L10 needs xrefs in
        `"use"` mode for the typed edges to drive equivalence.
    """

    enabling: bool
    n_docs: int
    n_existing_l10_edges: int
    entities_layer_on: bool
    xrefs_layer_use: bool

    @property
    def summary(self) -> str:
        if not self.enabling:
            return (
                f"Disable L10 cross_doc_xrefs. "
                f"{self.n_existing_l10_edges} existing "
                ":RELATED_BY_XREF edges in the graph; flipping the "
                "flag off does NOT remove them. Per-doc ingest will "
                "stop refreshing the focal doc's L10 edges, but the "
                "edges sit there until the next graph-wide recompute "
                "or until each focal doc is deleted (DETACH DELETE "
                "cascade)."
            )

        # Enabling: dependency checks first.
        deps = []
        if not self.entities_layer_on:
            deps.append("`entities=true`")
        if not self.xrefs_layer_use:
            deps.append("`xrefs=\"use\"`")
        if deps:
            missing = " AND ".join(deps)
            return (
                "Enable L10 cross_doc_xrefs — but the corpus config "
                f"does not yet satisfy the dependency: needs {missing}. "
                f"Set those first; the `CorpusConfig` model validator "
                f"will refuse to load until the dependency chain is "
                f"complete."
            )

        # Happy path: dependencies satisfied.
        return (
            f"Enable L10 cross_doc_xrefs. "
            f"{self.n_docs} :Document + :Artifact nodes in the corpus. "
            f"{self.n_existing_l10_edges} pre-existing "
            ":RELATED_BY_XREF edges (stale from a previous run, if "
            "any). After the flip: per-doc ingest writes the focal "
            "doc's L10 edges; the `recompute_cross_doc_xrefs` "
            "bulk_op rebuilds graph-wide. Expect minutes on a corpus "
            "with thousands of docs; the per-doc cost stays small."
        )


async def install_cross_doc_xrefs_plan(
    client,
    *,
    enabling: bool,
    entities_layer_on: bool,
    xrefs_layer_use: bool,
) -> InstallCrossDocXrefsPlan:
    """Build the read-only plan summarising an L10 layer flip.

    `entities_layer_on` + `xrefs_layer_use` come from the caller's
    view of the CURRENT corpus config (NOT the proposed config —
    the plan surfaces the gap when the user is about to flip L10 on
    without first satisfying the dependency).

    Fail-soft: when a Cypher count primitive returns None, the field
    is filled with 0 — the plan still builds and the dialog shows
    the partial picture.
    """
    n_docs = _count_focal_nodes(client)
    n_l10 = _count_l10_edges(client)
    return InstallCrossDocXrefsPlan(
        enabling=enabling,
        n_docs=n_docs if n_docs is not None else 0,
        n_existing_l10_edges=n_l10 if n_l10 is not None else 0,
        entities_layer_on=entities_layer_on,
        xrefs_layer_use=xrefs_layer_use,
    )


def _count_focal_nodes(client) -> int | None:
    """Count `:Document` + `:Artifact` nodes in the KG.

    Used by `install_cross_doc_xrefs_plan` to surface the L10 rebuild
    cost. Returns the int count, or None on Cypher error (logged).
    """
    try:
        with client.driver.session() as session:
            result = session.run(
                "MATCH (d:Document|Artifact) RETURN count(d) AS n"
            )
            row = result.single()
            return int(row["n"]) if row else 0
    except Exception as exc:
        logger.warning("_count_focal_nodes failed: %r", exc)
        return None


def _count_l10_edges(client) -> int | None:
    """Count `:RELATED_BY_XREF` edges in the KG.

    Used by `install_cross_doc_xrefs_plan` to surface "this many L10
    edges already in place". Returns the int count, or None on Cypher
    error (logged).
    """
    try:
        with client.driver.session() as session:
            result = session.run(
                "MATCH ()-[r:RELATED_BY_XREF]-() "
                "RETURN count(r) AS n"
            )
            row = result.single()
            return int(row["n"]) if row else 0
    except Exception as exc:
        logger.warning("_count_l10_edges failed: %r", exc)
        return None


# ---------------------------------------------------------------------------
# Cross-link surface (extractor ↔ ontology pairing for install dialogs).
#
# The mapping is computed from `OntologyProvenance.covers_labels`
# (declared per-ontology — see kg/ontology_provenance.py) joined
# against each extractor's `emitted_labels` (declared per-adapter in
# the EXTRACTOR_REGISTRY from entity_extractors/extractor_lifecycle.py).
# Both registries are leaf data — putting the join here keeps the
# coupling one-way (kg/ → entity_extractors/) and avoids touching the
# extractor side beyond the lazy imports below.
#
# Lazy imports inside each helper so this module can still be imported
# before `entity_extractors` is even installable (the registry's
# heavy_warning copies don't trigger Flair / PyTorch — just plain
# tuples — so the cost is minimal).
# ---------------------------------------------------------------------------


def get_canonicalization_candidates(
    extractor_name: str,
) -> dict[str, tuple[str, ...]]:
    """For each label this extractor emits, return the ontology names
    whose `covers_labels` include it.

    Returns a dict `{label: (ontology_name, ...)}` where each
    ontology-name tuple is sorted alphabetically for determinism. A
    label with no matching ontology gets an empty tuple value (NOT
    omitted from the dict) so the dialog can surface "no shipped
    ontology covers this" per-label.

    Open-vocabulary extractors (LLM, `emitted_labels=None`) return an
    empty dict — the cross-link mapping is undefined for runtime-chosen
    label vocabularies. Callers showing the LLM install plan should
    render a "any ontology, lexically" disclaimer instead.

    Raises `ValueError` for an unknown extractor (registry miss).
    """
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        EXTRACTOR_REGISTRY,
    )
    if extractor_name not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"get_canonicalization_candidates: unknown extractor "
            f"{extractor_name!r}. Known: {sorted(EXTRACTOR_REGISTRY)}."
        )
    emitted = EXTRACTOR_REGISTRY[extractor_name].get("emitted_labels")
    if emitted is None:
        return {}

    out: dict[str, tuple[str, ...]] = {}
    for label in emitted:
        matches: list[str] = []
        for ont_name, entry in ONTOLOGY_REGISTRY.items():
            prov = entry.get("provenance")
            if prov is None:
                continue
            if label in prov.covers_labels:
                matches.append(ont_name)
        out[label] = tuple(sorted(matches))
    return out


def get_extractor_candidates(
    ontology_name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """For one ontology, return the extractors whose `emitted_labels`
    overlap with its `covers_labels`.

    Returns a tuple of `(extractor_name, matching_labels)` pairs,
    sorted by overlap size (best match first) then by extractor name
    (alphabetical tie-break for determinism).

    Open-vocabulary extractors (LLM, `emitted_labels=None`) are
    SKIPPED in this surface — they technically match every ontology
    lexically at runtime, but listing them under every ontology's
    "pairs well with" would be noise. The install dialog should
    mention LLM separately as a domain-agnostic fallback.

    Returns an empty tuple when:
      - the ontology has `covers_labels=()` (no extractor pairings
        possible; ECO / FOODON / ENVO / OBI / FIBO case)
      - no shipped extractor emits any of the labels (transitional
        state — could happen if a new ontology declares a label no
        adapter emits yet)

    Raises `ValueError` for an unknown ontology (registry miss).
    """
    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        EXTRACTOR_REGISTRY,
    )
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"get_extractor_candidates: unknown ontology "
            f"{ontology_name!r}. Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    prov = ONTOLOGY_REGISTRY[ontology_name].get("provenance")
    if prov is None:
        return ()
    covers = set(prov.covers_labels)
    if not covers:
        return ()

    candidates: list[tuple[str, tuple[str, ...]]] = []
    for extractor_name, entry in EXTRACTOR_REGISTRY.items():
        emitted = entry.get("emitted_labels")
        if emitted is None:
            continue
        overlap = sorted(set(emitted) & covers)
        if overlap:
            candidates.append((extractor_name, tuple(overlap)))

    # Sort by overlap size (desc) then name (asc) for determinism.
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    return tuple(candidates)
