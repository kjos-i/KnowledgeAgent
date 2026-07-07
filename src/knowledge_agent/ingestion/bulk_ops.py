"""Layer 3 (UI-facing) bulk operations.

Every UI button maps to a function in this module. bulk_ops composes the
per-doc primitives from `pipeline.py` (Layer 2) and adds the plan/execute
contract + iteration + confirmation surface that a GUI needs.

The plan/execute contract:

  plan   = bulk_ops.<op>_plan(*args)     # read-only; cheap; returns a Plan
  # GUI shows `plan` in a confirmation dialog; user clicks Confirm
  result = bulk_ops.<op>_execute(plan)   # performs the work

Plans carry the counts + filenames + summary string the dialog renders.
Trust the plan: there is no re-validation between plan and execute, so
the dialog -> execute round trip is deterministic from the user's view
(the underlying data is allowed to drift; see the deferred "plan expiry"
note in the bulk_ops design memory).

This module never calls Layer 1 primitives (`parse_document`,
`embed_texts`, KG / LanceDB clients) directly - everything goes through
Layer 2 (`pipeline.py`). That keeps the dependency chain
`bulk_ops -> pipeline -> primitives` one-directional and avoids circular
imports.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from knowledge_agent.ingestion import parse, pipeline
from knowledge_agent.ingestion.ids import compute_doc_id
from knowledge_agent.ingestion.metadata import is_doi_eligible
from knowledge_agent.ingestion.sync_diff import (
    DiskFile,
    IndexedDoc,
    SyncBuckets,
    classify,
)
from knowledge_agent.kg import (
    cross_doc_xrefs_writes,
    ontology_xrefs,
)
from knowledge_agent.kg.client import get_kg_client
from knowledge_agent.kg.corpus_config import CorpusConfig
from knowledge_agent.kg.ontology_linking import ONTOLOGY_REGISTRY
from knowledge_agent.kg.reconcile import (
    reconcile_cross_doc_to_config,
    reconcile_cross_doc_xrefs_to_config,
    reconcile_entities_to_config,
    reconcile_ontologies_to_config,
    reconcile_triples_to_config,
)
from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS
from knowledge_agent.search.client import get_search_client

logger = logging.getLogger(__name__)


# ---- shared helpers (private) ----


def _validate_folder(folder: Path | None) -> Path:
    """Validate a user-supplied folder argument. Raises ValueError on bad input."""
    if folder is None:
        raise ValueError("folder is required")
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory")
    return folder


def _walk_and_hash(folder: Path) -> list[tuple[Path, str, int]]:
    """Recursively walk `folder`; yield supported files with hash + size.

    Returns list of `(path, doc_id, size_bytes)` tuples sorted by path.
    Shared by `ingest_folder_plan` and `sync_plan` so the walk + hash
    pass is the single source of truth for "what files are in this
    folder."

    Hashing every file is the expensive step (~50 MB/s on commodity
    SSDs); for a 100-paper folder of 10 MB PDFs that's ~20s of CPU
    before the plan dialog appears. Acceptable trade-off for accurate
    counts.
    """
    extensions = parse.supported_extensions()
    out: list[tuple[Path, str, int]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower().lstrip(".") not in extensions:
            continue
        out.append((path, compute_doc_id(path), path.stat().st_size))
    return out


# ---- delete_doc (op #9 in the bulk_ops UI-intent list) ----


@dataclass(frozen=True)
class DeleteDocPlan:
    """Plan for the per-doc Delete operation.

    Built by `delete_doc_plan(doc_id)`, rendered in a confirmation
    dialog, then passed to `delete_doc_execute(plan)` to perform the
    delete across LanceDB + Neo4j.

    Fields are what the dialog needs to display:
      - `doc_id`: the hash the delete will target.
      - `title`: doc title from LanceDB metadata cache, or None.
      - `n_chunks`: count of chunk rows the LanceDB delete will remove.
      - `source_path`: path stored at ingest time, for user recognition.

    Empty fields mean "nothing to read from LanceDB" (e.g., doc was
    never ingested or chunks were dropped) - the execute path is still
    safe (every underlying primitive is idempotent).
    """

    doc_id: str
    title: str | None
    n_chunks: int
    source_path: str | None

    @property
    def summary(self) -> str:
        """One-line summary the GUI renders in the dialog header."""
        name = self.title or self.source_path or f"doc {self.doc_id[:12]}"
        return (
            f"Delete '{name}'? Removes {self.n_chunks} chunks plus "
            f"all entities and metadata from both stores."
        )


@dataclass(frozen=True)
class DeleteDocResult:
    """Outcome of `delete_doc_execute(plan)`.

    `ok` is True only when all underlying primitives (LanceDB chunk
    delete + 3 KG deletes) succeeded. `ok=False` means at least one
    store reported an error (already logged at the primitive layer);
    state may be partially cleaned but a retry is safe (idempotent).
    """

    doc_id: str
    ok: bool


async def delete_doc_plan(doc_id: str) -> DeleteDocPlan:
    """Read-only inventory of what `delete_doc_execute` will remove.

    Pulls one chunk row from LanceDB to populate the dialog fields
    (title, source_path, chunk count). Cheap - one filter query, no
    KG calls. Safe when the doc isn't in LanceDB: returns a plan with
    `n_chunks=0` and None metadata; execute is still valid.
    """
    if not doc_id:
        raise ValueError("delete_doc_plan called with empty doc_id")

    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception:
        # Read failure → treat as "no info" so the plan dialog can still
        # render. Execute path will surface the real error if there is one.
        chunk_rows = []
    if not chunk_rows:
        return DeleteDocPlan(doc_id=doc_id, title=None, n_chunks=0, source_path=None)

    first = chunk_rows[0]
    return DeleteDocPlan(
        doc_id=doc_id,
        title=first.get("title"),
        n_chunks=len(chunk_rows),
        source_path=first.get("source_path"),
    )


# ---- bulk_resolve_openalex (op #3 bulk in the UI-intent list) ----


@dataclass(frozen=True)
class BulkResolveOpenAlexPlan:
    """Plan for the bulk Resolve OpenAlex operation.

    DOI / OpenAlex enrichment applies only to scholarly documents, so a
    doc is a target only when its sub-label is DOI-eligible (Paper — the
    same `is_doi_eligible` gate the ingest pipeline uses, kept single-
    sourced). Non-eligible docs are captured in `skipped_non_paper` for
    the dialog rather than silently dropped.

    Among the eligible docs, those whose `metadata_status` is "manual"
    are further held back when `skip_manual=True` (the default) and
    captured in `skipped_manual` so the user sees what protection they
    have. With `skip_manual=False` every eligible doc is a target
    (manual edits will be overwritten). Non-Paper docs are skipped
    regardless of `skip_manual`.
    """

    target_doc_ids: tuple[str, ...]
    skipped_manual: tuple[IndexedDoc, ...]
    skip_manual: bool
    skipped_non_paper: tuple[IndexedDoc, ...] = ()

    @property
    def n_targets(self) -> int:
        return len(self.target_doc_ids)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped_manual)

    @property
    def n_skipped_non_paper(self) -> int:
        return len(self.skipped_non_paper)

    @property
    def summary(self) -> str:
        s = f"Resolve OpenAlex for {self.n_targets} docs"
        notes: list[str] = []
        if self.n_skipped_non_paper:
            notes.append(
                f"{self.n_skipped_non_paper} non-Paper docs skipped "
                "(DOI/OpenAlex enrichment is Paper-only)"
            )
        if self.skip_manual and self.n_skipped:
            notes.append(f"{self.n_skipped} manual docs skipped")
        if notes:
            s += ". " + "; ".join(notes)
        return s + "."


@dataclass(frozen=True)
class BulkResolveOpenAlexResult:
    """Outcome of bulk Resolve OpenAlex.

    Three buckets: `n_resolved` (OpenAlex returned a work + writes
    succeeded), `n_no_work` (lookup returned None - leave state alone),
    `n_failed` (an exception escaped). `failures` carries doc_id +
    error repr for the failed bucket.
    """

    n_resolved: int
    n_no_work: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]


async def bulk_resolve_openalex_plan(
    *,
    skip_manual: bool = True,
) -> BulkResolveOpenAlexPlan:
    """List all indexed docs; split into targets vs non-Paper vs manual.

    DOI / OpenAlex enrichment is scholarly-only, so a doc is a target
    only when `is_doi_eligible(sub_label)` — the same Paper gate the
    ingest pipeline applies (`pipeline.ingest_document`). Non-eligible
    docs go to `skipped_non_paper` and are never resolved regardless of
    `skip_manual`.

    `skip_manual=True` (default) additionally excludes eligible docs
    whose `metadata_status` is "manual"; flip to False to overwrite
    manual edits intentionally (per the bulk_ops design memory - bulk
    Resolve protects manual by default; the dialog's checkbox lets the
    user opt out).
    """
    search_client = get_search_client()
    # LanceDB read errors now propagate to the caller (typed-errors
    # contract); the prior explicit `is None` check is unnecessary.
    indexed = await search_client.list_indexed_docs()

    target_ids: list[str] = []
    skipped: list[IndexedDoc] = []
    skipped_non_paper: list[IndexedDoc] = []
    for d in indexed:
        idx = IndexedDoc(
            doc_id=d["doc_id"],
            stored_path=d.get("source_path"),
            title=d.get("title"),
            metadata_status=d.get("metadata_status"),
            n_chunks=d.get("n_chunks", 0),
        )
        # Paper gate first: a non-Paper doc is never a resolve target,
        # even with skip_manual=False (mirrors the ingest-time gate).
        if not is_doi_eligible(d.get("sub_label")):
            skipped_non_paper.append(idx)
        elif skip_manual and d.get("metadata_status") == "manual":
            skipped.append(idx)
        else:
            target_ids.append(d["doc_id"])

    return BulkResolveOpenAlexPlan(
        target_doc_ids=tuple(target_ids),
        skipped_manual=tuple(skipped),
        skip_manual=skip_manual,
        skipped_non_paper=tuple(skipped_non_paper),
    )


async def bulk_resolve_openalex_execute(
    plan: BulkResolveOpenAlexPlan,
) -> BulkResolveOpenAlexResult:
    """Iterate `plan.target_doc_ids`; call `pipeline.resolve_openalex`.

    Per-doc resolve_openalex is invoked with `skip_manual=False` -
    plan-side filtering already excluded the manual docs (or chose to
    include them via the opt-out). Bucketed counts let the post-op UI
    show "X resolved / Y not in OpenAlex / Z errors".
    """
    n_resolved = 0
    n_no_work = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.resolve_openalex(doc_id, skip_manual=False)
            if result.get("work_resolved"):
                n_resolved += 1
            else:
                n_no_work += 1
        except Exception as exc:
            logger.warning(
                "bulk_resolve_openalex_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkResolveOpenAlexResult(
        n_resolved=n_resolved,
        n_no_work=n_no_work,
        n_failed=n_failed,
        failures=tuple(failures),
    )


# ---- bulk_re_embed (op #4 bulk in the UI-intent list) ----


@dataclass(frozen=True)
class BulkReEmbedPlan:
    """Plan for the bulk Re-Embed operation - every indexed doc."""

    target_doc_ids: tuple[str, ...]
    total_chunks: int  # informational total across all targets

    @property
    def n_targets(self) -> int:
        return len(self.target_doc_ids)

    @property
    def summary(self) -> str:
        return f"Re-embed {self.n_targets} docs (~{self.total_chunks} chunks total)."


@dataclass(frozen=True)
class BulkReEmbedResult:
    n_succeeded: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]


async def bulk_re_embed_plan() -> BulkReEmbedPlan:
    """List every indexed doc; report total chunk count."""
    search_client = get_search_client()
    # LanceDB read errors propagate (typed-errors contract).
    indexed = await search_client.list_indexed_docs()

    target_ids = tuple(d["doc_id"] for d in indexed)
    total_chunks = sum(d.get("n_chunks", 0) for d in indexed)
    return BulkReEmbedPlan(
        target_doc_ids=target_ids,
        total_chunks=total_chunks,
    )


async def bulk_re_embed_execute(
    plan: BulkReEmbedPlan,
    config: CorpusConfig,
) -> BulkReEmbedResult:
    """Iterate `plan.target_doc_ids`; call `pipeline.re_embed`.

    `config` is threaded through so per-doc re-embed reads the
    corpus's `optimize_indexes_per_ingest` flag (whether to rebuild
    the vector index after each write).

    A doc whose embed call fails (Voyage outage, etc.) is counted as
    failed and skipped; the loop keeps going so transient failures on
    one doc don't block re-embedding the rest of the corpus.
    """
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.re_embed(doc_id, config)
            if result.get("embed_ok") and result.get("lancedb_ok"):
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "embed_ok or lancedb_ok was False"))
        except Exception as exc:
            logger.warning(
                "bulk_re_embed_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkReEmbedResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


# ---- bulk_backfill_* (ops #5/6/7 bulk in the UI-intent list) ----


@dataclass(frozen=True)
class BulkBackfillPlan:
    """Shared plan dataclass for the three bulk backfill ops.

    `layer_name` is the user-facing label ("chunks", "entities",
    "ontology") that the summary string and the executor's logging
    use - the per-doc Layer 2 function the executor actually calls
    is decided by which `bulk_backfill_<layer>_execute` is invoked.
    """

    target_doc_ids: tuple[str, ...]
    layer_name: str

    @property
    def n_targets(self) -> int:
        return len(self.target_doc_ids)

    @property
    def summary(self) -> str:
        return f"Backfill {self.layer_name} for {self.n_targets} docs."


@dataclass(frozen=True)
class BulkBackfillResult:
    """Shared result dataclass for the three bulk backfill ops."""

    n_succeeded: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]


async def _bulk_backfill_plan(layer_name: str) -> BulkBackfillPlan:
    """Shared plan-building helper: list every indexed doc."""
    search_client = get_search_client()
    # LanceDB read errors propagate (typed-errors contract).
    indexed = await search_client.list_indexed_docs()
    return BulkBackfillPlan(
        target_doc_ids=tuple(d["doc_id"] for d in indexed),
        layer_name=layer_name,
    )


async def bulk_backfill_chunks_plan() -> BulkBackfillPlan:
    """List every indexed doc for the bulk KG L5+ rebuild op."""
    return await _bulk_backfill_plan("chunks")


async def bulk_backfill_chunks_execute(
    plan: BulkBackfillPlan,
    config: CorpusConfig,
) -> BulkBackfillResult:
    """Iterate targets; call `await pipeline.backfill_chunks(doc_id, config)`.

    Success criterion is `result["chunks_ok"]` - the per-doc function
    returns False when the chunks layer is disabled in the corpus
    config OR LanceDB has no chunks for the doc; those count as
    failures here.
    """
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.backfill_chunks(doc_id, config)
            if result.get("chunks_ok"):
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "chunks_ok was False"))
        except Exception as exc:
            logger.warning(
                "bulk_backfill_chunks_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkBackfillResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


async def bulk_backfill_entities_plan() -> BulkBackfillPlan:
    """List every indexed doc for the bulk KG L6+ rebuild op."""
    return await _bulk_backfill_plan("entities")


async def bulk_backfill_entities_execute(
    plan: BulkBackfillPlan,
    config: CorpusConfig,
) -> BulkBackfillResult:
    """Iterate targets; call `await pipeline.backfill_entities(doc_id, config)`.

    Reconciles entities-to-config at the top: wipes all :Entity
    corpus-wide if `layers.entities=false`, or wipes entities of
    types no longer in `config.entities.entity_types` when the layer
    is on but the type list narrowed. Fail-hard on wipe error.
    """
    await reconcile_entities_to_config(get_kg_client(), config)
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.backfill_entities(doc_id, config)
            if result.get("entities_ok"):
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "entities_ok was False"))
        except Exception as exc:
            logger.warning(
                "bulk_backfill_entities_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkBackfillResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


async def bulk_backfill_ontology_plan() -> BulkBackfillPlan:
    """List every indexed doc for the bulk KG L7 re-link op."""
    return await _bulk_backfill_plan("ontology")


async def bulk_backfill_ontology_execute(
    plan: BulkBackfillPlan,
    config: CorpusConfig,
) -> BulkBackfillResult:
    """Iterate targets; call `await pipeline.backfill_ontology(doc_id, config)`.

    Reconciles ontologies-to-config at the top: any ontology no
    longer in `config._enabled_ontology_layers()` but still present
    in the KG has its term nodes wiped (DETACH DELETE cascades to
    :CANONICAL_TO and xref edges). Fail-hard on wipe error.

    Success criterion: at least one enabled ontology successfully
    imported AND produced canonical links. A doc with zero enabled
    ontologies (`config._enabled_ontology_layers()` empty) counts as a
    no-op success (no failure to report).
    """
    await reconcile_ontologies_to_config(get_kg_client(), config)
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.backfill_ontology(doc_id, config)
            if not result:
                # No ontologies enabled - this is a no-op success.
                n_succeeded += 1
                continue
            any_imported = any(v.get("import_ok") for v in result.values())
            if any_imported:
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "no enabled ontology successfully imported"))
        except Exception as exc:
            logger.warning(
                "bulk_backfill_ontology_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkBackfillResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


async def bulk_backfill_triples_plan() -> BulkBackfillPlan:
    """List every indexed doc for the bulk KG L8 rebuild op."""
    return await _bulk_backfill_plan("triples")


async def bulk_backfill_triples_execute(
    plan: BulkBackfillPlan,
    config: CorpusConfig,
) -> BulkBackfillResult:
    """Iterate targets; call `await pipeline.backfill_triples(doc_id, config)`.

    Reconciles triples-to-config at the top: wipes all triple edges
    corpus-wide if `layers.triples=false`. Fail-hard on wipe error.

    Success criterion: the per-doc `triples_ok` flag is True. A doc
    that yielded zero triples still counts as success - empty output
    is a valid outcome when L6 found no entities to relate.

    `n_triples == 0 AND triples_ok == True` IS a success (the LLM
    found nothing worth extracting), not a failure - distinct from
    `triples_ok == False` which means a write/Cypher error.
    """
    await reconcile_triples_to_config(get_kg_client(), config)
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.backfill_triples(doc_id, config)
            if result.get("triples_ok"):
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "triples_ok was False"))
        except Exception as exc:
            logger.warning(
                "bulk_backfill_triples_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkBackfillResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


async def bulk_backfill_cross_doc_plan() -> BulkBackfillPlan:
    """List every indexed doc for the bulk KG L9 rebuild op."""
    return await _bulk_backfill_plan("cross_doc")


async def bulk_backfill_cross_doc_execute(
    plan: BulkBackfillPlan,
    config: CorpusConfig,
) -> BulkBackfillResult:
    """Iterate targets; call `await pipeline.backfill_cross_doc(doc_id, config)`.

    Reconciles cross-doc-to-config at the top: wipes all :RELATED_TO
    edges corpus-wide if `layers.cross_doc=false`. Fail-hard on wipe
    error.

    Success criterion: `cross_doc_ok=True`. A doc that ends up with
    zero `:RELATED_TO` edges (no other doc met the threshold) is
    still a success - empty output is a valid outcome, distinct from
    `cross_doc_ok=False` which means the recompute Cypher crashed.
    """
    await reconcile_cross_doc_to_config(get_kg_client(), config)
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []

    for doc_id in plan.target_doc_ids:
        try:
            result = await pipeline.backfill_cross_doc(doc_id, config)
            if result.get("cross_doc_ok"):
                n_succeeded += 1
            else:
                n_failed += 1
                failures.append((doc_id, "cross_doc_ok was False"))
        except Exception as exc:
            logger.warning(
                "bulk_backfill_cross_doc_execute: %s failed: %r",
                doc_id,
                exc,
            )
            failures.append((doc_id, repr(exc)))
            n_failed += 1

    return BulkBackfillResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


async def delete_doc_execute(plan: DeleteDocPlan) -> DeleteDocResult:
    """Perform the delete promised by `plan`.

    Pure delegation to `await pipeline.delete_doc(plan.doc_id)` - the
    composition is the single source of truth (also called by
    `ingest_document`'s delete-then-write step).
    """
    ok = await pipeline.delete_doc(plan.doc_id)
    return DeleteDocResult(doc_id=plan.doc_id, ok=ok)


# ---- ingest_folder (op #1 in the bulk_ops UI-intent list) ----


@dataclass(frozen=True)
class IngestFolderItem:
    """One file's planned-ingest info: path, hash, existing-DB state.

    Computed in `ingest_folder_plan` by hashing the file and querying
    LanceDB. Used by the dialog summary properties and by execute.
    """

    path: Path
    doc_id: str
    size_bytes: int
    exists_in_db: bool
    metadata_status: str | None  # None when the doc isn't in DB


@dataclass(frozen=True)
class IngestFolderPlan:
    """Plan for the Full Ingest operation.

    Built by `ingest_folder_plan(folder, main_label, sub_label)` after
    recursively scanning the folder, hashing each supported file, and
    checking LanceDB for each `doc_id`. Recursive scan is fine: the
    dialog shows the file count so the user catches a "300 instead of 2"
    surprise before any work happens.

    Fields the dialog renders via the summary property:
      - file count + total bytes
      - count of files that will overwrite existing docs
      - count of files whose stored `metadata_status` is "manual"
        (those manual edits will be replaced - full ingest is the
        nuclear option per the bulk_ops design memory).

    `main_label` + `sub_label` are folder-level: every file in the
    folder is ingested with the same label pair (matches the GUI's
    "doc type" selector flow).
    """

    folder: Path
    main_label: str
    sub_label: str | None
    items: tuple[IngestFolderItem, ...]

    @property
    def n_files(self) -> int:
        return len(self.items)

    @property
    def n_overwrites(self) -> int:
        return sum(1 for i in self.items if i.exists_in_db)

    @property
    def n_manual(self) -> int:
        return sum(1 for i in self.items if i.metadata_status == "manual")

    @property
    def total_bytes(self) -> int:
        return sum(i.size_bytes for i in self.items)

    @property
    def summary(self) -> str:
        size_mb = self.total_bytes / (1024 * 1024)
        parts = [f"Ingest {self.n_files} files ({size_mb:.1f} MB on disk)"]
        if self.n_overwrites:
            parts.append(f"will overwrite {self.n_overwrites} already in DB")
        if self.n_manual:
            parts.append(f"{self.n_manual} have manual metadata that will be replaced")
        return ". ".join(parts) + "."


@dataclass(frozen=True)
class IngestFolderResult:
    """Outcome of `ingest_folder_execute(plan, config)`.

    Per-file failures are recorded in `failures` (list of (filename,
    error repr) tuples) and counted in `n_failed`, but they don't
    abort the loop - the rest of the folder still gets ingested.
    """

    n_succeeded: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]


async def ingest_folder_plan(
    folder: Path,
    main_label: str,
    sub_label: str | None = None,
) -> IngestFolderPlan:
    """Recursively scan `folder` for supported files; hash each + check DB.

    Hashing every file is the price of an accurate "will overwrite K
    already in DB" dialog line. For a 100-file folder of 10 MB papers
    that's ~5-10s of CPU before the user sees the confirm dialog -
    acceptable trade-off for catching a wrong-folder mistake.

    Recursive descent is deliberate: the plan dialog shows the file
    count, so a user who accidentally selected `~/Documents/` will see
    the unrealistic number and cancel before any work happens.

    Raises `ValueError` for an empty `folder` argument or a path that
    isn't a directory.
    """
    try:
        folder = _validate_folder(folder)
    except ValueError as exc:
        raise ValueError(f"ingest_folder_plan: {exc}") from exc

    search_client = get_search_client()
    items: list[IngestFolderItem] = []

    for path, doc_id, size_bytes in await asyncio.to_thread(_walk_and_hash, folder):
        # Per-file read errors are tolerated so one corrupt row
        # doesn't kill the plan: treat the lookup as a miss and let
        # the execute path surface the real error.
        try:
            existing = await search_client.get_chunks_by_doc_id(doc_id)
        except Exception:
            existing = []
        if existing:
            exists_in_db = True
            metadata_status = existing[0].get("metadata_status")
        else:
            exists_in_db = False
            metadata_status = None

        items.append(
            IngestFolderItem(
                path=path,
                doc_id=doc_id,
                size_bytes=size_bytes,
                exists_in_db=exists_in_db,
                metadata_status=metadata_status,
            )
        )

    return IngestFolderPlan(
        folder=folder,
        main_label=main_label,
        sub_label=sub_label,
        items=tuple(items),
    )


# ---- add (op #3 in the bulk_ops UI-intent list) ----


@dataclass(frozen=True)
class AddPlan:
    """Plan for the Add operation.

    Add is "purely additive sync": ingest files whose `doc_id` is NOT
    already in the index; ignore everything else (no path patches,
    no deletes, no orphan handling). Cheaper than Sync because it
    skips the `list_indexed_docs` call and the diff classification.

    `new_items` lists only the files that WILL be ingested.
    `n_skipped` records the files that exist on disk AND already have
    their `doc_id` in the DB - shown in the dialog as informational
    ("M already in DB will be skipped").
    """

    folder: Path
    main_label: str
    sub_label: str | None
    new_items: tuple[IngestFolderItem, ...]
    n_skipped: int

    @property
    def n_new(self) -> int:
        return len(self.new_items)

    @property
    def total_bytes(self) -> int:
        return sum(i.size_bytes for i in self.new_items)

    @property
    def summary(self) -> str:
        size_mb = self.total_bytes / (1024 * 1024)
        s = f"Add {self.n_new} new files ({size_mb:.1f} MB on disk)"
        if self.n_skipped:
            s += f". {self.n_skipped} already in DB will be skipped"
        return s + "."


@dataclass(frozen=True)
class AddResult:
    """Outcome of `add_execute(plan, config)`.

    Per-file fail-soft: one failure doesn't stop the loop. Matches
    `IngestFolderResult` shape so the UI can reuse the same renderer.
    """

    n_succeeded: int
    n_failed: int
    failures: tuple[tuple[str, str], ...]


async def add_plan(
    folder: Path,
    main_label: str,
    sub_label: str | None = None,
) -> AddPlan:
    """Walk `folder`; classify each supported file as NEW or skipped.

    For each disk file, one `get_chunks_by_doc_id` check against
    LanceDB decides which bucket. Cheaper than `sync_plan` because we
    don't enumerate the whole index (no MOVED/EDITED/ORPHAN buckets to
    produce).
    """
    try:
        folder = _validate_folder(folder)
    except ValueError as exc:
        raise ValueError(f"add_plan: {exc}") from exc

    search_client = get_search_client()
    new_items: list[IngestFolderItem] = []
    n_skipped = 0

    for path, doc_id, size_bytes in await asyncio.to_thread(_walk_and_hash, folder):
        # Per-file read errors are tolerated so one corrupt row
        # doesn't kill the plan: treat the lookup as a miss.
        try:
            existing = await search_client.get_chunks_by_doc_id(doc_id)
        except Exception:
            existing = []
        if existing:
            n_skipped += 1
            continue
        new_items.append(
            IngestFolderItem(
                path=path,
                doc_id=doc_id,
                size_bytes=size_bytes,
                exists_in_db=False,
                metadata_status=None,
            )
        )

    return AddPlan(
        folder=folder,
        main_label=main_label,
        sub_label=sub_label,
        new_items=tuple(new_items),
        n_skipped=n_skipped,
    )


async def add_execute(
    plan: AddPlan,
    config: CorpusConfig,
    preserve_existing_labels: bool = True,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
) -> AddResult:
    """Ingest each file in `plan.new_items`. Per-file fail-soft.

    `preserve_existing_labels` is passed through to `ingest_document`.
    Semantically a no-op here because `add_plan` filters to new-only
    docs, but plumbed through for uniformity with sync / ingest_folder.

    `progress_cb`, when given, is called `(done, total)` after each file
    (success or failure) so a caller can drive a determinate progress
    bar: `total` is the file count, `done` counts files processed.
    """
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []
    total = len(plan.new_items)

    for done, item in enumerate(plan.new_items, start=1):
        try:
            await pipeline.ingest_document(
                item.path,
                config,
                plan.main_label,
                plan.sub_label,
                preserve_existing_labels=preserve_existing_labels,
            )
            n_succeeded += 1
        except Exception as exc:
            logger.warning(
                "add_execute: %s failed: %r",
                item.path.name,
                exc,
            )
            failures.append((item.path.name, repr(exc)))
            n_failed += 1
        if progress_cb is not None:
            progress_cb(done, total)

    return AddResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


# ---- sync (op #2 in the bulk_ops UI-intent list) ----


@dataclass(frozen=True)
class SyncPlan:
    """Plan for the Sync operation.

    Holds the 5-bucket diff (`buckets: SyncBuckets`) that classifies
    every file on disk + every doc in the index. `summary` and the
    `n_*` properties are what the confirmation dialog renders;
    `orphan_display_names` gives the dialog the explicit list of
    docs that Sync will delete from the DB (per the safe-by-default
    "show each orphan filename" rule).

    `main_label` + `sub_label` are folder-level: every NEW and EDITED
    file gets ingested with the same label pair (matches the GUI's
    "doc type" selector flow).
    """

    folder: Path
    main_label: str
    sub_label: str | None
    buckets: SyncBuckets

    @property
    def n_new(self) -> int:
        return len(self.buckets.new)

    @property
    def n_unchanged(self) -> int:
        return len(self.buckets.unchanged)

    @property
    def n_moved(self) -> int:
        return len(self.buckets.moved)

    @property
    def n_edited(self) -> int:
        return len(self.buckets.edited)

    @property
    def n_orphans(self) -> int:
        return len(self.buckets.orphan)

    @property
    def summary(self) -> str:
        parts: list[str] = []
        if self.n_new:
            parts.append(f"{self.n_new} new (ingest)")
        if self.n_moved:
            parts.append(f"{self.n_moved} moved (update paths)")
        if self.n_edited:
            parts.append(f"{self.n_edited} edited (replace old)")
        if self.n_orphans:
            parts.append(f"{self.n_orphans} orphans to DELETE")
        if not parts:
            return "Nothing to sync - folder matches DB."
        return "Sync will: " + ", ".join(parts) + "."

    @property
    def orphan_display_names(self) -> tuple[str, ...]:
        """User-facing names for each orphan, shown one-per-line in the
        confirmation dialog (the "explicit per-file confirmation" rule
        from the bulk_ops design memory)."""
        return tuple(
            o.title or o.stored_path or f"doc {o.doc_id[:12]}" for o in self.buckets.orphan
        )


@dataclass(frozen=True)
class SyncResult:
    """Outcome of `sync_execute(plan, config)`.

    Per-bucket counts so the post-sync UI can show "X new ingested,
    Y orphans deleted, Z failures". `failures` is a flat list of
    `(label, error repr)` tuples; the label prefix tells the user which
    bucket the failure was in (NEW, EDITED, MOVED, ORPHAN).
    """

    n_new_ingested: int
    n_new_failed: int
    n_moved: int
    n_edited_succeeded: int
    n_edited_failed: int
    n_orphans_deleted: int
    failures: tuple[tuple[str, str], ...]


async def sync_plan(
    folder: Path,
    main_label: str,
    sub_label: str | None = None,
) -> SyncPlan:
    """Build a Sync plan: walk + hash disk, list indexed docs, classify.

    Raises:
      - `ValueError` for an empty or non-directory folder.
      - Whatever LanceDB raises if `list_indexed_docs` fails - sync
        without index data could silently mark everything as orphan,
        so we surface the error and let the caller retry rather than
        catch and continue with an empty list.
    """
    try:
        folder = _validate_folder(folder)
    except ValueError as exc:
        raise ValueError(f"sync_plan: {exc}") from exc

    disk_files = [
        DiskFile(path=p, doc_id=did)
        for p, did, _ in await asyncio.to_thread(_walk_and_hash, folder)
    ]

    search_client = get_search_client()
    # LanceDB read errors propagate (typed-errors contract).
    indexed_dicts = await search_client.list_indexed_docs()

    indexed_docs = [
        IndexedDoc(
            doc_id=d["doc_id"],
            stored_path=d.get("source_path"),
            title=d.get("title"),
            metadata_status=d.get("metadata_status"),
            n_chunks=d.get("n_chunks", 0),
        )
        for d in indexed_dicts
    ]

    buckets = classify(disk_files, indexed_docs)
    return SyncPlan(
        folder=folder,
        main_label=main_label,
        sub_label=sub_label,
        buckets=buckets,
    )


async def sync_execute(
    plan: SyncPlan,
    config: CorpusConfig,
    preserve_existing_labels: bool = True,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
) -> SyncResult:
    """Apply changes per bucket: ingest NEW, patch MOVED, replace EDITED,
    delete ORPHAN.

    Failure policy: per-item fail-soft. Each bucket iteration catches
    exceptions, logs, records in `failures`, and continues. The caller
    gets a complete result even when half the items failed.

    The caller MUST have shown the orphan confirmation dialog before
    invoking this - sync_execute will delete every orphan in
    `plan.buckets.orphan` without further prompting.

    `preserve_existing_labels` (default True): when the EDITED bucket
    replaces a doc whose content changed, look up its old
    `(main_label, sub_label)` from Neo4j BEFORE the delete and pass
    those to the fresh ingest — the doc's category (Paper / Note /...)
    doesn't change when its content edits. Read must precede the
    delete because the new content has a new content-hash doc_id, so
    `ingest_document`'s own preserve check would find nothing.
    """
    n_new_ingested = 0
    n_new_failed = 0
    n_moved = 0
    n_edited_succeeded = 0
    n_edited_failed = 0
    n_orphans_deleted = 0
    failures: list[tuple[str, str]] = []
    # One progress step per item across all four buckets, so a caller's
    # determinate bar fills smoothly over the whole sync.
    total = (
        len(plan.buckets.new)
        + len(plan.buckets.moved)
        + len(plan.buckets.edited)
        + len(plan.buckets.orphan)
    )
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        if progress_cb is not None:
            progress_cb(done, total)

    search_client = get_search_client()

    # NEW: full ingest via Layer 2. New doc_ids never have prior
    # labels, so `preserve_existing_labels` is a semantic no-op here.
    for disk in plan.buckets.new:
        try:
            await pipeline.ingest_document(
                disk.path,
                config,
                plan.main_label,
                plan.sub_label,
                preserve_existing_labels=preserve_existing_labels,
            )
            n_new_ingested += 1
        except Exception as exc:
            logger.warning(
                "sync_execute: NEW %s failed: %r",
                disk.path.name,
                exc,
            )
            failures.append((f"NEW {disk.path.name}", repr(exc)))
            n_new_failed += 1
        _tick()

    # MOVED: just patch source_path on the existing doc.
    for disk, old in plan.buckets.moved:
        try:
            await search_client.update_doc_metadata(
                old.doc_id,
                {"source_path": disk.path.as_posix()},
            )
            n_moved += 1
        except Exception as exc:
            logger.warning(
                "sync_execute: MOVED %s failed: %r",
                disk.path.name,
                exc,
            )
            failures.append((f"MOVED {disk.path.name}", repr(exc)))
        _tick()

    # EDITED: delete the old doc_id, ingest the new content.
    # The new content produces a fresh content-hash doc_id, so
    # `ingest_document`'s built-in preserve lookup can't help — we
    # read the old labels here + carry them into the fresh ingest.
    # `kg_client` obtained lazily so plans with no EDITED items or
    # preserve=False don't touch the KG driver.
    kg_client_for_edited = (
        get_kg_client() if (preserve_existing_labels and plan.buckets.edited) else None
    )
    for disk, old in plan.buckets.edited:
        try:
            if kg_client_for_edited is not None:
                old_main, old_sub = await kg_client_for_edited.get_focal_labels_by_doc_id(
                    old.doc_id
                )
                use_main = old_main or plan.main_label
                use_sub = old_sub if old_main else plan.sub_label
            else:
                use_main = plan.main_label
                use_sub = plan.sub_label
            await pipeline.delete_doc(old.doc_id)
            await pipeline.ingest_document(
                disk.path,
                config,
                use_main,
                use_sub,
                preserve_existing_labels=False,
            )
            n_edited_succeeded += 1
        except Exception as exc:
            logger.warning(
                "sync_execute: EDITED %s failed: %r",
                disk.path.name,
                exc,
            )
            failures.append((f"EDITED {disk.path.name}", repr(exc)))
            n_edited_failed += 1
        _tick()

    # ORPHAN: delete - dialog confirmation is the caller's responsibility.
    for orphan in plan.buckets.orphan:
        if await pipeline.delete_doc(orphan.doc_id):
            n_orphans_deleted += 1
        else:
            label = orphan.title or orphan.stored_path or orphan.doc_id[:12]
            failures.append((f"ORPHAN {label}", "delete returned False"))
        _tick()

    return SyncResult(
        n_new_ingested=n_new_ingested,
        n_new_failed=n_new_failed,
        n_moved=n_moved,
        n_edited_succeeded=n_edited_succeeded,
        n_edited_failed=n_edited_failed,
        n_orphans_deleted=n_orphans_deleted,
        failures=tuple(failures),
    )


async def ingest_folder_execute(
    plan: IngestFolderPlan,
    config: CorpusConfig,
    preserve_existing_labels: bool = True,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
) -> IngestFolderResult:
    """Iterate `plan.items`; call `pipeline.ingest_document` per file.

    Failures fail-soft: a single file raising doesn't abort the loop.
    The exception is logged, recorded in `result.failures` as
    `(filename, repr(exc))`, and the iteration continues. This matches
    how `ingest_document` already handles per-stage failures internally
    (continue on partial).

    `config` is read at execute time, not stored in the plan. If the
    user changes corpus settings between plan and execute, the new
    settings apply.

    `preserve_existing_labels` (default True): when a file in the folder
    is already indexed (same content-hash `doc_id`), keep its stored
    labels rather than overwriting with `plan.main_label` /
    `plan.sub_label`. Passed through to `ingest_document`.
    """
    n_succeeded = 0
    n_failed = 0
    failures: list[tuple[str, str]] = []
    total = len(plan.items)

    for done, item in enumerate(plan.items, start=1):
        try:
            await pipeline.ingest_document(
                item.path,
                config,
                plan.main_label,
                plan.sub_label,
                preserve_existing_labels=preserve_existing_labels,
            )
            n_succeeded += 1
        except Exception as exc:
            logger.warning(
                "ingest_folder_execute: %s failed: %r",
                item.path.name,
                exc,
            )
            failures.append((item.path.name, repr(exc)))
            n_failed += 1
        if progress_cb is not None:
            progress_cb(done, total)

    return IngestFolderResult(
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        failures=tuple(failures),
    )


# ---- backfill_xrefs (L7 cross-ontology xref resolution) ----


@dataclass(frozen=True)
class BackfillXrefsPlan:
    """Plan for the `backfill_xrefs` op.

    Walks every shipped ontology's `dangling_xrefs` lists, materialises
    `:<X>_XREF` edges for entries whose targets are present, and
    strips resolved entries from the lists. When the corpus has
    `cross_doc_xrefs=true` ALSO triggers a graph-wide L10 recompute
    so the concept-level cross-doc edges pick up the freshly-resolved
    xref equivalence — that's the one-button "make L7+L10 consistent
    again" path.

    Fields:
      - `xrefs_mode`: the corpus's current `layers.xrefs` value. Drives
        whether the op is even useful — `"none"` short-circuits at
        execute time (nothing to backfill).
      - `n_dangling_sources`: total source-term count across all 18
        sub-labels whose `dangling_xrefs` list is non-empty. Cheap
        diagnostic via `count_dangling_xrefs`.
      - `will_recompute_l10`: True when `layers.cross_doc_xrefs=true`.
        The summary uses this to tell the user the op will be heavier
        than a pure xref resolution.
      - `l10_threshold`: forwarded to the L10 recompute call when
        applicable. Drawn from `config.cross_doc_xrefs.threshold`
        (auto-populated by validator when the layer is on).
    """

    xrefs_mode: str
    n_dangling_sources: int
    will_recompute_l10: bool
    l10_threshold: int

    @property
    def summary(self) -> str:
        if self.xrefs_mode == "none":
            return (
                'L7 xrefs layer is "none" — backfill is a no-op. '
                'Set xrefs to "collect_only" or "use" first; '
                "imports will then start storing the data this op "
                "would resolve."
            )
        head = (
            f"Resolve dangling xrefs across all 18 shipped ontologies: "
            f"{self.n_dangling_sources} terms currently hold "
            f"unresolved xref strings. Edges are MERGEd (idempotent); "
            f"resolved entries are stripped from each source's "
            f"`dangling_xrefs` list."
        )
        if self.will_recompute_l10:
            return (
                f"{head} L10 cross_doc_xrefs layer is on — the op will "
                f"ALSO rebuild :RELATED_BY_XREF edges graph-wide "
                f"(threshold={self.l10_threshold}). Heavier; expect "
                "minutes on a corpus with thousands of docs."
            )
        return head


@dataclass(frozen=True)
class BackfillXrefsResult:
    """Outcome of `backfill_xrefs_execute`.

    `xrefs_layer_skipped` is True when the corpus had `xrefs="none"`
    at plan time — the execute is a documented no-op, NOT a failure.

    `per_ontology_counts` is the dict returned by
    `kg.ontology_xrefs.backfill_resolved_xrefs` — keyed by term
    sub-label, each value `{"n_edges_attempted": int,
    "n_sources_cleaned": int}`. None when the underlying call hit
    a session-level error.

    `n_l10_edges_written` carries the graph-wide L10 recompute's
    return value. None when L10 wasn't requested (layer off) OR the
    recompute failed; the boolean `l10_attempted` disambiguates.
    """

    xrefs_layer_skipped: bool
    per_ontology_counts: dict[str, dict[str, int]] | None
    l10_attempted: bool
    n_l10_edges_written: int | None


async def backfill_xrefs_plan(config: CorpusConfig) -> BackfillXrefsPlan:
    """Build the plan for the `backfill_xrefs` bulk_op.

    Cheap: one count query per ontology sub-label (sum across 18) +
    config introspection. No mutation.
    """
    kg_client = get_kg_client()
    total_dangling = 0
    for term_label in ONTOLOGY_SUB_LABELS:
        try:
            total_dangling += await ontology_xrefs.count_dangling_xrefs(
                kg_client,
                term_label,
            )
        except Exception as exc:
            logger.warning(
                "backfill_xrefs_plan: count_dangling_xrefs(%s) failed: %r",
                term_label,
                exc,
            )

    will_recompute_l10 = bool(config.layers.cross_doc_xrefs)
    if will_recompute_l10:
        # CorpusConfig validator auto-populates a default threshold;
        # falling back defensively to 2 in case a future caller
        # constructed a CorpusConfig without the validator (shouldn't
        # happen via load_corpus_config).
        threshold = (
            config.cross_doc_xrefs.threshold
            if config.cross_doc_xrefs is not None
            else cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD
        )
    else:
        threshold = cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD

    return BackfillXrefsPlan(
        xrefs_mode=config.layers.xrefs,
        n_dangling_sources=total_dangling,
        will_recompute_l10=will_recompute_l10,
        l10_threshold=threshold,
    )


async def backfill_xrefs_execute(
    plan: BackfillXrefsPlan,
    config: CorpusConfig,
) -> BackfillXrefsResult:
    """Perform the backfill promised by `plan`.

    Two-phase:
      1. Always: `await ontology_xrefs.backfill_resolved_xrefs(client)` —
         the per-ontology resolve + strip passes.
      2. Conditional on `plan.will_recompute_l10`:
         `await cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(
         client, plan.l10_threshold)`.

    `xrefs_mode == "none"` short-circuits to a skipped result. The
    plan's summary surfaces that case so the user knows clicking
    Confirm won't do anything; the execute mirrors that for callers
    that don't show the plan first.
    """
    if plan.xrefs_mode == "none":
        return BackfillXrefsResult(
            xrefs_layer_skipped=True,
            per_ontology_counts=None,
            l10_attempted=False,
            n_l10_edges_written=None,
        )

    kg_client = get_kg_client()
    try:
        per_ontology = await ontology_xrefs.backfill_resolved_xrefs(kg_client)
    except Exception as exc:
        logger.warning(
            "backfill_xrefs_execute: backfill_resolved_xrefs failed: %r",
            exc,
        )
        per_ontology = None

    n_l10 = None
    if plan.will_recompute_l10:
        try:
            n_l10 = await cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(
                kg_client,
                plan.l10_threshold,
            )
        except Exception as exc:
            logger.warning(
                "backfill_xrefs_execute: recompute_cross_doc_xrefs_global failed: %r",
                exc,
            )
            n_l10 = None

    return BackfillXrefsResult(
        xrefs_layer_skipped=False,
        per_ontology_counts=per_ontology,
        l10_attempted=plan.will_recompute_l10,
        n_l10_edges_written=n_l10,
    )


# ---- recompute_cross_doc_xrefs (L10 graph-wide rebuild) ----


@dataclass(frozen=True)
class RecomputeCrossDocXrefsPlan:
    """Plan for the standalone L10 graph-wide recompute.

    Use case: the user wants to refresh L10 edges without backfilling
    xrefs (e.g. they manually edited the ontology graph and want L10
    to reflect that, OR they're recomputing with a new threshold).

    Fields:
      - `enabled`: corpus's `layers.cross_doc_xrefs` setting. Drives
        the no-op short-circuit at execute time.
      - `n_existing_l10_edges`: live count of `:RELATED_BY_XREF` edges
        before the rebuild. The execute wipes + rewrites, so this is
        the "current state" snapshot the user sees in the dialog.
      - `threshold`: forwarded to the recompute call. Drawn from
        `config.cross_doc_xrefs.threshold`.
    """

    enabled: bool
    n_existing_l10_edges: int
    threshold: int

    @property
    def summary(self) -> str:
        if not self.enabled:
            return (
                "L10 cross_doc_xrefs layer is off — recompute is a "
                "no-op. Enable the layer in `corpus.toml` first."
            )
        return (
            f"Rebuild L10 :RELATED_BY_XREF edges graph-wide "
            f"(threshold={self.threshold}). "
            f"{self.n_existing_l10_edges} existing edges will be "
            f"wiped and rewritten from scratch via the shared-concept-"
            f"via-xref query over every doc pair. Expect minutes on a "
            f"corpus with thousands of docs."
        )


@dataclass(frozen=True)
class RecomputeCrossDocXrefsResult:
    """Outcome of `recompute_cross_doc_xrefs_execute`.

    `layer_skipped` is True when the corpus had `cross_doc_xrefs=false`
    at plan time — execute is a documented no-op.

    `n_edges_written` is the integer count returned by the underlying
    `recompute_cross_doc_xrefs_global` call; None when skipped or
    the Cypher failed.
    """

    layer_skipped: bool
    n_edges_written: int | None


async def recompute_cross_doc_xrefs_plan(
    config: CorpusConfig,
) -> RecomputeCrossDocXrefsPlan:
    """Build the plan for the standalone L10 recompute.

    Cheap: one count query for the live L10 edge total + config
    introspection. No mutation.
    """
    enabled = bool(config.layers.cross_doc_xrefs)
    if enabled:
        kg_client = get_kg_client()
        # Reuse the same diagnostic the install plan uses.
        with kg_client.driver.session() as session:
            try:
                result = session.run("MATCH ()-[r:RELATED_BY_XREF]-() RETURN count(r) AS n")
                row = result.single()
                n_edges = int(row["n"]) if row else 0
            except Exception as exc:
                logger.warning(
                    "recompute_cross_doc_xrefs_plan: edge count failed: %r",
                    exc,
                )
                n_edges = 0
        threshold = (
            config.cross_doc_xrefs.threshold
            if config.cross_doc_xrefs is not None
            else cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD
        )
    else:
        n_edges = 0
        threshold = cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD

    return RecomputeCrossDocXrefsPlan(
        enabled=enabled,
        n_existing_l10_edges=n_edges,
        threshold=threshold,
    )


async def recompute_cross_doc_xrefs_execute(
    plan: RecomputeCrossDocXrefsPlan,
    config: CorpusConfig | None = None,
) -> RecomputeCrossDocXrefsResult:
    """Perform the L10 rebuild promised by `plan`.

    When `config` is passed AND `layers.cross_doc_xrefs=false`, the
    reconcile wipes existing :RELATED_BY_XREF edges corpus-wide before
    returning `layer_skipped=True`. Without `config` (legacy callers)
    the old skip-without-wipe behavior is preserved.
    """
    if not plan.enabled:
        if config is not None:
            await reconcile_cross_doc_xrefs_to_config(
                get_kg_client(),
                config,
            )
        return RecomputeCrossDocXrefsResult(
            layer_skipped=True,
            n_edges_written=None,
        )
    kg_client = get_kg_client()
    try:
        n = await cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(
            kg_client,
            plan.threshold,
        )
    except Exception as exc:
        logger.warning(
            "recompute_cross_doc_xrefs_execute: failed: %r",
            exc,
        )
        n = None
    return RecomputeCrossDocXrefsResult(
        layer_skipped=False,
        n_edges_written=n,
    )


# ---- clear_xref_edges (per-ontology xref wipe) ----


@dataclass(frozen=True)
class ClearXrefEdgesPlan:
    """Plan for wiping one ontology's xref surface.

    Use cases:
      - Maintenance: the user wants to clear MeSH's xref edges (e.g.
        before re-running `backfill_xrefs` after manual ontology
        cleanup).
      - Reset: the user is switching `xrefs="use"` -> `"none"` and
        wants the existing edges gone too.

    Fields:
      - `ontology_name`: the user-facing key (`"mesh"`, `"go"`, ...).
      - `term_label`: the corresponding `:<X>Term` sub-label, mapped
        via `ONTOLOGY_REGISTRY`. Stored on the plan so execute doesn't
        need to re-map.
      - `n_existing_edges`: count of outbound `:<X>_XREF` edges from
        this ontology's terms.
      - `n_dangling_sources`: source terms with non-empty
        `dangling_xrefs` (which will also be cleared).
    """

    ontology_name: str
    term_label: str
    n_existing_edges: int
    n_dangling_sources: int

    @property
    def summary(self) -> str:
        return (
            f"Clear {self.ontology_name} xref surface: delete "
            f"{self.n_existing_edges} outgoing :<X>_XREF edges and "
            f"strip `dangling_xrefs` from {self.n_dangling_sources} "
            f"source terms. INBOUND xref edges from OTHER ontologies "
            f"pointing at {self.ontology_name} terms are LEFT IN "
            f"PLACE — they belong to the other ontology's data. "
            f"Run this op for those ontologies in turn to wipe both "
            f"directions."
        )


@dataclass(frozen=True)
class ClearXrefEdgesResult:
    """Outcome of `clear_xref_edges_execute`.

    `n_cleared` is the sum of (edges deleted + properties removed)
    returned by `ontology_xrefs.clear_xref_edges_for_ontology`. None
    on Cypher failure (logged).
    """

    ontology_name: str
    n_cleared: int | None


async def clear_xref_edges_plan(ontology_name: str) -> ClearXrefEdgesPlan:
    """Build the plan for clearing one ontology's xref surface.

    Cheap: one edge count + one dangling-source count.

    Raises `ValueError` for an unknown ontology name (registry miss),
    matching the `import_ontology_plan` / `delete_ontology_plan`
    contract for unknown-ontology calls.
    """
    if ontology_name not in ONTOLOGY_REGISTRY:
        raise ValueError(
            f"clear_xref_edges_plan: unknown ontology "
            f"{ontology_name!r}. Known: {sorted(ONTOLOGY_REGISTRY)}."
        )
    entry = ONTOLOGY_REGISTRY[ontology_name]
    term_label = entry["term_label"]

    kg_client = get_kg_client()
    try:
        n_edges = await ontology_xrefs.count_xref_edges(kg_client, term_label)
    except Exception as exc:
        logger.warning(
            "clear_xref_edges_plan: count_xref_edges(%s) failed: %r",
            term_label,
            exc,
        )
        n_edges = 0
    try:
        n_dangling = await ontology_xrefs.count_dangling_xrefs(
            kg_client,
            term_label,
        )
    except Exception as exc:
        logger.warning(
            "clear_xref_edges_plan: count_dangling_xrefs(%s) failed: %r",
            term_label,
            exc,
        )
        n_dangling = 0
    return ClearXrefEdgesPlan(
        ontology_name=ontology_name,
        term_label=term_label,
        n_existing_edges=n_edges,
        n_dangling_sources=n_dangling,
    )


async def clear_xref_edges_execute(
    plan: ClearXrefEdgesPlan,
) -> ClearXrefEdgesResult:
    """Perform the wipe promised by `plan`."""
    kg_client = get_kg_client()
    try:
        n = await ontology_xrefs.clear_xref_edges_for_ontology(
            kg_client,
            plan.term_label,
        )
    except Exception as exc:
        logger.warning(
            "clear_xref_edges_execute (%s): failed: %r",
            plan.term_label,
            exc,
        )
        n = None
    return ClearXrefEdgesResult(
        ontology_name=plan.ontology_name,
        n_cleared=n,
    )
