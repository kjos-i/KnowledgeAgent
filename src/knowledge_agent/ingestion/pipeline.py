"""End-to-end ingestion of one document.

Composes the single-purpose helpers in this package into one call:

    parse -> ids -> metadata -> embed -> write to LanceDB + Neo4j

Stage-by-stage policy:

  - parse: raises on failure. If the file can't be parsed there's nothing
    downstream can do; the caller (CLI, script, future GUI) decides
    whether to skip or report.
  - metadata: returns None when no DOI candidate is found OR the candidate
    doesn't resolve. The pipeline continues - chunks still get written to
    LanceDB with `metadata_status = "baseline"` or `"pending"` and no KG
    write happens for that document.
  - embed: returns None on Voyage API failure. The pipeline aborts the
    LanceDB write for that document (chunks need vectors) but still
    attempts the KG write if metadata resolved.
  - LanceDB / KG writes: fail-soft (return False); the result reports it.

Idempotent on `doc_id`: re-ingesting the same file deletes its existing
LanceDB rows first, then rewrites. KG writes use MERGE patterns that are
inherently idempotent.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from knowledge_agent.config import get_settings
from knowledge_agent.entity_extractors import (
    get_extractor,
    validate_entity_types,
)
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.errors import ErrorDetail
from knowledge_agent.ingestion.embed import embed_texts
from knowledge_agent.ingestion.ids import compute_doc_id, make_chunk_id
from knowledge_agent.ingestion.metadata import (
    extract_doi_candidates,
    resolve_doi,
    resolve_metadata,
)
# OpenAlex resolution lives in its own module since 2026-06-29 — the
# orchestrator (this file) now only USES these functions; the
# implementations live next door. Re-exported below for backward
# compatibility so `pipeline.resolve_openalex(...)` / `lookup_known_doi`
# still work (bulk_ops + tests reach them through pipeline).
from knowledge_agent.ingestion.metadata_resolution import (
    _AUTHORS_DISPLAY_MAX,
    _apply_resolved_work,
    _build_authors_display,
    _doc_metadata_fields_from_work,
    lookup_known_doi,
    resolve_openalex,
)
from knowledge_agent.ingestion import triples_extractor
from knowledge_agent.ingestion.parse import (
    ParsedChunk,
    parse_document,
    supported_extensions,
)
from knowledge_agent.kg.client import get_kg_client
from knowledge_agent.kg.corpus_config import CorpusConfig
from knowledge_agent.kg.openalex_writes import (
    _clean_doi_for_storage,
    _extract_openalex_id,
)
from knowledge_agent.kg.schema import (
    MAIN_LABELS,
    PAPER_LABEL,
    SUB_LABEL_TO_MAIN,
)
from knowledge_agent.kg.triples_writes import ExtractedTriple
from knowledge_agent.search.client import get_search_client

logger = logging.getLogger(__name__)


async def _try_kg_write(coro_fn, *args, log_prefix: str, doc_id: str):
    """Run an async KG write, returning (ok, error). Logs on failure.

    Used by `ingest_document` for the L1-L4 OpenAlex writes that fan
    out via `asyncio.gather`. Each write needs the same try/except
    shape but as a coroutine, so callers pass `coro_fn` + args.
    """
    try:
        await coro_fn(*args)
        return True, None
    except Exception as exc:
        logger.warning(
            "ingest_document (%s): %s failed: %r",
            doc_id, log_prefix, exc,
        )
        return False, ErrorDetail.from_exception(exc)


@dataclass
class IngestResult:
    """Outcome of `ingest_document(path, config)` - one entry per pipeline stage."""

    doc_id: str
    path: Path
    n_chunks: int
    metadata_status: str  # "enriched" / "pending" / "baseline"
    work: dict[str, Any] | None
    embed_ok: bool
    embed_error: ErrorDetail | None
    lancedb_ok: bool
    lancedb_error: ErrorDetail | None
    kg_citations_ok: bool
    kg_citations_error: ErrorDetail | None
    kg_authorships_ok: bool
    kg_authorships_error: ErrorDetail | None
    kg_venue_ok: bool
    kg_venue_error: ErrorDetail | None
    kg_topics_ok: bool
    kg_topics_error: ErrorDetail | None
    kg_chunks_ok: bool
    kg_chunks_error: ErrorDetail | None
    kg_entities_ok: bool
    kg_entities_error: ErrorDetail | None
    n_entity_mentions: int  # total mentions written across all chunks (0 when layer off)
    # L7 per-ontology linking results. Keyed by ontology name (e.g. "mesh", "go").
    # Each entry: dict with "imported" (True if newly imported this ingest, False if
    # already present), "import_ok" (False if import failed), and "n_links"
    # (count of :CANONICAL_TO edges written by this run's linking pass).
    kg_ontology_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    # L8 triples results. `kg_triples_ok` is True when the layer ran cleanly
    # (including the all-empty case where no chunk yielded any triples).
    # `n_triples_written` is the total count of typed-relation edges submitted
    # across all chunks. Zero when the layer is off OR every LLM call returned
    # an empty list OR every triple was dropped by the vocabulary check.
    kg_triples_ok: bool = False
    kg_triples_error: ErrorDetail | None = None
    n_triples_written: int = 0
    # L9 cross-doc synthesis results. `kg_cross_doc_ok` is True when the
    # recompute Cypher ran cleanly. `n_cross_doc_edges_written` is the
    # number of `:RELATED_TO` edges this doc has after the rebuild
    # (the recompute pass wipes old edges incident to the doc first, so
    # this reflects current state, not deltas). Zero when the layer is
    # off OR no other doc met the shared-entity threshold.
    kg_cross_doc_ok: bool = False
    kg_cross_doc_error: ErrorDetail | None = None
    n_cross_doc_edges_written: int = 0
    # L10 concept-level cross-doc synthesis via xref equivalence.
    # `kg_cross_doc_xrefs_ok` is True when the recompute Cypher ran
    # cleanly. `n_cross_doc_xrefs_edges_written` is the number of
    # `:RELATED_BY_XREF` edges this doc has after the rebuild. Zero
    # when the layer is off OR no other doc met the shared-concept
    # threshold (the latter is a valid outcome, distinct from a
    # crash). Layer requires `entities=true` AND `xrefs="use"` —
    # both enforced by `CorpusConfig` validators.
    kg_cross_doc_xrefs_ok: bool = False
    kg_cross_doc_xrefs_error: ErrorDetail | None = None
    n_cross_doc_xrefs_edges_written: int = 0


async def delete_doc(doc_id: str) -> bool:
    """Wipe a doc's data across LanceDB + Neo4j. Idempotent.

    The canonical per-doc delete composition: LanceDB chunks, then KG
    L1-L4 focal :Document (with orphan GC for authors / venues / topics),
    then KG L5 :Chunk nodes, then KG L6a entity orphan GC (which also
    drops L7 :CANONICAL_TO edges via DETACH DELETE).

    Used by:
      - `bulk_ops.delete_doc` (Layer 3 UI button) - wraps this with
        plan/confirm dialog.
      - `ingest_document`'s delete-then-write step (single source of
        truth for the delete sequence; no inline duplication).

    Every step runs even if an earlier one returned False, so a single
    failure doesn't leave the other stores half-cleaned. The return is
    `all(results)` so the caller sees "everything OK" vs "at least one
    store reported a problem (already logged)".
    """
    search_client = get_search_client()
    kg_client = get_kg_client()

    # All store primitives (LanceDB + KG) now raise on failure under the
    # typed-errors contract. `_safe` awaits each coroutine so a single
    # failure doesn't abort the remaining delete steps; one missing
    # step is better than no cleanup at all.
    async def _safe(label: str, coro) -> bool:
        try:
            await coro
            return True
        except Exception as exc:
            logger.warning("delete_doc (%s): %s failed: %r", doc_id, label, exc)
            return False

    # Order is load-bearing: L8 triples wipe MUST come before the
    # entity orphan GC. L8 edges live BETWEEN :Entity nodes (not
    # anchored to :Chunk), so chunk delete doesn't cascade them; if
    # any survive when the entity GC's plain DELETE fires, the GC step
    # errors out trying to delete a node that still has relationships.
    # Sequential awaits preserve the load-bearing ordering above; do
    # NOT change to asyncio.gather without re-checking that constraint.
    results = [
        await _safe("lancedb.delete_chunks",
                    search_client.delete_chunks_by_doc_id(doc_id)),
        await _safe("kg.delete_doc", kg_client.delete_doc(doc_id)),
        await _safe("kg.delete_chunks",
                    kg_client.delete_chunks_by_doc_id(doc_id)),
        await _safe("kg.delete_triples",
                    kg_client.delete_triples_by_doc_id(doc_id)),
        await _safe("kg.delete_entities",
                    kg_client.delete_entities_by_doc_id(doc_id)),
    ]
    return all(results)


async def re_embed(doc_id: str) -> dict[str, Any]:
    """Re-embed one doc's existing chunks; update LanceDB vector column.

    Per-doc partial op. Use when:
      - The embedding model was swapped (Voyage -> Voyage-2, or to a
        different provider) and the doc's vectors need refreshing.
      - Embedding settings changed (truncation, dimension reduction).

    Preconditions:
      - Doc's chunks must already be in LanceDB (this op reads from
        LanceDB, not from the source file).
      - Embedding model must produce vectors of the dimension declared
        in `settings.embedding_dims` - LanceDB schema enforces this.

    Side effects:
      - Re-embeds the chunk text via the configured embedding provider.
      - Delete-then-insert in LanceDB to update the vector column (LanceDB
        has no in-place column update; rewriting all doc-level fields
        is the trade-off, acceptable because re-embed is rare).
      - Rebuilds vector index when `settings.optimize_indexes_per_ingest`
        is True.

    Downstream NOT affected: embeddings are isolated to LanceDB; nothing
    in KG (`:Chunk`, `:Entity`, ontology links) depends on the vector.

    Returns dict with:
      - "embed_ok" (bool): Voyage (or other provider) call success.
      - "lancedb_ok" (bool): write success.
      - "n_chunks" (int): number of chunks processed.
    """
    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "re_embed (%s): LanceDB read failed: %r; aborting", doc_id, exc
        )
        return {"embed_ok": False, "lancedb_ok": False, "n_chunks": 0}
    if not chunk_rows:
        logger.warning(
            "re_embed (%s): no chunks in LanceDB; nothing to re-embed", doc_id
        )
        return {"embed_ok": False, "lancedb_ok": False, "n_chunks": 0}

    n_chunks = len(chunk_rows)
    texts = [row["text"] for row in chunk_rows]
    try:
        new_embeddings = await embed_texts(texts)
    except Exception as exc:
        logger.warning("re_embed (%s): embed_texts failed: %r", doc_id, exc)
        return {"embed_ok": False, "lancedb_ok": False, "n_chunks": n_chunks}

    # Mutate the existing rows in place: swap embeddings, refresh
    # `ingested_at` (semantically "last written"). All other fields -
    # metadata, label mirrors, source_path - are preserved.
    new_ingested_at = datetime.now()
    for row, embedding in zip(chunk_rows, new_embeddings, strict=True):
        row["embedding"] = embedding
        row["ingested_at"] = new_ingested_at

    try:
        await search_client.delete_chunks_by_doc_id(doc_id)
        await search_client.write_chunks(chunk_rows)
        lancedb_ok = True
    except Exception as exc:
        logger.warning("re_embed (%s): LanceDB rewrite failed: %r", doc_id, exc)
        lancedb_ok = False
    if lancedb_ok and get_settings().optimize_indexes_per_ingest:
        try:
            await search_client.ensure_indexes()
        except Exception as exc:
            logger.warning(
                "re_embed (%s): ensure_indexes failed: %r", doc_id, exc
            )

    return {
        "embed_ok": True,
        "lancedb_ok": lancedb_ok,
        "n_chunks": n_chunks,
    }


async def backfill_chunks(
    doc_id: str, config: CorpusConfig
) -> dict[str, Any]:
    """Re-write KG L5 (`:Chunk`) + downstream layers from existing LanceDB
    chunks. Per-doc partial op.

    Use when:
      - KG was wiped (drop, migration) but LanceDB chunks survived.
      - The corpus's chunks layer was enabled after initial LanceDB-only
        ingest (rare; chunks defaults to True).

    Preconditions:
      - Doc's chunks must already be in LanceDB.
      - `config.layers.chunks` must be True (no-op otherwise).

    Side effects:
      - Wipes existing KG `:Chunk` nodes for this doc.
      - Re-writes `:Chunk` nodes + `:PART_OF` edges from LanceDB row data.
      - Chains into `backfill_entities` (which chains into
        `backfill_ontology`) so the downstream layers stay aligned.

    `main_label` and `sub_label` are recovered from the first LanceDB
    row - all chunks of one doc share the same doc-level fields.

    Returns dict with:
      - "chunks_ok" (bool): KG L5 write success.
      - "entities" (dict): nested `backfill_entities` result (empty when
        the entities layer is disabled OR `chunks_ok` is False).
    """
    if not config.layers.chunks:
        logger.info(
            "backfill_chunks (%s): chunks layer disabled; skip", doc_id
        )
        return {"chunks_ok": False, "entities": {}}

    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_chunks (%s): LanceDB read failed: %r; aborting",
            doc_id, exc,
        )
        return {"chunks_ok": False, "entities": {}}
    if not chunk_rows:
        logger.warning(
            "backfill_chunks (%s): no chunks in LanceDB; nothing to backfill",
            doc_id,
        )
        return {"chunks_ok": False, "entities": {}}

    # Recover label state from the first row. All chunks of one doc share
    # the same doc-level fields (`main_label` / `sub_label` written
    # identically on every row by `_build_lance_rows`).
    main_label = chunk_rows[0]["main_label"]
    sub_label = chunk_rows[0].get("sub_label")

    chunks = [
        ParsedChunk(
            chunk_index=row["chunk_index"],
            text=row["text"],
            section=row.get("section"),
            page=row.get("page"),
            content_type=row.get("content_type", "text"),
        )
        for row in chunk_rows
    ]

    kg_client = get_kg_client()
    try:
        await kg_client.delete_chunks_by_doc_id(doc_id)
        await kg_client.write_chunks(doc_id, chunks, main_label, sub_label)
        chunks_ok = True
    except Exception as exc:
        logger.warning(
            "backfill_chunks (%s): KG L5 write failed: %r", doc_id, exc
        )
        chunks_ok = False

    entities_result: dict[str, Any] = {}
    if chunks_ok:
        entities_result = backfill_entities(doc_id, config)

    return {"chunks_ok": chunks_ok, "entities": entities_result}


async def backfill_entities(
    doc_id: str, config: CorpusConfig
) -> dict[str, Any]:
    """Re-extract L6a entities for one doc's existing chunks; re-link L7.

    Per-doc partial op. Use when:
      - The corpus's `entity_types` list or `extractor` was changed and
        you want to re-extract from existing chunks.
      - L6a was enabled in `corpus.toml` after the doc was ingested.

    Preconditions:
      - Doc's chunks (L5) must already be in LanceDB; this op reads
        chunk text from LanceDB, NOT by re-parsing the source file.
      - `config.layers.entities` must be True (no-op otherwise).

    Side effects:
      - Deletes existing :Entity orphans for this doc.
      - Writes new :Entity nodes + :MENTIONS edges.
      - Chains into `backfill_ontology(doc_id, config)` so :CANONICAL_TO
        edges stay aligned with the freshly-extracted entities.

    Returns a dict with:
      - "entities_ok" (bool): whether the entity write succeeded.
      - "n_mentions" (int): total mentions written.
      - "ontology" (dict): per-ontology results (empty when entities_ok=False).
    """
    if not config.layers.entities:
        logger.info(
            "backfill_entities (%s): entities layer disabled; skip", doc_id,
        )
        return {"entities_ok": False, "n_mentions": 0, "ontology": {}}

    assert config.entities is not None  # noqa: S101 - guaranteed by config validator

    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_entities (%s): LanceDB read failed: %r; aborting",
            doc_id, exc,
        )
        return {"entities_ok": False, "n_mentions": 0, "ontology": {}}
    if not chunk_rows:
        logger.warning(
            "backfill_entities (%s): no chunks in LanceDB; nothing to backfill",
            doc_id,
        )
        return {"entities_ok": False, "n_mentions": 0, "ontology": {}}

    kg_client = get_kg_client()
    try:
        await kg_client.delete_entities_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_entities (%s): delete_entities failed: %r", doc_id, exc
        )
        return {"entities_ok": False, "n_mentions": 0, "ontology": {}}

    extractor = get_extractor(config.entities.extractor)
    chunk_mentions: list[tuple[str, list[Mention]]] = []
    n_mentions = 0
    for row in chunk_rows:
        chunk_id = row["chunk_id"]
        try:
            mentions = await extractor.extract(
                row["text"], config.entities.entity_types
            )
        except Exception as exc:
            logger.warning(
                "backfill_entities (%s): extraction failed for chunk %s: %r; skipping",
                doc_id, chunk_id, exc,
            )
            mentions = []
        chunk_mentions.append((chunk_id, mentions))
        n_mentions += len(mentions)

    try:
        await kg_client.write_entities(doc_id, chunk_mentions)
        entities_ok = True
    except Exception as exc:
        logger.warning(
            "backfill_entities (%s): write_entities failed: %r", doc_id, exc
        )
        entities_ok = False

    ontology_results: dict[str, dict[str, Any]] = {}
    triples_result: dict[str, Any] = {}
    cross_doc_result: dict[str, Any] = {}
    if entities_ok:
        ontology_results = backfill_ontology(doc_id, config)
        # L8 depends on L6a entities only, so re-running entities
        # invalidates this doc's existing triples - rebuild them too.
        # No-op when layers.triples is off.
        triples_result = backfill_triples(doc_id, config)
        # L9 depends on L6a entities (not canonicals), so re-running
        # entities invalidates the doc's :RELATED_TO edges - rebuild.
        # No-op when layers.cross_doc is off.
        cross_doc_result = backfill_cross_doc(doc_id, config)

    return {
        "entities_ok": entities_ok,
        "n_mentions": n_mentions,
        "ontology": ontology_results,
        "triples": triples_result,
        "cross_doc": cross_doc_result,
    }


async def backfill_ontology(
    doc_id: str, config: CorpusConfig
) -> dict[str, dict[str, Any]]:
    """Re-run L7 ontology linking for one doc's existing entities.

    Per-doc partial op. Use when:
      - A new ontology was enabled in `corpus.toml` after the doc was ingested
        and you want the existing entities linked.
      - The `matching` strategy was switched ("exact" -> "fuzzy") and you want
        the doc's `:CANONICAL_TO` edges rebuilt.

    Preconditions:
      - Doc's entities (L6a) already in the KG; this op only LINKS, never
        re-extracts. If entities are missing, run `backfill_entities` first.
      - Ontology layers enabled in `config.layers.ontology_*` drive which
        ontologies are linked.

    Each enabled ontology is processed independently: if `ensure_imported`
    fails for one, the others still run. Returns per-ontology dict with
    "imported" (True if newly imported this call), "import_ok", "n_links".
    Shape matches `IngestResult.kg_ontology_results` for symmetry with
    full ingest.
    """
    kg_client = get_kg_client()
    results: dict[str, dict[str, Any]] = {}
    for ontology_name in config._enabled_ontology_layers():
        ontology_cfg = config.ontology[ontology_name]
        # Per-ontology resilience boundary: one failed ontology import
        # mustn't abort the walk of the other enabled ontologies. We
        # catch here even though `ensure_ontology_imported` now raises
        # under the typed-errors contract — same pattern as the L7
        # block inside `ingest_document`.
        import_ok = True
        was_imported = False
        try:
            # Same `xrefs_mode` plumbing as the full-ingest L7 path:
            # the corpus-config flag flows through to the underlying
            # `write_ontology_terms` helper.
            was_imported = await kg_client.ensure_ontology_imported(
                ontology_name,
                xrefs_mode=config.layers.xrefs,
            )
        except Exception as exc:
            logger.warning(
                "backfill_ontology (%s): ensure_imported failed: %r",
                ontology_name, exc,
            )
            import_ok = False

        n_links = 0
        if import_ok:
            try:
                # Backfill is always per-doc. Even if the ontology was just
                # imported by this call, we scope linking to this doc - the
                # caller is asking about ONE doc, not the whole corpus.
                # bulk backfill (Layer 3) iterates over docs instead.
                n_links = await kg_client.link_entities_to_ontology(
                    ontology_name,
                    ontology_cfg.matching,
                    doc_id=doc_id,
                )
            except Exception as exc:
                logger.warning(
                    "backfill_ontology (%s): linking pass failed: %r",
                    ontology_name, exc,
                )

        results[ontology_name] = {
            "imported": not was_imported,
            "import_ok": import_ok,
            "n_links": n_links,
        }
    return results


async def backfill_triples(
    doc_id: str, config: CorpusConfig
) -> dict[str, Any]:
    """Re-extract L8 triples for one doc from chunk text + L6a entities.

    Per-doc partial op. Use when:
      - `layers.triples` was enabled in `corpus.toml` after the doc was
        ingested and you want triples extracted from existing chunks.
      - L6a entities were re-extracted (chained automatically from
        `backfill_entities`) and the triples need to follow.
      - You want to retry triple extraction after an LLM outage.

    Preconditions:
      - Doc's chunks (L5) must already be in LanceDB - the LLM reads
        chunk text from LanceDB rows, NOT by re-parsing the source.
      - Doc's L6a entities (`:Entity` + `:MENTIONS`) must already be in
        Neo4j - the per-chunk entity vocabulary comes from KG, NOT by
        re-running the entity extractor. If entities are missing or
        stale, run `backfill_entities` instead (it chains here).
      - `config.layers.triples` must be True (no-op otherwise).

    Side effects:
      - Wipes existing L8 edges for this doc via
        `delete_triples_by_doc_id` (any predicate type, edges with
        matching `doc_id` property).
      - Calls the LLM once per chunk that has at least one L6a entity
        in this doc's vocabulary. Skips chunks with zero entities (no
        possible triples).
      - Writes new typed edges via `write_triples`.

    Per-chunk failures are caught + logged so one bad LLM call doesn't
    abort the doc.

    Returns a dict with:
      - "triples_ok" (bool): whether the write succeeded.
      - "n_triples" (int): total typed edges written.
    """
    if not config.layers.triples:
        logger.info(
            "backfill_triples (%s): triples layer disabled; skip", doc_id,
        )
        return {"triples_ok": False, "n_triples": 0}

    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_triples (%s): LanceDB read failed: %r; aborting",
            doc_id, exc,
        )
        return {"triples_ok": False, "n_triples": 0}
    if not chunk_rows:
        logger.warning(
            "backfill_triples (%s): no chunks in LanceDB; nothing to backfill",
            doc_id,
        )
        return {"triples_ok": False, "n_triples": 0}

    kg_client = get_kg_client()
    # Build {chunk_id: [(key, type), ...]} from the current KG state.
    # A read failure now raises (typed-errors contract); caller's
    # backfill is aborted rather than running with stale vocab.
    try:
        chunk_entities = await kg_client.get_entities_by_chunk(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_triples (%s): get_entities_by_chunk failed: %r",
            doc_id, exc,
        )
        return {"triples_ok": False, "n_triples": 0}

    # Wipe stale triples before writing fresh ones.
    try:
        await kg_client.delete_triples_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "backfill_triples (%s): delete_triples failed: %r", doc_id, exc
        )
        return {"triples_ok": False, "n_triples": 0}

    chunk_triples: list[tuple[str, list[ExtractedTriple]]] = []
    n_triples = 0
    for row in chunk_rows:
        chunk_id = row["chunk_id"]
        entity_vocab = chunk_entities.get(chunk_id, [])
        if not entity_vocab:
            chunk_triples.append((chunk_id, []))
            continue
        try:
            triples = await triples_extractor.extract(row["text"], entity_vocab)
        except Exception as exc:
            logger.warning(
                "backfill_triples (%s): extraction failed for chunk %s: %r; skipping",
                doc_id, chunk_id, exc,
            )
            triples = []
        chunk_triples.append((chunk_id, triples))
        n_triples += len(triples)

    try:
        await kg_client.write_triples(doc_id, chunk_triples)
        triples_ok = True
    except Exception as exc:
        logger.warning(
            "backfill_triples (%s): write_triples failed: %r", doc_id, exc
        )
        triples_ok = False

    return {"triples_ok": triples_ok, "n_triples": n_triples}


async def backfill_cross_doc(
    doc_id: str, config: CorpusConfig
) -> dict[str, Any]:
    """Wipe + rewrite L9 `:RELATED_TO` edges for one doc.

    Per-doc partial op. Use when:
      - `layers.cross_doc` was enabled in `corpus.toml` after the doc
        was ingested and you want the cross-doc edges materialised.
      - L6a entities for this doc were re-extracted (chained
        automatically from `backfill_entities`).

    Preconditions:
      - Doc's L6a entities (`:Entity` + `:MENTIONS`) must already be
        in Neo4j - the overlap query relies on them. If entities are
        missing, run `backfill_entities` first (which chains here).
      - `config.layers.cross_doc` must be True (no-op otherwise).

    Side effects:
      - Wipes all `:RELATED_TO` edges incident to this doc (undirected
        match catches both directions).
      - Runs the shared-entity overlap query and MERGEs one edge per
        (this doc, other doc) pair whose shared distinct entity count
        meets the threshold.

    Returns a dict with:
      - "cross_doc_ok" (bool): whether the recompute Cypher succeeded.
      - "n_edges" (int): number of `:RELATED_TO` edges this doc has
        after the rebuild. Zero is a valid outcome (no other doc met
        the threshold) - distinct from `cross_doc_ok=False` which
        means the query crashed.
    """
    if not config.layers.cross_doc:
        logger.info(
            "backfill_cross_doc (%s): cross_doc layer disabled; skip", doc_id,
        )
        return {"cross_doc_ok": False, "n_edges": 0}

    kg_client = get_kg_client()
    threshold = config.cross_doc.threshold  # auto-populated by validator
    try:
        n = await kg_client.recompute_cross_doc_edges(doc_id, threshold)
    except Exception as exc:
        logger.warning(
            "backfill_cross_doc (%s): recompute failed: %r", doc_id, exc,
        )
        return {"cross_doc_ok": False, "n_edges": 0}
    return {"cross_doc_ok": True, "n_edges": n}


async def backfill_cross_doc_xrefs(
    doc_id: str, config: CorpusConfig
) -> dict[str, Any]:
    """Wipe + rewrite L10 `:RELATED_BY_XREF` edges for one doc.

    Per-doc partial op. Use when:
      - `layers.cross_doc_xrefs` was enabled in `corpus.toml` after
        the doc was ingested and you want the concept-level cross-doc
        edges materialised.
      - L7 xref edges or L6a entities for this doc were churned and
        the L10 view needs a refresh.

    Preconditions (enforced by `CorpusConfig` validators upstream;
    re-checked here):
      - `config.layers.cross_doc_xrefs` must be True (no-op otherwise).
      - Doc's L6a entities + `:CANONICAL_TO` edges must already exist —
        the equivalence join needs them.
      - `:<X>_XREF` edges must exist in the graph (`xrefs="use"`).
        Without them, L10 collapses to L9-like identity-only behaviour
        but still produces SOME edges; the validator already prevents
        the misconfiguration so reaching this branch implies xref
        edges are populated.

    Side effects:
      - Wipes all `:RELATED_BY_XREF` edges incident to this doc.
      - Runs the shared-concept-via-xref query and MERGEs one edge
        per (this doc, other doc) pair whose distinct shared-concept
        count meets the threshold.

    Returns a dict with:
      - "cross_doc_xrefs_ok" (bool): whether the recompute Cypher
        succeeded.
      - "n_edges" (int): number of `:RELATED_BY_XREF` edges this doc
        has after the rebuild. Zero is a valid outcome (no other doc
        met the threshold) - distinct from `cross_doc_xrefs_ok=False`
        which means the query crashed.
    """
    if not config.layers.cross_doc_xrefs:
        logger.info(
            "backfill_cross_doc_xrefs (%s): cross_doc_xrefs layer "
            "disabled; skip", doc_id,
        )
        return {"cross_doc_xrefs_ok": False, "n_edges": 0}

    kg_client = get_kg_client()
    threshold = config.cross_doc_xrefs.threshold  # auto-populated
    try:
        n = await kg_client.recompute_cross_doc_xrefs_edges(doc_id, threshold)
    except Exception as exc:
        logger.warning(
            "backfill_cross_doc_xrefs (%s): recompute failed: %r",
            doc_id, exc,
        )
        return {"cross_doc_xrefs_ok": False, "n_edges": 0}
    return {"cross_doc_xrefs_ok": True, "n_edges": n}


async def ingest_document(
    path: Path,
    config: CorpusConfig,
    main_label: str,
    sub_label: str | None = None,
) -> IngestResult:
    """Native-async ingest of one document.

    Same contract + IngestResult shape as the sync `ingest_document`.
    The native implementation parallelises the embarrassingly-parallel
    stages:

      - delete_doc: LanceDB + KG side wipes run concurrently
      - L1-L4 OpenAlex writes (4 independent writes) gather concurrently
      - L6a per-chunk entity extraction fans out via
        `asyncio.gather` bounded by an `asyncio.Semaphore` of size
        `settings.pipeline_max_concurrent_chunks`
      - L8 per-chunk triples extraction fans out the same way
      - L9 + L10 cross-doc recomputes run concurrently

    Sequential stages (CPU-bound, single API call, or state-
    dependent):

      - parse, _build_lance_rows (CPU-bound; wrapped in
        `asyncio.to_thread`)
      - OpenAlex metadata resolution (one HTTP call)
      - embed (one batched Voyage call)
      - LanceDB write (one batch insert)
      - L5 chunks write (one batch write)
      - L7 ontology linking (state-dependent: was_imported chains
        into the link-globally vs link-per-doc decision)

    Error semantics match the sync version: per-stage try/except,
    fail-soft for individual layer writes (each populates its own
    `*_error` field on the result), per-chunk fail-soft for L6a and
    L8 (one bad chunk doesn't poison the doc).
    """
    path = Path(path)

    # ---- 0. Validate inputs before any side effects.
    if main_label not in MAIN_LABELS:
        raise ValueError(
            f"main_label must be one of {MAIN_LABELS}, got {main_label!r}"
        )
    if sub_label is not None:
        if sub_label not in config.allowed_types:
            raise ValueError(
                f"sub_label {sub_label!r} is not in this corpus's "
                f"allowed_types {config.allowed_types}. Either pick a "
                f"different sub_label or add it to corpus.toml."
            )
        expected_main = SUB_LABEL_TO_MAIN.get(sub_label)
        if expected_main != main_label:
            raise ValueError(
                f"sub_label {sub_label!r} belongs under "
                f":{expected_main}, not :{main_label}."
            )
    ext = path.suffix.lower().lstrip(".")
    if ext not in supported_extensions():
        raise ValueError(
            f"No parser available for extension {ext!r} (path={path}). "
            f"Supported: {sorted(supported_extensions())}."
        )
    if config.layers.entities:
        assert config.entities is not None  # noqa: S101
        validate_entity_types(
            config.entities.extractor, config.entities.entity_types
        )

    # ---- 1. Identity ----
    doc_id = compute_doc_id(path)
    logger.info("aingest %s -> doc_id=%s", path.name, doc_id[:12])

    settings = get_settings()
    search_client = get_search_client()
    kg_client = get_kg_client()

    # ---- 2. Parse (CPU-bound, threaded) ----
    chunks = await asyncio.to_thread(parse_document, path)

    # ---- 2b. Delete stale across BOTH stores in parallel ----
    # adelete_doc on pipeline.py would re-call sync delete via thread;
    # split into the two underlying calls and gather instead for true
    # 2-way parallelism.
    await asyncio.gather(
        search_client.delete_chunks_by_doc_id(doc_id),
        kg_client.delete_doc(doc_id),
    )

    # ---- 3. Metadata resolution (one HTTP call) ----
    work = await asyncio.to_thread(resolve_metadata, chunks)
    if work is not None:
        metadata_status = "enriched"
    elif extract_doi_candidates(chunks):
        metadata_status = "pending"
    else:
        metadata_status = "baseline"

    # ---- 4. Embed (one batched Voyage call) ----
    embed_ok = False
    embed_error: ErrorDetail | None = None
    embeddings: list[list[float]] | None = None
    try:
        embeddings = await embed_texts([c.text for c in chunks])
        embed_ok = True
    except Exception as exc:
        logger.warning(
            "aingest_document (%s): aembed_texts failed: %r", doc_id, exc
        )
        embed_error = ErrorDetail.from_exception(exc)

    # ---- 5. LanceDB write ----
    lancedb_ok = False
    lancedb_error: ErrorDetail | None = None
    if embed_ok and embeddings is not None:
        rows = await asyncio.to_thread(
            _build_lance_rows,
            doc_id, chunks, embeddings, work, metadata_status,
            main_label, sub_label, path,
        )
        try:
            await search_client.ensure_schema()
            await search_client.write_chunks(rows)
            lancedb_ok = True
        except Exception as exc:
            logger.warning(
                "aingest_document (%s): LanceDB write failed: %r",
                doc_id, exc,
            )
            lancedb_error = ErrorDetail.from_exception(exc)
        if lancedb_ok and settings.optimize_indexes_per_ingest:
            try:
                await search_client.ensure_indexes()
            except Exception as exc:
                logger.warning(
                    "aingest_document (%s): aensure_indexes failed: %r",
                    doc_id, exc,
                )

    # ---- 6. KG writes ----
    try:
        await kg_client.ensure_constraints()
    except Exception as exc:
        logger.warning(
            "aingest_document (%s): aensure_constraints failed: %r",
            doc_id, exc,
        )

    # 6a. L1-L4 OpenAlex writes (4 independent writes -> gather)
    kg_citations_ok: bool = False
    kg_citations_error: ErrorDetail | None = None
    kg_authorships_ok: bool = False
    kg_authorships_error: ErrorDetail | None = None
    kg_venue_ok: bool = False
    kg_venue_error: ErrorDetail | None = None
    kg_topics_ok: bool = False
    kg_topics_error: ErrorDetail | None = None
    if (
        sub_label == PAPER_LABEL
        and config.layers.openalex_papers
        and work is not None
    ):
        results = await asyncio.gather(
            _try_kg_write(
                kg_client.write_citations, doc_id, work,
                log_prefix="KG L1 citations", doc_id=doc_id,
            ),
            _try_kg_write(
                kg_client.write_authorships, doc_id, work,
                log_prefix="KG L2 authorships", doc_id=doc_id,
            ),
            _try_kg_write(
                kg_client.write_venue, doc_id, work,
                log_prefix="KG L3 venue", doc_id=doc_id,
            ),
            _try_kg_write(
                kg_client.write_topics, doc_id, work,
                log_prefix="KG L4 topics", doc_id=doc_id,
            ),
        )
        (kg_citations_ok, kg_citations_error) = results[0]
        (kg_authorships_ok, kg_authorships_error) = results[1]
        (kg_venue_ok, kg_venue_error) = results[2]
        (kg_topics_ok, kg_topics_error) = results[3]

    # 6b. L5 chunks
    kg_chunks_ok = False
    kg_chunks_error: ErrorDetail | None = None
    if config.layers.chunks:
        try:
            await kg_client.write_chunks(
                doc_id, chunks, main_label, sub_label
            )
            kg_chunks_ok = True
        except Exception as exc:
            logger.warning(
                "aingest_document (%s): KG L5 write failed: %r",
                doc_id, exc,
            )
            kg_chunks_error = ErrorDetail.from_exception(exc)

    # ---- 7. L6a entity extraction — PARALLEL per-chunk fan-out ----
    kg_entities_ok = False
    kg_entities_error: ErrorDetail | None = None
    n_entity_mentions = 0
    chunk_mentions: list[tuple[str, list[Mention]]] = []
    if config.layers.entities and kg_chunks_ok:
        assert config.entities is not None  # noqa: S101
        extractor = get_extractor(config.entities.extractor)
        entity_types = config.entities.entity_types
        sem = asyncio.Semaphore(settings.pipeline_max_concurrent_chunks)

        async def _extract_entities_one(chunk):
            chunk_id = make_chunk_id(doc_id, chunk.chunk_index)
            async with sem:
                try:
                    mentions = await extractor.extract(
                        chunk.text, entity_types
                    )
                except Exception as exc:
                    logger.warning(
                        "L6a: extraction failed for chunk %s: %r; "
                        "skipping",
                        chunk_id, exc,
                    )
                    mentions = []
            return chunk_id, mentions

        chunk_mentions = list(
            await asyncio.gather(
                *(_extract_entities_one(c) for c in chunks)
            )
        )
        n_entity_mentions = sum(len(m) for _, m in chunk_mentions)
        try:
            await kg_client.write_entities(doc_id, chunk_mentions)
            kg_entities_ok = True
        except Exception as exc:
            logger.warning(
                "aingest_document (%s): KG L6a write failed: %r",
                doc_id, exc,
            )
            kg_entities_error = ErrorDetail.from_exception(exc)

    # ---- 8. L7 ontology linking — sequential per ontology
    #         (was_imported chains into link-globally vs link-per-doc) ----
    kg_ontology_results: dict[str, dict[str, Any]] = {}
    if kg_entities_ok:
        for ontology_name in config._enabled_ontology_layers():
            ontology_cfg = config.ontology[ontology_name]
            import_ok = True
            was_imported = False
            try:
                was_imported = await kg_client.ensure_ontology_imported(
                    ontology_name,
                    xrefs_mode=config.layers.xrefs,
                )
            except Exception as exc:
                logger.warning(
                    "L7 (%s): ensure_imported failed: %r",
                    ontology_name, exc,
                )
                import_ok = False

            n_links = 0
            if import_ok:
                try:
                    link_doc_id = None if not was_imported else doc_id
                    n_links = await kg_client.link_entities_to_ontology(
                        ontology_name,
                        ontology_cfg.matching,
                        doc_id=link_doc_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "L7 (%s): linking pass failed: %r",
                        ontology_name, exc,
                    )

            kg_ontology_results[ontology_name] = {
                "imported": not was_imported,
                "import_ok": import_ok,
                "n_links": n_links,
            }

    # ---- 9. L8 triples extraction — PARALLEL per-chunk fan-out ----
    kg_triples_ok = False
    kg_triples_error: ErrorDetail | None = None
    n_triples_written = 0
    if config.layers.triples and kg_entities_ok:
        sem_l8 = asyncio.Semaphore(
            settings.pipeline_max_concurrent_chunks
        )
        chunk_text_by_id = {
            make_chunk_id(doc_id, c.chunk_index): c.text for c in chunks
        }

        async def _extract_triples_one(chunk_id, mentions):
            entity_vocab = [
                (m.raw_text.lower(), m.entity_type) for m in mentions
            ]
            if not entity_vocab:
                return chunk_id, []
            async with sem_l8:
                try:
                    triples = await triples_extractor.extract(
                        chunk_text_by_id[chunk_id], entity_vocab,
                    )
                except Exception as exc:
                    logger.warning(
                        "L8: extraction failed for chunk %s: %r; "
                        "skipping",
                        chunk_id, exc,
                    )
                    triples = []
            return chunk_id, triples

        chunk_triples = list(
            await asyncio.gather(
                *(
                    _extract_triples_one(cid, m)
                    for cid, m in chunk_mentions
                )
            )
        )
        n_triples_written = sum(len(t) for _, t in chunk_triples)
        try:
            await kg_client.write_triples(doc_id, chunk_triples)
            kg_triples_ok = True
        except Exception as exc:
            logger.warning(
                "aingest_document (%s): KG L8 write_triples failed: %r",
                doc_id, exc,
            )
            kg_triples_error = ErrorDetail.from_exception(exc)

    # ---- 10/11. L9 + L10 cross-doc — concurrent recomputes ----
    kg_cross_doc_ok = False
    kg_cross_doc_error: ErrorDetail | None = None
    n_cross_doc_edges_written = 0
    kg_cross_doc_xrefs_ok = False
    kg_cross_doc_xrefs_error: ErrorDetail | None = None
    n_cross_doc_xrefs_edges_written = 0
    if kg_entities_ok:
        do_l9 = config.layers.cross_doc
        do_l10 = config.layers.cross_doc_xrefs
        tasks: list = []
        if do_l9:
            tasks.append(
                await kg_client.recompute_cross_doc_edges(
                    doc_id, config.cross_doc.threshold,
                )
            )
        if do_l10:
            tasks.append(
                await kg_client.recompute_cross_doc_xrefs_edges(
                    doc_id, config.cross_doc_xrefs.threshold,
                )
            )
        if tasks:
            cross_results = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            i = 0
            if do_l9:
                r = cross_results[i]; i += 1
                if isinstance(r, Exception):
                    logger.warning(
                        "aingest_document (%s): KG L9 recompute failed: %r",
                        doc_id, r,
                    )
                    kg_cross_doc_error = ErrorDetail.from_exception(r)
                else:
                    n_cross_doc_edges_written = r
                    kg_cross_doc_ok = True
            if do_l10:
                r = cross_results[i]
                if isinstance(r, Exception):
                    logger.warning(
                        "aingest_document (%s): KG L10 recompute failed: %r",
                        doc_id, r,
                    )
                    kg_cross_doc_xrefs_error = ErrorDetail.from_exception(r)
                else:
                    n_cross_doc_xrefs_edges_written = r
                    kg_cross_doc_xrefs_ok = True

    return IngestResult(
        doc_id=doc_id,
        path=path,
        n_chunks=len(chunks),
        metadata_status=metadata_status,
        work=work,
        embed_ok=embed_ok,
        embed_error=embed_error,
        lancedb_ok=lancedb_ok,
        lancedb_error=lancedb_error,
        kg_citations_ok=kg_citations_ok,
        kg_citations_error=kg_citations_error,
        kg_authorships_ok=kg_authorships_ok,
        kg_authorships_error=kg_authorships_error,
        kg_venue_ok=kg_venue_ok,
        kg_venue_error=kg_venue_error,
        kg_topics_ok=kg_topics_ok,
        kg_topics_error=kg_topics_error,
        kg_chunks_ok=kg_chunks_ok,
        kg_chunks_error=kg_chunks_error,
        kg_entities_ok=kg_entities_ok,
        kg_entities_error=kg_entities_error,
        n_entity_mentions=n_entity_mentions,
        kg_ontology_results=kg_ontology_results,
        kg_triples_ok=kg_triples_ok,
        kg_triples_error=kg_triples_error,
        n_triples_written=n_triples_written,
        kg_cross_doc_ok=kg_cross_doc_ok,
        kg_cross_doc_error=kg_cross_doc_error,
        n_cross_doc_edges_written=n_cross_doc_edges_written,
        kg_cross_doc_xrefs_ok=kg_cross_doc_xrefs_ok,
        kg_cross_doc_xrefs_error=kg_cross_doc_xrefs_error,
        n_cross_doc_xrefs_edges_written=(
            n_cross_doc_xrefs_edges_written
        ),
    )


def _build_lance_rows(
    doc_id: str,
    chunks: list[ParsedChunk],
    embeddings: list[list[float]],
    work: dict[str, Any] | None,
    metadata_status: str,
    main_label: str,
    sub_label: str | None,
    source_path: Path,
) -> list[dict[str, Any]]:
    """Assemble chunk dicts shaped for the LanceDB chunks-table schema."""
    # Single source of truth: derive doc-level OpenAlex fields once.
    doc_fields = _doc_metadata_fields_from_work(work)

    ingested_at = datetime.now()
    # `.as_posix()` so stored paths use forward slashes on every OS.
    # Comparing stored vs current at sync time then works without
    # OS-specific separator normalization.
    source_path_str = source_path.as_posix()
    rows: list[dict[str, Any]] = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        rows.append(
            {
                "chunk_id": make_chunk_id(doc_id, chunk.chunk_index),
                "doc_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "page": chunk.page,
                "char_start": None,
                "char_end": None,
                "content_type": chunk.content_type,
                "image_ref": None,
                "text": chunk.text,
                "embedding": embedding,
                # Mirror the Neo4j top-level and sub-labels. main_label
                # is always set; sub_label is None when the caller didn't
                # pick a subtype (loose-file case).
                "main_label": main_label,
                "sub_label": sub_label,
                **doc_fields,
                "metadata_status": metadata_status,
                "source_path": source_path_str,
                "ingested_at": ingested_at,
            }
        )
    return rows


# `_doc_metadata_fields_from_work` and `_build_authors_display` moved
# to `metadata_resolution.py` 2026-06-29. Re-imported at top of file so
# `_build_lance_rows` and any test patches still find them at the
# original `pipeline.X` path.
