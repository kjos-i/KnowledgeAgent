"""LanceDB connection wrapper for the hybrid retrieval store.

Owns the connection lifecycle (lazy open), the schema + write methods used
by the ingestion pipeline, and the search methods read by the agent.

Error policy: raise-at-leaves across writes, maintenance, AND search.
All methods raise on preconditions (`ValueError` for empty `doc_id`
etc.) and propagate Lance / Voyage / disk failures to the caller. The
ingestion pipeline + bulk_ops + LangGraph nodes are the orchestrator
boundaries that catch and report via:
  - `IngestResult.lancedb_error` (ingest pipeline)
  - per-op `failures` lists (bulk_ops)
  - `AgentState.lancedb_retrieval_error` (nodes.lancedb_retriever_node)

Public methods are async. LanceDB is embedded (local disk, no
network), so most calls return in milliseconds and the bodies run
inline on the event loop. A future polish pass could wrap them in
`asyncio.to_thread` for hard-bound responsiveness; deferred until
profiling shows real contention.

LanceDB is embedded - the "connection" is a handle to an on-disk directory.
No process, no port. Each `LanceClient` is cheap; one per process is the
expected pattern (cached via `get_search_client`).

Idempotency note: `write_chunks` is a plain append. Re-ingesting the same
doc_id will create duplicate rows unless the caller first deletes that
doc_id's existing rows. The ingestion pipeline owns the dedup contract.

Search modes (mirroring ResearchArticlesAgent's `retrieval.py`):
  - hybrid  - BM25 + vector kNN fused via LanceDB's native RRF
  - fts     - BM25 only
  - vector  - kNN cosine only
Cross-store RRF (for the future `parallel_fused` agent mode) lives in the
agent code, not here.
"""

import asyncio
import logging
import math
from functools import lru_cache
from typing import Any, Literal

import lancedb
from lancedb import AsyncConnection
from lancedb.rerankers import RRFReranker

from knowledge_agent.config import Settings, get_settings
from knowledge_agent.ingestion.embed import embed_texts
from knowledge_agent.models import RetrievedChunk
from knowledge_agent.search.schema import CHUNKS_TABLE, chunks_schema

logger = logging.getLogger(__name__)

# LanceDB's `search().where()` builder requires an explicit `.limit()`. The
# methods that scan the chunks table for sync/backfill workflows
# (`get_chunks_by_doc_id`, `list_indexed_docs`) need ALL matching rows, not
# a top-K. We pass a very large limit instead of paging - at our scale
# (10K-100K chunks per corpus) this is one round-trip without surprises;
# the real ceiling lives in `LanceClient.write_chunks` and a corpus that
# exceeds this would already be hitting other scalability limits.
_LANCEDB_SCAN_LIMIT = 10_000_000

# Per-mode score column. LanceDB stores the ranking value under different
# column names depending on the search type:
#   vector  -> _distance         (cosine distance, lower = closer)
#   fts     -> _score            (BM25 score, higher = better)
#   hybrid  -> _relevance_score  (RRF fused score, higher = better)
# The _row_to_chunk helper normalises these into one `score` field on
# RetrievedChunk (higher = better in all modes).
_SCORE_COLUMN: dict[str, str] = {
    "vector": "_distance",
    "fts": "_score",
    "hybrid": "_relevance_score",
}


# Sentinel `LANCEDB_PATH` the GUI sets when no corpus is active — the
# backend requires a non-empty path (see `config.Settings.lancedb_path`),
# so we hand it this marker rather than a real writable folder.
# `_ensure_conn` refuses to open it, so nothing can create a phantom
# store or read/write data with no corpus. Never a real location.
NO_CORPUS_SENTINEL = "__no_active_corpus__"


class NoActiveCorpusError(RuntimeError):
    """A LanceDB op was attempted with no active corpus.

    Raised by `LanceClient._ensure_conn` when `lancedb_path` is the
    `NO_CORPUS_SENTINEL` marker. The GUI already gates its LanceDB
    access on an active corpus; this is the backstop that makes
    reading/writing with no corpus impossible at the one choke point
    every op passes through.
    """


class LanceClient:
    """Thin sync wrapper around the LanceDB connection.

    Lazy-opens the connection on first use. Writes + maintenance methods
    raise on failure; the caller (pipeline / bulk_ops orchestrator) is
    the boundary that catches them. The search-path methods still
    fail-soft (return `[]`) for the agent's read path. `LANCEDB_PATH`
    is required at `Settings` load, so a LanceClient always has a valid
    path.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._conn: AsyncConnection | None = None
        # Serialises the first-connect: every read + write funnels through
        # `_ensure_conn` (often several at once via asyncio.gather), so without
        # this two cold callers would each open a connection and leak the loser.
        # Binds to the running loop on first await; the client lives on one loop
        # per app / CLI run (cache_clear drops it on corpus switch).
        self._conn_lock = asyncio.Lock()

    # ---- lifecycle ----

    async def _ensure_conn(self) -> AsyncConnection:
        """Lazy native-async LanceDB connection (a directory handle, no network).

        `lancedb.connect_async` returns an `AsyncConnection`; every table
        op on it is `async def`. The path directory itself is created
        sync (one mkdir at first connect) — disk-only, no event-loop
        impact.

        Refuses the `NO_CORPUS_SENTINEL` path — the marker the GUI sets
        when no corpus is active — by raising `NoActiveCorpusError`
        BEFORE any mkdir, so a corpus-less read/write can never create a
        phantom store. As the one choke point every read and write flows
        through, this single guard covers them all.
        """
        if self._conn is not None:
            return self._conn
        async with self._conn_lock:
            # Re-check under the lock: a caller we queued behind may have opened
            # the connection while we were awaiting the lock.
            if self._conn is None:
                if str(self._settings.lancedb_path) == NO_CORPUS_SENTINEL:
                    raise NoActiveCorpusError(
                        "No active corpus — create or select one in Library before using the index."
                    )
                self._settings.lancedb_path.mkdir(parents=True, exist_ok=True)
                self._conn = await lancedb.connect_async(
                    str(self._settings.lancedb_path),
                )
        return self._conn

    async def close(self) -> None:
        """Drop the cached connection handle.

        LanceDB's embedded connection has no resources to release - this is
        a hygiene method that lets `get_search_client.cache_clear()` paths
        produce fresh clients in tests.
        """
        self._conn = None

    # ---- schema ----

    async def ensure_schema(self) -> None:
        """Create the chunks table with the project schema if it doesn't exist.

        Idempotent - safe to call every startup. Does not attempt to
        migrate an existing table's schema - that's a bigger lift
        (drop + re-embed) and should be deliberate, not implicit (see
        `drop_chunks_table` for the explicit-reset path).

        Raises whatever LanceDB / disk failure occurs; the ingestion
        orchestrator boundary (`pipeline.ingest_document`) catches and
        reports via `IngestResult.lancedb_error`.
        """
        conn = await self._ensure_conn()
        if CHUNKS_TABLE in await conn.table_names():
            return
        await conn.create_table(CHUNKS_TABLE, schema=chunks_schema())
        logger.info("LanceDB: created table %r", CHUNKS_TABLE)

    async def drop_chunks_table(self) -> None:
        """Drop the chunks table entirely. Idempotent.

        Intended for explicit schema-reset paths (smoke scripts after a
        schema change, manual cleanup). The pipeline's per-doc
        `delete_chunks_by_doc_id` is the right call for normal re-ingest -
        it preserves other docs' rows. This call wipes ALL rows and the
        table itself; the next `ensure_schema` will recreate it from the
        current `chunks_schema()`.

        Returns when the table is gone (dropped or didn't exist). LanceDB
        / disk failures propagate to the caller.
        """
        conn = await self._ensure_conn()
        if CHUNKS_TABLE not in await conn.table_names():
            return
        await conn.drop_table(CHUNKS_TABLE)
        logger.info("LanceDB: dropped table %r", CHUNKS_TABLE)

    # ---- writes ----

    async def write_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Append one batch of chunk rows to the chunks table.

        Each row must match the field set declared in `chunks_schema()` -
        required fields non-null, optional ones may be missing. Empty
        batch is a no-op success.

        LanceDB / schema-mismatch / disk failures propagate to the caller.
        """
        if not chunks:
            return
        conn = await self._ensure_conn()
        table = await conn.open_table(CHUNKS_TABLE)
        await table.add(chunks)
        logger.info("LanceDB: wrote %d chunks", len(chunks))

    async def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        """Remove every chunk row for a given doc_id. Idempotent.

        Used by the ingestion pipeline to make re-ingest of the same file
        replace its rows instead of accumulating duplicates. Treats the
        missing-table case as a no-op success - "there are no rows to
        delete because there is no table" is the expected post-state of
        a fresh corpus or a smoke-test clear.

        Raises `ValueError` for empty `doc_id`. LanceDB / disk failures
        propagate.
        """
        if not doc_id:
            raise ValueError("LanceDB: delete_chunks_by_doc_id called with no doc_id")
        conn = await self._ensure_conn()
        if CHUNKS_TABLE not in await conn.table_names():
            return
        table = await conn.open_table(CHUNKS_TABLE)
        await table.delete(f"doc_id = '{doc_id}'")

    async def update_doc_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        """Patch doc-level columns for every chunk row of a doc. Idempotent.

        Used by `pipeline.resolve_openalex` and `pipeline.lookup_known_doi`
        to refresh the denormalized metadata cache (title, authors_display,
        year, doi, openalex_id, venue, source_url, language,
        metadata_status) without touching chunk text or embeddings.

        Uses LanceDB's native partial update so the heavy columns
        (`embedding`, `text`) are NOT rewritten - just the listed fields.
        Much faster than read-modify-write for corpora with sizable
        embedding vectors.

        `fields` keys must be doc-level columns. Patching chunk-level
        fields (`chunk_id`, `text`, `embedding`, `chunk_index`) would
        break invariants; the caller is responsible for not doing so.

        Raises `ValueError` for empty `doc_id` or empty `fields`
        (programming error, not a no-op). Raises `RuntimeError` when
        the chunks table doesn't exist - asking to patch a doc whose
        row doesn't exist is a real caller mistake (vs. delete, where
        "nothing to delete" is a fine outcome). LanceDB / disk failures
        propagate.
        """
        if not doc_id:
            raise ValueError("LanceDB: update_doc_metadata called with no doc_id")
        if not fields:
            raise ValueError("LanceDB: update_doc_metadata called with no fields")
        conn = await self._ensure_conn()
        if CHUNKS_TABLE not in await conn.table_names():
            raise RuntimeError("LanceDB: update_doc_metadata: chunks table doesn't exist")
        table = await conn.open_table(CHUNKS_TABLE)
        # lancedb 0.33 `AsyncTable.update` takes `updates=` for the literal
        # column→value map. `values=` is the *sync* Table API's name and is
        # rejected here with a TypeError — passing it silently broke every
        # doc-metadata write (bulk resolve, Edit-metadata modal, sync-moved
        # source_path patch) because the callers swallowed the error and
        # reported false success.
        await table.update(where=f"doc_id = '{doc_id}'", updates=fields)

    async def list_indexed_docs(self) -> list[dict[str, Any]]:
        """One entry per unique `doc_id` with doc-level metadata + counts.

        Used by `bulk_ops.sync_plan` (compare corpus index vs files on
        disk) AND the Library → Documents browser. Projects the
        doc-level columns both need via lancedb's native search builder
        - no `pylance` dependency required.

        Per-`doc_id` fields returned: `source_path`, `title`,
        `sub_label` (doc type), `main_label`, `metadata_status`,
        `ingested_at`, the editable metadata-cache columns (`doi`,
        `year`, `authors_display`, `venue`, `source_url`, `language` -
        consumed by the Documents browser's Edit modal), the read-only
        `openalex_id`, plus `n_chunks` (total chunk rows) and `n_figures`
        (rows with `content_type = 'figure'`). Aggregated in
        Python after the projection - keeps the first chunk row's
        metadata + counts. At moderate corpus scale (10K-100K chunks
        across 100s-1000s of docs) this is fast; a native groupby is
        deferred until larger scale needs it.

        Returns empty list when the chunks table doesn't yet exist (fresh
        corpus / post-clear state) OR has no rows. LanceDB / disk read
        errors propagate to the caller.
        """
        conn = await self._ensure_conn()
        if CHUNKS_TABLE not in await conn.table_names():
            return []
        table = await conn.open_table(CHUNKS_TABLE)
        query = (
            table.query()
            .select(
                [
                    "doc_id",
                    "source_path",
                    "title",
                    "sub_label",
                    "main_label",
                    "metadata_status",
                    "ingested_at",
                    "content_type",
                    "doi",
                    "openalex_id",
                    "year",
                    "authors_display",
                    "venue",
                    "source_url",
                    "language",
                ]
            )
            .limit(_LANCEDB_SCAN_LIMIT)
        )
        rows = await query.to_list()

        seen: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        figure_counts: dict[str, int] = {}
        for r in rows:
            did = r["doc_id"]
            if did not in seen:
                seen[did] = {
                    "doc_id": did,
                    "source_path": r.get("source_path"),
                    "title": r.get("title"),
                    "sub_label": r.get("sub_label"),
                    "main_label": r.get("main_label"),
                    "metadata_status": r.get("metadata_status"),
                    "ingested_at": r.get("ingested_at"),
                    "doi": r.get("doi"),
                    "openalex_id": r.get("openalex_id"),
                    "year": r.get("year"),
                    "authors_display": r.get("authors_display"),
                    "venue": r.get("venue"),
                    "source_url": r.get("source_url"),
                    "language": r.get("language"),
                }
            counts[did] = counts.get(did, 0) + 1
            if r.get("content_type") == "figure":
                figure_counts[did] = figure_counts.get(did, 0) + 1

        return [
            {
                **d,
                "n_chunks": counts[d["doc_id"]],
                "n_figures": figure_counts.get(d["doc_id"], 0),
            }
            for d in seen.values()
        ]

    async def get_chunks_by_doc_id(self, doc_id: str) -> list[dict[str, Any]]:
        """Fetch every chunk row for one doc_id, sorted by chunk_index.

        Used by partial-pipeline ops (`backfill_chunks`, `backfill_entities`,
        `re_embed`) that need to re-process already-ingested content without
        re-parsing the source file.

        Reads via lancedb's native `search().where()` builder so no
        separate `pylance` dependency is needed - lancedb's own Rust
        engine pushes the filter down. Returns chunk dicts (each
        matches `chunks_schema()`) sorted by chunk_index for
        deterministic downstream processing.

        Empty list = either no table (fresh corpus / post-clear) OR doc
        was never ingested.

        Raises `ValueError` for empty `doc_id`. LanceDB / disk read
        errors propagate to the caller.
        """
        if not doc_id:
            raise ValueError("LanceDB: get_chunks_by_doc_id called with no doc_id")
        conn = await self._ensure_conn()
        if CHUNKS_TABLE not in await conn.table_names():
            return []
        table = await conn.open_table(CHUNKS_TABLE)
        query = table.query().where(f"doc_id = '{doc_id}'").limit(_LANCEDB_SCAN_LIMIT)
        rows = await query.to_list()
        rows.sort(key=lambda r: r["chunk_index"])
        return rows

    # ---- indexes ----

    async def ensure_indexes(self) -> None:
        """Create or incrementally optimize the vector + FTS indexes.

        Vector index: attempted only when row count >= `min_rows_for_vector_index`
        (LanceDB's default IVF_PQ needs ~256 rows to train; below that
        brute force scan is fast enough). Skipped with a clear INFO log
        otherwise.

        FTS index: always attempted (no row-count threshold).

        Both creations are idempotent (the inner try/except swallows
        "already exists" - the same domain-aware pattern as
        per-ontology import resilience). `table.optimize()` then folds
        any rows written since the last call into whichever indexes
        exist - cheap incremental operation, not a rebuild.

        Returns even when the vector index is intentionally skipped -
        search still works (LanceDB falls back to brute-force scan).
        Real LanceDB / disk failures (table missing, optimize crash)
        propagate to the caller.
        """
        conn = await self._ensure_conn()
        table = await conn.open_table(CHUNKS_TABLE)
        row_count = await table.count_rows()
        threshold = self._settings.min_rows_for_vector_index

        # ---- Vector index (gated by row count threshold).
        if row_count < threshold:
            logger.info(
                "LanceDB: vector index skipped - %d rows < %d threshold "
                "(brute force scan is fast enough at this scale)",
                row_count,
                threshold,
            )
        else:
            try:
                await table.create_index(
                    column="embedding",
                    # Cosine so ranking matches the query-side distance_type and
                    # is magnitude-invariant — non-normalized (e.g. HF) embeddings
                    # rank wrong under the default L2 metric.
                    config=lancedb.index.IvfPq(distance_type="cosine"),
                )
                logger.info("LanceDB: created vector index (%d rows)", row_count)
            except Exception as exc:
                # Domain-aware swallow: most common cause is "index
                # already exists". Idempotent path; let other errors
                # surface via the outer optimize() call instead of
                # raising here.
                logger.info("LanceDB: vector index create skipped: %r", exc)

        # ---- FTS index (always attempted, no threshold).
        try:
            await table.create_index(
                column="text",
                config=lancedb.index.FTS(),
            )
            logger.info("LanceDB: created FTS index (%d rows)", row_count)
        except Exception as exc:
            # Same idempotent-already-exists swallow as the vector index.
            logger.info("LanceDB: FTS index create skipped: %r", exc)

        # ---- Fold new rows into existing indexes. Failures here
        # (disk full, lock contention) are real and must propagate.
        await table.optimize()

    # ---- search ----

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: Literal["hybrid", "fts", "vector"] | None = None,
        use_mmr: bool = False,
        num_candidates: int | None = None,
        rrf_k: int | None = None,
        mmr_lambda: float | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Single dispatch entry point. Picks one of the three search
        methods based on `mode`.

        `mode=None` falls back to `settings.lancedb_search_mode`. The tuning
        knobs (`num_candidates`, `rrf_k`, `mmr_lambda`) are forwarded to the
        chosen method, each falling back to its `settings.*` value there when
        None; `rrf_k` applies to hybrid only, `mmr_lambda` only when MMR runs.
        `use_mmr` is silently ignored for `fts` mode (no vectors to compare);
        a warning is logged in that case.
        """
        mode = mode or self._settings.lancedb_search_mode
        if mode == "hybrid":
            return await self.hybrid_search(
                query,
                top_k,
                num_candidates=num_candidates,
                rrf_k=rrf_k,
                mmr_lambda=mmr_lambda,
                filters=filters,
                use_mmr=use_mmr,
            )
        if mode == "fts":
            if use_mmr:
                logger.warning("LanceDB: use_mmr=True ignored in fts mode (no vectors)")
            return await self.fts_search(query, top_k, filters=filters)
        if mode == "vector":
            return await self.vector_search(
                query,
                top_k,
                num_candidates=num_candidates,
                mmr_lambda=mmr_lambda,
                filters=filters,
                use_mmr=use_mmr,
            )
        logger.warning("LanceDB: unknown retrieval mode %r; falling back to hybrid", mode)
        return await self.hybrid_search(
            query,
            top_k,
            num_candidates=num_candidates,
            rrf_k=rrf_k,
            mmr_lambda=mmr_lambda,
            filters=filters,
            use_mmr=use_mmr,
        )

    async def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        num_candidates: int | None = None,
        rrf_k: int | None = None,
        mmr_lambda: float | None = None,
        filters: dict[str, Any] | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        """Hybrid BM25 + vector search, fused via LanceDB's native RRF.

        The BM25 + vector legs each retrieve `num_candidates` rows, which
        `RRFReranker(K=rrf_k)` fuses (score `1/(rrf_k + rank)`) into one
        ranked candidate pool. `num_candidates`/`rrf_k` fall back to
        `settings.num_candidates`/`settings.rrf_rank_constant` when the
        caller doesn't pass them (None-check, not `or`, so a per-invoke
        override can never be silently dropped).

        With `use_mmr=True`, the `num_candidates` pool (LanceDB returns
        each row's embedding so MMR can run) is Python-side MMR-reranked
        down to `top_k`. Without MMR, the fused pool is truncated to the
        first `top_k` (`num_candidates >= top_k` is enforced by
        `Settings._validate_retrieval_windows`).

        Returns an empty list when Voyage produces no query vector (e.g.
        empty input). LanceDB / Voyage failures propagate to the caller
        under the typed-errors contract; `lancedb_retriever_node` is the
        orchestrator boundary that catches and populates
        `AgentState.lancedb_retrieval_error`.
        """
        settings = self._settings
        top_k = top_k if top_k is not None else settings.top_k
        num_candidates = num_candidates if num_candidates is not None else settings.num_candidates
        rrf_k = rrf_k if rrf_k is not None else settings.rrf_rank_constant
        mmr_lambda = mmr_lambda if mmr_lambda is not None else settings.mmr_lambda
        query_vector = await _embed_query(query)
        if query_vector is None:
            return []
        conn = await self._ensure_conn()
        table = await conn.open_table(CHUNKS_TABLE)
        search = (
            table.query()
            .nearest_to(query_vector)
            # Cosine for the vector leg (see vector_search) so its RRF input
            # ranks match the index; FTS/BM25 leg is unaffected.
            .distance_type("cosine")
            .nearest_to_text(query)
            .rerank(RRFReranker(K=rrf_k))
            .limit(num_candidates)
        )
        if filters:
            where = _filters_to_sql(filters)
            if where:
                search = search.where(where)
        rows = await search.to_list()
        if use_mmr:
            return _mmr_rerank_rows(
                rows,
                query_vector,
                top_k,
                mmr_lambda,
                "hybrid",
            )
        return [_row_to_chunk(r, "hybrid") for r in rows[:top_k]]

    async def fts_search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Lexical-only BM25 search over the `text` column.

        Useful when exact terms matter (gene names, methods) and a
        query-side embedding cost is unwanted. No MMR variant - there
        are no vectors at query time to drive diversity. LanceDB
        failures propagate (typed-errors contract).
        """
        settings = self._settings
        top_k = top_k or settings.top_k
        conn = await self._ensure_conn()
        table = await conn.open_table(CHUNKS_TABLE)
        search = table.query().nearest_to_text(query).limit(top_k)
        if filters:
            where = _filters_to_sql(filters)
            if where:
                search = search.where(where)
        rows = await search.to_list()
        return [_row_to_chunk(r, "fts") for r in rows]

    async def vector_search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        num_candidates: int | None = None,
        mmr_lambda: float | None = None,
        filters: dict[str, Any] | None = None,
        use_mmr: bool = False,
    ) -> list[RetrievedChunk]:
        """Dense-vector kNN search (cosine).

        Useful for conceptual queries where lexical overlap is weak. The
        kNN retrieves `num_candidates` rows (falling back to
        `settings.num_candidates` via a None-check when the caller doesn't
        pass one). With `use_mmr=True`, that pool is Python-side MMR-reranked
        down to `top_k`; without MMR it is truncated to the first `top_k`
        (`num_candidates >= top_k` is enforced by
        `Settings._validate_retrieval_windows`). LanceDB / Voyage failures
        propagate (typed-errors contract).
        """
        settings = self._settings
        top_k = top_k if top_k is not None else settings.top_k
        num_candidates = num_candidates if num_candidates is not None else settings.num_candidates
        mmr_lambda = mmr_lambda if mmr_lambda is not None else settings.mmr_lambda
        query_vector = await _embed_query(query)
        if query_vector is None:
            return []
        conn = await self._ensure_conn()
        table = await conn.open_table(CHUNKS_TABLE)
        search = (
            table.query()
            .nearest_to(query_vector)
            # Cosine distance: magnitude-invariant so non-normalized (e.g. HF)
            # embeddings rank by direction. The default L2 metric ranks wrong
            # for them, and _row_to_chunk's `1 - distance` score assumes cosine.
            .distance_type("cosine")
            .limit(num_candidates)
        )
        if filters:
            where = _filters_to_sql(filters)
            if where:
                search = search.where(where)
        rows = await search.to_list()
        if use_mmr:
            return _mmr_rerank_rows(
                rows,
                query_vector,
                top_k,
                mmr_lambda,
                "vector",
            )
        return [_row_to_chunk(r, "vector") for r in rows[:top_k]]


@lru_cache(maxsize=1)
def get_search_client() -> LanceClient:
    """Process-wide singleton `LanceClient`. Mirrors the cached KG client."""
    return LanceClient()


# =========================================================================
# Module-level helpers (placed after the class per the top-down convention
# used in kg/client.py and ingestion/parse.py).
# =========================================================================


async def _embed_query(text: str) -> list[float] | None:
    """Embed a query string with `input_type='query'` via the ingest embedder.

    Returns the single 1024-dim vector, or None on Voyage failure / empty
    result. Voyage exceptions are caught here so the agent's search path
    stays fail-soft (returns `[]` on embed failure); the typed-errors
    wiring for the read path lives in `nodes.py` (Phase D). The SAME
    multimodal model used at ingest time produces query vectors so they
    share a latent space with the indexed chunks.
    """
    try:
        embeddings = await embed_texts([text], input_type="query")
    except Exception as exc:
        logger.warning("LanceDB: _embed_query Voyage call failed: %r", exc)
        return None
    if not embeddings:
        return None
    return embeddings[0]


def _filters_to_sql(filters: dict[str, Any]) -> str:
    """Translate `{field: value | [values]}` dict to a LanceDB WHERE string.

    Strings are single-quoted with internal quotes doubled (SQL escape
    convention). Ints / floats are written as-is. Lists become `IN (...)`
    clauses. Multiple keys join with AND. None values and empty lists are
    skipped. Returns "" when the dict is empty after filtering.
    """
    clauses: list[str] = []
    for field, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            items = [v for v in value if v is not None]
            if not items:
                continue
            parts = ", ".join(_sql_literal(v) for v in items)
            clauses.append(f"{field} IN ({parts})")
        else:
            clauses.append(f"{field} = {_sql_literal(value)}")
    return " AND ".join(clauses)


def _sql_literal(value: Any) -> str:
    """Render one Python value as a SQL literal for LanceDB's WHERE."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _row_to_chunk(row: dict[str, Any], mode: str) -> RetrievedChunk:
    """Map one LanceDB row to a `RetrievedChunk`; normalises the score field.

    Vector mode stores cosine DISTANCE (lower=closer); we convert to
    similarity (higher=closer) so the `score` field is comparable across
    hits within a single result set in every mode.
    """
    score_col = _SCORE_COLUMN.get(mode)
    raw = row.get(score_col) if score_col else None
    if raw is None:
        score: float | None = None
    elif mode == "vector":
        score = 1.0 - float(raw)
    else:
        score = float(raw)
    return RetrievedChunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        text=row["text"],
        score=score,
        title=row.get("title"),
        year=row.get("year"),
        authors_display=row.get("authors_display"),
        doi=row.get("doi"),
        openalex_id=row.get("openalex_id"),
        venue=row.get("venue"),
        source_url=row.get("source_url"),
        section=row.get("section"),
        page=row.get("page"),
        # Multimodal fields. LanceDB rows carry both since 2026-07-03
        # (schema always had the columns; ingest populates them). Text
        # chunks have `content_type='text'` and `image_ref=None`;
        # figure chunks have `content_type='figure'` and `image_ref`
        # set to the picture file path.
        content_type=row.get("content_type"),
        image_ref=row.get("image_ref"),
    )


def _l2_norm(v: list[float]) -> float:
    """L2 norm; returns 1.0 for a zero vector to avoid divide-by-zero."""
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _cosine_sim(a: list[float], b: list[float], na: float, nb: float) -> float:
    """Cosine similarity with pre-computed L2 norms (no per-call re-norm)."""
    return sum(x * y for x, y in zip(a, b, strict=False)) / (na * nb)


def _mmr_rerank_rows(
    rows: list[dict[str, Any]],
    query_vector: list[float],
    top_k: int,
    lambda_: float,
    mode: str,
) -> list[RetrievedChunk]:
    """Re-rank rows by Maximal Marginal Relevance, return the top_k chunks.

    Iteratively picks the row maximising

        score = lambda_ * sim(query, c) - (1 - lambda_) * max_sim(c, selected)

    where `sim` is cosine on each row's `embedding` column. `lambda_=1.0`
    is pure relevance, `lambda_=0.0` is pure diversity. Pure Python -
    pool sizes are small enough (~20-50) that the cost is negligible next
    to the LanceDB round-trip.
    """
    pool_size = len(rows)
    if pool_size == 0 or top_k <= 0:
        return []
    top_k = min(top_k, pool_size)

    vectors = [r.get("embedding") for r in rows]
    if any(v is None for v in vectors):
        # Defensive: a row missing its embedding can't be MMR-ranked.
        # Fall back to the original (already-relevance-sorted) order.
        logger.warning(
            "LanceDB: MMR pool had rows missing 'embedding'; falling back to original order"
        )
        return [_row_to_chunk(r, mode) for r in rows[:top_k]]

    q_norm = _l2_norm(query_vector)
    cand_norms = [_l2_norm(v) for v in vectors]
    sim_to_query = [
        _cosine_sim(query_vector, v, q_norm, cn) for v, cn in zip(vectors, cand_norms, strict=False)
    ]

    selected: list[int] = []
    is_selected = [False] * pool_size
    max_sim_to_selected = [0.0] * pool_size

    for _ in range(top_k):
        best_idx = -1
        best_score = float("-inf")
        for i in range(pool_size):
            if is_selected[i]:
                continue
            score = lambda_ * sim_to_query[i] - (1.0 - lambda_) * max_sim_to_selected[i]
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            break
        selected.append(best_idx)
        is_selected[best_idx] = True
        new_vec = vectors[best_idx]
        new_norm = cand_norms[best_idx]
        for i in range(pool_size):
            if is_selected[i]:
                continue
            s = _cosine_sim(vectors[i], new_vec, cand_norms[i], new_norm)
            if s > max_sim_to_selected[i]:
                max_sim_to_selected[i] = s

    return [_row_to_chunk(rows[i], mode) for i in selected]
