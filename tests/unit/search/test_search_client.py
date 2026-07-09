"""Tests for search.client - LanceClient - and search.schema.

Uses a recording Connection/Table stub so tests don't touch a real LanceDB
dataset. The schema function is exercised by reading the returned `pa.Schema`.
"""

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pytest
from lancedb.rerankers import RRFReranker

from knowledge_agent.config import Settings
from knowledge_agent.search.client import (
    NO_CORPUS_SENTINEL,
    LanceClient,
    NoActiveCorpusError,
    _filters_to_sql,
    _sql_literal,
)
from knowledge_agent.search.schema import (
    CHUNKS_TABLE,
    CONTENT_TYPES,
    METADATA_STATUS_VALUES,
    chunks_schema,
)

# ---- Test harness ----


@dataclass
class _RecordingArrowTable:
    """Stand-in for a pyarrow Table - only `to_pylist` is exercised."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_pylist(self) -> list[dict[str, Any]]:
        return list(self.rows)


@dataclass
class _RecordingQueryBuilder:
    """Stand-in for lancedb's search builder. Captures filter / select /
    limit / vector / text / reranker on the fluent chain, then
    materialises to a stub row list on `to_list()`."""

    rows_to_return: list[dict[str, Any]] = field(default_factory=list)
    last_filter: str | None = None
    last_columns: list[str] | None = None
    last_limit: int | None = None
    last_vector: list[float] | None = None
    last_text: str | None = None
    last_reranker: Any = None
    raise_on_to_arrow: Exception | None = None

    def where(self, filter_sql: str) -> "_RecordingQueryBuilder":
        self.last_filter = filter_sql
        return self

    def select(self, columns: list[str]) -> "_RecordingQueryBuilder":
        self.last_columns = list(columns)
        return self

    def limit(self, n: int) -> "_RecordingQueryBuilder":
        self.last_limit = n
        return self

    def nearest_to(self, qv: list[float]) -> "_RecordingQueryBuilder":
        self.last_vector = list(qv)
        return self

    def nearest_to_text(self, t: str) -> "_RecordingQueryBuilder":
        self.last_text = t
        return self

    def rerank(self, reranker: Any) -> "_RecordingQueryBuilder":
        # Mirrors AsyncHybridQuery.rerank — records the reranker instance
        # so hybrid_search tests can assert the RRFReranker's K equals the
        # configured `rrf_rank_constant` (the dead-knob wiring fix).
        self.last_reranker = reranker
        return self

    async def to_list(self) -> list[dict[str, Any]]:
        if self.raise_on_to_arrow is not None:
            raise self.raise_on_to_arrow
        return list(self.rows_to_return)


@dataclass
class RecordingTable:
    """Stand-in for a LanceDB Table - captures batches passed to add()."""

    added: list[Any] = field(default_factory=list)
    raise_on_add: Exception | None = None
    deleted_filters: list[str] = field(default_factory=list)
    raise_on_delete: Exception | None = None
    # update() calls captured as (where, updates) tuples.
    updates: list[tuple[str | None, dict[str, Any] | None]] = field(default_factory=list)
    raise_on_update: Exception | None = None
    # Stubbed search-builder returned by `search()` - lets tests
    # pre-load rows or force errors on `to_arrow()`.
    query_builder: _RecordingQueryBuilder | None = None
    # `search(*args, **kwargs)` capture lets search-method tests assert
    # the call shape (e.g. `query_type="fts"` for FTS, a query_vector
    # positional for vector search).
    last_search_args: tuple[Any, ...] = ()
    last_search_kwargs: dict[str, Any] = field(default_factory=dict)

    async def add(self, rows: Any) -> None:
        if self.raise_on_add is not None:
            raise self.raise_on_add
        self.added.append(rows)

    async def delete(self, where: str) -> None:
        if self.raise_on_delete is not None:
            raise self.raise_on_delete
        self.deleted_filters.append(where)

    async def update(
        self,
        updates: dict[str, Any] | None = None,
        *,
        where: str | None = None,
        updates_sql: dict[str, Any] | None = None,
    ) -> None:
        # Mirrors lancedb 0.33 `AsyncTable.update(updates, *, where,
        # updates_sql)` exactly. Keeping the real signature is the point:
        # a caller that reverts to the sync API's `values=` kwarg raises
        # TypeError here (regression guard for the silent no-op bug), same
        # as it would against a live async table.
        if self.raise_on_update is not None:
            raise self.raise_on_update
        self.updates.append((where, updates))

    def query(self) -> _RecordingQueryBuilder:
        if self.query_builder is None:
            self.query_builder = _RecordingQueryBuilder()
        return self.query_builder

    async def search(self, *args: Any, **kwargs: Any) -> _RecordingQueryBuilder:
        self.last_search_args = args
        self.last_search_kwargs = dict(kwargs)
        if self.query_builder is None:
            self.query_builder = _RecordingQueryBuilder()
        return self.query_builder

    async def count_rows(self) -> int:
        return 0

    async def create_index(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def optimize(self) -> None:
        pass


@dataclass
class RecordingConnection:
    """Stand-in for a LanceDB DBConnection - tracks tables created / opened."""

    tables: dict[str, RecordingTable] = field(default_factory=dict)
    raise_on_create: Exception | None = None
    raise_on_open: Exception | None = None
    raise_on_drop: Exception | None = None

    async def table_names(self) -> list[str]:
        return list(self.tables.keys())

    async def create_table(self, name: str, schema: Any = None, **_kwargs: Any) -> RecordingTable:
        if self.raise_on_create is not None:
            raise self.raise_on_create
        table = RecordingTable()
        self.tables[name] = table
        return table

    async def open_table(self, name: str) -> RecordingTable:
        if self.raise_on_open is not None:
            raise self.raise_on_open
        return self.tables[name]

    async def drop_table(self, name: str) -> None:
        if self.raise_on_drop is not None:
            raise self.raise_on_drop
        del self.tables[name]


def _configured_settings(**overrides: Any) -> Settings:
    """Build Settings with all required fields filled in."""
    defaults: dict[str, Any] = {
        "anthropic_api_key": "fake-anthropic-key",
        "voyage_api_key": "fake-voyage-key",
        "neo4j_uri": "neo4j://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "fake-neo4j-password",
        "lancedb_path": "/tmp/fake-lancedb-path",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _client_with_conn(settings: Settings, conn: RecordingConnection) -> LanceClient:
    """LanceClient with a pre-injected stub connection (skips lazy open)."""
    client = LanceClient(settings=settings)
    client._conn = conn  # type: ignore[assignment]
    return client


# ---- lifecycle ----


def test_conn_lazy_attribute_starts_none():
    client = LanceClient(settings=_configured_settings())
    assert client._conn is None


async def test_close_resets_conn_to_none():
    conn = RecordingConnection()
    client = _client_with_conn(_configured_settings(), conn)
    await client.close()
    assert client._conn is None


async def test_close_is_safe_when_conn_never_created():
    client = LanceClient(settings=_configured_settings())
    await client.close()  # must not raise


async def test_ensure_conn_refuses_no_corpus_sentinel():
    """`_ensure_conn` refuses the no-active-corpus sentinel path.

    The GUI sets `LANCEDB_PATH` to `NO_CORPUS_SENTINEL` when no corpus is
    active (the backend requires *some* path, so it gets this marker
    rather than a real folder). `_ensure_conn` — the single choke point
    every read and write passes through — must raise `NoActiveCorpusError`
    instead of `mkdir`-ing and connecting, so no phantom store is created
    and no data can be read/written with no corpus.

    The guard fires BEFORE any `mkdir`, so this test never touches disk
    (crucially: it does not create a stray `__no_active_corpus__` folder).
    """
    client = LanceClient(settings=_configured_settings(lancedb_path=NO_CORPUS_SENTINEL))
    with pytest.raises(NoActiveCorpusError):
        await client._ensure_conn()
    # Nothing was cached — a later call with a real path would still work.
    assert client._conn is None


# ---- ensure_schema ----


async def test_ensure_schema_creates_table_when_missing():
    conn = RecordingConnection()
    client = _client_with_conn(_configured_settings(), conn)
    # Success: returns None (typed-errors contract).
    assert await client.ensure_schema() is None
    assert CHUNKS_TABLE in conn.tables


async def test_ensure_schema_no_op_when_table_exists():
    """Existing table is left untouched - no recreate, no error."""
    pre_existing = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: pre_existing})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.ensure_schema() is None
    # Same instance - the table was not recreated.
    assert conn.tables[CHUNKS_TABLE] is pre_existing


async def test_ensure_schema_propagates_lance_exception():
    conn = RecordingConnection(raise_on_create=RuntimeError("boom"))
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.ensure_schema()


# ---- drop_chunks_table ----


async def test_drop_chunks_table_removes_existing_table():
    """Schema-reset path: existing table is dropped so the next
    ensure_schema can recreate it with the current schema."""
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable()})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.drop_chunks_table() is None
    assert CHUNKS_TABLE not in conn.tables


async def test_drop_chunks_table_no_op_when_table_absent():
    """Idempotent: dropping a non-existent table is success, not error.
    Lets smoke scripts call drop unconditionally without worrying about
    state."""
    conn = RecordingConnection()  # no tables
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.drop_chunks_table() is None


async def test_drop_chunks_table_propagates_lance_exception():
    """LanceDB error propagates - typed-errors contract; smoke
    scripts can wrap if they want fail-soft semantics."""
    conn = RecordingConnection(
        tables={CHUNKS_TABLE: RecordingTable()},
        raise_on_drop=RuntimeError("boom"),
    )
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.drop_chunks_table()


# ---- write_chunks ----


async def test_write_chunks_empty_is_noop_without_opening_table():
    """Empty input is a no-op; no need to open the table."""
    conn = RecordingConnection()
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.write_chunks([]) is None
    # No side effect on the connection.
    assert CHUNKS_TABLE not in conn.tables


async def test_write_chunks_appends_rows_to_table():
    table = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    assert await client.write_chunks(chunks) is None
    # The batch was passed through to table.add().
    assert table.added == [chunks]


async def test_write_chunks_propagates_lance_exception():
    table = RecordingTable(raise_on_add=RuntimeError("boom"))
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_chunks([{"chunk_id": "a"}])


# ---- update_doc_metadata ----


async def test_update_doc_metadata_passes_doc_id_filter_and_values():
    """Update wires the doc_id filter + values dict through to LanceDB's API."""
    table = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    fields = {"title": "New Title", "year": 2026, "doi": "10.1/abc"}
    assert await client.update_doc_metadata("abc-123", fields) is None
    assert table.updates == [("doc_id = 'abc-123'", fields)]


async def test_update_doc_metadata_uses_async_updates_kwarg_not_sync_values():
    """Regression guard for the silent-no-op bug: `update_doc_metadata`
    must call lancedb's async API with `updates=` (a column→value map),
    NOT the sync Table API's `values=`. The stub mirrors the real async
    signature, so a `values=` slip raises TypeError instead of quietly
    "succeeding" against a permissive mock while writing nothing live."""
    table = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    await client.update_doc_metadata("abc-123", {"title": "T"})
    # Recorded as (where, updates) — the map lands in the `updates` slot.
    where, updates = table.updates[0]
    assert where == "doc_id = 'abc-123'"
    assert updates == {"title": "T"}

    # And prove the guard bites: the old sync kwarg is rejected outright.
    with pytest.raises(TypeError):
        await table.update(where="doc_id = 'x'", values={"title": "y"})


async def test_update_doc_metadata_empty_doc_id_raises():
    """Empty doc_id is a programming error; raise ValueError."""
    table = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.update_doc_metadata("", {"title": "x"})
    assert table.updates == []


async def test_update_doc_metadata_empty_fields_raises():
    """Empty fields dict is a programming error - patching nothing is a bug."""
    table = RecordingTable()
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(ValueError, match="no fields"):
        await client.update_doc_metadata("abc-123", {})
    assert table.updates == []


async def test_update_doc_metadata_propagates_lance_exception():
    table = RecordingTable(raise_on_update=RuntimeError("boom"))
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.update_doc_metadata("abc-123", {"title": "x"})


# ---- list_indexed_docs ----


async def test_list_indexed_docs_aggregates_unique_doc_ids():
    """Multiple chunk rows per doc -> one summary entry per doc_id."""
    qb = _RecordingQueryBuilder(
        rows_to_return=[
            {
                "doc_id": "d1",
                "source_path": "/x/1.pdf",
                "title": "T1",
                "metadata_status": "enriched",
            },
            {
                "doc_id": "d1",
                "source_path": "/x/1.pdf",
                "title": "T1",
                "metadata_status": "enriched",
            },
            {"doc_id": "d2", "source_path": "/x/2.pdf", "title": "T2", "metadata_status": "manual"},
        ]
    )
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    docs = await client.list_indexed_docs()
    by_id = {d["doc_id"]: d for d in docs}
    assert by_id["d1"]["n_chunks"] == 2
    assert by_id["d1"]["title"] == "T1"
    assert by_id["d1"]["source_path"] == "/x/1.pdf"
    assert by_id["d2"]["n_chunks"] == 1
    assert by_id["d2"]["metadata_status"] == "manual"


async def test_list_indexed_docs_projects_only_doc_level_columns():
    """Heavy embedding + text columns are NOT loaded - only the doc-level
    columns we need are pushed down via the search builder's `select()`."""
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    await client.list_indexed_docs()
    assert qb.last_columns == [
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
    # Heavy columns must NOT be in the projection.
    assert "text" not in qb.last_columns
    assert "embedding" not in qb.last_columns
    # Limit must be passed too (search builder requires one) - check the
    # value is large enough to cover any realistic corpus.
    assert qb.last_limit is not None and qb.last_limit >= 1_000_000


async def test_list_indexed_docs_returns_type_date_and_figure_count():
    """The extended projection surfaces sub_label (type), ingested_at
    (date), and an n_figures count (content_type='figure' rows)."""
    qb = _RecordingQueryBuilder(
        rows_to_return=[
            {
                "doc_id": "d1",
                "source_path": "/x/1.pdf",
                "title": "T1",
                "sub_label": "Paper",
                "main_label": "Document",
                "metadata_status": "enriched",
                "ingested_at": "2026-07-03",
                "content_type": "text",
            },
            {
                "doc_id": "d1",
                "source_path": "/x/1.pdf",
                "title": "T1",
                "sub_label": "Paper",
                "main_label": "Document",
                "metadata_status": "enriched",
                "ingested_at": "2026-07-03",
                "content_type": "figure",
            },
        ]
    )
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    docs = await client.list_indexed_docs()
    d1 = {d["doc_id"]: d for d in docs}["d1"]
    assert d1["sub_label"] == "Paper"
    assert d1["main_label"] == "Document"
    assert d1["ingested_at"] == "2026-07-03"
    assert d1["n_chunks"] == 2
    assert d1["n_figures"] == 1


async def test_list_indexed_docs_returns_editable_metadata_fields():
    """The Documents-browser Edit modal prefills from these doc-level
    columns, so the aggregation must surface them per doc_id."""
    qb = _RecordingQueryBuilder(
        rows_to_return=[
            {
                "doc_id": "d1",
                "source_path": "/x/1.pdf",
                "title": "T1",
                "sub_label": "Paper",
                "main_label": "Document",
                "metadata_status": "enriched",
                "ingested_at": "2026-07-03",
                "content_type": "text",
                "doi": "10.1/x",
                "year": 2020,
                "authors_display": "Jane Doe; John Smith",
                "venue": "Science",
                "source_url": "https://doi.org/10.1/x",
                "language": "en",
            },
        ]
    )
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    docs = await client.list_indexed_docs()
    d1 = {d["doc_id"]: d for d in docs}["d1"]
    assert d1["doi"] == "10.1/x"
    assert d1["year"] == 2020
    assert d1["authors_display"] == "Jane Doe; John Smith"
    assert d1["venue"] == "Science"
    assert d1["source_url"] == "https://doi.org/10.1/x"
    assert d1["language"] == "en"


async def test_list_indexed_docs_empty_table_returns_empty_list():
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.list_indexed_docs() == []


async def test_list_indexed_docs_propagates_lance_exception():
    qb = _RecordingQueryBuilder(raise_on_to_arrow=RuntimeError("boom"))
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.list_indexed_docs()


# ---- get_chunks_by_doc_id ----


async def test_get_chunks_by_doc_id_returns_rows_sorted_by_chunk_index():
    """Rows from lancedb can come back in any order; the method must sort."""
    qb = _RecordingQueryBuilder(
        rows_to_return=[
            {"chunk_id": "doc1#2", "doc_id": "doc1", "chunk_index": 2},
            {"chunk_id": "doc1#0", "doc_id": "doc1", "chunk_index": 0},
            {"chunk_id": "doc1#1", "doc_id": "doc1", "chunk_index": 1},
        ]
    )
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    rows = await client.get_chunks_by_doc_id("doc1")
    assert [r["chunk_index"] for r in rows] == [0, 1, 2]
    assert [r["chunk_id"] for r in rows] == ["doc1#0", "doc1#1", "doc1#2"]


async def test_get_chunks_by_doc_id_passes_doc_id_in_filter():
    """The where() clause must constrain to the requested doc_id."""
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    await client.get_chunks_by_doc_id("abc-123")
    assert qb.last_filter == "doc_id = 'abc-123'"


async def test_get_chunks_by_doc_id_empty_input_raises():
    """Empty doc_id is a programming error; raise ValueError."""
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable()})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.get_chunks_by_doc_id("")


async def test_get_chunks_by_doc_id_no_match_returns_empty_list():
    """Doc not in LanceDB -> empty list, distinguishable from error
    paths (which raise under typed-errors)."""
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.get_chunks_by_doc_id("missing-doc") == []


async def test_get_chunks_by_doc_id_propagates_lance_exception():
    """Search builder error propagates so the caller's orchestrator
    boundary catches; the prior None-sentinel path is gone."""
    qb = _RecordingQueryBuilder(raise_on_to_arrow=RuntimeError("boom"))
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.get_chunks_by_doc_id("doc1")


# ---- missing-table behavior (one test per read/delete/update/list method) ----


async def test_get_chunks_by_doc_id_missing_table_returns_empty_quietly():
    """No chunks table -> [] (same as no rows). No warning, no error.

    Fresh corpus or post-`drop_chunks_table` state must not log
    spurious "Table 'chunks' was not found" warnings - the read just
    has nothing to find."""
    conn = RecordingConnection(tables={})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.get_chunks_by_doc_id("doc1") == []


async def test_delete_chunks_by_doc_id_missing_table_is_noop_success():
    """No chunks table -> returns None (nothing to delete is success)."""
    conn = RecordingConnection(tables={})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.delete_chunks_by_doc_id("doc1") is None


async def test_delete_chunks_by_doc_id_empty_doc_id_raises():
    """Empty doc_id is a programming error; raise ValueError."""
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable()})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.delete_chunks_by_doc_id("")


async def test_list_indexed_docs_missing_table_returns_empty_quietly():
    """No chunks table -> empty list. No warning."""
    conn = RecordingConnection(tables={})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.list_indexed_docs() == []


async def test_update_doc_metadata_missing_table_raises():
    """No chunks table -> RuntimeError. Updating a non-existent row IS
    a real caller mistake (vs. delete, where 'nothing to delete' is fine)."""
    conn = RecordingConnection(tables={})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="table doesn't exist"):
        await client.update_doc_metadata("doc1", {"title": "x"})


# ---- search methods (hybrid / fts / vector / retrieve) ----
#
# Parity with the write/maintenance test block above. The search-path
# methods got migrated to raise-at-leaves in Phase D (typed-errors
# contract) — these tests pin the leaf contract while
# `tests/unit/test_nodes.py::test_lancedb_retriever_captures_typed_error_on_failure`
# pins the orchestrator-boundary catch in the LangGraph node.
#
# `_embed_query` is patched directly because hybrid + vector search
# call it before any lance work; we want to test the lance code path
# separately from the Voyage call.


def _chunk_row(chunk_id: str, text: str) -> dict[str, Any]:
    """Minimal lance row matching the fields `_row_to_chunk` reads."""
    return {
        "chunk_id": chunk_id,
        "doc_id": "doc1",
        "text": text,
        "title": "T",
        "year": 2026,
        "authors_display": "A",
        "doi": None,
        "openalex_id": None,
        "venue": None,
        "source_url": None,
        "section": None,
        "page": None,
        # Mode-specific score column is added per-test.
    }


async def test_fts_search_returns_chunks_on_success():
    """Happy path: stub lance returns 2 rows; fts_search maps each to
    a `RetrievedChunk` carrying the `_score` field as the `score`."""
    rows = [
        {**_chunk_row("c1", "alpha"), "_score": 12.5},
        {**_chunk_row("c2", "beta"), "_score": 7.0},
    ]
    qb = _RecordingQueryBuilder(rows_to_return=rows)
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    hits = await client.fts_search("alpha beta", top_k=5)

    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].score == 12.5
    # FTS mode wires the text query through nearest_to_text — no
    # vector set, only text.
    assert qb.last_text == "alpha beta"
    assert qb.last_vector is None
    assert qb.last_limit == 5


async def test_fts_search_propagates_lance_exception():
    """Lance failure -> raises (typed-errors contract). The agent's
    `lancedb_retriever_node` boundary catches and populates
    `AgentState.lancedb_retrieval_error`."""
    qb = _RecordingQueryBuilder(raise_on_to_arrow=RuntimeError("boom"))
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="boom"):
        await client.fts_search("alpha", top_k=5)


async def test_vector_search_returns_empty_when_voyage_returns_none(monkeypatch):
    """Empty / failed Voyage embedding -> early return [] WITHOUT
    touching lance. Documented success no-op for the empty-query path
    (separate from the lance-failure path which raises)."""

    async def _fake_embed(_text: str) -> None:
        return None

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable()})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.vector_search("query", top_k=5) == []


async def test_vector_search_propagates_lance_exception(monkeypatch):
    """Lance failure during vector search -> raises (typed-errors
    contract). Voyage succeeds first; the failure surfaces from the
    lance query builder."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.1] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    qb = _RecordingQueryBuilder(raise_on_to_arrow=RuntimeError("lance boom"))
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)
    with pytest.raises(RuntimeError, match="lance boom"):
        await client.vector_search("query", top_k=5)


async def test_hybrid_search_returns_chunks_on_success(monkeypatch):
    """Happy path: hybrid mode wires both the query vector AND the
    text into the search builder, and maps rows to `RetrievedChunk`
    with the `_relevance_score` field as `score`."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    rows = [
        {**_chunk_row("c1", "alpha"), "_relevance_score": 0.91},
    ]
    qb = _RecordingQueryBuilder(rows_to_return=rows)
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    hits = await client.hybrid_search("alpha query", top_k=3)

    assert len(hits) == 1
    assert hits[0].score == 0.91
    # Hybrid mode chains BOTH nearest_to(vector) AND nearest_to_text(text)
    # on the builder.
    assert qb.last_vector == [0.5] * 1024
    assert qb.last_text == "alpha query"


async def test_hybrid_search_wires_rrf_reranker_and_num_candidates(monkeypatch):
    """The dead-knob fix: hybrid_search must attach an explicit
    `RRFReranker(K=settings.rrf_rank_constant)` to the query chain AND
    set the candidate-pool `.limit()` to `settings.num_candidates`.

    Before the fix, neither knob reached LanceDB — `rrf_rank_constant`
    was masked by LanceDB's default `RRFReranker(K=60)` and the pool was
    just `top_k`. These assertions pin that both settings now flow
    through the builder.
    """

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    settings = _configured_settings(rrf_rank_constant=17, num_candidates=42, top_k=5)
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(settings, conn)

    await client.hybrid_search("alpha query", top_k=5)

    # An explicit RRFReranker with K == the configured rrf_rank_constant.
    assert isinstance(qb.last_reranker, RRFReranker)
    assert qb.last_reranker.K == 17
    # Candidate pool = num_candidates (NOT top_k).
    assert qb.last_limit == 42


async def test_hybrid_search_uses_settings_when_params_omitted(monkeypatch):
    """When the caller omits num_candidates / rrf_k, the None-check
    fallback resolves them from settings (not a silent 0/top_k)."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    settings = _configured_settings(rrf_rank_constant=99, num_candidates=88, top_k=5)
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(settings, conn)

    # No num_candidates / rrf_k passed → settings supply both.
    await client.hybrid_search("q")

    assert qb.last_reranker.K == 99
    assert qb.last_limit == 88


async def test_hybrid_search_explicit_params_override_settings(monkeypatch):
    """Explicit num_candidates / rrf_k win over settings (per-invoke
    override path the eval case will use)."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    settings = _configured_settings(rrf_rank_constant=60, num_candidates=100, top_k=5)
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(settings, conn)

    await client.hybrid_search("q", top_k=5, num_candidates=7, rrf_k=3)

    assert qb.last_reranker.K == 3
    assert qb.last_limit == 7


async def test_hybrid_search_truncates_non_mmr_pool_to_top_k(monkeypatch):
    """Non-MMR hybrid returns AT MOST top_k even though the pool
    (`num_candidates`) is larger. The fused pool is sliced `[:top_k]`."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    # 8 candidate rows come back; top_k=3 must cap the result at 3.
    rows = [{**_chunk_row(f"c{i}", f"t{i}"), "_relevance_score": 1.0 - i / 10} for i in range(8)]
    qb = _RecordingQueryBuilder(rows_to_return=rows)
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(num_candidates=50), conn)

    hits = await client.hybrid_search("q", top_k=3)

    assert len(hits) == 3
    assert [h.chunk_id for h in hits] == ["c0", "c1", "c2"]


async def test_vector_search_uses_num_candidates_pool_and_truncates(monkeypatch):
    """vector_search sets `.limit(num_candidates)` and truncates the
    non-MMR result to top_k (no RRF reranker on the vector-only path)."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)
    rows = [{**_chunk_row(f"c{i}", f"t{i}"), "_distance": i / 10} for i in range(6)]
    qb = _RecordingQueryBuilder(rows_to_return=rows)
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(num_candidates=30, top_k=2), conn)

    hits = await client.vector_search("q", top_k=2)

    assert qb.last_limit == 30
    assert qb.last_reranker is None  # vector-only path: no RRF fusion
    assert len(hits) == 2


async def test_retrieve_dispatches_to_mode_specific_method(monkeypatch):
    """`retrieve(mode=...)` is a pure dispatcher: each mode value
    routes to the matching `*_search` method. Patch the three and
    assert exactly one fires per call."""
    calls: list[str] = []

    async def fake_hybrid(self, *a, **kw):
        calls.append("hybrid")
        return []

    async def fake_fts(self, *a, **kw):
        calls.append("fts")
        return []

    async def fake_vector(self, *a, **kw):
        calls.append("vector")
        return []

    monkeypatch.setattr(LanceClient, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(LanceClient, "fts_search", fake_fts)
    monkeypatch.setattr(LanceClient, "vector_search", fake_vector)

    client = LanceClient(settings=_configured_settings())

    await client.retrieve("q", mode="hybrid")
    await client.retrieve("q", mode="fts")
    await client.retrieve("q", mode="vector")
    # Unknown mode falls back to hybrid (per the warn-then-default
    # branch in `retrieve()`).
    await client.retrieve("q", mode="bogus")  # type: ignore[arg-type]

    assert calls == ["hybrid", "fts", "vector", "hybrid"]


async def test_retrieve_forwards_tuning_knobs_per_mode(monkeypatch):
    """`retrieve()` forwards the per-invoke tuning knobs to the chosen
    method: num_candidates/rrf_k/mmr_lambda → hybrid; num_candidates/
    mmr_lambda (NO rrf_k) → vector. This is the node→client hop the eval
    case rides through."""
    seen: dict[str, dict[str, Any]] = {}

    async def fake_hybrid(self, query, top_k=None, **kw):
        seen["hybrid"] = kw
        return []

    async def fake_vector(self, query, top_k=None, **kw):
        seen["vector"] = kw
        return []

    monkeypatch.setattr(LanceClient, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(LanceClient, "vector_search", fake_vector)
    client = LanceClient(settings=_configured_settings())

    await client.retrieve("q", mode="hybrid", num_candidates=11, rrf_k=7, mmr_lambda=0.3)
    assert seen["hybrid"]["num_candidates"] == 11
    assert seen["hybrid"]["rrf_k"] == 7
    assert seen["hybrid"]["mmr_lambda"] == 0.3

    await client.retrieve("q", mode="vector", num_candidates=9, mmr_lambda=0.4)
    assert seen["vector"]["num_candidates"] == 9
    assert seen["vector"]["mmr_lambda"] == 0.4
    assert "rrf_k" not in seen["vector"]  # vector-only path has no RRF fusion


@pytest.mark.parametrize("method", ["hybrid_search", "vector_search"])
async def test_mmr_lambda_reaches_reranker_override_then_settings(monkeypatch, method):
    """With use_mmr=True, an explicit mmr_lambda wins; when omitted it
    falls back to settings.mmr_lambda (None-check, not a silent default).
    Covers both MMR-capable methods."""

    async def _fake_embed(_text: str) -> list[float]:
        return [0.5] * 1024

    monkeypatch.setattr("knowledge_agent.search.client._embed_query", _fake_embed)

    captured: dict[str, float] = {}

    def _fake_mmr(rows, query_vector, top_k, mmr_lambda, mode):
        captured["lambda"] = mmr_lambda
        return []

    monkeypatch.setattr("knowledge_agent.search.client._mmr_rerank_rows", _fake_mmr)

    settings = _configured_settings(mmr_lambda=0.6, num_candidates=50, top_k=5)
    qb = _RecordingQueryBuilder(rows_to_return=[{**_chunk_row("c1", "a"), "_relevance_score": 1.0}])
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable(query_builder=qb)})
    client = _client_with_conn(settings, conn)
    search = getattr(client, method)

    await search("q", top_k=5, use_mmr=True, mmr_lambda=0.2)
    assert captured["lambda"] == 0.2  # explicit override

    await search("q", top_k=5, use_mmr=True)
    assert captured["lambda"] == 0.6  # settings fallback


# ---- schema ----


def test_chunks_schema_has_required_fields():
    schema = chunks_schema()
    names = set(schema.names)
    expected = {
        "chunk_id",
        "doc_id",
        "chunk_index",
        "text",
        "embedding",
        "content_type",
        "main_label",
        "sub_label",
        "metadata_status",
        "ingested_at",
    }
    assert expected.issubset(names)


def test_chunks_schema_embedding_is_fixed_size_list_float32():
    schema = chunks_schema()
    embedding_field = schema.field("embedding")
    assert pa.types.is_fixed_size_list(embedding_field.type)
    assert embedding_field.type.value_type == pa.float32()
    # The list size matches settings.embedding_dims (default 1024).
    assert embedding_field.type.list_size == 1024


def test_chunks_schema_required_fields_non_nullable():
    schema = chunks_schema()
    required = [
        "chunk_id",
        "doc_id",
        "chunk_index",
        "text",
        "embedding",
        "content_type",
        "main_label",
        "metadata_status",
        "ingested_at",
    ]
    for name in required:
        assert schema.field(name).nullable is False, f"{name} must be non-nullable"


def test_chunks_schema_sub_label_is_nullable():
    """sub_label is optional - a file ingested with no chosen subtype
    still produces valid chunk rows (main_label set, sub_label None)."""
    schema = chunks_schema()
    assert schema.field("sub_label").nullable is True


def test_chunks_schema_optional_fields_nullable():
    schema = chunks_schema()
    optional = [
        "section",
        "page",
        "char_start",
        "char_end",
        "image_ref",
        "doi",
        "openalex_id",
        "title",
        "year",
        "authors_display",
        "venue",
        "source_url",
        "language",
        "source_path",
    ]
    for name in optional:
        assert schema.field(name).nullable is True, f"{name} must be nullable"


def test_content_types_constants():
    """Writers and readers both reference this tuple - check the expected
    set is present."""
    assert set(CONTENT_TYPES) >= {"text", "figure", "table"}


def test_metadata_status_values_constants():
    assert set(METADATA_STATUS_VALUES) >= {
        "enriched",
        "pending",
        "baseline",
        "manual",
    }


# ---- _filters_to_sql (mode 4 doc_id filter depends on this) ----


def test_filters_to_sql_single_string_value():
    assert _filters_to_sql({"doc_id": "W1"}) == "doc_id = 'W1'"


def test_filters_to_sql_list_becomes_in_clause():
    """The mode 4 (neo4j_then_lancedb) doc_id filter relies on this."""
    sql = _filters_to_sql({"doc_id": ["W1", "W2", "W3"]})
    assert sql == "doc_id IN ('W1', 'W2', 'W3')"


def test_filters_to_sql_empty_list_skips_field():
    """An empty doc_id list must skip the filter, not produce `doc_id IN ()`."""
    sql = _filters_to_sql({"doc_id": [], "year": 2024})
    assert sql == "year = 2024"


def test_filters_to_sql_none_value_skipped():
    sql = _filters_to_sql({"doc_id": None, "year": 2024})
    assert sql == "year = 2024"


def test_filters_to_sql_empty_dict_returns_empty_string():
    assert _filters_to_sql({}) == ""


def test_filters_to_sql_multi_field_joins_with_and():
    sql = _filters_to_sql({"doc_id": "W1", "year": 2024})
    assert "doc_id = 'W1'" in sql
    assert "year = 2024" in sql
    assert " AND " in sql


def test_filters_to_sql_skips_none_in_list():
    sql = _filters_to_sql({"doc_id": ["W1", None, "W2"]})
    assert sql == "doc_id IN ('W1', 'W2')"


# ---- _sql_literal ----


def test_sql_literal_string_escapes_single_quotes():
    """SQL-injection defence: a value containing a single quote is escaped."""
    assert _sql_literal("W1'evil") == "'W1''evil'"


def test_sql_literal_int_unquoted():
    assert _sql_literal(42) == "42"


def test_sql_literal_bool_lowercase():
    assert _sql_literal(True) == "true"
    assert _sql_literal(False) == "false"


async def test_fts_search_passes_content_type_and_image_ref_through():
    """LanceDB row's `content_type` and `image_ref` land on
    RetrievedChunk. Multimodal: retrieval side of the round trip so the
    synthesizer can enrich ChunkSource / GUI can render thumbnails."""
    fig_row = {
        **_chunk_row("c1", "caption"),
        "content_type": "figure",
        "image_ref": "/corpus/figures/doc1/0.png",
        "_score": 5.0,
    }
    txt_row = {
        **_chunk_row("c2", "body para"),
        "content_type": "text",
        "image_ref": None,
        "_score": 4.0,
    }
    qb = _RecordingQueryBuilder(rows_to_return=[fig_row, txt_row])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    hits = await client.fts_search("q", top_k=5)

    assert hits[0].content_type == "figure"
    assert hits[0].image_ref == "/corpus/figures/doc1/0.png"
    assert hits[1].content_type == "text"
    assert hits[1].image_ref is None
