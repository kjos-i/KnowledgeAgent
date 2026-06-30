"""Tests for search.client - LanceClient - and search.schema.

Uses a recording Connection/Table stub so tests don't touch a real LanceDB
dataset. The schema function is exercised by reading the returned `pa.Schema`.
"""

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pytest

from knowledge_agent.config import Settings
from knowledge_agent.search.client import (
    LanceClient,
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
    limit / vector / text on the fluent chain, then materialises to a
    stub Arrow table on `to_arrow()`."""

    rows_to_return: list[dict[str, Any]] = field(default_factory=list)
    last_filter: str | None = None
    last_columns: list[str] | None = None
    last_limit: int | None = None
    last_vector: list[float] | None = None
    last_text: str | None = None
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
    # update() calls captured as (where, values) tuples.
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
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

    async def update(self, where: str, values: dict[str, Any]) -> None:
        if self.raise_on_update is not None:
            raise self.raise_on_update
        self.updates.append((where, values))

    def query(self) -> _RecordingQueryBuilder:
        if self.query_builder is None:
            self.query_builder = _RecordingQueryBuilder()
        return self.query_builder

    async def search(
        self, *args: Any, **kwargs: Any
    ) -> _RecordingQueryBuilder:
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

    async def create_table(
        self, name: str, schema: Any = None, **_kwargs: Any
    ) -> RecordingTable:
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


def _client_with_conn(
    settings: Settings, conn: RecordingConnection
) -> LanceClient:
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
            {"doc_id": "d1", "source_path": "/x/1.pdf", "title": "T1", "metadata_status": "enriched"},
            {"doc_id": "d1", "source_path": "/x/1.pdf", "title": "T1", "metadata_status": "enriched"},
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
    """Heavy embedding + text columns are NOT loaded - only the 4 we need
    are pushed down via the search builder's `select()` call."""
    qb = _RecordingQueryBuilder(rows_to_return=[])
    table = RecordingTable(query_builder=qb)
    conn = RecordingConnection(tables={CHUNKS_TABLE: table})
    client = _client_with_conn(_configured_settings(), conn)

    await client.list_indexed_docs()
    assert qb.last_columns == [
        "doc_id", "source_path", "title", "metadata_status",
    ]
    # Limit must be passed too (search builder requires one) - check the
    # value is large enough to cover any realistic corpus.
    assert qb.last_limit is not None and qb.last_limit >= 1_000_000


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
        "chunk_id": chunk_id, "doc_id": "doc1", "text": text,
        "title": "T", "year": 2026, "authors_display": "A",
        "doi": None, "openalex_id": None, "venue": None,
        "source_url": None, "section": None, "page": None,
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
    monkeypatch.setattr(
        "knowledge_agent.search.client._embed_query", _fake_embed
    )
    conn = RecordingConnection(tables={CHUNKS_TABLE: RecordingTable()})
    client = _client_with_conn(_configured_settings(), conn)
    assert await client.vector_search("query", top_k=5) == []


async def test_vector_search_propagates_lance_exception(monkeypatch):
    """Lance failure during vector search -> raises (typed-errors
    contract). Voyage succeeds first; the failure surfaces from the
    lance query builder."""
    async def _fake_embed(_text: str) -> list[float]:
        return [0.1] * 1024
    monkeypatch.setattr(
        "knowledge_agent.search.client._embed_query", _fake_embed
    )
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
    monkeypatch.setattr(
        "knowledge_agent.search.client._embed_query", _fake_embed
    )
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
        assert schema.field(name).nullable is False, (
            f"{name} must be non-nullable"
        )


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
        assert schema.field(name).nullable is True, (
            f"{name} must be nullable"
        )


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
