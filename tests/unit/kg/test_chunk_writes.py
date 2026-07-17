"""Tests for kg.chunk_writes (L5) - write_chunks, delete_chunks_by_doc_id,
and get_focal_labels_by_doc_id.

Uses the same RecordingDriver harness as `test_entity_writes.py` /
`test_kg_client.py` so the Cypher strings + parameters sent to
session.run() are captured without hitting a real Neo4j instance. The
harness here additionally lets `session.run()` return a result stub with
an async `.single()`, since `get_focal_labels_by_doc_id` reads a row back
(the write/delete primitives ignore the return value).

We verify:
  - Typed-errors contract: bad inputs raise ValueError before any
    session opens; driver exceptions propagate (caught at the
    orchestrator boundary). Success returns None.
  - write_chunks: focal MERGE keeps labels stable (ON CREATE only),
    sub-label co-application, UNWIND batching into a single round trip,
    chunk_id shape via make_chunk_id, full property payload, and the
    back-compat `image_ref` getattr default.
  - delete_chunks_by_doc_id: single DETACH DELETE scoped by doc_id.
  - get_focal_labels_by_doc_id: (main, sub) label splitting, the
    (None, None) missing-doc signal, and unknown-label tolerance.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.ingestion.ids import make_chunk_id
from knowledge_agent.kg import chunk_writes
from knowledge_agent.kg.client import Neo4jClient

# ---- Test harness (mirrors test_entity_writes.py, plus a result stub) ----


@dataclass
class _RecordingResult:
    """Stand-in for `neo4j.AsyncResult`. `.single()` returns the row the
    driver was configured with (None = no matching row)."""

    single_row: Any = None

    async def single(self) -> Any:
        return self.single_row


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # What `session.run(...)` -> `result.single()` yields. Only
    # get_focal_labels_by_doc_id reads it; writes/deletes ignore it.
    single_row: Any = None

    async def run(self, query: str, **params: Any) -> _RecordingResult:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        return _RecordingResult(single_row=self.single_row)

    async def __aenter__(self) -> "RecordingSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    single_row: Any = None
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(
            raise_on_run=self.raise_on_run,
            single_row=self.single_row,
        )
        self.sessions.append(sess)
        return sess

    async def close(self) -> None:
        self.closed = True


def _configured_settings() -> Settings:
    return Settings(
        anthropic_api_key="fake-anthropic-key",  # pragma: allowlist secret
        voyage_api_key="fake-voyage-key",  # pragma: allowlist secret
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="fake-neo4j-password",  # pragma: allowlist secret
        lancedb_path="/tmp/fake-lancedb-path",
    )


def _client_with_driver(driver: RecordingDriver) -> Neo4jClient:
    client = Neo4jClient(settings=_configured_settings())
    client._driver = driver  # type: ignore[assignment]
    return client


DOC_ID = "test-doc-l5"


@dataclass
class FakeChunk:
    """Duck-typed stand-in for `ingestion.parse.ParsedChunk`.

    Deliberately has NO `image_ref` attribute so it exercises the
    back-compat `getattr(c, "image_ref", None)` path in write_chunks.
    """

    chunk_index: int
    section: str | None
    page: int | None
    content_type: str = "text"


@dataclass
class FakeFigureChunk:
    """ParsedChunk variant that DOES carry `image_ref` (figure chunks)."""

    chunk_index: int
    section: str | None
    page: int | None
    content_type: str = "figure"
    image_ref: str | None = None


# ---- write_chunks: input validation + empty-input handling ----


async def test_write_chunks_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await chunk_writes.write_chunks(client, "", [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    # Bailed before touching the driver.
    assert driver.sessions == []


async def test_write_chunks_unknown_main_label_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="invalid main_label"):
        await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "NotALabel")
    assert driver.sessions == []


async def test_write_chunks_sub_label_not_under_main_raises():
    """`Dataset` is an :Artifact sub-label; pairing it with :Document is a
    programmer error caught before any write."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="doesn't belong under"):
        await chunk_writes.write_chunks(
            client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Dataset"
        )
    assert driver.sessions == []


async def test_write_chunks_unknown_sub_label_raises():
    """A sub-label not in the schema at all also fails the belongs-under
    check (SUB_LABEL_TO_MAIN.get(...) is None != main_label)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="doesn't belong under"):
        await chunk_writes.write_chunks(
            client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Bogus"
        )
    assert driver.sessions == []


async def test_write_chunks_validation_precedes_empty_chunks_noop():
    """Bad label raises even when chunks is empty - validation runs before
    the empty-list no-op short-circuit."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="invalid main_label"):
        await chunk_writes.write_chunks(client, DOC_ID, [], "NotALabel")
    assert driver.sessions == []


async def test_write_chunks_empty_chunks_is_noop_returning_none():
    """Empty chunks list is a documented success no-op: returns None, no
    Cypher dispatched (focal is written via a separate path)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await chunk_writes.write_chunks(client, DOC_ID, [], "Document", "Paper") is None
    assert driver.sessions == []


# ---- write_chunks: Cypher shape + payload ----


async def test_write_chunks_sends_one_session_one_call():
    """UNWIND batches every chunk into a single round trip."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    chunks = [FakeChunk(0, "Intro", 1), FakeChunk(1, "Methods", 2)]
    assert await chunk_writes.write_chunks(client, DOC_ID, chunks, "Document", "Paper") is None
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1


async def test_write_chunks_batches_all_chunks_in_one_payload():
    """Many chunks -> still one call, all present in the `chunks` param."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    chunks = [FakeChunk(i, f"S{i}", i) for i in range(5)]
    await chunk_writes.write_chunks(client, DOC_ID, chunks, "Document", "Paper")
    _query, params = driver.sessions[0].calls[0]
    assert len(params["chunks"]) == 5
    assert [c["chunk_index"] for c in params["chunks"]] == [0, 1, 2, 3, 4]


async def test_write_chunks_chunk_ids_use_make_chunk_id_format():
    """chunk_id = '{doc_id}#{chunk_index}' via ingestion.ids.make_chunk_id.
    Non-contiguous indices are preserved verbatim."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    chunks = [FakeChunk(0, "Intro", 1), FakeChunk(2, "Methods", 3)]
    await chunk_writes.write_chunks(client, DOC_ID, chunks, "Document", "Paper")
    _query, params = driver.sessions[0].calls[0]
    assert params["chunks"][0]["chunk_id"] == make_chunk_id(DOC_ID, 0)
    assert params["chunks"][0]["chunk_id"] == f"{DOC_ID}#0"
    assert params["chunks"][1]["chunk_id"] == f"{DOC_ID}#2"


async def test_write_chunks_passes_all_chunk_properties():
    """Identity + structural props flow onto each row; None section/page
    (e.g. a figure chunk) pass through as None."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    chunks = [
        FakeChunk(0, "Intro", 1, content_type="text"),
        FakeChunk(1, None, None, content_type="figure"),
    ]
    await chunk_writes.write_chunks(client, DOC_ID, chunks, "Document", "Paper")
    _query, params = driver.sessions[0].calls[0]
    assert params["doc_id"] == DOC_ID
    assert params["chunks"][0]["chunk_index"] == 0
    assert params["chunks"][0]["section"] == "Intro"
    assert params["chunks"][0]["page"] == 1
    assert params["chunks"][0]["content_type"] == "text"
    assert params["chunks"][1]["section"] is None
    assert params["chunks"][1]["page"] is None
    assert params["chunks"][1]["content_type"] == "figure"


async def test_write_chunks_missing_image_ref_defaults_to_none():
    """A chunk object without an `image_ref` attribute (older parser
    producer) yields image_ref=None via getattr back-compat."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    _query, params = driver.sessions[0].calls[0]
    assert params["chunks"][0]["image_ref"] is None


async def test_write_chunks_present_image_ref_is_carried_onto_row():
    """A figure chunk that DOES carry `image_ref` writes the path string,
    and the SET clause binds it onto the node."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    fig = FakeFigureChunk(0, None, 3, content_type="figure", image_ref="figures/doc/0.png")
    await chunk_writes.write_chunks(client, DOC_ID, [fig], "Document", "Paper")
    query, params = driver.sessions[0].calls[0]
    assert params["chunks"][0]["image_ref"] == "figures/doc/0.png"
    assert "c.image_ref = ch.image_ref" in query


async def test_write_chunks_creates_focal_via_merge_with_in_corpus_on_create():
    """Focal :Document is MERGEd; in_corpus=true is SET on create only -
    so a doc ingested without OpenAlex still has a focal node, and a
    pre-existing focal keeps whatever flag it had."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    query, _params = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in query
    assert "ON CREATE SET d.in_corpus = true" in query


async def test_write_chunks_sub_label_co_applied_on_create_only():
    """A provided sub-label is added inside ON CREATE SET (so an existing
    focal left by write_citations keeps its labels)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    query, _params = driver.sessions[0].calls[0]
    assert "ON CREATE SET d.in_corpus = true, d:Paper" in query


async def test_write_chunks_no_sub_label_omits_sub_label_clause():
    """sub_label=None -> only in_corpus is set on create; no extra label
    clause is injected between it and the WITH."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Artifact", None)
    query, _params = driver.sessions[0].calls[0]
    assert "ON CREATE SET d.in_corpus = true WITH d" in query
    assert "d:Paper" not in query


async def test_write_chunks_uses_artifact_main_label():
    """main_label='Artifact' keys the focal MERGE on :Artifact."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(
        client, DOC_ID, [FakeChunk(0, "Data", 1)], "Artifact", "Dataset"
    )
    query, _params = driver.sessions[0].calls[0]
    assert "MERGE (d:Artifact {doc_id: $doc_id})" in query
    assert "ON CREATE SET d.in_corpus = true, d:Dataset" in query


async def test_write_chunks_unwind_merges_chunks_and_part_of_edges():
    """One UNWIND MERGEs each :Chunk by chunk_id and its :PART_OF edge to
    the focal - MERGE (not CREATE) is what makes the write idempotent."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    query, _params = driver.sessions[0].calls[0]
    assert "UNWIND $chunks AS ch" in query
    assert "MERGE (c:Chunk {chunk_id: ch.chunk_id})" in query
    assert "MERGE (c)-[:PART_OF]->(d)" in query
    assert "SET c.doc_id = $doc_id" in query


async def test_write_chunks_propagates_cypher_exception():
    """await session.run() raises -> propagate; the orchestrator boundary
    catches and stores on `IngestResult.kg_chunks_error`."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await chunk_writes.write_chunks(
            client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper"
        )


# ---- delete_chunks_by_doc_id ----


async def test_delete_chunks_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await chunk_writes.delete_chunks_by_doc_id(client, "")
    assert driver.sessions == []


async def test_delete_chunks_one_round_trip_returning_none():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await chunk_writes.delete_chunks_by_doc_id(client, DOC_ID) is None
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1


async def test_delete_chunks_uses_chunk_label_with_doc_id_filter():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.delete_chunks_by_doc_id(client, DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (c:Chunk {doc_id: $doc_id})" in query
    assert "DETACH DELETE c" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_chunks_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await chunk_writes.delete_chunks_by_doc_id(client, DOC_ID)


async def test_delete_then_rewrite_dispatches_both_on_same_client():
    """The pipeline's re-ingest pattern: delete_chunks then write_chunks on
    one client. Each opens its own session; the mock records both. (True
    node-level idempotency is a Neo4j MERGE property, covered by the
    integration test.)"""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await chunk_writes.delete_chunks_by_doc_id(client, DOC_ID)
    await chunk_writes.write_chunks(client, DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    assert len(driver.sessions) == 2
    delete_query, _ = driver.sessions[0].calls[0]
    write_query, _ = driver.sessions[1].calls[0]
    assert "DETACH DELETE c" in delete_query
    assert "MERGE (c:Chunk {chunk_id: ch.chunk_id})" in write_query


# ---- get_focal_labels_by_doc_id ----


async def test_get_focal_labels_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await chunk_writes.get_focal_labels_by_doc_id(client, "")
    assert driver.sessions == []


async def test_get_focal_labels_returns_none_none_when_no_row():
    """No focal for this doc_id -> (None, None): the pipeline's
    'first ingest' signal."""
    driver = RecordingDriver(single_row=None)
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == (None, None)


async def test_get_focal_labels_returns_main_and_sub():
    driver = RecordingDriver(single_row={"labels": ["Document", "Paper"]})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == ("Document", "Paper")


async def test_get_focal_labels_returns_main_only_when_no_sub():
    driver = RecordingDriver(single_row={"labels": ["Artifact"]})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == ("Artifact", None)


async def test_get_focal_labels_ignores_unknown_extra_labels():
    """Any label matching neither the main nor sub group is dropped, so
    future auxiliary labels don't break the split."""
    driver = RecordingDriver(single_row={"labels": ["Document", "Paper", "FutureAuxLabel"]})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == ("Document", "Paper")


async def test_get_focal_labels_sub_without_main_returns_none_sub():
    """The main/sub lookups are independent next() scans: a node carrying
    only a sub-label yields (None, sub)."""
    driver = RecordingDriver(single_row={"labels": ["Paper"]})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == (None, "Paper")


async def test_get_focal_labels_empty_label_list_returns_none_none():
    driver = RecordingDriver(single_row={"labels": []})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == (None, None)


async def test_get_focal_labels_null_labels_value_returns_none_none():
    """`labels(d)` coming back null (`row["labels"] or []`) is tolerated."""
    driver = RecordingDriver(single_row={"labels": None})
    client = _client_with_driver(driver)
    assert await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID) == (None, None)


async def test_get_focal_labels_query_shape_and_params():
    driver = RecordingDriver(single_row={"labels": ["Document"]})
    client = _client_with_driver(driver)
    await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d) WHERE d.doc_id = $doc_id" in query
    assert "d:Document OR d:Artifact" in query
    assert "RETURN labels(d) AS labels" in query
    assert params == {"doc_id": DOC_ID}


async def test_get_focal_labels_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await chunk_writes.get_focal_labels_by_doc_id(client, DOC_ID)


# ---- Neo4jClient delegate (get_focal_labels only) ----
#
# write_chunks / delete_chunks_by_doc_id delegates are already covered in
# test_kg_client.py; get_focal_labels_by_doc_id had no unit-level delegate
# test, so it's filled here.


async def test_client_get_focal_labels_delegates_to_module():
    driver = RecordingDriver(single_row={"labels": ["Document", "Paper"]})
    client = _client_with_driver(driver)
    assert await client.get_focal_labels_by_doc_id(DOC_ID) == ("Document", "Paper")
    assert len(driver.sessions[0].calls) == 1
