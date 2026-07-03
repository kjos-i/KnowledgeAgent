"""Tests for kg.entity_writes - write_entities + delete_entities_by_doc_id.

Uses the same RecordingDriver harness as `test_kg_client.py` so the
Cypher strings + parameters sent to session.run() are captured without
hitting a real Neo4j instance.

We verify:
  - Typed-errors contract: bad inputs raise ValueError; driver
    exceptions propagate (caught at the orchestrator boundary). Success
    returns None.
  - Cypher includes the right labels, relationships, and MERGE shape.
  - Lowercase-merge-key contract: raw_text is lowercased to compute
    the :Entity node's `key`.
  - delete pipeline: drops MENTIONS edges THEN GCs orphan entities.
  - Empty-input handling: returns None without sending any Cypher.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.kg import entity_writes
from knowledge_agent.kg.client import Neo4jClient

# ---- Test harness (mirrors test_kg_client.py) ----


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None

    async def run(self, query: str, **params: Any) -> None:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))

    async def __aenter__(self) -> "RecordingSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(raise_on_run=self.raise_on_run)
        self.sessions.append(sess)
        return sess

    async def close(self) -> None:
        self.closed = True


def _configured_settings() -> Settings:
    return Settings(
        anthropic_api_key="fake-anthropic-key",
        voyage_api_key="fake-voyage-key",
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="fake-neo4j-password",
        lancedb_path="/tmp/fake-lancedb-path",
    )


def _client_with_driver(driver: RecordingDriver) -> Neo4jClient:
    client = Neo4jClient(settings=_configured_settings())
    client._driver = driver  # type: ignore[assignment]
    return client


DOC_ID = "test-doc-l6"


# ---- write_entities: input validation + exception propagation ----


async def test_write_entities_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await entity_writes.write_entities(client, "", [])
    # No session opened either - we bailed before touching the driver.
    assert driver.sessions == []


async def test_write_entities_empty_chunk_mentions_is_noop():
    """No chunks to process -> no Cypher sent; success no-op (None)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await entity_writes.write_entities(client, DOC_ID, []) is None
    assert driver.sessions == []


async def test_write_entities_all_chunks_empty_is_noop():
    """Each chunk yielded zero mentions -> no Cypher sent; success no-op."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = await entity_writes.write_entities(
        client,
        DOC_ID,
        [("c0", []), ("c1", []), ("c2", [])],
    )
    assert result is None
    assert driver.sessions == []


async def test_write_entities_propagates_cypher_exception():
    """await session.run() raises -> propagate; orchestrator boundary catches
    and stores on `IngestResult.kg_entities_error`."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await entity_writes.write_entities(
            client,
            DOC_ID,
            [("c0", [Mention(raw_text="BRCA1", entity_type="GENE")])],
        )


# ---- write_entities: Cypher shape ----


async def _run_write(mentions_per_chunk: list[tuple[str, list[Mention]]]) -> RecordingDriver:
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await entity_writes.write_entities(client, DOC_ID, mentions_per_chunk) is None
    return driver


async def test_write_entities_sends_one_session_one_call():
    """All mentions batched into a single UNWIND - one round trip."""
    driver = await _run_write(
        [
            ("c0", [Mention("BRCA1", "GENE"), Mention("p53", "GENE")]),
            ("c1", [Mention("aspirin", "CHEMICAL")]),
        ]
    )
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1


async def test_write_entities_cypher_includes_entity_label_and_mentions_rel():
    driver = await _run_write(
        [("c0", [Mention(raw_text="BRCA1", entity_type="GENE")])]
    )
    cypher, _params = driver.sessions[0].calls[0]
    # The composite-key MERGE on :Entity.
    assert ":Entity" in cypher
    assert ":MENTIONS" in cypher
    assert "row.key" in cypher
    assert "row.entity_type" in cypher
    # canonicalised flag set on creation.
    assert "canonicalised = false" in cypher


async def test_write_entities_lowercases_raw_text_into_key():
    """raw_text 'BRCA1' (capital) becomes key 'brca1' in the merge param."""
    driver = await _run_write(
        [("c0", [Mention(raw_text="BRCA1", entity_type="GENE")])]
    )
    _cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert rows[0]["key"] == "brca1"
    assert rows[0]["entity_type"] == "GENE"


async def test_write_entities_collapses_case_variants_to_same_key():
    """'BRCA1', 'Brca1', 'brca1' all reduce to key 'brca1'. Different
    entity_types still produce different rows (composite key)."""
    driver = await _run_write(
        [
            ("c0", [Mention("BRCA1", "GENE"), Mention("Brca1", "GENE")]),
            ("c1", [Mention("brca1", "GENE")]),
        ]
    )
    _cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert [r["key"] for r in rows] == ["brca1", "brca1", "brca1"]


async def test_write_entities_preserves_offset_and_confidence_in_rows():
    """NER mentions carry offset + confidence onto the :MENTIONS edge
    via ON CREATE SET. Verify the row payload includes them verbatim."""
    driver = await _run_write(
        [
            (
                "c0",
                [
                    Mention(
                        raw_text="BRCA1",
                        entity_type="GENE",
                        offset=42,
                        confidence=0.95,
                    )
                ],
            )
        ]
    )
    _cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert rows[0]["offset"] == 42
    assert rows[0]["confidence"] == 0.95


async def test_write_entities_llm_mentions_carry_none_offset_and_confidence():
    """LLM mentions have offset=None, confidence=None - rows carry None
    through so the Cypher sets them to null on the edge."""
    driver = await _run_write(
        [("c0", [Mention(raw_text="brca1", entity_type="GENE")])]
    )
    _cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert rows[0]["offset"] is None
    assert rows[0]["confidence"] is None


async def test_write_entities_carries_union_sources_onto_edge():
    """A mention stamped with `sources` (which extractor(s) found it in
    the priority-ordered union) writes those onto the :MENTIONS edge,
    and the Cypher sets `m.sources`."""
    driver = await _run_write(
        [
            (
                "c0",
                [
                    Mention(
                        raw_text="Aspirin",
                        entity_type="CHEMICAL",
                        sources=("hunflair2", "llm"),
                    )
                ],
            )
        ]
    )
    cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert rows[0]["sources"] == ["hunflair2", "llm"]
    assert "m.sources = row.sources" in cypher


async def test_write_entities_sources_defaults_empty_when_unset():
    """A plain mention with no `sources` (e.g. a hand-built one) writes an
    empty list, not null - keeps the edge property shape consistent."""
    driver = await _run_write(
        [("c0", [Mention(raw_text="brca1", entity_type="GENE")])]
    )
    _cypher, params = driver.sessions[0].calls[0]
    rows = params["rows"]
    assert rows[0]["sources"] == []


async def test_write_entities_uses_chunk_id_to_link_mentions_edge():
    """The chunk_id in each row binds the :MENTIONS edge to the right
    chunk via MATCH (c:Chunk {chunk_id: row.chunk_id})."""
    driver = await _run_write(
        [
            ("alpha", [Mention("BRCA1", "GENE")]),
            ("beta", [Mention("TP53", "GENE")]),
        ]
    )
    cypher, params = driver.sessions[0].calls[0]
    assert "Chunk" in cypher
    assert "row.chunk_id" in cypher
    rows = params["rows"]
    assert {r["chunk_id"] for r in rows} == {"alpha", "beta"}


# ---- delete_entities_by_doc_id ----


async def test_delete_entities_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await entity_writes.delete_entities_by_doc_id(client, "")
    assert driver.sessions == []


async def test_delete_entities_runs_two_queries():
    """First: delete MENTIONS edges from this doc's chunks.
    Second: GC orphan :Entity nodes globally."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await entity_writes.delete_entities_by_doc_id(client, DOC_ID) is None
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 2


async def test_delete_entities_first_query_drops_mentions_for_this_doc():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await entity_writes.delete_entities_by_doc_id(client, DOC_ID)
    cypher, params = driver.sessions[0].calls[0]
    assert ":MENTIONS" in cypher
    assert "DELETE m" in cypher
    assert "doc_id" in cypher
    assert params == {"doc_id": DOC_ID}


async def test_delete_entities_second_query_gcs_orphans():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await entity_writes.delete_entities_by_doc_id(client, DOC_ID)
    cypher, params = driver.sessions[0].calls[1]
    assert ":Entity" in cypher
    assert "NOT ()" in cypher  # WHERE-NOT orphan pattern
    assert "MENTIONS" in cypher
    assert "DELETE e" in cypher
    # GC is global - no doc_id parameter on this query.
    assert params == {}


async def test_delete_entities_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await entity_writes.delete_entities_by_doc_id(client, DOC_ID)


# ---- Neo4jClient delegate methods ----


async def test_client_write_entities_delegates_to_module():
    """Client.write_entities is a 1-line wrapper. Verify it calls through."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = await client.write_entities(
        DOC_ID, [("c0", [Mention("BRCA1", "GENE")])]
    )
    assert result is None
    # The underlying call landed: one Cypher round trip with our payload.
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1


async def test_client_delete_entities_by_doc_id_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = await client.delete_entities_by_doc_id(DOC_ID)
    assert result is None
    # Underlying delete runs its two queries.
    assert len(driver.sessions[0].calls) == 2
