"""Tests for kg.triples_writes - L8 typed-relation writes + delete +
get_entities_by_chunk + client delegates.

Reuses the RecordingDriver harness pattern from test_entity_writes.py.
The driver also supports returning canned `data()` results for
session.run().data() reads (used by get_entities_by_chunk).

We verify:
  - Fail-soft on empty doc_id, driver exceptions, all-empty input.
  - One Cypher per predicate (NOT one per row): rows grouped by
    predicate before write.
  - Unknown predicates are dropped (logged) before Cypher fires.
  - delete uses the pipe-joined predicate union (one DELETE query for
    all 15 types).
  - get_entities_by_chunk groups (key, entity_type) by chunk_id.
  - Client delegates pass through correctly.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import triples_writes
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.schema import TRIPLE_PREDICATE_RELS
from knowledge_agent.kg.triples_writes import ExtractedTriple


# ---- Test harness ----


@dataclass
class _RunResult:
    """Stand-in for the Neo4j Result.data() return shape."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def data(self) -> list[dict[str, Any]]:
        return self.rows


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # Queued return values for session.run().data() calls. Pop one per
    # call; queries that don't need data() ignore this.
    data_returns: list[list[dict[str, Any]]] = field(default_factory=list)

    def run(self, query: str, **params: Any) -> _RunResult:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        if self.data_returns:
            return _RunResult(rows=self.data_returns.pop(0))
        return _RunResult(rows=[])

    def __enter__(self) -> "RecordingSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    data_returns: list[list[dict[str, Any]]] = field(default_factory=list)
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(
            raise_on_run=self.raise_on_run,
            data_returns=list(self.data_returns),
        )
        self.sessions.append(sess)
        return sess

    def close(self) -> None:
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


DOC_ID = "test-doc-l8"


def _t(
    subject: str = "brca1",
    predicate: str = "INHIBITS",
    obj: str = "tp53",
    *,
    subject_type: str = "GENE",
    object_type: str = "GENE",
    evidence: str = "BRCA1 inhibits TP53 expression.",
) -> ExtractedTriple:
    """Convenience constructor for tests."""
    return ExtractedTriple(
        subject_key=subject,
        subject_entity_type=subject_type,
        predicate=predicate,
        object_key=obj,
        object_entity_type=object_type,
        evidence_span=evidence,
    )


# ---- ExtractedTriple shape ----


def test_extracted_triple_carries_provenance_and_vocab():
    t = _t()
    assert t.subject_key == "brca1"
    assert t.predicate == "INHIBITS"
    assert t.object_key == "tp53"
    assert "BRCA1 inhibits" in t.evidence_span


def test_extracted_triple_is_frozen_dataclass():
    t = _t()
    try:
        t.subject_key = "changed"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ExtractedTriple should be frozen")


# ---- delete_triples_by_doc_id ----


def test_delete_triples_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        triples_writes.delete_triples_by_doc_id(client, "")
    assert driver.sessions == []


def test_delete_triples_runs_single_cypher_for_all_predicates():
    """Pipe-joined predicate union -> one DELETE query, not 15."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert triples_writes.delete_triples_by_doc_id(client, DOC_ID) is None
    assert len(driver.sessions[0].calls) == 1
    cypher, params = driver.sessions[0].calls[0]
    # All 15 predicates appear in the rel union.
    for pred in TRIPLE_PREDICATE_RELS:
        assert pred in cypher
    # Pipe syntax used (not 15 separate edge labels).
    assert "|" in cypher
    assert "DELETE r" in cypher
    assert params == {"doc_id": DOC_ID}


def test_delete_triples_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        triples_writes.delete_triples_by_doc_id(client, DOC_ID)


# ---- write_triples: fail-soft + empty paths ----


def test_write_triples_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        triples_writes.write_triples(client, "", [])
    assert driver.sessions == []


def test_write_triples_empty_chunk_triples_is_noop():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert triples_writes.write_triples(client, DOC_ID, []) is None
    assert driver.sessions == []


def test_write_triples_all_chunks_empty_is_noop():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = triples_writes.write_triples(
        client,
        DOC_ID,
        [("c0", []), ("c1", []), ("c2", [])],
    )
    assert result is None
    assert driver.sessions == []


def test_write_triples_only_invalid_predicates_is_noop():
    """Every triple has a bogus predicate -> all dropped, no Cypher."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    bad = _t(predicate="BOGUS_VERB")
    result = triples_writes.write_triples(
        client, DOC_ID, [("c0", [bad])]
    )
    assert result is None
    assert driver.sessions == []


def test_write_triples_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        triples_writes.write_triples(client, DOC_ID, [("c0", [_t()])])


# ---- write_triples: Cypher shape + batching ----


def _run_write(
    chunk_triples: list[tuple[str, list[ExtractedTriple]]],
) -> RecordingDriver:
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert triples_writes.write_triples(client, DOC_ID, chunk_triples) is None
    return driver


def test_write_triples_one_predicate_one_cypher():
    """All rows share a predicate -> single UNWIND."""
    driver = _run_write(
        [
            ("c0", [_t(), _t(subject="palb2")]),
            ("c1", [_t(subject="atm")]),
        ]
    )
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1


def test_write_triples_multiple_predicates_get_separate_cypher_calls():
    """One Cypher per predicate type (Neo4j can't param a rel type)."""
    driver = _run_write(
        [
            (
                "c0",
                [
                    _t(predicate="INHIBITS"),
                    _t(predicate="ACTIVATES", obj="other"),
                    _t(predicate="BINDS_TO", obj="another"),
                ],
            )
        ]
    )
    assert len(driver.sessions[0].calls) == 3
    # Each call hardcodes its predicate as the rel label.
    cyphers = [c for c, _ in driver.sessions[0].calls]
    assert any(":INHIBITS" in c for c in cyphers)
    assert any(":ACTIVATES" in c for c in cyphers)
    assert any(":BINDS_TO" in c for c in cyphers)


def test_write_triples_cypher_uses_create_not_merge():
    """One edge per chunk assertion - CREATE matches the policy.
    MERGE would dedupe identical triples and lose the per-chunk
    provenance count."""
    driver = _run_write([("c0", [_t()])])
    cypher, _ = driver.sessions[0].calls[0]
    assert "CREATE" in cypher
    assert "MERGE (s)" not in cypher  # not merging on the edge


def test_write_triples_edge_carries_chunk_id_doc_id_evidence_span():
    driver = _run_write([("c0", [_t()])])
    cypher, params = driver.sessions[0].calls[0]
    assert "chunk_id: row.chunk_id" in cypher
    assert "doc_id: row.doc_id" in cypher
    assert "evidence_span: row.evidence_span" in cypher
    row = params["rows"][0]
    assert row["chunk_id"] == "c0"
    assert row["doc_id"] == DOC_ID
    assert "BRCA1 inhibits" in row["evidence_span"]


def test_write_triples_matches_entities_by_composite_key():
    """Subject + object MATCH on (key, entity_type) composite NODE KEY."""
    driver = _run_write([("c0", [_t()])])
    cypher, params = driver.sessions[0].calls[0]
    assert "key: row.subject_key" in cypher
    assert "entity_type: row.subject_entity_type" in cypher
    assert "key: row.object_key" in cypher
    assert "entity_type: row.object_entity_type" in cypher
    row = params["rows"][0]
    assert row["subject_key"] == "brca1"
    assert row["subject_entity_type"] == "GENE"
    assert row["object_key"] == "tp53"
    assert row["object_entity_type"] == "GENE"


def test_write_triples_drops_unknown_predicate_but_keeps_valid_ones():
    """Unknown predicates filtered out per-row; valid triples still write."""
    valid = _t(predicate="INHIBITS")
    invalid = _t(predicate="BOGUS_VERB", obj="other")
    driver = _run_write([("c0", [valid, invalid])])
    # Only one Cypher fired (for the valid predicate), with one row.
    assert len(driver.sessions[0].calls) == 1
    cypher, params = driver.sessions[0].calls[0]
    assert ":INHIBITS" in cypher
    assert "BOGUS_VERB" not in cypher
    assert len(params["rows"]) == 1


# ---- get_entities_by_chunk ----


def test_get_entities_by_chunk_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        triples_writes.get_entities_by_chunk(client, "")
    assert driver.sessions == []


def test_get_entities_by_chunk_groups_rows_by_chunk_id():
    """3 rows for 2 chunks -> 2-key dict with the right vocab per chunk."""
    driver = RecordingDriver(
        data_returns=[
            [
                {"chunk_id": "c0", "key": "brca1", "entity_type": "GENE"},
                {"chunk_id": "c0", "key": "tp53", "entity_type": "GENE"},
                {"chunk_id": "c1", "key": "aspirin", "entity_type": "CHEMICAL"},
            ]
        ]
    )
    client = _client_with_driver(driver)
    result = triples_writes.get_entities_by_chunk(client, DOC_ID)
    assert set(result) == {"c0", "c1"}
    assert ("brca1", "GENE") in result["c0"]
    assert ("tp53", "GENE") in result["c0"]
    assert result["c1"] == [("aspirin", "CHEMICAL")]


def test_get_entities_by_chunk_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        triples_writes.get_entities_by_chunk(client, DOC_ID)


# ---- Client delegate methods ----


def test_client_write_triples_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = client.write_triples(DOC_ID, [("c0", [_t()])])
    assert result is None
    assert len(driver.sessions[0].calls) == 1


def test_client_delete_triples_by_doc_id_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    result = client.delete_triples_by_doc_id(DOC_ID)
    assert result is None
    assert len(driver.sessions[0].calls) == 1


def test_client_get_entities_by_chunk_delegates_to_module():
    driver = RecordingDriver(
        data_returns=[
            [{"chunk_id": "c0", "key": "brca1", "entity_type": "GENE"}]
        ]
    )
    client = _client_with_driver(driver)
    result = client.get_entities_by_chunk(DOC_ID)
    assert result == {"c0": [("brca1", "GENE")]}
