"""Tests for kg.cross_doc_writes - L9 :RELATED_TO recompute + delete +
client delegate.

The recompute uses two session.run() calls (DELETE then MATCH/MERGE
with RETURN count). The harness reuses the RecordingDriver pattern
from test_triples_writes.py + adds support for `.single()` returns
so we can simulate the count(r) read.

We verify:
  - Fail-soft on empty doc_id, bad threshold, driver exceptions, and
    missing RETURN row.
  - Two-step Cypher: DELETE first, then MATCH/MERGE.
  - Cypher carries the :Document|:Artifact label disjunction so
    artifacts participate on equal footing with documents.
  - WHERE filters self-edges; threshold parameter binds to query.
  - MERGE on undirected edge, SET stamps shared_entities + count +
    timestamp.
  - Client delegate passes through correctly.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import cross_doc_writes
from knowledge_agent.kg.client import Neo4jClient


# ---- Test harness ----


@dataclass
class _RunResult:
    """Stand-in for the Neo4j Result, supporting both .data() and .single()."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    single_value: dict[str, Any] | None = None

    def data(self) -> list[dict[str, Any]]:
        return self.rows

    def single(self) -> dict[str, Any] | None:
        if self.single_value is not None:
            return self.single_value
        if self.rows:
            return self.rows[0]
        return None


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # Pre-queued .single() returns, in call order. None elements are
    # treated as "no row available" (.single() returns None).
    single_returns: list[dict[str, Any] | None] = field(default_factory=list)

    def run(self, query: str, **params: Any) -> _RunResult:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        if self.single_returns:
            sv = self.single_returns.pop(0)
            return _RunResult(single_value=sv)
        return _RunResult()

    def __enter__(self) -> "RecordingSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    single_returns: list[dict[str, Any] | None] = field(default_factory=list)
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(
            raise_on_run=self.raise_on_run,
            single_returns=list(self.single_returns),
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


DOC_ID = "test-doc-l9"


def _driver_with_count(n: int) -> RecordingDriver:
    """RecordingDriver pre-queued with one DELETE result then one
    {n: <count>} single() result for the recompute pass."""
    # DELETE returns nothing; recompute returns count.
    return RecordingDriver(single_returns=[None, {"n": n}])


# ---- Fail-soft paths ----


def test_recompute_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        cross_doc_writes.recompute_cross_doc_edges(client, "")
    assert driver.sessions == []


def test_recompute_threshold_zero_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match=">= 1"):
        cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID, threshold=0)
    assert driver.sessions == []


def test_recompute_threshold_negative_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match=">= 1"):
        cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID, threshold=-1)
    assert driver.sessions == []


def test_recompute_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)


def test_recompute_returns_zero_when_single_returns_no_row():
    """RETURN count(r) always emits one row in practice. The defensive
    fallback path returns 0 (not None) since we know nothing was
    written; the typed-errors contract reserves raising for real
    failures."""
    driver = RecordingDriver(single_returns=[None, None])
    client = _client_with_driver(driver)
    assert (
        cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID) == 0
    )


# ---- Two-step Cypher ----


def test_recompute_runs_two_queries_delete_then_merge():
    """First call: DELETE. Second call: MATCH ... MERGE ... RETURN."""
    driver = _driver_with_count(3)
    client = _client_with_driver(driver)
    result = cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    assert result == 3
    assert len(driver.sessions[0].calls) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "DELETE r" in delete_cypher
    assert "MERGE" in merge_cypher
    assert "RETURN count(r)" in merge_cypher


def test_recompute_zero_edges_is_valid_outcome():
    """No other doc met the threshold -> count=0, ok=True (not None)."""
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    result = cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    assert result == 0


# ---- Cypher shape ----


def test_recompute_delete_query_uses_undirected_pattern():
    """Wipe is undirected so it catches edges written in either
    direction (MERGE on an undirected pattern stores in arbitrary
    direction)."""
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    delete_cypher, params = driver.sessions[0].calls[0]
    # Undirected: the right-hand side has no `>` after the bracket.
    assert ":RELATED_TO]-(" in delete_cypher
    # Doc lookup uses Document|Artifact disjunction so artifacts
    # also participate.
    assert "Document|Artifact" in delete_cypher
    assert params == {"doc_id": DOC_ID}


def test_recompute_merge_query_carries_label_disjunction_on_both_sides():
    """Both focal AND `other` use :Document|:Artifact - cross-type
    linking ('dataset relates to paper') participates."""
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    # Disjunction appears at LEAST twice (focal + other).
    assert merge_cypher.count("Document|Artifact") >= 2


def test_recompute_merge_query_filters_self_edges():
    """`other.doc_id <> $doc_id` keeps the focal from linking to
    itself via the symmetric MENTIONS pattern."""
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    merge_cypher, params = driver.sessions[0].calls[1]
    assert "other.doc_id <> $doc_id" in merge_cypher
    assert params["doc_id"] == DOC_ID


def test_recompute_merge_query_uses_threshold_param():
    driver = _driver_with_count(5)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(
        client, DOC_ID, threshold=4
    )
    merge_cypher, params = driver.sessions[0].calls[1]
    assert "size(shared) >= $threshold" in merge_cypher
    assert params["threshold"] == 4


def test_recompute_merge_query_sets_shared_entities_count_and_timestamp():
    driver = _driver_with_count(2)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "r.shared_entities = shared" in merge_cypher
    assert "r.shared_count = size(shared)" in merge_cypher
    assert "r.computed_at = datetime()" in merge_cypher


def test_recompute_merge_query_uses_undirected_merge():
    """`MERGE (d)-[r:RELATED_TO]-(other)` (no `>`) matches in either
    direction, so the edge is genuinely undirected."""
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "MERGE (d)-[r:RELATED_TO]-(other)" in merge_cypher


def test_recompute_uses_default_threshold_when_not_provided():
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    cross_doc_writes.recompute_cross_doc_edges(client, DOC_ID)
    _merge_cypher, params = driver.sessions[0].calls[1]
    assert params["threshold"] == cross_doc_writes.DEFAULT_SHARED_COUNT_THRESHOLD
    assert params["threshold"] == 2  # belt + suspenders on the constant value


# ---- Client delegate ----


def test_client_recompute_cross_doc_edges_delegates_to_module():
    driver = _driver_with_count(7)
    client = _client_with_driver(driver)
    result = client.recompute_cross_doc_edges(DOC_ID)
    assert result == 7
    # Both queries fired through the delegate.
    assert len(driver.sessions[0].calls) == 2


def test_client_recompute_cross_doc_edges_passes_through_threshold():
    driver = _driver_with_count(0)
    client = _client_with_driver(driver)
    client.recompute_cross_doc_edges(DOC_ID, threshold=5)
    _merge, params = driver.sessions[0].calls[1]
    assert params["threshold"] == 5
