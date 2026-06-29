"""Tests for kg.cross_doc_xrefs_writes - L10 :RELATED_BY_XREF
recompute (per-doc + graph-wide) + client delegate.

Mirrors `test_cross_doc_writes.py` (the L9 tests) — same harness
shape, same fail-soft coverage, same Cypher-shape assertions. The
L10 surface adds:

  - Equivalence-clause check (`t1 = t2 OR (t1)-[<pipe>]-(t2)` where
    the pipe carries all 18 `:<X>_XREF` types).
  - `recompute_cross_doc_xrefs_global` — graph-wide variant with
    `id(this) < id(other)` deduplication.
  - Per-doc client delegate (parallel to L9's).
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import cross_doc_xrefs_writes
from knowledge_agent.kg.client import Neo4jClient


# ---- Test harness (mirrors cross_doc_writes tests) ----


@dataclass
class _RunResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    single_value: dict[str, Any] | None = None

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


DOC_ID = "test-doc-l10"


def _driver_with_per_doc_count(n: int) -> RecordingDriver:
    """Pre-queued: DELETE returns nothing, recompute returns {n: <count>}."""
    return RecordingDriver(single_returns=[None, {"n": n}])


# ---- Per-doc: fail-soft paths ----


def test_per_doc_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, "")
    assert driver.sessions == []


def test_per_doc_threshold_zero_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match=">= 1"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(
            client, DOC_ID, threshold=0,
        )
    assert driver.sessions == []


def test_per_doc_threshold_negative_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match=">= 1"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(
            client, DOC_ID, threshold=-1,
        )
    assert driver.sessions == []


def test_per_doc_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)


def test_per_doc_returns_zero_when_single_returns_no_row():
    """Defensive fallback when the driver returns no row — returns 0
    (not None), matching the typed-errors contract that reserves
    raising for real failures."""
    driver = RecordingDriver(single_returns=[None, None])
    client = _client_with_driver(driver)
    assert (
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
        == 0
    )


# ---- Per-doc: two-step Cypher ----


def test_per_doc_runs_two_queries_delete_then_merge():
    driver = _driver_with_per_doc_count(3)
    client = _client_with_driver(driver)
    result = cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(
        client, DOC_ID,
    )
    assert result == 3
    assert len(driver.sessions[0].calls) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "DELETE r" in delete_cypher
    assert "MERGE" in merge_cypher
    assert "RETURN count(r)" in merge_cypher


def test_per_doc_zero_edges_is_valid_outcome():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    result = cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(
        client, DOC_ID,
    )
    assert result == 0


# ---- Per-doc: Cypher shape ----


def test_per_doc_delete_query_uses_related_by_xref_and_undirected():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    delete_cypher, params = driver.sessions[0].calls[0]
    assert ":RELATED_BY_XREF]-(" in delete_cypher
    assert "Document|Artifact" in delete_cypher
    assert params == {"doc_id": DOC_ID}


def test_per_doc_merge_query_carries_label_disjunction_on_both_sides():
    """Both focal AND `other` use :Document|:Artifact."""
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert merge_cypher.count("Document|Artifact") >= 2


def test_per_doc_merge_query_walks_canonical_to_ontology_term_path():
    """The path must reach :OntologyTerm via :CANONICAL_TO on both
    sides (not the L9 raw-entity shape)."""
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert ":CANONICAL_TO" in merge_cypher
    assert ":OntologyTerm" in merge_cypher
    # Both sides walk to OntologyTerm.
    assert merge_cypher.count(":OntologyTerm") >= 2


def test_per_doc_merge_query_uses_xref_equivalence_clause():
    """The WHERE clause checks BOTH same-node identity AND the
    pipe-union of all 18 xref edge types."""
    from knowledge_agent.kg.schema import ONTOLOGY_XREF_RELS

    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    # Identity branch.
    assert "t1 = t2 OR EXISTS" in merge_cypher
    # All 18 xref relationship types appear in the pipe union.
    for rel in ONTOLOGY_XREF_RELS:
        assert rel in merge_cypher


def test_per_doc_merge_query_filters_self_edges():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, params = driver.sessions[0].calls[1]
    assert "other.doc_id <> $doc_id" in merge_cypher
    assert params["doc_id"] == DOC_ID


def test_per_doc_merge_query_uses_threshold_param():
    driver = _driver_with_per_doc_count(5)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(
        client, DOC_ID, threshold=4,
    )
    merge_cypher, params = driver.sessions[0].calls[1]
    assert "size(shared_concepts) >= $threshold" in merge_cypher
    assert params["threshold"] == 4


def test_per_doc_merge_query_sets_shared_concepts_count_and_timestamp():
    driver = _driver_with_per_doc_count(2)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "r.shared_concepts = shared_concepts" in merge_cypher
    assert "r.shared_count = size(shared_concepts)" in merge_cypher
    assert "r.computed_at = datetime()" in merge_cypher


def test_per_doc_merge_query_uses_undirected_merge():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "MERGE (this)-[r:RELATED_BY_XREF]-(other)" in merge_cypher


def test_per_doc_uses_default_threshold_when_not_provided():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges(client, DOC_ID)
    _merge_cypher, params = driver.sessions[0].calls[1]
    assert (
        params["threshold"]
        == cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD
    )
    assert params["threshold"] == 2


# ---- Global recompute: fail-soft + Cypher shape ----


def test_global_threshold_zero_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match=">= 1"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(
            client, threshold=0,
        )
    assert driver.sessions == []


def test_global_runs_two_queries_delete_then_merge():
    """Global wipes ALL :RELATED_BY_XREF edges before recompute."""
    driver = RecordingDriver(single_returns=[None, {"n": 17}])
    client = _client_with_driver(driver)
    result = cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(client)
    assert result == 17
    assert len(driver.sessions[0].calls) == 2

    delete_cypher, _ = driver.sessions[0].calls[0]
    # No doc_id filter on the global wipe.
    assert "DELETE r" in delete_cypher
    assert "$doc_id" not in delete_cypher
    assert "RELATED_BY_XREF" in delete_cypher


def test_global_merge_uses_id_pair_dedup():
    """`id(this) < id(other)` deduplicates the unordered doc pair."""
    driver = RecordingDriver(single_returns=[None, {"n": 0}])
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(client)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "id(this) < id(other)" in merge_cypher


def test_global_merge_uses_equivalence_clause():
    """The xref equivalence WHERE clause appears identically in the
    global query."""
    from knowledge_agent.kg.schema import ONTOLOGY_XREF_RELS

    driver = RecordingDriver(single_returns=[None, {"n": 0}])
    client = _client_with_driver(driver)
    cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(client)
    merge_cypher, _ = driver.sessions[0].calls[1]
    assert "t1 = t2 OR EXISTS" in merge_cypher
    for rel in ONTOLOGY_XREF_RELS:
        assert rel in merge_cypher


def test_global_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(client)


def test_global_returns_zero_when_single_returns_no_row():
    """Defensive fallback when the driver returns no row — returns 0
    (not None)."""
    driver = RecordingDriver(single_returns=[None, None])
    client = _client_with_driver(driver)
    assert (
        cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global(client)
        == 0
    )


# ---- Client delegate ----


def test_client_recompute_cross_doc_xrefs_edges_delegates_to_module():
    driver = _driver_with_per_doc_count(7)
    client = _client_with_driver(driver)
    result = client.recompute_cross_doc_xrefs_edges(DOC_ID)
    assert result == 7
    assert len(driver.sessions[0].calls) == 2


def test_client_recompute_cross_doc_xrefs_edges_passes_through_threshold():
    driver = _driver_with_per_doc_count(0)
    client = _client_with_driver(driver)
    client.recompute_cross_doc_xrefs_edges(DOC_ID, threshold=5)
    _merge, params = driver.sessions[0].calls[1]
    assert params["threshold"] == 5
