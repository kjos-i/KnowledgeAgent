"""Tests for kg.ontology_go_writes - the L7 GO adapter.

Less ground to cover than the MeSH tests because GO follows the
standard OBO Foundry layout, so we delegate extraction to the shared
`extract_terms_obo` helper (already tested in test_ontology_helpers).
Tests here focus on:
  - DOMAIN_TAGS metadata
  - Cypher write shape (:OntologyTerm:GOTerm + :GO_IS_A)
  - is_imported / import_go / delete_imported lifecycle
  - Neo4jClient delegate methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from unittest.mock import patch

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_go_writes
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontology_helpers import OntologyTerm

# ---- Test harness ----


@dataclass
class _RecordingResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results: list[_RecordingResult] = field(default_factory=list)

    def run(self, query: str, **params: Any):
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        idx = len(self.calls) - 1
        if idx < len(self.canned_results):
            return self.canned_results[idx]
        return _RecordingResult()

    def __enter__(self) -> "RecordingSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results_per_session: list[list[_RecordingResult]] = field(
        default_factory=list
    )
    closed: bool = False

    def session(self) -> RecordingSession:
        idx = len(self.sessions)
        canned = (
            self.canned_results_per_session[idx]
            if idx < len(self.canned_results_per_session)
            else []
        )
        sess = RecordingSession(
            raise_on_run=self.raise_on_run, canned_results=canned
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


def _term(id_: str, label: str, synonyms=(), parents=()) -> OntologyTerm:
    return OntologyTerm(
        id=id_,
        label=label,
        synonyms=tuple(synonyms),
        parents=tuple(parents),
        definition=None,
    )


# ---- DOMAIN_TAGS ----


def test_domain_tags_declared():
    """The wizard suggests GO for cell biology / biochemistry corpora."""
    assert isinstance(ontology_go_writes.DOMAIN_TAGS, tuple)
    assert len(ontology_go_writes.DOMAIN_TAGS) > 0
    assert "biology" in ontology_go_writes.DOMAIN_TAGS


# ---- is_imported ----


def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert ontology_go_writes.is_imported(client) is True


def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    assert ontology_go_writes.is_imported(client) is False


def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("connection lost"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="connection lost"):
        ontology_go_writes.is_imported(client)


def test_is_imported_query_uses_goterm_label():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    ontology_go_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":GOTerm" in cypher


# ---- write_terms ----


def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert ontology_go_writes.write_terms(client, []) is None
    assert driver.sessions == []


def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [_term("GO:0008150", "biological_process")]
    assert ontology_go_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 1


def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("GO:0008150", "biological_process"),
        _term(
            "GO:0007154",
            "cell communication",
            parents=("GO:0008150",),
        ),
    ]
    assert ontology_go_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 2


def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term(
            "GO:0008150",
            "biological_process",
            synonyms=("bp", "biological process"),
        ),
    ]
    ontology_go_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[0]

    # Multi-label: both :OntologyTerm and :GOTerm.
    assert ":OntologyTerm" in cypher
    assert ":GOTerm" in cypher
    # Sets label/synonyms/definition from row.
    assert "row.label" in cypher
    assert "row.synonyms" in cypher
    assert "row.definition" in cypher

    rows = params["rows"]
    # GO IDs carry the "GO:" prefix verbatim from pronto.
    assert rows == [
        {
            "id": "GO:0008150",
            "label": "biological_process",
            "synonyms": ["bp", "biological process"],
            "definition": None,
        }
    ]


def test_write_terms_edge_query_uses_go_is_a_rel():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("GO:0008150", "biological_process"),
        _term(
            "GO:0007154",
            "cell communication",
            parents=("GO:0008150",),
        ),
    ]
    ontology_go_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[1]

    # GO uses :GO_IS_A, not :MESH_BROADER.
    assert ":GO_IS_A" in cypher
    assert "row.child" in cypher
    assert "row.parent" in cypher
    assert params["rows"] == [
        {"child": "GO:0007154", "parent": "GO:0008150"}
    ]


def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        ontology_go_writes.write_terms(
            client, [_term("GO:0008150", "biological_process")]
        )


# ---- import_go ----


def test_import_go_short_circuits_when_already_imported():
    """force=False and GO already imported -> no download/parse/write."""
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        result = ontology_go_writes.import_go(client, force=False)

    assert result is False  # no-op: typed-errors contract
    mock_cache.assert_not_called()


def test_import_go_force_drops_then_reimports():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    fake_terms = [_term("GO:0008150", "biological_process")]
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/go-basic.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_go_writes._read_and_extract",
            return_value=fake_terms,
        ),
    ):
        result = ontology_go_writes.import_go(client, force=True)

    assert result is True
    # First session = delete_imported. Second = write_terms.
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":GOTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


def test_import_go_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/go-basic.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_go_writes._read_and_extract",
            return_value=[],
        ),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            ontology_go_writes.import_go(client, force=False)


def test_import_go_propagates_download_exception():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached",
        side_effect=RuntimeError("network down"),
    ):
        with pytest.raises(RuntimeError, match="network down"):
            ontology_go_writes.import_go(client, force=False)


# ---- delete_imported ----


def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert ontology_go_writes.delete_imported(client) is None
    assert len(driver.sessions[0].calls) == 1
    cypher, _ = driver.sessions[0].calls[0]
    assert ":GOTerm" in cypher
    assert "DETACH DELETE" in cypher


def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        ontology_go_writes.delete_imported(client)


# ---- Neo4jClient delegate methods ----


def test_client_is_go_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert client.is_go_imported() is True


def test_client_delete_go_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert client.delete_go() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":GOTerm" in cypher
    assert "DETACH DELETE" in cypher


def test_client_import_go_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        assert client.import_go(force=False) is False
    mock_cache.assert_not_called()
