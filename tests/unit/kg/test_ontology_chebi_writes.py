"""Tests for kg.ontology_chebi_writes - the L7 ChEBI adapter.

Mirrors the GO/HPO/UBERON/MONDO test pattern. ChEBI LITE follows the
standard OBO Foundry layout (parses cleanly via pronto), so we
delegate extraction to the shared `extract_terms_obo` helper. Tests
here focus on:
  - DOMAIN_TAGS metadata (chemistry primary + biochemistry secondary)
  - LITE variant URL pinned
  - Cypher write shape (:OntologyTerm:ChEBITerm + :CHEBI_IS_A)
  - is_imported / import_chebi / delete_imported lifecycle
  - Neo4jClient delegate methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from unittest.mock import patch

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_chebi_writes
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


def test_domain_tags_declared_chemistry_primary():
    """The wizard suggests ChEBI for chemistry corpora primarily,
    with biochemistry as secondary tag matching its biological-
    relevance filter."""
    assert isinstance(ontology_chebi_writes.DOMAIN_TAGS, tuple)
    assert "chemistry" in ontology_chebi_writes.DOMAIN_TAGS
    assert "biochemistry" in ontology_chebi_writes.DOMAIN_TAGS


# ---- LITE variant URL pinned ----


def test_lite_variant_url_pinned():
    """Pin the LITE variant URL - bumping to FULL silently would
    quadruple disk + memory."""
    assert "chebi_lite.obo" in ontology_chebi_writes.CHEBI_DOWNLOAD_URL


def test_cache_filename_uses_lite_variant():
    """The cache file name must match the URL variant to avoid
    silently shipping the wrong subset on subsequent runs."""
    assert ontology_chebi_writes.CHEBI_CACHE_FILENAME == "chebi_lite.obo"


# ---- is_imported ----


def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert ontology_chebi_writes.is_imported(client) is True


def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    assert ontology_chebi_writes.is_imported(client) is False


def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("connection lost"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="connection lost"):
        ontology_chebi_writes.is_imported(client)


def test_is_imported_query_uses_chebiterm_label():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    ontology_chebi_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ChEBITerm" in cypher


# ---- write_terms ----


def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert ontology_chebi_writes.write_terms(client, []) is None
    assert driver.sessions == []


def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [_term("CHEBI:24431", "chemical entity")]
    assert ontology_chebi_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 1


def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("CHEBI:24431", "chemical entity"),
        _term(
            "CHEBI:17234",
            "glucose",
            parents=("CHEBI:24431",),
        ),
    ]
    assert ontology_chebi_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 2


def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term(
            "CHEBI:17234",
            "glucose",
            synonyms=("dextrose", "D-glucose"),
        ),
    ]
    ontology_chebi_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[0]

    # Multi-label: both :OntologyTerm and :ChEBITerm.
    assert ":OntologyTerm" in cypher
    assert ":ChEBITerm" in cypher
    assert "row.label" in cypher
    assert "row.synonyms" in cypher
    assert "row.definition" in cypher

    rows = params["rows"]
    # ChEBI IDs carry the "CHEBI:" prefix verbatim from pronto.
    assert rows == [
        {
            "id": "CHEBI:17234",
            "label": "glucose",
            "synonyms": ["dextrose", "D-glucose"],
            "definition": None,
        }
    ]


def test_write_terms_edge_query_uses_chebi_is_a_rel():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("CHEBI:24431", "chemical entity"),
        _term(
            "CHEBI:17234",
            "glucose",
            parents=("CHEBI:24431",),
        ),
    ]
    ontology_chebi_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[1]

    # ChEBI uses :CHEBI_IS_A, not the other ontology hierarchy types.
    assert ":CHEBI_IS_A" in cypher
    assert "row.child" in cypher
    assert "row.parent" in cypher
    assert params["rows"] == [
        {"child": "CHEBI:17234", "parent": "CHEBI:24431"}
    ]


def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        ontology_chebi_writes.write_terms(
            client, [_term("CHEBI:24431", "chemical entity")]
        )


# ---- import_chebi ----


def test_import_chebi_short_circuits_when_already_imported():
    """force=False and ChEBI already imported -> no download/parse/write."""
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        result = ontology_chebi_writes.import_chebi(client, force=False)

    assert result is False  # no-op: typed-errors contract
    mock_cache.assert_not_called()


def test_import_chebi_force_drops_then_reimports():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    fake_terms = [_term("CHEBI:24431", "chemical entity")]
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/chebi_lite.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_chebi_writes._read_and_extract",
            return_value=fake_terms,
        ),
    ):
        result = ontology_chebi_writes.import_chebi(client, force=True)

    assert result is True
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":ChEBITerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


def test_import_chebi_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/chebi_lite.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_chebi_writes._read_and_extract",
            return_value=[],
        ),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            ontology_chebi_writes.import_chebi(client, force=False)


def test_import_chebi_propagates_download_exception():
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
            ontology_chebi_writes.import_chebi(client, force=False)


# ---- delete_imported ----


def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert ontology_chebi_writes.delete_imported(client) is None
    assert len(driver.sessions[0].calls) == 1
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ChEBITerm" in cypher
    assert "DETACH DELETE" in cypher


def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        ontology_chebi_writes.delete_imported(client)


# ---- Neo4jClient delegate methods ----


def test_client_is_chebi_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert client.is_chebi_imported() is True


def test_client_delete_chebi_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert client.delete_chebi() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ChEBITerm" in cypher
    assert "DETACH DELETE" in cypher


def test_client_import_chebi_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        assert client.import_chebi(force=False) is False
    mock_cache.assert_not_called()
