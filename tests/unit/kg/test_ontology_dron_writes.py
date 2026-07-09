"""Tests for kg.ontology_dron_writes - the L7 DRON adapter (OWL/rdflib path).

Mirrors the pronto-OBO test pattern but covers the rdflib path: tests
patch `_read_and_extract` directly so the underlying read_rdf +
extract_terms_owl pipeline isn't exercised here (that pipeline has
its own tests in test_ontology_helpers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_dron_writes
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontology_helpers import OntologyTerm


@dataclass
class _RecordingResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    async def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results: list[_RecordingResult] = field(default_factory=list)

    async def run(self, query: str, **params: Any):
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        idx = len(self.calls) - 1
        if idx < len(self.canned_results):
            return self.canned_results[idx]
        return _RecordingResult()

    async def __aenter__(self) -> RecordingSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results_per_session: list[list[_RecordingResult]] = field(default_factory=list)
    closed: bool = False

    def session(self) -> RecordingSession:
        idx = len(self.sessions)
        canned = (
            self.canned_results_per_session[idx]
            if idx < len(self.canned_results_per_session)
            else []
        )
        sess = RecordingSession(raise_on_run=self.raise_on_run, canned_results=canned)
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


def _term(id_: str, label: str, synonyms=(), parents=()) -> OntologyTerm:
    return OntologyTerm(
        id=id_,
        label=label,
        synonyms=tuple(synonyms),
        parents=tuple(parents),
        definition=None,
    )


def test_domain_tags_declared():
    assert isinstance(ontology_dron_writes.DOMAIN_TAGS, tuple)
    assert "pharmacology" in ontology_dron_writes.DOMAIN_TAGS


async def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await ontology_dron_writes.is_imported(_client_with_driver(driver)) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    assert await ontology_dron_writes.is_imported(_client_with_driver(driver)) is False


async def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("conn lost"))
    with pytest.raises(RuntimeError, match="conn lost"):
        await ontology_dron_writes.is_imported(_client_with_driver(driver))


async def test_is_imported_query_uses_term_label():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    await ontology_dron_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":DRONTerm" in cypher


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    assert await ontology_dron_writes.write_terms(_client_with_driver(driver), []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    terms = [_term("DRON:00000001", "drug")]
    assert await ontology_dron_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    terms = [
        _term("DRON:00000001", "drug"),
        _term("DRON:00000005", "aspirin", parents=("DRON:00000001",)),
    ]
    assert await ontology_dron_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    terms = [
        _term("DRON:00000005", "aspirin", synonyms=("acetylsalicylic acid", "asa")),
    ]
    await ontology_dron_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[0]
    assert ":OntologyTerm" in cypher
    assert ":DRONTerm" in cypher
    assert params["rows"] == [
        {
            "id": "DRON:00000005",
            "label": "aspirin",
            "synonyms": ["acetylsalicylic acid", "asa"],
            "definition": None,
        }
    ]


async def test_write_terms_edge_query_uses_is_a_rel():
    driver = RecordingDriver()
    terms = [
        _term("DRON:00000001", "drug"),
        _term("DRON:00000005", "aspirin", parents=("DRON:00000001",)),
    ]
    await ontology_dron_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[1]
    assert ":DRON_IS_A" in cypher
    assert params["rows"] == [{"child": "DRON:00000005", "parent": "DRON:00000001"}]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_dron_writes.write_terms(
            _client_with_driver(driver), [_term("DRON:00000001", "drug")]
        )


async def test_import_short_circuits_when_already_imported():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    with patch("knowledge_agent.kg.ontology_writes.require_cached") as mc:
        assert (
            await ontology_dron_writes.import_dron(_client_with_driver(driver), force=False)
            is False
        )
    mc.assert_not_called()


async def test_import_force_drops_then_reimports():
    driver = RecordingDriver()
    with (
        patch("knowledge_agent.kg.ontology_writes.require_cached", return_value="/fake/dron.owl"),
        patch(
            "knowledge_agent.kg.ontology_dron_writes._read_and_extract",
            return_value=[_term("DRON:00000001", "drug")],
        ),
    ):
        assert (
            await ontology_dron_writes.import_dron(_client_with_driver(driver), force=True) is True
        )
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":DRONTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


async def test_import_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch("knowledge_agent.kg.ontology_writes.require_cached", return_value="/fake/dron.owl"),
        patch("knowledge_agent.kg.ontology_dron_writes._read_and_extract", return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            await ontology_dron_writes.import_dron(_client_with_driver(driver), force=False)


async def test_import_propagates_download_exception():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.require_cached",
            side_effect=RuntimeError("network down"),
        ),
        pytest.raises(RuntimeError, match="network down"),
    ):
        await ontology_dron_writes.import_dron(_client_with_driver(driver), force=False)


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    assert await ontology_dron_writes.delete_imported(_client_with_driver(driver)) is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":DRONTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_dron_writes.delete_imported(_client_with_driver(driver))


def test_read_and_extract_uses_rdflib_not_pronto():
    """Critical invariant: this adapter's `_read_and_extract` must go
    through rdflib (read_rdf + extract_terms_owl), NOT pronto. Pronto
    silently drops oboInOwl synonyms from OWL files; rdflib preserves
    them."""
    from pathlib import Path

    with (
        patch("knowledge_agent.kg.ontology_dron_writes.read_rdf") as mock_read_rdf,
        patch(
            "knowledge_agent.kg.ontology_dron_writes.extract_terms_owl", return_value=[]
        ) as mock_extract,
    ):
        ontology_dron_writes._read_and_extract(Path("/fake/dron.owl"))
    mock_read_rdf.assert_called_once()
    mock_extract.assert_called_once()
    _, kwargs = mock_read_rdf.call_args
    assert kwargs.get("format") == "xml"


async def test_client_is_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await _client_with_driver(driver).is_dron_imported() is True


async def test_client_delete_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_dron() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":DRONTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    with patch("knowledge_agent.kg.ontology_writes.require_cached") as mc:
        assert await client.import_dron(force=False) is False
    mc.assert_not_called()
