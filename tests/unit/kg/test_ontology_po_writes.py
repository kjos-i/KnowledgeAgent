"""Tests for kg.ontology_po_writes - the L7 PO adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from unittest.mock import patch, AsyncMock

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_po_writes
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

    async def __aenter__(self) -> "RecordingSession":
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
            if idx < len(self.canned_results_per_session) else []
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
        id=id_, label=label,
        synonyms=tuple(synonyms), parents=tuple(parents),
        definition=None,
    )


def test_domain_tags_declared():
    assert isinstance(ontology_po_writes.DOMAIN_TAGS, tuple)
    assert "biology" in ontology_po_writes.DOMAIN_TAGS


async def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]])
    assert await ontology_po_writes.is_imported(_client_with_driver(driver)) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]])
    assert await ontology_po_writes.is_imported(_client_with_driver(driver)) is False


async def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("conn lost"))
    with pytest.raises(RuntimeError, match="conn lost"):
        await ontology_po_writes.is_imported(_client_with_driver(driver))


async def test_is_imported_query_uses_term_label():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]])
    client = _client_with_driver(driver)
    await ontology_po_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":POTerm" in cypher


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    assert await ontology_po_writes.write_terms(_client_with_driver(driver), []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    terms = [_term("PO:0025131", "plant anatomical entity")]
    assert await ontology_po_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    terms = [
        _term("PO:0025131", "plant anatomical entity"),
        _term("PO:0009001", "leaf", parents=("PO:0025131",)),
    ]
    assert await ontology_po_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    terms = [_term("PO:0009001", "leaf", synonyms=("foliage leaf",))]
    await ontology_po_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[0]
    assert ":OntologyTerm" in cypher
    assert ":POTerm" in cypher
    assert params["rows"] == [{"id": "PO:0009001", "label": "leaf",
                               "synonyms": ["foliage leaf"], "definition": None}]


async def test_write_terms_edge_query_uses_is_a_rel():
    driver = RecordingDriver()
    terms = [
        _term("PO:0025131", "plant anatomical entity"),
        _term("PO:0009001", "leaf", parents=("PO:0025131",)),
    ]
    await ontology_po_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[1]
    assert ":PO_IS_A" in cypher
    assert params["rows"] == [{"child": "PO:0009001", "parent": "PO:0025131"}]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_po_writes.write_terms(
            _client_with_driver(driver), [_term("PO:0025131", "plant anatomical entity")])


async def test_import_short_circuits_when_already_imported():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]])
    with patch("knowledge_agent.kg.ontology_writes.ensure_cached") as mc:
        assert await ontology_po_writes.import_po(_client_with_driver(driver), force=False) is False
    mc.assert_not_called()


async def test_import_force_drops_then_reimports():
    driver = RecordingDriver()
    with (
        patch("knowledge_agent.kg.ontology_writes.ensure_cached", return_value="/fake/po.obo"),
        patch("knowledge_agent.kg.ontology_po_writes._read_and_extract",
              return_value=[_term("PO:0025131", "plant anatomical entity")]),
    ):
        assert await ontology_po_writes.import_po(_client_with_driver(driver), force=True) is True
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":POTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


async def test_import_aborts_on_zero_terms():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]])
    with (
        patch("knowledge_agent.kg.ontology_writes.ensure_cached", return_value="/fake/po.obo"),
        patch("knowledge_agent.kg.ontology_po_writes._read_and_extract", return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            await ontology_po_writes.import_po(_client_with_driver(driver), force=False)


async def test_import_propagates_download_exception():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]])
    with patch("knowledge_agent.kg.ontology_writes.ensure_cached",
               side_effect=RuntimeError("network down")):
        with pytest.raises(RuntimeError, match="network down"):
            await ontology_po_writes.import_po(_client_with_driver(driver), force=False)


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    assert await ontology_po_writes.delete_imported(_client_with_driver(driver)) is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":POTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_po_writes.delete_imported(_client_with_driver(driver))


async def test_client_is_imported_delegates_to_module():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]])
    assert await _client_with_driver(driver).is_po_imported() is True


async def test_client_delete_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_po() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":POTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_delegates_to_module():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]])
    client = _client_with_driver(driver)
    with patch("knowledge_agent.kg.ontology_writes.ensure_cached") as mc:
        assert await client.import_po(force=False) is False
    mc.assert_not_called()
