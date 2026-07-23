"""Tests for kg.ontologies.foodon_writes - the L7 FOODON adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontologies import foodon_writes
from knowledge_agent.kg.ontologies.helpers import OntologyTerm


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
    assert isinstance(foodon_writes.DOMAIN_TAGS, tuple)
    assert "nutrition" in foodon_writes.DOMAIN_TAGS


async def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await foodon_writes.is_imported(_client_with_driver(driver)) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    assert await foodon_writes.is_imported(_client_with_driver(driver)) is False


async def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("conn lost"))
    with pytest.raises(RuntimeError, match="conn lost"):
        await foodon_writes.is_imported(_client_with_driver(driver))


async def test_is_imported_query_uses_term_label():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    await foodon_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FOODONTerm" in cypher


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    assert await foodon_writes.write_terms(_client_with_driver(driver), []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    terms = [_term("FOODON:00001002", "food product")]
    assert await foodon_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    terms = [
        _term("FOODON:00001002", "food product"),
        _term("FOODON:03301720", "apple", parents=("FOODON:00001002",)),
    ]
    assert await foodon_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    terms = [_term("FOODON:03301720", "apple", synonyms=("raw apple",))]
    await foodon_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[0]
    assert ":OntologyTerm" in cypher
    assert ":FOODONTerm" in cypher
    assert params["rows"] == [
        {"id": "FOODON:03301720", "label": "apple", "synonyms": ["raw apple"], "definition": None}
    ]


async def test_write_terms_edge_query_uses_is_a_rel():
    driver = RecordingDriver()
    terms = [
        _term("FOODON:00001002", "food product"),
        _term("FOODON:03301720", "apple", parents=("FOODON:00001002",)),
    ]
    await foodon_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[1]
    assert ":FOODON_IS_A" in cypher
    assert params["rows"] == [{"child": "FOODON:03301720", "parent": "FOODON:00001002"}]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await foodon_writes.write_terms(
            _client_with_driver(driver), [_term("FOODON:00001002", "food product")]
        )


async def test_import_short_circuits_when_already_imported():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    with patch("knowledge_agent.kg.ontologies.writes.require_cached") as mc:
        assert await foodon_writes.import_foodon(_client_with_driver(driver), force=False) is False
    mc.assert_not_called()


async def test_import_force_drops_then_reimports():
    driver = RecordingDriver()
    with (
        patch(
            "knowledge_agent.kg.ontologies.writes.require_cached", return_value="/fake/foodon.obo"
        ),
        patch(
            "knowledge_agent.kg.ontologies.foodon_writes._read_and_extract",
            return_value=[_term("FOODON:00001002", "food product")],
        ),
    ):
        assert await foodon_writes.import_foodon(_client_with_driver(driver), force=True) is True
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":FOODONTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


async def test_import_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch(
            "knowledge_agent.kg.ontologies.writes.require_cached", return_value="/fake/foodon.obo"
        ),
        patch("knowledge_agent.kg.ontologies.foodon_writes._read_and_extract", return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            await foodon_writes.import_foodon(_client_with_driver(driver), force=False)


async def test_import_propagates_download_exception():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch(
            "knowledge_agent.kg.ontologies.writes.require_cached",
            side_effect=RuntimeError("network down"),
        ),
        pytest.raises(RuntimeError, match="network down"),
    ):
        await foodon_writes.import_foodon(_client_with_driver(driver), force=False)


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    assert await foodon_writes.delete_imported(_client_with_driver(driver)) is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FOODONTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await foodon_writes.delete_imported(_client_with_driver(driver))


async def test_client_is_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await _client_with_driver(driver).is_foodon_imported() is True


async def test_client_delete_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_foodon() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FOODONTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    with patch("knowledge_agent.kg.ontologies.writes.require_cached") as mc:
        assert await client.import_foodon(force=False) is False
    mc.assert_not_called()
