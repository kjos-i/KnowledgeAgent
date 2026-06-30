"""Tests for kg.ontology_eco_writes - the L7 ECO adapter.

Mirrors the HPO/UBERON/MONDO/ChEBI test pattern. ECO follows the
standard OBO Foundry layout, so we delegate extraction to the shared
`extract_terms_obo` helper (already tested in test_ontology_helpers).
Tests here focus on:
  - DOMAIN_TAGS metadata
  - Cypher write shape (:OntologyTerm:ECOTerm + :ECO_IS_A)
  - is_imported / import_eco / delete_imported lifecycle
  - Neo4jClient delegate methods
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from unittest.mock import patch, AsyncMock

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_eco_writes
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontology_helpers import OntologyTerm

# ---- Test harness ----


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


# ---- DOMAIN_TAGS ----


def test_domain_tags_declared():
    """The wizard suggests ECO for experimental-methods corpora."""
    assert isinstance(ontology_eco_writes.DOMAIN_TAGS, tuple)
    assert len(ontology_eco_writes.DOMAIN_TAGS) > 0
    assert "experimental-methods" in ontology_eco_writes.DOMAIN_TAGS


# ---- is_imported ----


async def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert await ontology_eco_writes.is_imported(client) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    assert await ontology_eco_writes.is_imported(client) is False


async def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("connection lost"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="connection lost"):
        await ontology_eco_writes.is_imported(client)


async def test_is_imported_query_uses_ecoterm_label():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    await ontology_eco_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ECOTerm" in cypher


# ---- write_terms ----


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await ontology_eco_writes.write_terms(client, []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [_term("ECO:0000000", "evidence")]
    assert await ontology_eco_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("ECO:0000000", "evidence"),
        _term(
            "ECO:0000006",
            "experimental evidence",
            parents=("ECO:0000000",),
        ),
    ]
    assert await ontology_eco_writes.write_terms(client, terms) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term(
            "ECO:0000006",
            "experimental evidence",
            synonyms=("experimental data", "wet-lab evidence"),
        ),
    ]
    await ontology_eco_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[0]

    # Multi-label: both :OntologyTerm and :ECOTerm.
    assert ":OntologyTerm" in cypher
    assert ":ECOTerm" in cypher
    # Sets label/synonyms/definition from row.
    assert "row.label" in cypher
    assert "row.synonyms" in cypher
    assert "row.definition" in cypher

    rows = params["rows"]
    # ECO IDs carry the "ECO:" prefix verbatim from pronto.
    assert rows == [
        {
            "id": "ECO:0000006",
            "label": "experimental evidence",
            "synonyms": ["experimental data", "wet-lab evidence"],
            "definition": None,
        }
    ]


async def test_write_terms_edge_query_uses_eco_is_a_rel():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("ECO:0000000", "evidence"),
        _term(
            "ECO:0000006",
            "experimental evidence",
            parents=("ECO:0000000",),
        ),
    ]
    await ontology_eco_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[1]

    # ECO uses :ECO_IS_A, not :MESH_BROADER or :GO_IS_A.
    assert ":ECO_IS_A" in cypher
    assert "row.child" in cypher
    assert "row.parent" in cypher
    assert params["rows"] == [
        {"child": "ECO:0000006", "parent": "ECO:0000000"}
    ]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_eco_writes.write_terms(
            client, [_term("ECO:0000000", "evidence")]
        )


# ---- import_eco ----


async def test_import_eco_short_circuits_when_already_imported():
    """force=False and ECO already imported -> no download/parse/write."""
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        result = await ontology_eco_writes.import_eco(client, force=False)

    assert result is False  # no-op: typed-errors contract
    mock_cache.assert_not_called()


async def test_import_eco_force_drops_then_reimports():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    fake_terms = [_term("ECO:0000000", "evidence")]
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/eco.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_eco_writes._read_and_extract",
            return_value=fake_terms,
        ),
    ):
        result = await ontology_eco_writes.import_eco(client, force=True)

    assert result is True
    # First session = delete_imported. Second = write_terms.
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":ECOTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


async def test_import_eco_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    with (
        patch(
            "knowledge_agent.kg.ontology_writes.ensure_cached",
            return_value="/fake/eco.obo",
        ),
        patch(
            "knowledge_agent.kg.ontology_eco_writes._read_and_extract",
            return_value=[],
        ),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            await ontology_eco_writes.import_eco(client, force=False)


async def test_import_eco_propagates_download_exception():
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
            await ontology_eco_writes.import_eco(client, force=False)


# ---- delete_imported ----


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await ontology_eco_writes.delete_imported(client) is None
    assert len(driver.sessions[0].calls) == 1
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ECOTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_eco_writes.delete_imported(client)


# ---- Neo4jClient delegate methods ----


async def test_client_is_eco_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    assert await client.is_eco_imported() is True


async def test_client_delete_eco_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_eco() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":ECOTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_eco_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch(
        "knowledge_agent.kg.ontology_writes.ensure_cached"
    ) as mock_cache:
        assert await client.import_eco(force=False) is False
    mock_cache.assert_not_called()
