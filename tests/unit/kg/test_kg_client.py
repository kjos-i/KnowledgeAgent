"""Tests for kg.client - Neo4jClient + helpers.

Uses a recording Driver/Session stub so tests don't touch a real DB. Each
RecordingSession captures the Cypher strings and parameters passed to
session.run() so assertions can check exactly what was sent.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.openalex_writes import (
    _clean_doi_for_storage,
    _extract_openalex_id,
)

# ---- Test harness ----


@dataclass
class RecordingRecord:
    """Stand-in for `neo4j.Record`. `.data()` returns the row dict."""

    _data: dict[str, Any] = field(default_factory=dict)

    async def data(self) -> dict[str, Any]:
        return self._data


@dataclass
class _AsyncResultStub:
    """Stand-in for neo4j.AsyncResult with `.data()` returning a list."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    async def data(self) -> list[dict[str, Any]]:
        return self.rows


@dataclass
class RecordingTransaction:
    """Stand-in for the `Transaction` object passed to `execute_read`."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    records: list[RecordingRecord] = field(default_factory=list)
    raise_on_run: Exception | None = None

    async def run(self, query: str, **params: Any) -> _AsyncResultStub:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        return _AsyncResultStub(rows=[r._data for r in self.records])


@dataclass
class RecordingSession:
    """Captures `session.run(...)` and `session.execute_read(...)` calls."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # Stub transaction used by `execute_read`. When None, the session
    # builds an empty one on demand.
    read_transaction: RecordingTransaction | None = None

    async def run(self, query: str, **params: Any) -> None:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))

    def execute_read(self, fn: Any) -> Any:
        if self.read_transaction is None:
            self.read_transaction = RecordingTransaction()
        return fn(self.read_transaction)

    async def __aenter__(self) -> "RecordingSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    """Stand-in for neo4j.Driver. Returns a RecordingSession from session()."""

    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # Pre-built transaction stub injected into every session built here.
    # Tests use it to control what records `execute_read` sees.
    read_transaction: RecordingTransaction | None = None
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(
            raise_on_run=self.raise_on_run,
            read_transaction=self.read_transaction,
        )
        self.sessions.append(sess)
        return sess

    async def close(self) -> None:
        self.closed = True


def _configured_settings(**overrides: Any) -> Settings:
    """Build a Settings with all required fields set."""
    defaults: dict[str, Any] = {
        "anthropic_api_key": "fake-anthropic-key",
        "voyage_api_key": "fake-voyage-key",
        "neo4j_uri": "neo4j://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "fake-neo4j-password",
        "lancedb_path": "/tmp/fake-lancedb-path",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _client_with_driver(settings: Settings, driver: RecordingDriver) -> Neo4jClient:
    """Build a Neo4jClient with a pre-injected stub driver."""
    client = Neo4jClient(settings=settings)
    # Bypass the @property's lazy GraphDatabase.driver() call by setting the
    # backing attribute directly. The @property checks `is None` first, so a
    # pre-set value is returned as-is.
    client._driver = driver  # type: ignore[assignment]
    return client


# Test doc_id used by write_citations / write_authorships tests.
DOC_ID = "test-doc-1"


# ---- Helpers: _extract_openalex_id ----


def test_extract_openalex_id_strips_url():
    assert _extract_openalex_id("https://openalex.org/W1234") == "W1234"


def test_extract_openalex_id_returns_bare_unchanged():
    assert _extract_openalex_id("W1234") == "W1234"


def test_extract_openalex_id_handles_none():
    assert _extract_openalex_id(None) is None


def test_extract_openalex_id_handles_empty():
    assert _extract_openalex_id("") is None


# ---- Helpers: _clean_doi_for_storage ----


def test_clean_doi_strips_url_prefix():
    assert _clean_doi_for_storage("https://doi.org/10.1234/ABC") == "10.1234/abc"


def test_clean_doi_lowercases():
    assert _clean_doi_for_storage("10.1234/ABC") == "10.1234/abc"


def test_clean_doi_strips_trailing_punctuation():
    assert _clean_doi_for_storage("10.1234/abc.") == "10.1234/abc"


def test_clean_doi_handles_none():
    assert _clean_doi_for_storage(None) is None


# ---- close ----


async def test_close_closes_driver_and_resets_to_none():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.close()
    assert driver.closed is True
    assert client._driver is None


async def test_close_is_safe_when_driver_never_created():
    client = Neo4jClient(settings=_configured_settings())
    await client.close()  # must not raise even though no driver was ever set


# ---- ensure_constraints ----


async def test_ensure_constraints_applies_all_statements():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    # success: returns None (typed-errors contract)
    assert await client.ensure_constraints() is None
    # One session opened, all CONSTRAINT_STATEMENTS run inside it.
    assert len(driver.sessions) == 1
    # 27 constraints: document_doc_id, document_openalex_id,
    # document_doi, artifact_doc_id, author_openalex_id,
    # venue_openalex_id, topic_openalex_id, chunk_chunk_id,
    # entity_key_node_key (L6), mesh_term_id, go_term_id,
    # hpo_term_id, uberon_term_id, mondo_term_id, chebi_term_id,
    # eco_term_id, so_term_id, pr_term_id, cl_term_id, po_term_id,
    # foodon_term_id, envo_term_id, ncbitaxon_term_id, obi_term_id,
    # efo_term_id, dron_term_id, fibo_term_id (L7).
    assert len(driver.sessions[0].calls) == 27


async def test_ensure_constraints_propagates_driver_exception():
    """Cypher failures propagate under the typed-errors contract."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.ensure_constraints()


# ---- write_citations ----


WORK_WITH_CITATIONS: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/ABC",
    "referenced_works": [
        "https://openalex.org/W100",
        "https://openalex.org/W101",
    ],
}


async def test_write_citations_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.write_citations("", WORK_WITH_CITATIONS)
    assert len(driver.sessions) == 0


async def test_write_citations_three_round_trips():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    ok = await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)
    assert ok is None  # success: returns None (typed-errors contract)
    # Round trips: focal MERGE, shadow UNWIND-MERGE, citation edges.
    assert len(driver.sessions[0].calls) == 3


async def test_write_citations_with_openalex_id_keys_focal_on_doc_id_and_claims_openalex():
    """When openalex_id is known the focal MERGE keys on doc_id (NOT openalex_id
    — keying on openalex_id would clobber an existing in-corpus doc's doc_id and
    violate the doc_id uniqueness constraint; the openalex doc_id-clobber fix).
    openalex_id is then CLAIMED via a guarded subquery only if no other node
    already holds it; the :Paper subtype label + in_corpus + doi are set."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)
    focal_query, focal_params = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in focal_query
    assert "d:Paper" in focal_query
    assert "d.in_corpus = true" in focal_query
    # openalex_id is claimed via the guarded subquery, not the focal MERGE key.
    assert "SET d.openalex_id = $openalex_id" in focal_query
    assert focal_params["openalex_id"] == "W1"
    assert focal_params["doc_id"] == DOC_ID
    # DOI was normalized: lowercase, URL prefix stripped.
    assert focal_params["doi"] == "10.1/abc"


async def test_write_citations_no_openalex_id_keys_focal_on_doc_id():
    """When openalex_id is missing (non-scholarly doc) the focal MERGE keys
    on doc_id; no openalex_id param is sent. The :Paper subtype label is
    still added because write_citations is paper-specific."""
    work: dict[str, Any] = {
        # No `id` field => no openalex_id extractable.
        "doi": "10.1/abc",
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, work)
    focal_query, focal_params = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in focal_query
    assert "d:Paper" in focal_query
    assert "openalex_id" not in focal_params
    assert focal_params["doc_id"] == DOC_ID
    assert focal_params["doi"] == "10.1/abc"


async def test_write_citations_shadow_gets_paper_label():
    """Shadow nodes get :Paper subtype label since they come from referenced
    works (paper citations)."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)
    shadow_query, _ = driver.sessions[0].calls[1]
    assert "ON CREATE SET c:Paper" in shadow_query


async def test_write_citations_no_doi_skips_doi_set():
    work = dict(WORK_WITH_CITATIONS)
    work["doi"] = None
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, work)
    focal_query, focal_params = driver.sessions[0].calls[0]
    assert "d.doi" not in focal_query
    assert "doi" not in focal_params


async def test_write_citations_no_references_one_round_trip():
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "doi": "10.1/abc"}
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, work)
    # Only the focal document write; no shadow nodes or edges.
    assert len(driver.sessions[0].calls) == 1


async def test_write_citations_extracts_bare_reference_ids():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)
    _, shadow_params = driver.sessions[0].calls[1]
    assert shadow_params["cited_ids"] == ["W100", "W101"]


async def test_write_citations_edges_match_focal_by_doc_id():
    """The :CITES edge query MATCHes the focal by doc_id (set in step 1)."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)
    edge_query, edge_params = driver.sessions[0].calls[2]
    assert "MATCH (p:Document {doc_id: $doc_id})" in edge_query
    assert edge_params["doc_id"] == DOC_ID


async def test_write_citations_propagates_cypher_exception():
    """Cypher failures propagate; the orchestrator boundary catches
    and stores on `IngestResult.kg_citations_error`."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_citations(DOC_ID, WORK_WITH_CITATIONS)


# ---- write_authorships ----


WORK_WITH_AUTHORS: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "authorships": [
        {
            "author_position": "first",
            "author": {
                "id": "https://openalex.org/A1",
                "display_name": "Alice",
            },
            "is_corresponding": True,
        },
        {
            "author_position": "last",
            "author": {
                "id": "https://openalex.org/A2",
                "display_name": "Bob",
            },
            "is_corresponding": False,
        },
    ],
}


async def test_write_authorships_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.write_authorships("", WORK_WITH_AUTHORS)


async def test_write_authorships_two_round_trips():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    ok = await client.write_authorships(DOC_ID, WORK_WITH_AUTHORS)
    assert ok is None  # success: returns None (typed-errors contract)
    # Round trips: author nodes UNWIND-MERGE, AUTHORED edges UNWIND.
    assert len(driver.sessions[0].calls) == 2


async def test_write_authorships_skips_authors_without_id():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {
                "author_position": "first",
                "author": {},
                "is_corresponding": False,
            },
            {
                "author_position": "last",
                "author": {"id": "https://openalex.org/A2"},
                "is_corresponding": False,
            },
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_authorships(DOC_ID, work)
    _, author_params = driver.sessions[0].calls[0]
    assert len(author_params["authorships"]) == 1
    assert author_params["authorships"][0]["openalex_id"] == "A2"


async def test_write_authorships_returns_true_with_no_authorships():
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "authorships": []}
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_authorships(DOC_ID, work) is None  # success: returns None
    # No session opened when there's nothing to write.
    assert len(driver.sessions) == 0


async def test_write_authorships_edge_properties_propagate():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_authorships(DOC_ID, WORK_WITH_AUTHORS)
    _, edge_params = driver.sessions[0].calls[1]
    authorships = edge_params["authorships"]
    assert authorships[0]["position"] == "first"
    assert authorships[0]["is_corresponding"] is True
    assert authorships[1]["position"] == "last"
    assert authorships[1]["is_corresponding"] is False


async def test_write_authorships_edges_match_focal_by_doc_id():
    """The :AUTHORED edge query MATCHes the focal by doc_id."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_authorships(DOC_ID, WORK_WITH_AUTHORS)
    edge_query, edge_params = driver.sessions[0].calls[1]
    assert "MATCH (p:Document {doc_id: $doc_id})" in edge_query
    assert edge_params["doc_id"] == DOC_ID


async def test_write_authorships_is_corresponding_defaults_to_false():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {
                "author_position": "first",
                "author": {"id": "https://openalex.org/A1"},
                # is_corresponding key missing on purpose.
            },
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_authorships(DOC_ID, work)
    _, author_params = driver.sessions[0].calls[0]
    assert author_params["authorships"][0]["is_corresponding"] is False


async def test_write_authorships_propagates_cypher_exception():
    """Cypher failures propagate; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_authorships(DOC_ID, WORK_WITH_AUTHORS)


# ---- write_venue ----


WORK_WITH_VENUE: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S137773608",
            "display_name": "Nature",
            "type": "journal",
            "issn_l": "0028-0836",
        },
    },
}


async def test_write_venue_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.write_venue("", WORK_WITH_VENUE)
    assert len(driver.sessions) == 0


async def test_write_venue_no_primary_location_returns_true_without_writes():
    """Work payload without primary_location -> no-op success."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1"}
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_venue(DOC_ID, work) is None  # success: returns None
    assert len(driver.sessions) == 0


async def test_write_venue_no_source_returns_true_without_writes():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "primary_location": {},
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_venue(DOC_ID, work) is None  # success: returns None
    assert len(driver.sessions) == 0


async def test_write_venue_no_openalex_id_returns_true_without_writes():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "primary_location": {"source": {"display_name": "Some Venue"}},
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_venue(DOC_ID, work) is None  # success: returns None
    assert len(driver.sessions) == 0


async def test_write_venue_one_round_trip():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_venue(DOC_ID, WORK_WITH_VENUE) is None  # success: returns None
    assert len(driver.sessions[0].calls) == 1


async def test_write_venue_extracts_bare_openalex_id():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_venue(DOC_ID, WORK_WITH_VENUE)
    _, params = driver.sessions[0].calls[0]
    assert params["openalex_id"] == "S137773608"


async def test_write_venue_passes_name_type_issn():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_venue(DOC_ID, WORK_WITH_VENUE)
    _, params = driver.sessions[0].calls[0]
    assert params["name"] == "Nature"
    assert params["venue_type"] == "journal"
    assert params["issn"] == "0028-0836"


async def test_write_venue_matches_focal_by_doc_id():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_venue(DOC_ID, WORK_WITH_VENUE)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert params["doc_id"] == DOC_ID


async def test_write_venue_propagates_cypher_exception():
    """Cypher failures propagate; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_venue(DOC_ID, WORK_WITH_VENUE)


# ---- write_topics ----


WORK_WITH_TOPICS: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "topics": [
        {
            "id": "https://openalex.org/T11537",
            "display_name": "Vitamin D and Bone Health",
            "score": 0.97,
        },
        {
            "id": "https://openalex.org/T12086",
            "display_name": "Calcium Metabolism Disorders",
            "score": 0.71,
        },
    ],
}


async def test_write_topics_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.write_topics("", WORK_WITH_TOPICS)


async def test_write_topics_empty_topics_returns_true_without_writes():
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "topics": []}
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_topics(DOC_ID, work) is None  # success: returns None
    assert len(driver.sessions) == 0


async def test_write_topics_two_round_trips():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_topics(DOC_ID, WORK_WITH_TOPICS) is None  # success: returns None
    # Round trips: topic nodes UNWIND-MERGE, ABOUT_TOPIC edges UNWIND.
    assert len(driver.sessions[0].calls) == 2


async def test_write_topics_extracts_bare_openalex_ids():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_topics(DOC_ID, WORK_WITH_TOPICS)
    _, params = driver.sessions[0].calls[0]
    assert [t["openalex_id"] for t in params["topics"]] == [
        "T11537",
        "T12086",
    ]


async def test_write_topics_score_propagates_to_edge_property():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_topics(DOC_ID, WORK_WITH_TOPICS)
    edge_query, edge_params = driver.sessions[0].calls[1]
    topics = edge_params["topics"]
    assert topics[0]["score"] == 0.97
    assert topics[1]["score"] == 0.71
    assert "SET r.score = t.score" in edge_query


async def test_write_topics_skips_topics_without_id():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "topics": [
            {"display_name": "Has no id", "score": 0.5},
            {
                "id": "https://openalex.org/T12086",
                "display_name": "Calcium",
                "score": 0.71,
            },
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_topics(DOC_ID, work)
    _, params = driver.sessions[0].calls[0]
    assert len(params["topics"]) == 1
    assert params["topics"][0]["openalex_id"] == "T12086"


async def test_write_topics_edges_match_focal_by_doc_id():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_topics(DOC_ID, WORK_WITH_TOPICS)
    edge_query, edge_params = driver.sessions[0].calls[1]
    assert "MATCH (d:Document {doc_id: $doc_id})" in edge_query
    assert edge_params["doc_id"] == DOC_ID


async def test_write_topics_propagates_cypher_exception():
    """Cypher failures propagate; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_topics(DOC_ID, WORK_WITH_TOPICS)


# ---- delete_doc (L1-L4 wipe + GC orphans) ----


async def test_delete_doc_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.delete_doc("")
    assert len(driver.sessions) == 0


async def test_delete_doc_runs_focal_wipe_plus_four_gc_queries():
    """One DETACH DELETE on the focal + 4 orphan GC queries."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.delete_doc(DOC_ID) is None  # success: returns None
    # 1 focal wipe + 4 GC queries (authors, topics, venues, shadow docs).
    assert len(driver.sessions[0].calls) == 5


async def test_delete_doc_focal_query_matches_by_doc_id():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc(DOC_ID)
    focal_query, focal_params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document {doc_id: $doc_id})" in focal_query
    assert "DETACH DELETE d" in focal_query
    assert focal_params == {"doc_id": DOC_ID}


async def test_delete_doc_gc_authors_uses_authored_pattern():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc(DOC_ID)
    query, _ = driver.sessions[0].calls[1]
    assert "MATCH (a:Author)" in query
    assert "NOT (a)-[:AUTHORED]->()" in query
    assert "DELETE a" in query


async def test_delete_doc_gc_topics_uses_about_topic_pattern():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc(DOC_ID)
    query, _ = driver.sessions[0].calls[2]
    assert "MATCH (t:Topic)" in query
    assert "NOT ()-[:ABOUT_TOPIC]->(t)" in query


async def test_delete_doc_gc_venues_uses_published_in_pattern():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc(DOC_ID)
    query, _ = driver.sessions[0].calls[3]
    assert "MATCH (v:Venue)" in query
    assert "NOT ()-[:PUBLISHED_IN]->(v)" in query


async def test_delete_doc_gc_shadow_documents_filters_in_corpus_false():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc(DOC_ID)
    query, _ = driver.sessions[0].calls[4]
    assert "MATCH (d:Document)" in query
    assert "in_corpus = false" in query
    assert "NOT ()-[:CITES]->(d)" in query


async def test_delete_doc_propagates_cypher_exception():
    """Cypher failures propagate; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.delete_doc(DOC_ID)


# ---- delete_doc_l1_l4_edges (surgical wipe for resolve_openalex) ----


async def test_delete_doc_l1_l4_edges_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.delete_doc_l1_l4_edges("")
    assert len(driver.sessions) == 0


async def test_delete_doc_l1_l4_edges_runs_four_edge_wipes_plus_four_gcs():
    """4 targeted edge-only DELETEs + 4 orphan GC queries = 8 calls."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.delete_doc_l1_l4_edges(DOC_ID) is None  # success: returns None
    assert len(driver.sessions[0].calls) == 8


async def test_delete_doc_l1_l4_edges_does_not_detach_focal_document():
    """No DETACH DELETE on the focal :Document - that would orphan chunks."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    for query, _ in driver.sessions[0].calls:
        assert "DETACH DELETE d" not in query, (
            "delete_doc_l1_l4_edges must NOT DETACH DELETE the focal"
        )


async def test_delete_doc_l1_l4_edges_wipes_cites_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "-[r:CITES]->()" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_wipes_authored_incoming():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    query, params = driver.sessions[0].calls[1]
    assert "[r:AUTHORED]->" in query
    assert "(d:Document {doc_id: $doc_id})" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_wipes_published_in_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    query, _ = driver.sessions[0].calls[2]
    assert "-[r:PUBLISHED_IN]->()" in query
    assert "DELETE r" in query


async def test_delete_doc_l1_l4_edges_wipes_about_topic_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    query, _ = driver.sessions[0].calls[3]
    assert "-[r:ABOUT_TOPIC]->()" in query
    assert "DELETE r" in query


async def test_delete_doc_l1_l4_edges_gc_queries_mirror_delete_doc():
    """The four GC queries (authors / topics / venues / shadow docs) must
    match `delete_doc`'s GC so the post-state guarantee is identical."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_doc_l1_l4_edges(DOC_ID)
    gc_queries = [c[0] for c in driver.sessions[0].calls[4:8]]
    assert any("MATCH (a:Author)" in q and "NOT (a)-[:AUTHORED]->()" in q for q in gc_queries)
    assert any("MATCH (t:Topic)" in q and "NOT ()-[:ABOUT_TOPIC]->(t)" in q for q in gc_queries)
    assert any("MATCH (v:Venue)" in q and "NOT ()-[:PUBLISHED_IN]->(v)" in q for q in gc_queries)
    assert any("in_corpus = false" in q and "NOT ()-[:CITES]->(d)" in q for q in gc_queries)


async def test_delete_doc_l1_l4_edges_propagates_cypher_exception():
    """Cypher failures propagate; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.delete_doc_l1_l4_edges(DOC_ID)


# ---- delete_chunks_by_doc_id (L5) ----


async def test_delete_chunks_by_doc_id_no_doc_id_raises():
    """Empty doc_id is a programmer error — raise ValueError, never run."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.delete_chunks_by_doc_id("")
    assert len(driver.sessions) == 0


async def test_delete_chunks_by_doc_id_one_round_trip():
    """Success is signalled by returning None (no exception)."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.delete_chunks_by_doc_id(DOC_ID) is None
    assert len(driver.sessions[0].calls) == 1


async def test_delete_chunks_by_doc_id_uses_chunk_label_with_doc_id_filter():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.delete_chunks_by_doc_id(DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (c:Chunk {doc_id: $doc_id})" in query
    assert "DETACH DELETE c" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_chunks_by_doc_id_propagates_cypher_exception():
    """Cypher failures propagate to the orchestrator boundary, which
    catches and stores on the matching `_error: ErrorDetail | None`
    result field."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.delete_chunks_by_doc_id(DOC_ID)


# ---- write_chunks (L5) ----


@dataclass
class FakeChunk:
    """Duck-typed stand-in for ParsedChunk used by write_chunks tests."""

    chunk_index: int
    section: str | None
    page: int | None
    content_type: str = "text"


async def test_write_chunks_no_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await client.write_chunks("", [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    assert len(driver.sessions) == 0


async def test_write_chunks_unknown_main_label_raises():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(ValueError, match="invalid main_label"):
        await client.write_chunks(DOC_ID, [FakeChunk(0, "Intro", 1)], "NotALabel", "Paper")
    assert len(driver.sessions) == 0


async def test_write_chunks_empty_chunks_is_noop_returning_none():
    """Empty chunks list is a documented success no-op: returns None,
    no Cypher dispatched."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.write_chunks(DOC_ID, [], "Document", "Paper") is None
    assert len(driver.sessions) == 0


async def test_write_chunks_one_round_trip():
    """Success returns None; one UNWIND batch covers all chunks."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    chunks = [FakeChunk(0, "Intro", 1), FakeChunk(1, "Methods", 2)]
    assert await client.write_chunks(DOC_ID, chunks, "Document", "Paper") is None
    # UNWIND batches all chunks in one query.
    assert len(driver.sessions[0].calls) == 1


async def test_write_chunks_chunk_ids_use_make_chunk_id_format():
    """chunk_id = '{doc_id}#{chunk_index}' via ingestion.ids.make_chunk_id."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    chunks = [FakeChunk(0, "Intro", 1), FakeChunk(2, "Methods", 3)]
    await client.write_chunks(DOC_ID, chunks, "Document", "Paper")
    _, params = driver.sessions[0].calls[0]
    assert params["chunks"][0]["chunk_id"] == f"{DOC_ID}#0"
    assert params["chunks"][1]["chunk_id"] == f"{DOC_ID}#2"


async def test_write_chunks_passes_all_chunk_properties():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    chunks = [
        FakeChunk(0, "Intro", 1, content_type="text"),
        FakeChunk(1, None, None, content_type="figure"),
    ]
    await client.write_chunks(DOC_ID, chunks, "Document", "Paper")
    _, params = driver.sessions[0].calls[0]
    assert params["doc_id"] == DOC_ID
    assert params["chunks"][0]["chunk_index"] == 0
    assert params["chunks"][0]["section"] == "Intro"
    assert params["chunks"][0]["page"] == 1
    assert params["chunks"][0]["content_type"] == "text"
    assert params["chunks"][1]["section"] is None
    assert params["chunks"][1]["page"] is None
    assert params["chunks"][1]["content_type"] == "figure"


async def test_write_chunks_creates_focal_via_merge_with_in_corpus_on_create():
    """Focal :Document is MERGEd; in_corpus=true is SET on create only -
    so a doc ingested without OpenAlex still has a focal node."""
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_chunks(DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    query, _ = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in query
    assert "ON CREATE SET d.in_corpus = true" in query


async def test_write_chunks_unwind_merges_chunks_and_part_of_edges():
    driver = RecordingDriver()
    client = _client_with_driver(_configured_settings(), driver)
    await client.write_chunks(DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")
    query, _ = driver.sessions[0].calls[0]
    assert "UNWIND $chunks AS ch" in query
    assert "MERGE (c:Chunk {chunk_id: ch.chunk_id})" in query
    assert "MERGE (c)-[:PART_OF]->(d)" in query


async def test_write_chunks_propagates_cypher_exception():
    """Cypher failures propagate to the orchestrator boundary, which
    catches and stores on `IngestResult.kg_chunks_error`."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(_configured_settings(), driver)
    with pytest.raises(RuntimeError, match="boom"):
        await client.write_chunks(DOC_ID, [FakeChunk(0, "Intro", 1)], "Document", "Paper")


# ---- read_query ----


async def test_read_query_uses_execute_read_with_provided_cypher_and_params():
    """The Cypher + params must reach the inner tx.run via execute_read."""
    tx = RecordingTransaction(records=[])
    driver = RecordingDriver(read_transaction=tx)
    client = _client_with_driver(_configured_settings(), driver)
    await client.read_query("MATCH (n) RETURN n", x=1, y="two")
    assert tx.calls == [("MATCH (n) RETURN n", {"x": 1, "y": "two"})]


async def test_read_query_returns_records_as_dicts():
    """Record.data() output is what flows back from read_query."""
    tx = RecordingTransaction(
        records=[
            RecordingRecord(_data={"name": "Alice", "papers": 5}),
            RecordingRecord(_data={"name": "Bob", "papers": 3}),
        ]
    )
    driver = RecordingDriver(read_transaction=tx)
    client = _client_with_driver(_configured_settings(), driver)
    result = await client.read_query("MATCH (a:Author) RETURN a")
    assert result == [
        {"name": "Alice", "papers": 5},
        {"name": "Bob", "papers": 3},
    ]


async def test_read_query_no_records_returns_empty_list():
    tx = RecordingTransaction(records=[])
    driver = RecordingDriver(read_transaction=tx)
    client = _client_with_driver(_configured_settings(), driver)
    assert await client.read_query("MATCH (n) WHERE false RETURN n") == []


async def test_read_query_propagates_exception_does_not_fail_soft():
    """Per docstring: read_query raises on Cypher / connection failures.
    The fail-soft handler lives one level up in neo4j_retriever_node."""
    tx = RecordingTransaction(raise_on_run=RuntimeError("syntax error"))
    driver = RecordingDriver(read_transaction=tx)
    client = _client_with_driver(_configured_settings(), driver)
    try:
        await client.read_query("INVALID CYPHER")
    except RuntimeError as exc:
        assert "syntax error" in str(exc)
    else:
        raise AssertionError("read_query must propagate the exception")
