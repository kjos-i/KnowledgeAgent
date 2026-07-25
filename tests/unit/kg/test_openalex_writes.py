"""Unit tests for kg.openalex_writes - the L1-L4 OpenAlex-derived writes.

Exercises the module-level write functions directly (not through the
`Neo4jClient` wrapper methods, which are covered in `test_kg_client.py`)
using the same RecordingDriver harness as `test_entity_writes.py` /
`test_kg_client.py`, so the Cypher strings + parameters sent to
`session.run()` are captured without touching a real Neo4j instance.

This mirrors the integration suite (`tests/integration/kg/test_openalex_writes.py`)
at the unit level: same L1-L4 scenarios + the two delete primitives, but
asserting the *issued Cypher/params* and no-op behaviour instead of the
post-write graph state.

We verify, per function:
  - Typed-errors contract: empty `doc_id` raises ValueError before any
    session opens; driver exceptions propagate; success returns None.
  - No-op success: nothing-to-write payloads return None WITHOUT opening
    a session (or, for write_citations, without the 2nd/3rd round trip).
  - Cypher shape: correct labels, relationships, MERGE/keys, edge props.
  - Param payloads: bare-ID extraction, DOI normalisation, defensive
    filtering of entries missing an OpenAlex id, dedup/MERGE idempotency.

The string helpers `_extract_openalex_id` / `_clean_doi_for_storage` have
their own coverage in `test_kg_client.py`; this file focuses on the
Cypher-issuing write logic.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import openalex_writes
from knowledge_agent.kg.client import Neo4jClient

# ---- Test harness (mirrors test_entity_writes.py / test_kg_client.py) ----


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None

    async def run(self, query: str, **params: Any) -> None:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))

    async def __aenter__(self) -> "RecordingSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    closed: bool = False

    def session(self) -> RecordingSession:
        sess = RecordingSession(raise_on_run=self.raise_on_run)
        self.sessions.append(sess)
        return sess

    async def close(self) -> None:
        self.closed = True


def _configured_settings() -> Settings:
    return Settings(
        anthropic_api_key="fake-anthropic-key",  # pragma: allowlist secret
        voyage_api_key="fake-voyage-key",  # pragma: allowlist secret
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="fake-neo4j-password",  # pragma: allowlist secret
        lancedb_path="/tmp/fake-lancedb-path",
    )


def _client_with_driver(driver: RecordingDriver) -> Neo4jClient:
    client = Neo4jClient(settings=_configured_settings())
    client._driver = driver  # type: ignore[assignment]
    return client


DOC_ID = "test-doc-openalex"


# ---- Shared synthetic OpenAlex-shaped payloads ----


WORK_WITH_CITATIONS: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1/ABC",
    "referenced_works": [
        "https://openalex.org/W100",
        "https://openalex.org/W101",
    ],
}

WORK_WITH_AUTHORS: dict[str, Any] = {
    "id": "https://openalex.org/W1",
    "authorships": [
        {
            "author_position": "first",
            "author": {"id": "https://openalex.org/A1", "display_name": "Alice"},
            "is_corresponding": True,
        },
        {
            "author_position": "last",
            "author": {"id": "https://openalex.org/A2", "display_name": "Bob"},
            "is_corresponding": False,
        },
    ],
}

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


# ===========================================================================
# write_citations (L1)
# ===========================================================================


async def test_write_citations_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.write_citations(client, "", WORK_WITH_CITATIONS)
    # Bailed before touching the driver.
    assert driver.sessions == []


async def test_write_citations_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches + stores
    on `IngestResult.kg_citations_error`."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)


async def test_write_citations_full_path_three_round_trips():
    """Focal MERGE + shadow UNWIND-MERGE + CITES edges = 3 calls, one session."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS) is None
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 3


async def test_write_citations_focal_merges_by_doc_id_and_upgrades_shadow():
    """openalex_id known -> focal MERGEs by doc_id (never clobbers doc_id or
    collapses two files), absorbs any same-openalex_id shadow (re-point its
    :CITES onto the focal, delete it), and claims openalex_id only when no
    other :Document holds it. SETs :Paper + in_corpus + normalised doi."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)
    query, params = driver.sessions[0].calls[0]
    # Focal keyed by doc_id (the per-file identity), NOT by openalex_id.
    assert "MERGE (d:Document {doc_id: $doc_id})" in query
    assert "MERGE (d:Document {openalex_id: $openalex_id})" not in query
    # doc_id is the MERGE key now - never SET (that was the clobber bug).
    assert "d.doc_id = $doc_id" not in query
    assert "d:Paper" in query
    assert "d.in_corpus = true" in query
    assert "d.doi = $doi" in query
    # Shadow absorb: re-point incoming :CITES onto the focal, delete shadow.
    assert "MATCH (shadow:Document {openalex_id: $openalex_id})" in query
    assert "shadow.in_corpus = false" in query
    assert "MERGE (c)-[:CITES]->(d)" in query
    assert "DETACH DELETE shadow" in query
    # Claim openalex_id only when no OTHER node holds it (UNIQUE-safe).
    assert "NOT EXISTS" in query
    assert "SET d.openalex_id = $openalex_id" in query
    assert params["openalex_id"] == "W1"
    assert params["doc_id"] == DOC_ID
    # DOI normalised: URL prefix stripped + lowercased.
    assert params["doi"] == "10.1/abc"


async def test_write_citations_focal_keyed_on_doc_id_when_no_openalex_id():
    """No extractable openalex_id (non-scholarly doc) -> focal MERGE keys on
    doc_id; no openalex_id param, but :Paper label still applied."""
    work: dict[str, Any] = {"doi": "10.1/abc"}  # no `id` field
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    query, params = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in query
    assert "d:Paper" in query
    assert "openalex_id" not in params
    assert params["doc_id"] == DOC_ID
    assert params["doi"] == "10.1/abc"


async def test_write_citations_no_doi_skips_doi_clause_with_openalex_id():
    work = dict(WORK_WITH_CITATIONS)
    work["doi"] = None
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    query, params = driver.sessions[0].calls[0]
    assert "d.doi" not in query
    assert "doi" not in params


async def test_write_citations_no_doi_skips_doi_clause_without_openalex_id():
    """The doc_id-keyed branch also omits the doi clause + param when absent."""
    work: dict[str, Any] = {"referenced_works": []}  # no id, no doi
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    query, params = driver.sessions[0].calls[0]
    assert "MERGE (d:Document {doc_id: $doc_id})" in query
    assert "d.doi" not in query
    assert "doi" not in params


async def test_write_citations_no_references_single_round_trip():
    """No referenced_works -> only the focal write (no shadow, no edges)."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "doi": "10.1/abc"}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_citations(client, DOC_ID, work) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_citations_referenced_works_none_single_round_trip():
    """`referenced_works: None` is guarded by `or []` -> treated as none."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "referenced_works": None}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    assert len(driver.sessions[0].calls) == 1


async def test_write_citations_extracts_bare_reference_ids():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)
    _query, shadow_params = driver.sessions[0].calls[1]
    assert shadow_params["cited_ids"] == ["W100", "W101"]


async def test_write_citations_filters_empty_reference_ids():
    """Referenced entries that yield no bare id (None / empty) are dropped."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "referenced_works": [
            "https://openalex.org/W100",
            None,
            "",
            "https://openalex.org/W101",
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    _query, shadow_params = driver.sessions[0].calls[1]
    assert shadow_params["cited_ids"] == ["W100", "W101"]


async def test_write_citations_all_references_empty_single_round_trip():
    """Every reference filtered out -> behaves like the no-references case."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "referenced_works": [None, ""]}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, work)
    assert len(driver.sessions[0].calls) == 1


async def test_write_citations_shadow_nodes_merge_on_create_paper_shadow():
    """Shadow docs: MERGE on openalex_id, ON CREATE ONLY set :Paper +
    in_corpus=false (never downgrade an existing corpus doc)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)
    query, _params = driver.sessions[0].calls[1]
    assert "UNWIND $cited_ids AS cid" in query
    assert "MERGE (c:Document {openalex_id: cid})" in query
    assert "ON CREATE SET c:Paper" in query
    assert "c.in_corpus = false" in query


async def test_write_citations_cites_edge_matches_focal_by_doc_id():
    """CITES edge: focal matched by doc_id (SET in step 1), shadows by
    openalex_id; MERGE keeps it idempotent."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)
    query, params = driver.sessions[0].calls[2]
    assert "MATCH (p:Document {doc_id: $doc_id})" in query
    assert "MATCH (c:Document {openalex_id: cid})" in query
    assert "MERGE (p)-[:CITES]->(c)" in query
    assert params["doc_id"] == DOC_ID
    assert params["cited_ids"] == ["W100", "W101"]


async def test_write_citations_uses_merge_everywhere_for_idempotency():
    """Mirrors the integration idempotency test at the unit level: every
    round trip MERGEs (never CREATEs), so a re-run can't duplicate."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_citations(client, DOC_ID, WORK_WITH_CITATIONS)
    for query, _params in driver.sessions[0].calls:
        assert "MERGE" in query
        assert "CREATE (" not in query  # no bare node CREATE


# ===========================================================================
# write_authorships (L2)
# ===========================================================================


async def test_write_authorships_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.write_authorships(client, "", WORK_WITH_AUTHORS)
    assert driver.sessions == []


async def test_write_authorships_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.write_authorships(client, DOC_ID, WORK_WITH_AUTHORS)


async def test_write_authorships_full_path_two_round_trips():
    """Author nodes UNWIND-MERGE + AUTHORED edges UNWIND = 2 calls."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_authorships(client, DOC_ID, WORK_WITH_AUTHORS) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_authorships_empty_list_is_noop():
    """No authorships -> no session opened; success no-op (None)."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "authorships": []}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_authorships(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_authorships_none_is_noop():
    """`authorships: None` is guarded by `or []` -> no-op, no session."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "authorships": None}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_authorships(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_authorships_skips_author_without_id():
    """An authorship whose author has no id is dropped (defensive)."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {"author_position": "first", "author": {}, "is_corresponding": False},
            {
                "author_position": "last",
                "author": {"id": "https://openalex.org/A2"},
                "is_corresponding": False,
            },
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, work)
    _query, params = driver.sessions[0].calls[0]
    assert len(params["authorships"]) == 1
    assert params["authorships"][0]["openalex_id"] == "A2"


async def test_write_authorships_skips_missing_author_key():
    """An authorship with no `author` key at all is dropped (au.get -> {})."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {"author_position": "first"},  # no `author`
            {
                "author_position": "last",
                "author": {"id": "https://openalex.org/A2"},
            },
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, work)
    _query, params = driver.sessions[0].calls[0]
    assert [a["openalex_id"] for a in params["authorships"]] == ["A2"]


async def test_write_authorships_all_skipped_is_noop():
    """When every authorship lacks a usable id, it's a no-op (no session)."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [{"author": {}}, {"author_position": "middle"}],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_authorships(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_authorships_node_merge_sets_display_name():
    """Author nodes MERGE by openalex_id + unconditional SET display_name
    (so an OpenAlex name correction wins on re-ingest)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, WORK_WITH_AUTHORS)
    query, params = driver.sessions[0].calls[0]
    assert "UNWIND $authorships AS a" in query
    assert "MERGE (au:Author {openalex_id: a.openalex_id})" in query
    assert "SET au.display_name = a.display_name" in query
    assert [a["display_name"] for a in params["authorships"]] == ["Alice", "Bob"]


async def test_write_authorships_extracts_bare_author_ids():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, WORK_WITH_AUTHORS)
    _query, params = driver.sessions[0].calls[0]
    assert [a["openalex_id"] for a in params["authorships"]] == ["A1", "A2"]


async def test_write_authorships_edge_carries_position_and_is_corresponding():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, WORK_WITH_AUTHORS)
    query, params = driver.sessions[0].calls[1]
    assert "MATCH (p:Document {doc_id: $doc_id})" in query
    assert "MERGE (au)-[r:AUTHORED]->(p)" in query
    assert "r.position = a.position" in query
    assert "r.is_corresponding = a.is_corresponding" in query
    assert params["doc_id"] == DOC_ID
    authorships = params["authorships"]
    assert authorships[0]["position"] == "first"
    assert authorships[0]["is_corresponding"] is True
    assert authorships[1]["position"] == "last"
    assert authorships[1]["is_corresponding"] is False


async def test_write_authorships_is_corresponding_defaults_to_false():
    """Missing `is_corresponding` coerces to False via `bool(None)`."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "authorships": [
            {"author_position": "first", "author": {"id": "https://openalex.org/A1"}},
        ],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_authorships(client, DOC_ID, work)
    _query, params = driver.sessions[0].calls[0]
    assert params["authorships"][0]["is_corresponding"] is False


# ===========================================================================
# write_venue (L3)
# ===========================================================================


async def test_write_venue_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.write_venue(client, "", WORK_WITH_VENUE)
    assert driver.sessions == []


async def test_write_venue_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.write_venue(client, DOC_ID, WORK_WITH_VENUE)


async def test_write_venue_no_primary_location_is_noop():
    work: dict[str, Any] = {"id": "https://openalex.org/W1"}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_venue_primary_location_none_is_noop():
    """`primary_location: None` is guarded by `or {}` -> no-op."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "primary_location": None}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_venue_no_source_is_noop():
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "primary_location": {}}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_venue_source_none_is_noop():
    """`source: None` is guarded by `or {}` -> no-op."""
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "primary_location": {"source": None},
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_venue_no_openalex_id_is_noop():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "primary_location": {"source": {"display_name": "Some Venue"}},
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_venue_full_path_one_round_trip():
    """MATCH focal + MERGE venue + SET props + MERGE edge, all in one call."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_venue(client, DOC_ID, WORK_WITH_VENUE) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_venue_cypher_shape_and_params():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_venue(client, DOC_ID, WORK_WITH_VENUE)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "MERGE (v:Venue {openalex_id: $openalex_id})" in query
    assert "MERGE (d)-[:PUBLISHED_IN]->(v)" in query
    # Bare id extracted; name/type/issn threaded as params.
    assert params["doc_id"] == DOC_ID
    assert params["openalex_id"] == "S137773608"
    assert params["name"] == "Nature"
    assert params["venue_type"] == "journal"
    assert params["issn"] == "0028-0836"


# ===========================================================================
# write_topics (L4)
# ===========================================================================


async def test_write_topics_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.write_topics(client, "", WORK_WITH_TOPICS)
    assert driver.sessions == []


async def test_write_topics_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.write_topics(client, DOC_ID, WORK_WITH_TOPICS)


async def test_write_topics_empty_list_is_noop():
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "topics": []}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_topics(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_topics_none_is_noop():
    """`topics: None` is guarded by `or []` -> no-op."""
    work: dict[str, Any] = {"id": "https://openalex.org/W1", "topics": None}
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_topics(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_topics_full_path_two_round_trips():
    """Topic nodes UNWIND-MERGE + ABOUT_TOPIC edges UNWIND = 2 calls."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_topics(client, DOC_ID, WORK_WITH_TOPICS) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_topics_skips_topic_without_id():
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
    client = _client_with_driver(driver)
    await openalex_writes.write_topics(client, DOC_ID, work)
    _query, params = driver.sessions[0].calls[0]
    assert len(params["topics"]) == 1
    assert params["topics"][0]["openalex_id"] == "T12086"


async def test_write_topics_all_skipped_is_noop():
    work: dict[str, Any] = {
        "id": "https://openalex.org/W1",
        "topics": [{"display_name": "no id", "score": 0.5}],
    }
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.write_topics(client, DOC_ID, work) is None
    assert driver.sessions == []


async def test_write_topics_node_merge_sets_display_name_and_bare_ids():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_topics(client, DOC_ID, WORK_WITH_TOPICS)
    query, params = driver.sessions[0].calls[0]
    assert "UNWIND $topics AS t" in query
    assert "MERGE (n:Topic {openalex_id: t.openalex_id})" in query
    assert "SET n.display_name = t.display_name" in query
    assert [t["openalex_id"] for t in params["topics"]] == ["T11537", "T12086"]


async def test_write_topics_edge_carries_score_and_matches_focal():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.write_topics(client, DOC_ID, WORK_WITH_TOPICS)
    query, params = driver.sessions[0].calls[1]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "MERGE (d)-[r:ABOUT_TOPIC]->(n)" in query
    assert "SET r.score = t.score" in query
    assert params["doc_id"] == DOC_ID
    assert [t["score"] for t in params["topics"]] == [0.97, 0.71]


# ===========================================================================
# delete_doc (full L1-L4 wipe + GC orphans)
# ===========================================================================


async def test_delete_doc_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.delete_doc(client, "")
    assert driver.sessions == []


async def test_delete_doc_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.delete_doc(client, DOC_ID)


async def test_delete_doc_five_calls_focal_wipe_plus_four_gc():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.delete_doc(client, DOC_ID) is None
    # 1 focal DETACH DELETE + 4 GC (authors, topics, venues, shadow docs).
    assert len(driver.sessions[0].calls) == 5


async def test_delete_doc_focal_detach_delete_by_doc_id():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document|Artifact {doc_id: $doc_id})" in query
    assert "DETACH DELETE d" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_focal_wipe_covers_document_and_artifact():
    """Regression: the focal DETACH DELETE must match BOTH :Document and
    :Artifact, not :Document alone.

    An artifact's focal node is labelled :Artifact (write_chunks does
    MERGE (d:<main_label> ...) with main_label in {Document, Artifact}).
    A :Document-only focal wipe silently skipped artifact focal nodes,
    leaving the node AND its incident L9 :RELATED_TO / L10
    :RELATED_BY_XREF edges (which hang off the focal) behind after a
    delete. The label disjunction ensures artifacts are deleted too.
    """
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    focal_query, _params = driver.sessions[0].calls[0]
    assert "d:Document|Artifact" in focal_query
    assert "DETACH DELETE d" in focal_query


async def test_delete_doc_gc_authors():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    query, params = driver.sessions[0].calls[1]
    assert "MATCH (a:Author)" in query
    assert "NOT (a)-[:AUTHORED]->()" in query
    assert "DELETE a" in query
    # GC is global -> no doc_id param.
    assert params == {}


async def test_delete_doc_gc_topics():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    query, params = driver.sessions[0].calls[2]
    assert "MATCH (t:Topic)" in query
    assert "NOT ()-[:ABOUT_TOPIC]->(t)" in query
    assert "DELETE t" in query
    assert params == {}


async def test_delete_doc_gc_venues():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    query, params = driver.sessions[0].calls[3]
    assert "MATCH (v:Venue)" in query
    assert "NOT ()-[:PUBLISHED_IN]->(v)" in query
    assert "DELETE v" in query
    assert params == {}


async def test_delete_doc_gc_shadow_documents():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc(client, DOC_ID)
    query, params = driver.sessions[0].calls[4]
    assert "MATCH (d:Document)" in query
    assert "in_corpus = false" in query
    assert "NOT ()-[:CITES]->(d)" in query
    assert "DELETE d" in query
    assert params == {}


# ===========================================================================
# delete_doc_l1_l4_edges (surgical edges-only wipe for resolve_openalex)
# ===========================================================================


async def test_delete_doc_l1_l4_edges_empty_doc_id_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="no doc_id"):
        await openalex_writes.delete_doc_l1_l4_edges(client, "")
    assert driver.sessions == []


async def test_delete_doc_l1_l4_edges_propagates_cypher_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)


async def test_delete_doc_l1_l4_edges_eight_calls():
    """4 targeted edge-only DELETEs + 4 orphan GC queries = 8 calls."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID) is None
    assert len(driver.sessions[0].calls) == 8


async def test_delete_doc_l1_l4_edges_never_detaches_focal():
    """No DETACH DELETE on the focal :Document - that would orphan chunks."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    for query, _params in driver.sessions[0].calls:
        assert "DETACH DELETE d" not in query


async def test_delete_doc_l1_l4_edges_wipes_cites_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    query, params = driver.sessions[0].calls[0]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "-[r:CITES]->()" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_wipes_authored_incoming():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    query, params = driver.sessions[0].calls[1]
    assert "()-[r:AUTHORED]->(d:Document {doc_id: $doc_id})" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_wipes_published_in_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    query, params = driver.sessions[0].calls[2]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "-[r:PUBLISHED_IN]->()" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_wipes_about_topic_outgoing():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    query, params = driver.sessions[0].calls[3]
    assert "MATCH (d:Document {doc_id: $doc_id})" in query
    assert "-[r:ABOUT_TOPIC]->()" in query
    assert "DELETE r" in query
    assert params == {"doc_id": DOC_ID}


async def test_delete_doc_l1_l4_edges_gc_mirrors_delete_doc():
    """The four GC queries (authors / topics / venues / shadow docs) match
    `delete_doc`'s GC so the post-state guarantee is identical."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    await openalex_writes.delete_doc_l1_l4_edges(client, DOC_ID)
    gc_queries = [c[0] for c in driver.sessions[0].calls[4:8]]
    assert any("MATCH (a:Author)" in q and "NOT (a)-[:AUTHORED]->()" in q for q in gc_queries)
    assert any("MATCH (t:Topic)" in q and "NOT ()-[:ABOUT_TOPIC]->(t)" in q for q in gc_queries)
    assert any("MATCH (v:Venue)" in q and "NOT ()-[:PUBLISHED_IN]->(v)" in q for q in gc_queries)
    assert any("in_corpus = false" in q and "NOT ()-[:CITES]->(d)" in q for q in gc_queries)
