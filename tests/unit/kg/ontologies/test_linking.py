"""Tests for kg.ontologies.linking - the shared L7 linking pass.

Four concerns covered:
  1. **Registry** — ONTOLOGY_REGISTRY has the right shape per known
     ontology.
  2. **Matcher** — exact / fuzzy strategies, variant generation,
     case-insensitive lookup, multi-hit handling.
  3. **Index build + entity fetch** — Cypher queries shape correctly,
     unconfigured-client and exception fail-soft paths.
  4. **End-to-end link_entities** + the import-orchestration
     `ensure_ontology_imported` glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontologies import linking

# ---- Test harness (mirrors test_ontology_mesh_writes / _go_writes) ----


@dataclass
class _RecordingResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    async def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    async def data(self) -> list[dict[str, Any]]:
        return self.rows


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
        pass


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


# ---- Registry ----


def test_registry_has_known_ontologies():
    """All shipped pronto OBO + MeSH ontologies are in the registry."""
    expected = {
        "mesh",
        "go",
        "hpo",
        "uberon",
        "mondo",
        "chebi",
        "eco",
        "so",
        "pr",
        "cl",
        "po",
        "foodon",
        "envo",
        "ncbitaxon",
        "obi",
        "efo",
        "dron",
        "fibo",
    }
    assert set(linking.ONTOLOGY_REGISTRY) == expected
    for name in expected:
        entry = linking.ONTOLOGY_REGISTRY[name]
        assert "term_label" in entry
        assert "is_imported_fn" in entry
        assert "import_fn" in entry
        assert "delete_fn" in entry
        assert "download_size_mb" in entry
        assert callable(entry["is_imported_fn"])
        assert callable(entry["import_fn"])
        assert callable(entry["delete_fn"])
        assert isinstance(entry["download_size_mb"], int)


def test_registry_mesh_points_at_meshterm_label():
    assert linking.ONTOLOGY_REGISTRY["mesh"]["term_label"] == "MeSHTerm"


def test_registry_go_points_at_goterm_label():
    assert linking.ONTOLOGY_REGISTRY["go"]["term_label"] == "GOTerm"


def test_registry_hpo_points_at_hpoterm_label():
    assert linking.ONTOLOGY_REGISTRY["hpo"]["term_label"] == "HPOTerm"


def test_registry_uberon_points_at_uberonterm_label():
    assert linking.ONTOLOGY_REGISTRY["uberon"]["term_label"] == "UBERONTerm"


def test_registry_mondo_points_at_mondoterm_label():
    assert linking.ONTOLOGY_REGISTRY["mondo"]["term_label"] == "MONDOTerm"


def test_registry_chebi_points_at_chebiterm_label():
    assert linking.ONTOLOGY_REGISTRY["chebi"]["term_label"] == "ChEBITerm"


# ---- count_ontology_terms ----


async def test_count_ontology_terms_unknown_ontology_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="unknown ontology"):
        await linking.count_ontology_terms(client, "nope")
    # No Cypher should have run.
    assert driver.sessions == []


async def test_count_ontology_terms_runs_count_cypher_against_term_label():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"n": 30142}])]])
    client = _client_with_driver(driver)
    assert await linking.count_ontology_terms(client, "mesh") == 30142
    query, _ = driver.sessions[0].calls[0]
    assert "MATCH (n:MeSHTerm)" in query
    assert "count(n)" in query


async def test_count_ontology_terms_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await linking.count_ontology_terms(client, "mesh")


# ---- count_canonical_links ----


async def test_count_canonical_links_unknown_ontology_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    with pytest.raises(ValueError, match="unknown ontology"):
        await linking.count_canonical_links(client, "nope")
    assert driver.sessions == []


async def test_count_canonical_links_runs_count_cypher_against_canonical_to():
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[{"n": 18}])]])
    client = _client_with_driver(driver)
    assert await linking.count_canonical_links(client, "mesh") == 18
    query, _ = driver.sessions[0].calls[0]
    assert "[r:CANONICAL_TO]->" in query
    assert "(:MeSHTerm)" in query
    assert "count(r)" in query


async def test_count_canonical_links_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await linking.count_canonical_links(client, "mesh")


# ---- _match_entity_key + _fuzzy_variants ----


def test_match_exact_hit_returns_term_id_with_none_confidence():
    index = {"diabetes mellitus": ["MESH:D003920"]}
    matches = linking._match_entity_key("diabetes mellitus", index, "exact")
    assert matches == [("MESH:D003920", None)]


def test_match_exact_miss_returns_empty_in_exact_mode():
    index = {"diabetes mellitus": ["MESH:D003920"]}
    matches = linking._match_entity_key("diabetes mellituses", index, "exact")
    assert matches == []


def test_match_exact_multi_hit_returns_all_term_ids():
    """One lowercased text mapping to multiple term IDs - the index
    holds them as a list and the matcher returns one tuple per ID."""
    index = {"stress": ["MESH:D013315", "GO:0006950"]}
    matches = linking._match_entity_key("stress", index, "exact")
    assert matches == [
        ("MESH:D013315", None),
        ("GO:0006950", None),
    ]


def test_match_fuzzy_tries_singular_strip_when_plural():
    """'cancers' (plural) -> matches 'cancer' in the index via -s strip."""
    index = {"cancer": ["MESH:D009369"]}
    matches = linking._match_entity_key("cancers", index, "fuzzy")
    assert len(matches) == 1
    assert matches[0][0] == "MESH:D009369"
    # Fuzzy match carries a confidence (heuristic plural flip = 0.9).
    assert matches[0][1] == 0.9


def test_match_fuzzy_tries_plural_add_when_singular():
    """'cardiovascular disease' -> matches 'cardiovascular diseases' via +s."""
    index = {"cardiovascular diseases": ["MESH:D002318"]}
    matches = linking._match_entity_key("cardiovascular disease", index, "fuzzy")
    assert matches == [("MESH:D002318", 0.9)]


def test_match_fuzzy_tries_hyphen_space_swap():
    """'non-alcoholic fatty liver' -> matches 'non alcoholic fatty liver'
    via hyphen->space."""
    index = {"non alcoholic fatty liver": ["MESH:D065626"]}
    matches = linking._match_entity_key("non-alcoholic fatty liver", index, "fuzzy")
    assert matches == [("MESH:D065626", 0.85)]


def test_match_fuzzy_exact_wins_before_variants():
    """Even when fuzzy variants would also hit, an exact match should
    return WITHOUT trying variants - confidence stays None."""
    index = {
        "cancer": ["MESH:D009369"],
        "cancers": ["OTHER:1"],
    }
    matches = linking._match_entity_key("cancer", index, "fuzzy")
    assert matches == [("MESH:D009369", None)]


def test_match_fuzzy_falls_through_all_variants_no_hits():
    """No variant hits -> empty result."""
    index = {"completely different": ["X:1"]}
    matches = linking._match_entity_key("foo", index, "fuzzy")
    assert matches == []


def test_fuzzy_variants_for_plural_word():
    variants = dict(linking._fuzzy_variants("diseases"))
    # -s strip should be the highest-confidence candidate.
    assert variants.get("disease") == 0.9


def test_fuzzy_variants_for_hyphenated_text():
    variants = dict(linking._fuzzy_variants("non-alcoholic"))
    # Hyphen -> space swap candidate.
    assert variants.get("non alcoholic") == 0.85


def test_fuzzy_variants_no_duplicates():
    """Calling _fuzzy_variants on a key that's both plural AND
    hyphenated shouldn't duplicate variants."""
    variants = linking._fuzzy_variants("non-alcoholic-diseases")
    keys = [v for v, _ in variants]
    assert len(keys) == len(set(keys))


# ---- _build_term_index ----


async def test_build_term_index_maps_labels_and_synonyms_to_ids():
    """Cypher returns one row per term; the index gets one entry per
    label + one per synonym, all keyed by lowercased text."""
    rows = [
        {
            "id": "MESH:D003920",
            "label": "Diabetes Mellitus",
            "synonyms": ["diabetes"],
        },
        {
            "id": "MESH:D003924",
            "label": "Diabetes Mellitus, Type 2",
            "synonyms": ["niddm", "type 2 diabetes"],
        },
    ]
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=rows)]])
    client = _client_with_driver(driver)
    index = await linking._build_term_index(client, "MeSHTerm")

    # Primary labels lowercased.
    assert index["diabetes mellitus"] == ["MESH:D003920"]
    assert index["diabetes mellitus, type 2"] == ["MESH:D003924"]
    # Synonyms verbatim (already lowercased at import).
    assert index["diabetes"] == ["MESH:D003920"]
    assert index["niddm"] == ["MESH:D003924"]
    assert index["type 2 diabetes"] == ["MESH:D003924"]


async def test_build_term_index_handles_shared_synonyms_with_list():
    """When two terms share the same synonym text, both IDs land in the
    same list - the matcher returns both."""
    rows = [
        {"id": "X:1", "label": "Stress", "synonyms": ["stress"]},
        {
            "id": "X:2",
            "label": "Stress (physiology)",
            "synonyms": ["stress"],
        },
    ]
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=rows)]])
    client = _client_with_driver(driver)
    index = await linking._build_term_index(client, "MeSHTerm")
    assert sorted(index["stress"]) == ["X:1", "X:2"]


async def test_build_term_index_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await linking._build_term_index(client, "MeSHTerm")


# ---- _fetch_entities_to_link ----


async def test_fetch_entities_with_doc_id_uses_chunk_match():
    """Per-doc fetch filters via MATCH (chunk {doc_id})."""
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[{"key": "brca1", "entity_type": "GENE"}])]
        ]
    )
    client = _client_with_driver(driver)
    rows = await linking._fetch_entities_to_link(client, "MeSHTerm", "doc-abc")
    assert rows == [{"key": "brca1", "entity_type": "GENE"}]
    cypher, params = driver.sessions[0].calls[0]
    assert "MENTIONS" in cypher
    assert "doc_id" in cypher
    assert params == {"doc_id": "doc-abc"}


async def test_fetch_entities_global_uses_unlinked_filter():
    """Global fetch returns only entities NOT yet linked to the given
    ontology - check Cypher uses the WHERE NOT EXISTS pattern with
    the right term label."""
    driver = RecordingDriver(canned_results_per_session=[[_RecordingResult(rows=[])]])
    client = _client_with_driver(driver)
    await linking._fetch_entities_to_link(client, "MeSHTerm", None)
    cypher, params = driver.sessions[0].calls[0]
    assert "WHERE NOT EXISTS" in cypher
    assert ":MeSHTerm" in cypher
    assert "CANONICAL_TO" in cypher
    assert params == {}


# ---- _write_canonical_links ----


async def test_write_canonical_links_empty_returns_zero():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    n = await linking._write_canonical_links(client, [], "MeSHTerm")
    assert n == 0
    assert driver.sessions == []


async def test_write_canonical_links_writes_rows_and_returns_count():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    rows = [
        {
            "entity_key": "diabetes mellitus",
            "entity_type": "DISEASE",
            "term_id": "MESH:D003920",
            "strategy": "exact",
            "confidence": None,
        },
        {
            "entity_key": "cancers",
            "entity_type": "DISEASE",
            "term_id": "MESH:D009369",
            "strategy": "fuzzy",
            "confidence": 0.9,
        },
    ]
    n = await linking._write_canonical_links(client, rows, "MeSHTerm")
    assert n == 2
    cypher, params = driver.sessions[0].calls[0]
    assert ":CANONICAL_TO" in cypher
    assert ":MeSHTerm" in cypher
    assert "canonicalised = true" in cypher
    assert params["rows"] == rows


async def test_write_canonical_links_propagates_cypher_exception():
    """Driver failure propagates; orchestrator boundary catches."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await linking._write_canonical_links(
            client,
            [
                {
                    "entity_key": "k",
                    "entity_type": "T",
                    "term_id": "X:1",
                    "strategy": "exact",
                    "confidence": None,
                }
            ],
            "MeSHTerm",
        )


# ---- link_entities end-to-end (mocking the smaller helpers via the
#      session canned results) ----


async def test_link_entities_unknown_ontology_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    try:
        await linking.link_entities(client, "not_a_thing", "exact")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not_a_thing" in str(e)


async def test_link_entities_returns_zero_when_no_terms_in_index():
    """If the ontology index has no terms (the ontology hasn't been
    imported yet, or the ontology has no data), return 0 without
    fetching entities. Each Cypher call opens its own session, so we
    expect a single session containing one call."""
    driver = RecordingDriver(
        canned_results_per_session=[
            [_RecordingResult(rows=[])]  # session 1: build_term_index empty
        ]
    )
    client = _client_with_driver(driver)
    n = await linking.link_entities(client, "mesh", "exact")
    assert n == 0
    # One session opened (for the index build); no entity fetch ran.
    assert len(driver.sessions) == 1


async def test_link_entities_returns_zero_when_no_entities_to_link():
    """Index has terms but the entity fetch returns nothing -> 0."""
    driver = RecordingDriver(
        canned_results_per_session=[
            # Session 1: build_term_index
            [
                _RecordingResult(
                    rows=[
                        {
                            "id": "MESH:D003920",
                            "label": "Diabetes Mellitus",
                            "synonyms": [],
                        }
                    ]
                )
            ],
            # Session 2: _fetch_entities_to_link returns empty
            [_RecordingResult(rows=[])],
        ]
    )
    client = _client_with_driver(driver)
    n = await linking.link_entities(client, "mesh", "exact", doc_id="doc-1")
    assert n == 0


async def test_link_entities_happy_path_writes_canonical_to_edges():
    """End-to-end: index built, entities fetched, matches found, edges
    written. Each helper opens its OWN driver session, so the canned
    results are distributed across three sessions."""
    driver = RecordingDriver(
        canned_results_per_session=[
            # Session 1: build_term_index
            [
                _RecordingResult(
                    rows=[
                        {
                            "id": "MESH:D003920",
                            "label": "Diabetes Mellitus",
                            "synonyms": ["type 2 diabetes"],
                        }
                    ]
                )
            ],
            # Session 2: _fetch_entities_to_link (per-doc)
            [
                _RecordingResult(
                    rows=[
                        {
                            "key": "diabetes mellitus",
                            "entity_type": "DISEASE",
                        },
                        {
                            "key": "type 2 diabetes",
                            "entity_type": "DISEASE",
                        },
                        {
                            "key": "something else entirely",
                            "entity_type": "DISEASE",
                        },
                    ]
                )
            ],
            # Session 3: _write_canonical_links (no records consumed)
            [_RecordingResult(rows=[])],
        ]
    )
    client = _client_with_driver(driver)
    n = await linking.link_entities(client, "mesh", "exact", doc_id="doc-1")
    # Two entities matched (the third didn't); two edges written.
    assert n == 2
    # Verify the write call landed and had the right rows.
    write_cypher, write_params = driver.sessions[2].calls[0]
    assert ":CANONICAL_TO" in write_cypher
    rows = write_params["rows"]
    keys_written = {r["entity_key"] for r in rows}
    assert keys_written == {"diabetes mellitus", "type 2 diabetes"}


# ---- ensure_ontology_imported ----


async def test_ensure_ontology_imported_unknown_raises():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    try:
        await linking.ensure_ontology_imported(client, "not_real")
        assert False
    except ValueError as e:
        assert "not_real" in str(e)


async def test_ensure_ontology_imported_returns_already_imported_when_present():
    """When the is_imported check says yes, returns (True, True) without
    triggering the (expensive) import function."""
    with (
        patch(
            "knowledge_agent.kg.ontologies.linking.mesh_writes.is_imported",
            return_value=True,
        ),
        patch("knowledge_agent.kg.ontologies.linking.mesh_writes.import_mesh") as mock_import,
    ):
        # Re-build registry to pick up the patches.
        import importlib

        importlib.reload(linking)
        driver = RecordingDriver()
        client = _client_with_driver(driver)
        was = await linking.ensure_ontology_imported(client, "mesh")

    assert was is True
    mock_import.assert_not_called()


async def test_ensure_ontology_imported_triggers_import_when_absent():
    """When is_imported returns False, the import function fires; the
    return value is `was_already_imported` (False because the import
    just ran). Failures from the import_fn propagate under the
    typed-errors contract — the pipeline's per-ontology try/except is
    the boundary."""
    with (
        patch(
            "knowledge_agent.kg.ontologies.linking.mesh_writes.is_imported",
            return_value=False,
        ),
        patch(
            "knowledge_agent.kg.ontologies.linking.mesh_writes.import_mesh",
            return_value=True,
        ),
    ):
        import importlib

        importlib.reload(linking)
        driver = RecordingDriver()
        client = _client_with_driver(driver)
        was = await linking.ensure_ontology_imported(client, "mesh")

    assert was is False
