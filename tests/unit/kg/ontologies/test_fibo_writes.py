"""Tests for kg.ontologies.fibo_writes - the L7 FIBO adapter.

FIBO is the first ontology on the menu that ships modularly (no
single-file download); the module has a custom walker + multi-file
parser path. Tests cover:
  - Standard module-level lifecycle (is_imported / write_terms /
    delete_imported / import_fibo / Neo4jClient delegates) — same
    shape as DRON/OBI/EFO test files.
  - Walker-specific behaviour: GitHub tree filtering, per-file caching,
    error handling.
  - Multi-file parser: combining ~70 .rdf files into one rdflib graph
    + the FIBO id_extractor's "FIBO:" prefixing of class URIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontologies import fibo_writes
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


# ---- DOMAIN_TAGS ----


def test_domain_tags_declared():
    assert isinstance(fibo_writes.DOMAIN_TAGS, tuple)
    assert "finance" in fibo_writes.DOMAIN_TAGS


# ---- is_imported ----


async def test_is_imported_true_when_query_returns_present():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await fibo_writes.is_imported(_client_with_driver(driver)) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    assert await fibo_writes.is_imported(_client_with_driver(driver)) is False


async def test_is_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("conn lost"))
    with pytest.raises(RuntimeError, match="conn lost"):
        await fibo_writes.is_imported(_client_with_driver(driver))


async def test_is_imported_query_uses_fiboterm_label():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    await fibo_writes.is_imported(client)
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FIBOTerm" in cypher


# ---- write_terms ----


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    assert await fibo_writes.write_terms(_client_with_driver(driver), []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    driver = RecordingDriver()
    terms = [_term("FIBO:FinancialInstrument", "financial instrument")]
    assert await fibo_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    driver = RecordingDriver()
    terms = [
        _term("FIBO:FinancialInstrument", "financial instrument"),
        _term("FIBO:Security", "security", parents=("FIBO:FinancialInstrument",)),
    ]
    assert await fibo_writes.write_terms(_client_with_driver(driver), terms) is None
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_uses_multilabel_and_id_carries_prefix():
    driver = RecordingDriver()
    terms = [
        _term("FIBO:Money", "money", synonyms=("currency", "legal tender")),
    ]
    await fibo_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[0]
    assert ":OntologyTerm" in cypher
    assert ":FIBOTerm" in cypher
    assert params["rows"] == [
        {
            "id": "FIBO:Money",
            "label": "money",
            "synonyms": ["currency", "legal tender"],
            "definition": None,
        }
    ]


async def test_write_terms_edge_query_uses_fibo_is_a_rel():
    driver = RecordingDriver()
    terms = [
        _term("FIBO:FinancialInstrument", "financial instrument"),
        _term("FIBO:Security", "security", parents=("FIBO:FinancialInstrument",)),
    ]
    await fibo_writes.write_terms(_client_with_driver(driver), terms)
    cypher, params = driver.sessions[0].calls[1]
    assert ":FIBO_IS_A" in cypher
    assert params["rows"] == [{"child": "FIBO:Security", "parent": "FIBO:FinancialInstrument"}]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await fibo_writes.write_terms(
            _client_with_driver(driver), [_term("FIBO:FinancialInstrument", "financial instrument")]
        )


# ---- import_fibo ----


async def test_import_fibo_short_circuits_when_already_imported():
    """force=False and FIBO already imported -> no walker / parse / write."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    with (
        patch.object(fibo_writes, "require_cached") as mock_walk,
        patch.object(fibo_writes, "_read_and_extract") as mock_extract,
    ):
        result = await fibo_writes.import_fibo(_client_with_driver(driver), force=False)
    assert result is False  # no-op: typed-errors contract
    mock_walk.assert_not_called()
    mock_extract.assert_not_called()


async def test_import_fibo_force_drops_then_reimports():
    driver = RecordingDriver()
    fake_terms = [_term("FIBO:Money", "money")]
    with (
        patch.object(
            fibo_writes,
            "require_cached",
            return_value=Path("/fake/cache/fibo"),
        ),
        patch.object(fibo_writes, "_read_and_extract", return_value=fake_terms),
    ):
        result = await fibo_writes.import_fibo(_client_with_driver(driver), force=True)
    assert result is True
    # First session = delete_imported; second = write_terms.
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert ":FIBOTerm" in delete_cypher
    assert "DETACH DELETE" in delete_cypher


async def test_import_fibo_aborts_on_zero_terms():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch.object(
            fibo_writes,
            "require_cached",
            return_value=Path("/fake/cache/fibo"),
        ),
        patch.object(fibo_writes, "_read_and_extract", return_value=[]),
    ):
        with pytest.raises(RuntimeError, match="extracted 0 terms"):
            await fibo_writes.import_fibo(_client_with_driver(driver), force=False)


async def test_import_fibo_propagates_walker_exception():
    """Walker network failure propagates under the typed-errors
    contract; orchestrator boundary catches per-ontology."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    with (
        patch.object(
            fibo_writes,
            "require_cached",
            side_effect=RuntimeError("network down"),
        ),
        pytest.raises(RuntimeError, match="network down"),
    ):
        await fibo_writes.import_fibo(_client_with_driver(driver), force=False)


async def test_import_fibo_propagates_when_force_delete_fails():
    """force=True + delete_imported failure propagates; walker is not
    reached because the delete raises first."""
    driver = RecordingDriver(raise_on_run=RuntimeError("delete boom"))
    with (
        patch.object(fibo_writes, "require_cached") as mock_walk,
        pytest.raises(RuntimeError, match="delete boom"),
    ):
        await fibo_writes.import_fibo(_client_with_driver(driver), force=True)
    mock_walk.assert_not_called()


async def test_import_fibo_forwards_xrefs_mode_to_write_terms():
    """import_fibo must forward its `xrefs_mode` down to write_terms, else the
    L7 xref pass never runs and FIBO xref edges/props are silently dropped
    (write_terms defaults to 'none'). Regression: the write_terms call omitted
    xrefs_mode, so a caller's 'use'/'collect_only' was lost — FIBO was the ONLY
    ontology that failed to forward it (mesh + generic import_ontology_data
    do)."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    client = _client_with_driver(driver)
    fake_terms = [_term("FIBO:Money", "money")]
    with (
        patch.object(
            fibo_writes,
            "require_cached",
            return_value=Path("/fake/cache/fibo"),
        ),
        patch.object(fibo_writes, "_read_and_extract", return_value=fake_terms),
        patch.object(fibo_writes, "write_terms", new_callable=AsyncMock) as mock_write,
    ):
        result = await fibo_writes.import_fibo(client, force=False, xrefs_mode="use")
    assert result is True
    mock_write.assert_awaited_once_with(client, fake_terms, xrefs_mode="use")


# ---- delete_imported ----


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    assert await fibo_writes.delete_imported(_client_with_driver(driver)) is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FIBOTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await fibo_writes.delete_imported(_client_with_driver(driver))


# ---- Walker: _list_fibo_rdf_paths ----


def _mock_http_response(payload: dict[str, Any], status: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


_HTTP_PATCH = "knowledge_agent.kg.ontologies.fibo_writes._http_client.request"


async def test_list_paths_filters_only_rdf_blobs():
    """Walker keeps only `type:blob` + .rdf-suffix paths from the tree."""
    payload = {
        "tree": [
            {"path": "BE/Corporations/Corporations.rdf", "type": "blob"},
            {"path": "FND/Accounting/Currency.rdf", "type": "blob"},
            {"path": "BE/Corporations", "type": "tree"},
            {"path": "README.md", "type": "blob"},
            {"path": "AboutFIBOProd.rdf", "type": "blob"},
        ]
    }
    with patch(
        _HTTP_PATCH,
        new_callable=AsyncMock,
        return_value=_mock_http_response(payload),
    ):
        paths = await fibo_writes._list_fibo_rdf_paths()
    assert paths == [
        "AboutFIBOProd.rdf",
        "BE/Corporations/Corporations.rdf",
        "FND/Accounting/Currency.rdf",
    ]


async def test_list_paths_skips_etc_and_dotgithub_prefixes():
    """Test fixtures (etc/) + CI config (.github/) are filtered out."""
    payload = {
        "tree": [
            {"path": "etc/source/SomeTest.rdf", "type": "blob"},
            {".github/workflows/build.yml": "irrelevant"},
            {"path": ".github/CODEOWNERS.rdf", "type": "blob"},
            {"path": "FBC/FinancialInstruments/Securities.rdf", "type": "blob"},
        ]
    }
    with patch(
        _HTTP_PATCH,
        new_callable=AsyncMock,
        return_value=_mock_http_response(payload),
    ):
        paths = await fibo_writes._list_fibo_rdf_paths()
    # Only the non-etc, non-.github .rdf blob remains.
    assert paths == ["FBC/FinancialInstruments/Securities.rdf"]


async def test_list_paths_returns_sorted():
    """Determinism: walker order is sorted so caching is reproducible."""
    payload = {
        "tree": [
            {"path": "Z/Last.rdf", "type": "blob"},
            {"path": "A/First.rdf", "type": "blob"},
            {"path": "M/Middle.rdf", "type": "blob"},
        ]
    }
    with patch(
        _HTTP_PATCH,
        new_callable=AsyncMock,
        return_value=_mock_http_response(payload),
    ):
        paths = await fibo_writes._list_fibo_rdf_paths()
    assert paths == ["A/First.rdf", "M/Middle.rdf", "Z/Last.rdf"]


async def test_list_paths_raises_on_http_error():
    """Walker bubbles up HTTP failures so import_fibo's try/except catches them."""
    response = MagicMock()
    response.raise_for_status = MagicMock(side_effect=RuntimeError("404"))
    with patch(_HTTP_PATCH, new_callable=AsyncMock, return_value=response):
        with pytest.raises(RuntimeError, match="404"):
            await fibo_writes._list_fibo_rdf_paths()


# ---- Walker: _walk_and_cache_fibo ----


async def test_walk_and_cache_calls_ensure_cached_per_path(tmp_path):
    """Walker calls ensure_cached once per discovered module path."""
    paths = ["A/First.rdf", "B/Second.rdf", "C/Third.rdf"]
    with (
        patch.object(
            fibo_writes,
            "ensure_cached",
            new_callable=AsyncMock,
        ) as mock_cache,
        patch.object(
            fibo_writes,
            "get_downloads_dir",
            return_value=tmp_path,
        ),
    ):
        await fibo_writes._walk_and_cache_fibo(paths=paths)
    assert mock_cache.call_count == 3
    # Each call uses the raw.githubusercontent base + the FIBO cache subdir.
    for call_obj, repo_path in zip(mock_cache.call_args_list, paths, strict=True):
        url, cache_filename = call_obj.args
        assert url.endswith(repo_path)
        assert "raw.githubusercontent.com/edmcouncil/fibo" in url
        assert cache_filename == f"fibo/{repo_path}"


async def test_walk_and_cache_uses_listing_when_paths_omitted(tmp_path):
    """Production callers pass no paths -> walker uses _list_fibo_rdf_paths."""
    with (
        patch.object(
            fibo_writes,
            "_list_fibo_rdf_paths",
            new_callable=AsyncMock,
            return_value=["X/Single.rdf"],
        ) as mock_list,
        patch.object(
            fibo_writes,
            "ensure_cached",
            new_callable=AsyncMock,
        ),
        patch.object(
            fibo_writes,
            "get_downloads_dir",
            return_value=tmp_path,
        ),
    ):
        await fibo_writes._walk_and_cache_fibo()
    mock_list.assert_called_once()


async def test_walk_and_cache_returns_fibo_subdir(tmp_path):
    """Returned path is `<cache>/fibo` so _read_and_extract can rglob it."""
    with (
        patch.object(
            fibo_writes,
            "ensure_cached",
            new_callable=AsyncMock,
        ),
        patch.object(
            fibo_writes,
            "get_downloads_dir",
            return_value=tmp_path,
        ),
    ):
        result = await fibo_writes._walk_and_cache_fibo(paths=[])
    assert result == tmp_path / "fibo"


# ---- Multi-file parser: _read_and_extract ----


def test_read_and_extract_parses_every_rdf_file(tmp_path):
    """Parser walks the cache dir recursively for .rdf files."""
    # Build a fake FIBO cache with two minimal .rdf files in separate subdirs.
    a_dir = tmp_path / "FND"
    a_dir.mkdir()
    a_rdf = a_dir / "ModuleA.rdf"
    a_rdf.write_text(_minimal_owl_rdf(), encoding="utf-8")

    b_dir = tmp_path / "BE"
    b_dir.mkdir()
    b_rdf = b_dir / "ModuleB.rdf"
    b_rdf.write_text(_minimal_owl_rdf(), encoding="utf-8")

    terms = fibo_writes._read_and_extract(tmp_path)
    # The minimal RDF has one OWL class; loading both files yields the
    # same class twice (idempotent in rdflib) -> deduped at the OntologyTerm
    # level by the shared extractor.
    assert len(terms) == 1
    assert terms[0].id.startswith("FIBO:")


def test_read_and_extract_skips_one_bad_module_and_continues(tmp_path):
    """One module's parse failure logs + skips, doesn't kill the whole import."""
    good = tmp_path / "good.rdf"
    good.write_text(_minimal_owl_rdf(), encoding="utf-8")
    bad = tmp_path / "bad.rdf"
    bad.write_text("<not really rdf>", encoding="utf-8")

    terms = fibo_writes._read_and_extract(tmp_path)
    # Good module's term still extracted.
    assert any(t.id.startswith("FIBO:") for t in terms)


def test_read_and_extract_empty_cache_returns_empty_list(tmp_path):
    """No .rdf files -> empty term list (caller surfaces 'aborting write')."""
    assert fibo_writes._read_and_extract(tmp_path) == []


# ---- FIBO id_extractor ----


def test_fibo_id_extractor_prefixes_fibo_uris():
    """edmcouncil.org/fibo URIs gain a `FIBO:` prefix on the last segment."""
    uri = "https://spec.edmcouncil.org/fibo/ontology/FBC/FinancialInstruments/FinancialInstrument"
    assert fibo_writes._fibo_id_extractor(uri) == "FIBO:FinancialInstrument"


def test_fibo_id_extractor_handles_fragment_uris():
    """Some FIBO URIs use `#` fragments instead of `/` paths."""
    uri = "https://spec.edmcouncil.org/fibo/ontology/MD/SomeModule#SomeClass"
    assert fibo_writes._fibo_id_extractor(uri) == "FIBO:SomeClass"


def test_fibo_id_extractor_falls_through_for_non_fibo_uris():
    """Non-FIBO URIs (BFO, CMNS, owl built-ins) use the default extractor."""
    bfo_uri = "http://purl.obolibrary.org/obo/BFO_0000015"
    # _owl_id_extractor handles obo PURL -> "BFO:0000015"
    assert fibo_writes._fibo_id_extractor(bfo_uri) == "BFO:0000015"


# ---- Neo4jClient delegates ----


async def test_client_is_fibo_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    assert await _client_with_driver(driver).is_fibo_imported() is True


async def test_client_delete_fibo_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_fibo() is None
    cypher, _ = driver.sessions[0].calls[0]
    assert ":FIBOTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_fibo_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    with (
        patch.object(fibo_writes, "require_cached") as mock_walk,
    ):
        assert await client.import_fibo(force=False) is False
    mock_walk.assert_not_called()


# ---- Test helpers ----


def _minimal_owl_rdf() -> str:
    """A tiny OWL/RDF-XML document defining one FIBO class.

    Format compatible with rdflib's xml parser. Used in the multi-file
    parser tests so we exercise the real `extract_terms_owl` path
    without depending on the FIBO repo.
    """
    return """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Class rdf:about="https://spec.edmcouncil.org/fibo/ontology/FBC/SampleClass">
    <rdfs:label xml:lang="en">sample class</rdfs:label>
  </owl:Class>
</rdf:RDF>
"""
