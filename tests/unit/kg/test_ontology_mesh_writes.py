"""Tests for kg.ontology_mesh_writes - the L7 MeSH adapter.

Three concerns covered:
  1. **MeSH-specific extraction** — exercised against a small real
     rdflib graph using the meshv: vocabulary, so we verify the
     extractor walks Descriptor -> rdfs:label, Descriptor ->
     meshv:concept -> rdfs:label, and meshv:broaderDescriptor edges
     correctly.
  2. **Cypher writes** — `write_terms` produces the two-pass UNWIND
     MERGE pattern (nodes first, then edges). Driver is stubbed so we
     verify the Cypher contents + params without a real Neo4j.
  3. **Lifecycle** — `is_imported` reads a count > 0 from Neo4j;
     `import_mesh(force=False)` short-circuits when imported;
     `delete_imported` runs DETACH DELETE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from knowledge_agent.config import Settings
from knowledge_agent.kg import ontology_mesh_writes
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.ontology_helpers import OntologyTerm

# ---- Test harness (mirrors test_kg_client / test_entity_writes) ----


@dataclass
class _RecordingResult:
    """Stand-in for a neo4j Result. Returns canned records via .single()."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    async def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    # canned responses for .run() — keyed by the call index (0, 1, 2, ...).
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


# ---- DOMAIN_TAGS ----


def test_domain_tags_declared():
    """The wizard uses DOMAIN_TAGS to suggest MeSH when a relevant tag
    is selected. Verify the constant is set + non-empty."""
    assert isinstance(ontology_mesh_writes.DOMAIN_TAGS, tuple)
    assert len(ontology_mesh_writes.DOMAIN_TAGS) > 0
    # Medicine is the most directly relevant tag - it must be in the set.
    assert "medicine" in ontology_mesh_writes.DOMAIN_TAGS


# ---- is_imported ----


async def test_is_imported_true_when_query_returns_present():
    """is_imported returns True when the COUNT query says nodes exist."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    assert await ontology_mesh_writes.is_imported(client) is True


async def test_is_imported_false_when_query_returns_no_nodes():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    client = _client_with_driver(driver)
    assert await ontology_mesh_writes.is_imported(client) is False


async def test_is_imported_propagates_driver_exception():
    """Driver error during the count query propagates under the
    typed-errors contract; ensure_ontology_imported (the orchestrator
    boundary) catches per-ontology."""
    driver = RecordingDriver(raise_on_run=RuntimeError("connection lost"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="connection lost"):
        await ontology_mesh_writes.is_imported(client)


# ---- write_terms ----


def _term(id_: str, label: str, synonyms=(), parents=()) -> OntologyTerm:
    return OntologyTerm(
        id=id_,
        label=label,
        synonyms=tuple(synonyms),
        parents=tuple(parents),
        definition=None,
    )


async def test_write_terms_empty_input_returns_true_no_io():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await ontology_mesh_writes.write_terms(client, []) is None
    assert driver.sessions == []


async def test_write_terms_one_round_trip_when_no_hierarchy():
    """No parent edges -> only the node-MERGE query runs (one call)."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [_term("MESH:D003920", "Diabetes Mellitus")]
    assert await ontology_mesh_writes.write_terms(client, terms) is None
    assert len(driver.sessions) == 1
    # Just the node MERGE; no second query for edges.
    assert len(driver.sessions[0].calls) == 1


async def test_write_terms_two_round_trips_when_hierarchy_present():
    """Parents set -> a second UNWIND/MATCH/MERGE for the edges runs."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("MESH:D003920", "Diabetes Mellitus"),
        _term(
            "MESH:D003924",
            "Diabetes Mellitus, Type 2",
            parents=("MESH:D003920",),
        ),
    ]
    assert await ontology_mesh_writes.write_terms(client, terms) is None
    assert len(driver.sessions) == 1
    # Two queries: node MERGE + edge MERGE.
    assert len(driver.sessions[0].calls) == 2


async def test_write_terms_node_query_carries_label_synonyms_definition():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term(
            "MESH:D003920",
            "Diabetes Mellitus",
            synonyms=("diabetes", "dm"),
        ),
    ]
    await ontology_mesh_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[0]

    assert ":OntologyTerm" in cypher
    assert ":MeSHTerm" in cypher
    assert "row.label" in cypher
    assert "row.synonyms" in cypher
    assert "row.definition" in cypher

    rows = params["rows"]
    assert rows == [
        {
            "id": "MESH:D003920",
            "label": "Diabetes Mellitus",
            "synonyms": ["diabetes", "dm"],
            "definition": None,
        }
    ]


async def test_write_terms_edge_query_uses_mesh_broader_rel_and_id_match():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    terms = [
        _term("MESH:D003920", "Diabetes Mellitus"),
        _term(
            "MESH:D003924",
            "Diabetes Mellitus, Type 2",
            parents=("MESH:D003920",),
        ),
    ]
    await ontology_mesh_writes.write_terms(client, terms)
    cypher, params = driver.sessions[0].calls[1]

    assert ":MESH_BROADER" in cypher
    assert "row.child" in cypher
    assert "row.parent" in cypher
    assert params["rows"] == [{"child": "MESH:D003924", "parent": "MESH:D003920"}]


async def test_write_terms_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_mesh_writes.write_terms(client, [_term("MESH:D003920", "Diabetes Mellitus")])


# ---- import_mesh ----


async def test_import_mesh_short_circuits_when_already_imported():
    """When MeSH is already in Neo4j and force=False, import skips
    the heavy download + parse + write."""
    driver = RecordingDriver(
        canned_results_per_session=[
            # is_imported returns True
            [_RecordingResult(rows=[{"present": True}])]
        ]
    )
    client = _client_with_driver(driver)
    with patch("knowledge_agent.kg.ontology_mesh_writes.ensure_cached") as mock_cache:
        result = await ontology_mesh_writes.import_mesh(client, force=False)

    assert result is False  # no-op: typed-errors contract
    # ensure_cached must NOT have been called - we short-circuited.
    mock_cache.assert_not_called()


async def test_import_mesh_force_drops_then_reimports():
    """force=True: skip the is_imported check, drop existing data,
    download, parse, write."""
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    fake_terms = [_term("MESH:D003920", "Diabetes Mellitus")]
    with (
        patch(
            "knowledge_agent.kg.ontology_mesh_writes.ensure_cached",
            return_value="/fake/cache/mesh.nt.gz",
        ),
        patch(
            "knowledge_agent.kg.ontology_mesh_writes._read_and_extract",
            return_value=fake_terms,
        ),
    ):
        result = await ontology_mesh_writes.import_mesh(client, force=True)

    assert result is True
    # First session = delete_imported (one DETACH DELETE).
    # Second session = write_terms (one MERGE call since no hierarchy).
    assert len(driver.sessions) == 2
    delete_cypher, _ = driver.sessions[0].calls[0]
    assert "DETACH DELETE" in delete_cypher
    assert ":MeSHTerm" in delete_cypher


async def test_import_mesh_aborts_on_zero_terms():
    """When extraction returns 0 terms (file parsed but nothing
    matched), don't write - raise so the caller knows something went
    wrong (typed-errors contract; orchestrator catches per-ontology)."""
    driver = RecordingDriver(
        canned_results_per_session=[
            # is_imported -> False so we proceed past the short-circuit.
            [_RecordingResult(rows=[{"present": False}])]
        ]
    )
    client = _client_with_driver(driver)
    with (
        patch(
            "knowledge_agent.kg.ontology_mesh_writes.ensure_cached",
            return_value="/fake/cache/mesh.nt.gz",
        ),
        patch(
            "knowledge_agent.kg.ontology_mesh_writes._read_and_extract",
            return_value=[],
        ),
        pytest.raises(RuntimeError, match="extracted 0 terms"),
    ):
        await ontology_mesh_writes.import_mesh(client, force=False)


async def test_import_mesh_propagates_download_exception():
    """Network failure -> log + return False, don't raise."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": False}])]]
    )
    client = _client_with_driver(driver)
    with (
        patch(
            "knowledge_agent.kg.ontology_mesh_writes.ensure_cached",
            side_effect=RuntimeError("network down"),
        ),
        pytest.raises(RuntimeError, match="network down"),
    ):
        await ontology_mesh_writes.import_mesh(client, force=False)


# ---- delete_imported ----


async def test_delete_imported_runs_one_detach_delete_query():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await ontology_mesh_writes.delete_imported(client) is None
    assert len(driver.sessions) == 1
    assert len(driver.sessions[0].calls) == 1
    cypher, _ = driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_delete_imported_propagates_driver_exception():
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_mesh_writes.delete_imported(client)


# ---- MeSH extraction (real rdflib graph using meshv: vocabulary) ----


def _build_sample_mesh_graph():
    """Build a tiny meshv-shaped RDF graph:
    D003920 "Diabetes Mellitus" - root descriptor
              concept C100 "Diabetes Mellitus" (= primary, same as label)
              concept C101 "Diabetes" (alt name, becomes synonym)
    D003924 "Diabetes Mellitus, Type 2" - broader = D003920
              concept C200 "Type 2 Diabetes Mellitus"
              concept C201 "NIDDM"
    """
    import rdflib
    from rdflib import URIRef
    from rdflib.namespace import RDF, RDFS

    graph = rdflib.Graph()
    mesh_ns = "http://id.nlm.nih.gov/mesh/"
    meshv = URIRef("http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor")
    broader = URIRef("http://id.nlm.nih.gov/mesh/vocab#broaderDescriptor")
    concept = URIRef("http://id.nlm.nih.gov/mesh/vocab#concept")

    def add_descriptor(d_id: str, label: str, concept_pairs, parents=()):
        d_uri = URIRef(mesh_ns + d_id)
        graph.add((d_uri, RDF.type, meshv))
        graph.add((d_uri, RDFS.label, rdflib.Literal(label, lang="en")))
        for parent in parents:
            graph.add((d_uri, broader, URIRef(mesh_ns + parent)))
        for c_id, c_label in concept_pairs:
            c_uri = URIRef(mesh_ns + c_id)
            graph.add((d_uri, concept, c_uri))
            graph.add((c_uri, RDFS.label, rdflib.Literal(c_label, lang="en")))

    add_descriptor(
        "D003920",
        "Diabetes Mellitus",
        concept_pairs=[
            ("C100", "Diabetes Mellitus"),  # same as descriptor label
            ("C101", "Diabetes"),  # alternative name
        ],
    )
    add_descriptor(
        "D003924",
        "Diabetes Mellitus, Type 2",
        concept_pairs=[
            ("C200", "Type 2 Diabetes Mellitus"),
            ("C201", "NIDDM"),
        ],
        parents=["D003920"],
    )
    return graph


def test_extract_descriptors_returns_one_term_per_descriptor():
    graph = _build_sample_mesh_graph()
    terms = ontology_mesh_writes._extract_descriptors(graph)
    by_id = {t.id: t for t in terms}
    assert set(by_id) == {"MESH:D003920", "MESH:D003924"}


def test_extract_descriptors_uses_rdfs_label_for_primary():
    graph = _build_sample_mesh_graph()
    terms = ontology_mesh_writes._extract_descriptors(graph)
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003920"].label == "Diabetes Mellitus"
    assert by_id["MESH:D003924"].label == "Diabetes Mellitus, Type 2"


def test_extract_descriptors_collects_concept_labels_as_synonyms():
    """Concept labels become synonyms - except when they exactly match
    the descriptor label (avoiding trivial duplication)."""
    graph = _build_sample_mesh_graph()
    terms = ontology_mesh_writes._extract_descriptors(graph)
    by_id = {t.id: t for t in terms}
    # D003920 has two concepts: one matches the descriptor label exactly,
    # the other ("Diabetes") becomes a synonym.
    assert by_id["MESH:D003920"].synonyms == ("diabetes",)
    # D003924 has two concepts, neither matches its label.
    assert by_id["MESH:D003924"].synonyms == (
        "niddm",
        "type 2 diabetes mellitus",
    )


def test_extract_descriptors_picks_up_broader_parents():
    graph = _build_sample_mesh_graph()
    terms = ontology_mesh_writes._extract_descriptors(graph)
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003924"].parents == ("MESH:D003920",)
    assert by_id["MESH:D003920"].parents == ()


def test_extract_descriptors_skips_unlabelled_descriptors():
    """A descriptor without an rdfs:label is skipped - can't write a
    term without a label."""
    from rdflib import URIRef
    from rdflib.namespace import RDF

    graph = _build_sample_mesh_graph()
    meshv = URIRef("http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor")
    orphan = URIRef("http://id.nlm.nih.gov/mesh/D999999")
    graph.add((orphan, RDF.type, meshv))
    # No rdfs:label added.

    terms = ontology_mesh_writes._extract_descriptors(graph)
    ids = {t.id for t in terms}
    assert "MESH:D999999" not in ids


# ---- Neo4jClient delegate methods ----


async def test_client_is_mesh_imported_delegates_to_module():
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    assert await client.is_mesh_imported() is True


async def test_client_delete_mesh_delegates_to_module():
    driver = RecordingDriver()
    client = _client_with_driver(driver)
    assert await client.delete_mesh() is None
    assert len(driver.sessions) == 1
    cypher, _ = driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert "DETACH DELETE" in cypher


async def test_client_import_mesh_delegates_to_module():
    """Smoke check the delegate path: client.import_mesh runs the
    short-circuit branch when MeSH is already imported."""
    driver = RecordingDriver(
        canned_results_per_session=[[_RecordingResult(rows=[{"present": True}])]]
    )
    client = _client_with_driver(driver)
    with patch("knowledge_agent.kg.ontology_mesh_writes.ensure_cached") as mock_cache:
        assert await client.import_mesh(force=False) is False
    mock_cache.assert_not_called()


# ---- streaming parser sink ----


_SAMPLE_NTRIPLES = (
    # Two descriptors + one concept for D003920 + hierarchy edge D003924 -> D003920.
    b"<http://id.nlm.nih.gov/mesh/D003920> "
    b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
    b"<http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor> .\n"
    b"<http://id.nlm.nih.gov/mesh/D003920> "
    b"<http://www.w3.org/2000/01/rdf-schema#label> "
    b'"Diabetes Mellitus"@en .\n'
    b"<http://id.nlm.nih.gov/mesh/D003920> "
    b"<http://id.nlm.nih.gov/mesh/vocab#concept> "
    b"<http://id.nlm.nih.gov/mesh/C100> .\n"
    b"<http://id.nlm.nih.gov/mesh/C100> "
    b"<http://www.w3.org/2000/01/rdf-schema#label> "
    b'"Diabetes"@en .\n'
    b"<http://id.nlm.nih.gov/mesh/D003924> "
    b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
    b"<http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor> .\n"
    b"<http://id.nlm.nih.gov/mesh/D003924> "
    b"<http://www.w3.org/2000/01/rdf-schema#label> "
    b'"Diabetes Mellitus, Type 2"@en .\n'
    b"<http://id.nlm.nih.gov/mesh/D003924> "
    b"<http://id.nlm.nih.gov/mesh/vocab#broaderDescriptor> "
    b"<http://id.nlm.nih.gov/mesh/D003920> .\n"
    # Noise: a triple with an UNINTERESTING property - sink should drop it.
    b"<http://id.nlm.nih.gov/mesh/D003920> "
    b"<http://id.nlm.nih.gov/mesh/vocab#dateCreated> "
    b'"2025-01-01" .\n'
)


def _stream_parse(ntriples: bytes):
    from io import BytesIO

    from rdflib.plugins.parsers.ntriples import W3CNTriplesParser

    sink = ontology_mesh_writes._MeshStreamSink()
    parser = W3CNTriplesParser(sink=sink)
    parser.parse(BytesIO(ntriples))
    return sink


def test_stream_sink_counts_all_triples_processed():
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    # 7 relevant triples + 1 noise triple = 8 total seen.
    assert sink.triple_count == 8


def test_stream_sink_drops_irrelevant_properties():
    """The dateCreated noise triple gets counted but NOT stored in the
    record's fields. D003920's record should have no 'dateCreated' key
    leakage."""
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    record = sink.records["http://id.nlm.nih.gov/mesh/D003920"]
    # Only the 5 expected keys (xrefs added 2026-06-25 for the L7 xref ship).
    assert set(record.keys()) == {
        "types",
        "labels",
        "broader",
        "concepts",
        "xrefs",
    }


def test_stream_sink_records_descriptor_types_labels_and_concepts():
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    record = sink.records["http://id.nlm.nih.gov/mesh/D003920"]
    assert "http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor" in record["types"]
    # Label list of (text, lang) tuples.
    assert ("Diabetes Mellitus", "en") in record["labels"]
    # Concept link.
    assert "http://id.nlm.nih.gov/mesh/C100" in record["concepts"]


def test_stream_sink_records_broader_relationships():
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    record = sink.records["http://id.nlm.nih.gov/mesh/D003924"]
    assert record["broader"] == ["http://id.nlm.nih.gov/mesh/D003920"]


def test_stream_sink_records_concept_labels():
    """Concepts (not descriptors) also get retained because we'll
    walk through them later for synonym extraction."""
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    record = sink.records["http://id.nlm.nih.gov/mesh/C100"]
    assert ("Diabetes", "en") in record["labels"]


def test_build_terms_from_sink_produces_ontology_terms():
    """End-to-end: streaming parse + build produces the same OntologyTerm
    shape as the Graph-based `_extract_descriptors` path."""
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    terms = ontology_mesh_writes._build_terms_from_sink(sink)
    by_id = {t.id: t for t in terms}
    assert set(by_id) == {"MESH:D003920", "MESH:D003924"}

    # Descriptor D003920: label from rdfs:label, "diabetes" synonym from
    # the linked concept (lowercased).
    d003920 = by_id["MESH:D003920"]
    assert d003920.label == "Diabetes Mellitus"
    assert d003920.synonyms == ("diabetes",)
    assert d003920.parents == ()

    # Descriptor D003924: parent edge picked up.
    d003924 = by_id["MESH:D003924"]
    assert d003924.label == "Diabetes Mellitus, Type 2"
    assert d003924.parents == ("MESH:D003920",)


def test_build_terms_skips_subjects_without_descriptor_type():
    """Subjects that lack the TopicalDescriptor type are skipped
    (concepts, terms, anything not explicitly typed as a Descriptor)."""
    sink = _stream_parse(_SAMPLE_NTRIPLES)
    terms = ontology_mesh_writes._build_terms_from_sink(sink)
    # C100 is a Concept (not a Descriptor) and must not appear as a term.
    assert "MESH:C100" not in {t.id for t in terms}


def test_build_terms_skips_descriptors_without_label():
    """A subject typed as a Descriptor but with no rdfs:label is
    silently dropped (can't write a term without a label)."""
    no_label = (
        b"<http://id.nlm.nih.gov/mesh/D999999> "
        b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        b"<http://id.nlm.nih.gov/mesh/vocab#TopicalDescriptor> .\n"
    )
    sink = _stream_parse(no_label)
    terms = ontology_mesh_writes._build_terms_from_sink(sink)
    assert terms == []
