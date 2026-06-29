"""Tests for kg.ontology_helpers - the L7 import foundation.

Three concerns covered, with three test families:
  1. `OntologyTerm` dataclass — shape, frozen-ness, hashability.
  2. Cache + download — `get_cache_dir` + `ensure_cached`, with httpx
     stream patched to avoid real network calls.
  3. Term extraction — `extract_terms_skos` exercised against a real
     in-memory rdflib graph (small enough to build inline); pronto
     extraction patched via duck-typed mocks since building real
     pronto Ontologies in-memory is awkward.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    _validate_xrefs_mode,
    _xref_rel_from_term_label,
    ensure_cached,
    extract_terms_obo,
    extract_terms_owl,
    extract_terms_skos,
    get_cache_dir,
    read_rdf,
    write_ontology_terms,
)

# ---- OntologyTerm dataclass ----


def test_ontology_term_required_fields():
    t = OntologyTerm(
        id="MESH:D003920",
        label="Diabetes Mellitus",
        synonyms=("diabetes",),
        parents=(),
        definition=None,
    )
    assert t.id == "MESH:D003920"
    assert t.label == "Diabetes Mellitus"
    assert t.synonyms == ("diabetes",)
    assert t.parents == ()
    assert t.definition is None


def test_ontology_term_is_frozen():
    """Immutable: in-place mutation raises. Same pattern as Mention -
    lets readers / writers pass terms around without aliasing."""
    t = OntologyTerm(
        id="GO:0008150", label="biological_process",
        synonyms=(), parents=(), definition=None,
    )
    with pytest.raises(FrozenInstanceError):
        t.label = "changed"  # type: ignore[misc]


def test_ontology_term_hashable_and_value_equal():
    """Frozen dataclass is hashable; field-wise equality makes
    set/dict-of-terms operations clean."""
    a = OntologyTerm(
        id="GO:0008150", label="biological_process",
        synonyms=("bp",), parents=("GO:0000001",), definition=None,
    )
    b = OntologyTerm(
        id="GO:0008150", label="biological_process",
        synonyms=("bp",), parents=("GO:0000001",), definition=None,
    )
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


# ---- get_cache_dir + ensure_cached ----


def test_get_cache_dir_creates_directory(tmp_path: Path):
    """Cache directory is created on first call. Subsequent calls find
    it ready."""
    target = tmp_path / "ontology-cache"
    assert not target.exists()
    with patch(
        "knowledge_agent.kg.ontology_helpers.get_settings"
    ) as mock_settings:
        mock_settings.return_value.ontology_cache_dir = target
        cache = get_cache_dir()
    assert cache == target
    assert target.is_dir()


def test_ensure_cached_returns_existing_file(tmp_path: Path):
    """Cache hit: existing file path is returned, no download attempted."""
    target = tmp_path / "ontology-cache"
    target.mkdir()
    existing = target / "mesh.nt"
    existing.write_bytes(b"existing content")

    with (
        patch(
            "knowledge_agent.kg.ontology_helpers.get_settings"
        ) as mock_settings,
        patch("httpx.stream") as mock_stream,
    ):
        mock_settings.return_value.ontology_cache_dir = target
        result = ensure_cached("https://example.com/mesh.nt", "mesh.nt")

    assert result == existing
    assert result.read_bytes() == b"existing content"
    # httpx.stream NOT called - we short-circuited on the cache hit.
    mock_stream.assert_not_called()


def test_ensure_cached_downloads_when_missing(tmp_path: Path):
    """Cache miss: file is downloaded via httpx.stream and written."""
    target = tmp_path / "ontology-cache"
    target.mkdir()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.iter_raw = MagicMock(
        return_value=[b"chunk1", b"chunk2", b"chunk3"]
    )
    # httpx.stream is a context manager.
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_response)
    fake_cm.__exit__ = MagicMock(return_value=None)

    with (
        patch(
            "knowledge_agent.kg.ontology_helpers.get_settings"
        ) as mock_settings,
        patch("httpx.stream", return_value=fake_cm),
    ):
        mock_settings.return_value.ontology_cache_dir = target
        result = ensure_cached("https://example.com/go.obo", "go.obo")

    assert result == target / "go.obo"
    assert result.exists()
    assert result.read_bytes() == b"chunk1chunk2chunk3"


def test_ensure_cached_atomic_writes_via_tmp(tmp_path: Path):
    """The download writes to <name>.tmp first then renames. Verifies
    the .tmp file does NOT remain after a successful download."""
    target = tmp_path / "ontology-cache"
    target.mkdir()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.iter_raw = MagicMock(return_value=[b"hello"])
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_response)
    fake_cm.__exit__ = MagicMock(return_value=None)

    with (
        patch(
            "knowledge_agent.kg.ontology_helpers.get_settings"
        ) as mock_settings,
        patch("httpx.stream", return_value=fake_cm),
    ):
        mock_settings.return_value.ontology_cache_dir = target
        ensure_cached("https://example.com/x.nt", "x.nt")

    assert (target / "x.nt").exists()
    # .tmp must be cleaned up after successful write.
    assert not (target / "x.nt.tmp").exists()


def test_ensure_cached_cleans_up_tmp_on_failure(tmp_path: Path):
    """When the download fails partway, the partial .tmp file is
    removed so a retry starts clean (no half-written file masquerading
    as a valid cache)."""
    target = tmp_path / "ontology-cache"
    target.mkdir()

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()

    # Make iter_bytes yield one chunk then raise mid-stream.
    def _iter(chunk_size: int = 0):  # noqa: ARG001
        yield b"partial"
        raise RuntimeError("connection reset")

    fake_response.iter_raw = _iter
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_response)
    fake_cm.__exit__ = MagicMock(return_value=None)

    with (
        patch(
            "knowledge_agent.kg.ontology_helpers.get_settings"
        ) as mock_settings,
        patch("httpx.stream", return_value=fake_cm),
        pytest.raises(RuntimeError, match="connection reset"),
    ):
        mock_settings.return_value.ontology_cache_dir = target
        ensure_cached("https://example.com/x.nt", "x.nt")

    # No final file, no .tmp left behind.
    assert not (target / "x.nt").exists()
    assert not (target / "x.nt.tmp").exists()


def test_ensure_cached_force_redownloads(tmp_path: Path):
    """`force=True` bypasses the cache hit check and re-downloads,
    overwriting the existing cached file."""
    target = tmp_path / "ontology-cache"
    target.mkdir()
    existing = target / "mesh.nt"
    existing.write_bytes(b"old content")

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.iter_raw = MagicMock(return_value=[b"new content"])
    fake_cm = MagicMock()
    fake_cm.__enter__ = MagicMock(return_value=fake_response)
    fake_cm.__exit__ = MagicMock(return_value=None)

    with (
        patch(
            "knowledge_agent.kg.ontology_helpers.get_settings"
        ) as mock_settings,
        patch("httpx.stream", return_value=fake_cm),
    ):
        mock_settings.return_value.ontology_cache_dir = target
        result = ensure_cached(
            "https://example.com/mesh.nt", "mesh.nt", force=True
        )

    assert result == existing
    assert result.read_bytes() == b"new content"


# ---- extract_terms_skos (real rdflib graph, in-memory) ----


def _build_sample_skos_graph():
    """Build a tiny SKOS graph with three concepts modelling MeSH-style
    hierarchy:
      D003920 "Diabetes Mellitus" - root
      D003924 "Diabetes Mellitus, Type 2" - broader-of D003920
                                            altLabel "Type 2 Diabetes"
      D003922 "Diabetes Mellitus, Type 1" - broader-of D003920
    """
    import rdflib
    from rdflib.namespace import RDF, SKOS

    graph = rdflib.Graph()
    base = "http://id.nlm.nih.gov/mesh/"

    def add_concept(local_id: str, pref: str, parents=(), alt_labels=(), defn=None):
        uri = rdflib.URIRef(base + local_id)
        graph.add((uri, RDF.type, SKOS.Concept))
        graph.add((uri, SKOS.prefLabel, rdflib.Literal(pref, lang="en")))
        for parent_id in parents:
            graph.add((uri, SKOS.broader, rdflib.URIRef(base + parent_id)))
        for alt in alt_labels:
            graph.add((uri, SKOS.altLabel, rdflib.Literal(alt, lang="en")))
        if defn:
            graph.add((uri, SKOS.scopeNote, rdflib.Literal(defn, lang="en")))

    add_concept(
        "D003920", "Diabetes Mellitus",
        defn="A heterogeneous group of metabolic disorders.",
    )
    add_concept(
        "D003924", "Diabetes Mellitus, Type 2",
        parents=["D003920"],
        alt_labels=["Type 2 Diabetes", "NIDDM"],
    )
    add_concept(
        "D003922", "Diabetes Mellitus, Type 1",
        parents=["D003920"],
    )
    return graph


def test_extract_terms_skos_returns_ontology_terms():
    """Every skos:Concept in the graph becomes one OntologyTerm."""
    graph = _build_sample_skos_graph()
    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert set(by_id) == {
        "MESH:D003920",
        "MESH:D003924",
        "MESH:D003922",
    }


def test_extract_terms_skos_picks_pref_label_with_original_casing():
    """Primary label uses skos:prefLabel and preserves casing."""
    graph = _build_sample_skos_graph()
    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003920"].label == "Diabetes Mellitus"


def test_extract_terms_skos_collects_synonyms_lowercased():
    """All skos:altLabel values are gathered, lowercased, sorted."""
    graph = _build_sample_skos_graph()
    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003924"].synonyms == ("niddm", "type 2 diabetes")


def test_extract_terms_skos_extracts_parents():
    """skos:broader edges become parent IDs prefixed with id_prefix."""
    graph = _build_sample_skos_graph()
    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003924"].parents == ("MESH:D003920",)
    assert by_id["MESH:D003922"].parents == ("MESH:D003920",)
    assert by_id["MESH:D003920"].parents == ()


def test_extract_terms_skos_picks_up_definition():
    """skos:scopeNote becomes the OntologyTerm.definition."""
    graph = _build_sample_skos_graph()
    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert by_id["MESH:D003920"].definition == (
        "A heterogeneous group of metabolic disorders."
    )
    assert by_id["MESH:D003924"].definition is None


def test_extract_terms_skos_skips_concepts_without_label():
    """A skos:Concept that has no prefLabel is silently skipped -
    can't write a term without a label, and obsolete/incomplete
    entries shouldn't poison the import."""
    import rdflib
    from rdflib.namespace import RDF, SKOS

    graph = _build_sample_skos_graph()
    no_label = rdflib.URIRef("http://id.nlm.nih.gov/mesh/D999999")
    graph.add((no_label, RDF.type, SKOS.Concept))
    # No prefLabel added.

    terms = extract_terms_skos(graph, id_prefix="MESH")
    by_id = {t.id: t for t in terms}
    assert "MESH:D999999" not in by_id


# ---- extract_terms_owl (real rdflib graph, in-memory) ----


def _build_sample_owl_graph():
    """Build a tiny OWL graph mirroring OBO Foundry conventions:

      OBI:0000123 "process A" - root, no parents
        - hasExactSynonym "procA", "process-a" (different idioms)
        - IAO_0000115 definition "the canonical process A"
      OBI:0000456 "process B" - subclassOf OBI:0000123
        - hasRelatedSynonym "proc-b"
        - hasBroadSynonym "B-thing"
      OBI:0000789 "deprecated thing" - obsolete, should be skipped
        - owl:deprecated true
      OBI:0000999 - no label, should be skipped (no rdfs:label)
    """
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    graph = rdflib.Graph()
    obo = "http://purl.obolibrary.org/obo/"

    def class_uri(cid: str) -> rdflib.URIRef:
        return rdflib.URIRef(obo + cid)

    OBO_HAS_EXACT = rdflib.URIRef(
        "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
    )
    OBO_HAS_RELATED = rdflib.URIRef(
        "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym"
    )
    OBO_HAS_BROAD = rdflib.URIRef(
        "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym"
    )
    IAO_DEFINITION = rdflib.URIRef(
        "http://purl.obolibrary.org/obo/IAO_0000115"
    )

    # OBI:0000123 - root with two exact synonyms + definition
    a = class_uri("OBI_0000123")
    graph.add((a, RDF.type, OWL.Class))
    graph.add((a, RDFS.label, rdflib.Literal("process A", lang="en")))
    graph.add((a, OBO_HAS_EXACT, rdflib.Literal("procA", lang="en")))
    graph.add((a, OBO_HAS_EXACT, rdflib.Literal("process-a", lang="en")))
    graph.add(
        (a, IAO_DEFINITION,
         rdflib.Literal("the canonical process A", lang="en"))
    )

    # OBI:0000456 - child of A with related + broad synonyms
    b = class_uri("OBI_0000456")
    graph.add((b, RDF.type, OWL.Class))
    graph.add((b, RDFS.label, rdflib.Literal("process B", lang="en")))
    graph.add((b, RDFS.subClassOf, a))
    graph.add((b, OBO_HAS_RELATED, rdflib.Literal("proc-b", lang="en")))
    graph.add((b, OBO_HAS_BROAD, rdflib.Literal("B-thing", lang="en")))

    # OBI:0000789 - obsolete, must be filtered
    c = class_uri("OBI_0000789")
    graph.add((c, RDF.type, OWL.Class))
    graph.add((c, RDFS.label, rdflib.Literal("deprecated thing", lang="en")))
    graph.add(
        (c, OWL.deprecated,
         rdflib.Literal("true", datatype=rdflib.XSD.boolean))
    )

    # OBI:0000999 - no label, must be filtered
    d = class_uri("OBI_0000999")
    graph.add((d, RDF.type, OWL.Class))

    return graph


def test_extract_terms_owl_returns_one_term_per_owl_class():
    """Every owl:Class with rdfs:label becomes one OntologyTerm.
    Obsolete + label-less classes are skipped."""
    graph = _build_sample_owl_graph()
    terms = extract_terms_owl(graph, id_prefix="OBI")
    by_id = {t.id: t for t in terms}
    assert set(by_id) == {"OBI:0000123", "OBI:0000456"}


def test_extract_terms_owl_id_extractor_handles_obo_purl_convention():
    """Default id_extractor turns `<prefix>_<numeric>` -> `<prefix>:<numeric>`
    in the URI's last segment (the OBO PURL convention)."""
    graph = _build_sample_owl_graph()
    terms = extract_terms_owl(graph, id_prefix="OBI")
    assert any(t.id == "OBI:0000123" for t in terms)


def test_extract_terms_owl_pulls_all_synonym_idioms():
    """hasExactSynonym + hasRelatedSynonym + hasBroadSynonym are all
    collected, lowercased, deduped, sorted."""
    graph = _build_sample_owl_graph()
    by_id = {t.id: t for t in extract_terms_owl(graph, id_prefix="OBI")}
    # process A: two exact synonyms.
    assert by_id["OBI:0000123"].synonyms == ("proca", "process-a")
    # process B: one related + one broad.
    assert by_id["OBI:0000456"].synonyms == ("b-thing", "proc-b")


def test_extract_terms_owl_subclassof_becomes_parents():
    """rdfs:subClassOf to another owl:Class becomes a parent entry
    with the same prefixed ID format."""
    graph = _build_sample_owl_graph()
    by_id = {t.id: t for t in extract_terms_owl(graph, id_prefix="OBI")}
    assert by_id["OBI:0000456"].parents == ("OBI:0000123",)
    assert by_id["OBI:0000123"].parents == ()


def test_extract_terms_owl_picks_up_iao_definition():
    """IAO_0000115 (the OBO definition property) becomes
    OntologyTerm.definition."""
    graph = _build_sample_owl_graph()
    by_id = {t.id: t for t in extract_terms_owl(graph, id_prefix="OBI")}
    assert by_id["OBI:0000123"].definition == "the canonical process A"
    assert by_id["OBI:0000456"].definition is None


def test_extract_terms_owl_skips_blank_node_parents():
    """rdfs:subClassOf BNode (typical OWL restriction encoding) must
    NOT appear in the parents tuple - those describe restrictions,
    not is_a edges."""
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    graph = rdflib.Graph()
    cls = rdflib.URIRef("http://purl.obolibrary.org/obo/OBI_0000001")
    graph.add((cls, RDF.type, OWL.Class))
    graph.add((cls, RDFS.label, rdflib.Literal("thing", lang="en")))
    # BNode parent = OWL restriction; must be filtered.
    graph.add((cls, RDFS.subClassOf, rdflib.BNode()))

    terms = extract_terms_owl(graph, id_prefix="OBI")
    assert len(terms) == 1
    assert terms[0].parents == ()


def test_extract_terms_owl_skips_obsolete_classes():
    """owl:deprecated=true classes are filtered out entirely - no
    point linking entities to deprecated canonical concepts."""
    graph = _build_sample_owl_graph()
    by_id = {t.id: t for t in extract_terms_owl(graph, id_prefix="OBI")}
    assert "OBI:0000789" not in by_id


def test_extract_terms_owl_skips_blank_node_classes():
    """owl:Class targets that are blank nodes (anonymous restrictions
    / equivalence-class descriptions) have no canonical ID and must
    be skipped."""
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    graph = _build_sample_owl_graph()
    bnode = rdflib.BNode()
    graph.add((bnode, RDF.type, OWL.Class))
    graph.add((bnode, RDFS.label, rdflib.Literal("anonymous", lang="en")))

    terms = extract_terms_owl(graph, id_prefix="OBI")
    # Still only the 2 valid OBI classes.
    assert len(terms) == 2


# ---- read_rdf (gzip-aware) ----


def test_read_rdf_handles_gzipped_source(tmp_path: Path):
    """A gzipped RDF file (identified by `1f 8b` magic bytes) is
    transparently decompressed before parsing.

    Mirrors the production case: obo PURL endpoints often serve
    OWL/RDF with `Content-Encoding: gzip` on already-gzipped files,
    and `ensure_cached` preserves those bytes via `iter_raw`."""
    import gzip

    nt_content = (
        "<http://example.org/A> "
        "<http://www.w3.org/2000/01/rdf-schema#label> "
        '"alpha" .\n'
    )
    gz_path = tmp_path / "data.nt"
    with gzip.open(gz_path, "wb") as gz:
        gz.write(nt_content.encode("utf-8"))

    graph = read_rdf(gz_path, format="nt")
    # The triple parsed through the gzip transparency.
    assert len(graph) == 1


def test_read_rdf_handles_plain_source(tmp_path: Path):
    """A non-gzipped RDF file parses through the original path - the
    magic-bytes check leaves regular files untouched."""
    nt_content = (
        "<http://example.org/B> "
        "<http://www.w3.org/2000/01/rdf-schema#label> "
        '"beta" .\n'
    )
    nt_path = tmp_path / "data.nt"
    nt_path.write_text(nt_content, encoding="utf-8")

    graph = read_rdf(nt_path, format="nt")
    assert len(graph) == 1


# ---- extract_terms_obo (mocked pronto Ontology) ----


class _FakeSynonym:
    def __init__(self, description: str) -> None:
        self.description = description


class _FakeXref:
    """Duck-typed stand-in for `pronto.Xref` (the OBO `xref:` line type)."""

    def __init__(self, id_: str) -> None:
        self.id = id_


class _FakeTerm:
    """Duck-typed stand-in for `pronto.Term`."""

    def __init__(
        self,
        id_: str,
        name: str,
        synonyms: tuple[str, ...] = (),
        parent_ids: tuple[str, ...] = (),
        definition: str | None = None,
        obsolete: bool = False,
        xrefs: tuple[str, ...] = (),
    ) -> None:
        self.id = id_
        self.name = name
        self.synonyms = [_FakeSynonym(s) for s in synonyms]
        self._parent_ids = parent_ids
        self.definition = definition
        self.obsolete = obsolete
        self.xrefs = [_FakeXref(x) for x in xrefs]

    def superclasses(self, distance: int, with_self: bool):
        """Mimic pronto's superclasses() -> iterable of Term-likes."""
        for pid in self._parent_ids:
            yield _FakeTerm(pid, f"label-{pid}")


class _FakeOntology:
    def __init__(self, terms: list[_FakeTerm]) -> None:
        self._terms = terms

    def terms(self):
        return iter(self._terms)


def test_extract_terms_obo_returns_ontology_terms():
    ontology = _FakeOntology(
        [
            _FakeTerm("GO:0008150", "biological_process"),
            _FakeTerm(
                "GO:0007154", "cell communication",
                parent_ids=("GO:0008150",),
                definition="Any process that mediates interactions...",
            ),
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    by_id = {t.id: t for t in terms}
    assert set(by_id) == {"GO:0008150", "GO:0007154"}


def test_extract_terms_obo_lowercases_synonyms():
    ontology = _FakeOntology(
        [
            _FakeTerm(
                "GO:0001234", "kinase activity",
                synonyms=("Kinase Activity", "Phosphotransferase"),
            ),
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    assert terms[0].synonyms == ("kinase activity", "phosphotransferase")


def test_extract_terms_obo_extracts_is_a_parents():
    ontology = _FakeOntology(
        [
            _FakeTerm(
                "GO:0007154", "cell communication",
                parent_ids=("GO:0008150", "GO:0050794"),
            ),
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    assert terms[0].parents == ("GO:0008150", "GO:0050794")


def test_extract_terms_obo_skips_obsolete_terms():
    """Obsolete pronto terms are skipped from the output."""
    ontology = _FakeOntology(
        [
            _FakeTerm("GO:0008150", "biological_process"),
            _FakeTerm("GO:OBSOLETE", "old thing", obsolete=True),
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    ids = {t.id for t in terms}
    assert "GO:OBSOLETE" not in ids
    assert "GO:0008150" in ids


def test_extract_terms_obo_skips_terms_missing_label():
    """Terms with no name are skipped - same reason as the SKOS case
    (can't usefully write without a primary label)."""
    ontology = _FakeOntology(
        [
            _FakeTerm("GO:0008150", "biological_process"),
            _FakeTerm("GO:NOLABEL", ""),  # empty name
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    ids = {t.id for t in terms}
    assert "GO:NOLABEL" not in ids


def test_extract_terms_obo_definition_passes_through():
    ontology = _FakeOntology(
        [
            _FakeTerm(
                "GO:0001234", "kinase activity",
                definition="Catalysis of the transfer of a phosphate group...",
            ),
        ]
    )
    terms = extract_terms_obo(ontology, id_prefix="GO")
    assert terms[0].definition == (
        "Catalysis of the transfer of a phosphate group..."
    )


# ---- _xref_rel_from_term_label ----


def test_xref_rel_from_term_label_strips_term_and_uppercases():
    """The 18 derived strings must match the 18 `<X>_XREF_REL` constants
    declared in `kg/schema.py` (their construction rule)."""
    assert _xref_rel_from_term_label("MeSHTerm") == "MESH_XREF"
    assert _xref_rel_from_term_label("GOTerm") == "GO_XREF"
    assert _xref_rel_from_term_label("ChEBITerm") == "CHEBI_XREF"
    assert _xref_rel_from_term_label("NCBITaxonTerm") == "NCBITAXON_XREF"
    assert _xref_rel_from_term_label("FIBOTerm") == "FIBO_XREF"


def test_xref_rel_matches_all_18_shipped_schema_constants():
    """Every sub-label in ONTOLOGY_SUB_LABELS must derive to its
    corresponding ONTOLOGY_XREF_RELS entry by construction."""
    from knowledge_agent.kg.schema import (
        ONTOLOGY_SUB_LABELS,
        ONTOLOGY_XREF_RELS,
    )
    derived = tuple(_xref_rel_from_term_label(lbl) for lbl in ONTOLOGY_SUB_LABELS)
    assert derived == ONTOLOGY_XREF_RELS


# ---- _validate_xrefs_mode ----


def test_validate_xrefs_mode_accepts_three_states():
    """None of the three valid modes raise."""
    for mode in ("none", "collect_only", "use"):
        _validate_xrefs_mode(mode)


def test_validate_xrefs_mode_rejects_unknown():
    """Anything other than the 3-state literal raises ValueError so
    direct callers (smoke scripts, tests) get a clear boundary error."""
    with pytest.raises(ValueError, match="xrefs_mode must be"):
        _validate_xrefs_mode("yes")
    with pytest.raises(ValueError, match="xrefs_mode must be"):
        _validate_xrefs_mode("")
    with pytest.raises(ValueError, match="xrefs_mode must be"):
        _validate_xrefs_mode("NONE")  # case-sensitive


# ---- write_ontology_terms + xrefs_mode 3rd pass ----


from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402


@dataclass
class _StubResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class _StubSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    canned_results: list[_StubResult] = field(default_factory=list)

    def run(self, query: str, **params: Any):
        self.calls.append((query, params))
        idx = len(self.calls) - 1
        if idx < len(self.canned_results):
            return self.canned_results[idx]
        return _StubResult()

    def __enter__(self) -> "_StubSession":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


@dataclass
class _StubDriver:
    session_obj: _StubSession = field(default_factory=_StubSession)

    def session(self) -> _StubSession:
        return self.session_obj


@dataclass
class _StubClient:
    driver: _StubDriver = field(default_factory=_StubDriver)


def _term_with_xrefs(
    id_: str, parents=(), xrefs=()
) -> OntologyTerm:
    return OntologyTerm(
        id=id_,
        label=id_,
        synonyms=(),
        parents=tuple(parents),
        definition=None,
        xrefs=tuple(xrefs),
    )


def test_write_ontology_terms_default_xrefs_mode_is_none_no_extra_pass():
    """Default `xrefs_mode="none"`: only the 2 mandatory passes
    (nodes + hierarchy edges) run. No `dangling_xrefs` writes."""
    client = _StubClient()
    terms = [
        _term_with_xrefs("X:1"),
        _term_with_xrefs("X:2", parents=("X:1",), xrefs=("Y:42",)),
    ]
    ok = write_ontology_terms(
        client, terms,
        term_label="XTerm",
        hierarchy_rel="X_IS_A",
        ontology_name="X",
    )
    assert ok is None  # success: returns None (typed-errors contract)
    # 2 calls: nodes + hierarchy. No xref pass.
    assert len(client.driver.session_obj.calls) == 2


def test_write_ontology_terms_collect_only_writes_dangling_xrefs_no_edges():
    """`xrefs_mode="collect_only"`: 3rd pass stores dangling_xrefs but
    no resolved edges are written (no MERGE-edge pass)."""
    client = _StubClient()
    terms = [
        _term_with_xrefs("X:1", xrefs=("Y:42", "Z:99")),
        _term_with_xrefs("X:2"),  # no xrefs - row filtered out
    ]
    ok = write_ontology_terms(
        client, terms,
        term_label="XTerm",
        hierarchy_rel="X_IS_A",
        ontology_name="X",
        xrefs_mode="collect_only",
    )
    assert ok is None  # success: returns None (typed-errors contract)
    calls = client.driver.session_obj.calls
    # 2 mandatory passes + 1 dangling_xrefs pass. No resolved-edge pass.
    assert len(calls) == 2
    # Pass 1 = nodes (no hierarchy on these terms), pass 2 = dangling_xrefs.
    dangling_cypher, dangling_params = calls[1]
    assert "SET s.dangling_xrefs" in dangling_cypher
    # Only the term that has xrefs makes it into the payload.
    assert dangling_params["rows"] == [
        {"source_id": "X:1", "xrefs": ["Y:42", "Z:99"]},
    ]


def test_write_ontology_terms_use_mode_writes_dangling_and_resolved_edges():
    """`xrefs_mode="use"`: both the 3a dangling_xrefs pass AND the 3b
    resolved-edge MERGE pass run. Resolved-edge Cypher uses the
    derived `<X>_XREF` type."""
    client = _StubClient()
    # Pre-populate canned result for the resolved-edges query: pretend
    # 2 edges were actually MERGEd.
    client.driver.session_obj.canned_results = [
        _StubResult(),  # nodes pass
        _StubResult(),  # hierarchy pass (skipped - no parents)
        _StubResult(),  # dangling_xrefs pass
        _StubResult(rows=[{"n": 2}]),  # resolved-edges pass
    ]
    terms = [_term_with_xrefs("X:1", xrefs=("Y:42", "Z:99"))]
    ok = write_ontology_terms(
        client, terms,
        term_label="XTerm",
        hierarchy_rel="X_IS_A",
        ontology_name="X",
        xrefs_mode="use",
    )
    assert ok is None  # success: returns None (typed-errors contract)
    calls = client.driver.session_obj.calls
    # nodes + dangling_xrefs + resolved-edges. No hierarchy (no parents).
    assert len(calls) == 3
    resolved_cypher, resolved_params = calls[2]
    assert "MERGE (s)-[r:X_XREF]->(t)" in resolved_cypher
    assert "MATCH (t:OntologyTerm {id: xref_id})" in resolved_cypher
    assert resolved_params["rows"] == [
        {"source_id": "X:1", "xrefs": ["Y:42", "Z:99"]},
    ]


def test_write_ontology_terms_use_mode_no_xrefs_skips_pass_3():
    """`xrefs_mode="use"` with terms that have NO xrefs: pass 3 is
    skipped entirely (no `dangling_xrefs`, no resolved-edges)."""
    client = _StubClient()
    terms = [_term_with_xrefs("X:1"), _term_with_xrefs("X:2")]
    ok = write_ontology_terms(
        client, terms,
        term_label="XTerm",
        hierarchy_rel="X_IS_A",
        ontology_name="X",
        xrefs_mode="use",
    )
    assert ok is None  # success: returns None (typed-errors contract)
    # Just the nodes pass.
    assert len(client.driver.session_obj.calls) == 1


def test_write_ontology_terms_rejects_unknown_xrefs_mode():
    """The boundary check rejects unrecognised modes before any
    Cypher runs."""
    client = _StubClient()
    with pytest.raises(ValueError, match="xrefs_mode must be"):
        write_ontology_terms(
            client, [_term_with_xrefs("X:1")],
            term_label="XTerm",
            hierarchy_rel="X_IS_A",
            ontology_name="X",
            xrefs_mode="invalid",
        )
    # No session was opened.
    assert client.driver.session_obj.calls == []


def test_write_ontology_terms_use_mode_resolved_query_uses_correct_xref_rel():
    """Confirms the resolved-edges Cypher uses the derived xref edge
    type matching `_xref_rel_from_term_label(term_label)`. Verifies
    real ontology sub-labels (not just synthetic XTerm)."""
    client = _StubClient()
    client.driver.session_obj.canned_results = [
        _StubResult(),
        _StubResult(rows=[{"n": 1}]),
    ]
    terms = [_term_with_xrefs("MESH:D003920", xrefs=("DOID:9352",))]
    write_ontology_terms(
        client, terms,
        term_label="MeSHTerm",
        hierarchy_rel="MESH_BROADER",
        ontology_name="MeSH",
        xrefs_mode="use",
    )
    # 3 calls: nodes, dangling, resolved.
    resolved_cypher, _ = client.driver.session_obj.calls[2]
    assert ":MESH_XREF]" in resolved_cypher
