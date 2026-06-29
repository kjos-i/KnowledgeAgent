"""rdflib-based RDF/OWL/SKOS ontology reading + term extraction.

One of three format families split out of the historical
`ontology_helpers.py` (2026-06-29):

  - `ontology_pronto.py` — OBO Foundry via `pronto` library
  - `ontology_rdf.py` (THIS FILE) — RDF/OWL/SKOS via `rdflib` library
  - `ontology_writes.py` — Neo4j Cypher writes (format-agnostic)

rdflib is lazy-imported so corpora that only use OBO ontologies don't
pay the rdflib import cost.

Three extraction shapes covered here:

  - `extract_terms_owl` — OBO Foundry OWL distributions (OBI, DRON,
    EFO). Reads `owl:Class` subjects + `oboInOwl:has*Synonym` family +
    `obo:IAO_0000115` definitions + `rdfs:subClassOf` hierarchy. Built
    because pronto silently drops oboInOwl synonyms on OWL inputs
    (verified against OBI: 1377 source annotations -> 0 via pronto).
  - `extract_terms_skos` — SKOS thesauri (MeSH, AGROVOC, EuroVoc,
    GeoNames). Reads `skos:Concept` subjects + `skos:altLabel` family
    + `skos:scopeNote` definitions + `skos:broader` hierarchy.
  - `read_rdf` — generic RDF loader covering OWL/RDF-XML, Turtle,
    N-Triples, JSON-LD. Handles gzipped source files transparently.

The `id_extractor` callbacks (`_owl_id_extractor`,
`_default_id_extractor`) and the language-tag filter
(`_first_english_literal`) are private helpers used by both extractor
families.
"""

from __future__ import annotations

import gzip
import logging
from typing import TYPE_CHECKING, Any

from knowledge_agent.kg.ontology_helpers import OntologyTerm

if TYPE_CHECKING:
    from pathlib import Path

    import rdflib


logger = logging.getLogger(__name__)


def read_rdf(path, format: str | None = None):  # type: ignore[no-untyped-def]
    """Parse an RDF file via rdflib.

    `format` is rdflib's format hint ('nt' for N-Triples, 'turtle',
    'xml' for RDF/XML, 'json-ld'). When None, rdflib infers from the
    file extension. The per-ontology module typically passes an
    explicit format because rdflib's extension detection is loose.

    Returns the populated `rdflib.Graph`; the caller uses graph
    iteration or SPARQL to extract terms. Lazy-imports rdflib.

    Handles gzipped sources transparently: many obo PURL endpoints
    serve OWL/RDF with `Content-Encoding: gzip` on already-gzipped
    files, and `ensure_cached` preserves those bytes via `iter_raw`.
    A magic-bytes check (`1f 8b`) decides whether to pass a gzip
    file object or the raw path to rdflib.

    Heavy: a 600 MB MeSH N-Triples file takes ~30-60 seconds to parse
    and a few GB of RAM while loaded. Called once per ontology import.
    """
    import rdflib

    logger.info("ontology: parsing RDF file %s (format=%s)", path, format)
    graph = rdflib.Graph()

    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        logger.info("ontology: %s is gzip-compressed, streaming decompressed", path)
        with gzip.open(path, "rb") as gz:
            graph.parse(gz, format=format)
    else:
        graph.parse(str(path), format=format)
    return graph


def extract_terms_owl(
    graph,  # type: ignore[no-untyped-def]
    id_prefix: str,
    *,
    synonym_properties: tuple[str, ...] = (
        "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
        "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
        "http://purl.obolibrary.org/obo/IAO_0000118",  # "alternative term"
    ),
    xref_properties: tuple[str, ...] = (
        "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
    ),
    definition_property: str = "http://purl.obolibrary.org/obo/IAO_0000115",
    id_extractor: Any = None,
) -> list[OntologyTerm]:
    """Extract `OntologyTerm` records from an OWL-shaped RDF graph.

    Covers OBO Foundry OWL distributions (OBI, DRON, EFO, ...) and any
    other ontology that uses the standard OWL/RDFS idioms - subjects
    typed as `owl:Class`, `rdfs:label`, `oboInOwl:has*Synonym` family,
    `obo:IAO_0000115` definition, `rdfs:subClassOf` hierarchy.

    Why this exists alongside `extract_terms_obo` (pronto-based) AND
    `extract_terms_skos` (rdflib-based SKOS): pronto reads OWL but
    silently drops oboInOwl synonyms (verified against OBI: 1377
    synonym annotations in the source -> 0 extracted via pronto). OWL
    canonicalisation needs the synonym surface, so we read OWL via
    rdflib + this function. The Neo4j-write side is shared - the
    `write_ontology_terms` family helpers are format-agnostic.

    Skipped subjects:
      - Blank nodes (anonymous restrictions / equivalent-class
        descriptions; have no canonical ID).
      - Classes with no `rdfs:label` (anonymous unions, intersections,
        and ontology-control classes like `owl:Thing`).
      - Classes marked `owl:deprecated true` (obsolete terms).

    Args:
      graph: an rdflib Graph already populated with the ontology
        (typically via `read_rdf(path, format="xml")` for OWL/RDF-XML).
      id_prefix: the ontology's canonical prefix (e.g. "OBI"). Used in
        log messages; the actual term IDs come from the URI via
        `id_extractor`. Foreign-import classes (e.g. BFO terms imported
        into OBI) keep their native prefix - this matches the pronto
        behaviour and is correct for canonical matching.
      synonym_properties: RDF properties to collect as synonyms.
        Default covers the 4 standard oboInOwl synonym predicates +
        IAO's "alternative term". Pass extras for ontologies with
        non-standard synonym idioms.
      xref_properties: RDF properties whose literal objects are
        cross-ontology xrefs (prefixed canonical IDs of equivalent /
        related concepts in OTHER ontologies). Default: just
        `oboInOwl:hasDbXref` (the OBO-in-OWL standard). Stored as
        literal strings on the OntologyTerm's `xrefs` field; later
        materialised as `:<X>_XREF` edges by `write_ontology_terms`
        when the `xrefs` layer is in "use" mode.
      definition_property: RDF property for the textual definition.
        Default: IAO_0000115 (the OBO Foundry convention).
      id_extractor: function `URIRef -> str` producing a colon-prefixed
        ID (e.g. URIRef "http://purl.obolibrary.org/obo/OBI_0000489" ->
        "OBI:0000489"). Defaults to `_owl_id_extractor` which handles
        both the obo PURL convention and the EBI EFO convention.

    Returns a list of OntologyTerm with prefixed IDs. Hierarchy parents
    skip blank-node `subClassOf` targets (those describe restrictions,
    not class-to-class is_a edges).
    """
    import rdflib
    from rdflib.namespace import OWL, RDF, RDFS

    if id_extractor is None:
        id_extractor = _owl_id_extractor

    synonym_props = [rdflib.URIRef(p) for p in synonym_properties]
    xref_props = [rdflib.URIRef(p) for p in xref_properties]
    def_prop = rdflib.URIRef(definition_property)
    deprecated_prop = OWL.deprecated

    terms: list[OntologyTerm] = []
    for class_uri in graph.subjects(RDF.type, OWL.Class):
        # Anonymous classes (blank nodes) describe restrictions /
        # equivalence axioms - skip them, they have no canonical ID.
        if not isinstance(class_uri, rdflib.URIRef):
            continue

        # Skip obsolete classes.
        deprecated = next(
            iter(graph.objects(class_uri, deprecated_prop)), None
        )
        if deprecated is not None and str(deprecated).lower() == "true":
            continue

        # Need a label to be useful at canonical-matching time.
        label = _first_english_literal(graph, class_uri, RDFS.label)
        if not label:
            continue

        class_id = id_extractor(class_uri)
        if not class_id:
            continue

        # Synonyms across all configured properties, lowercased + deduped.
        synonyms: set[str] = set()
        for prop in synonym_props:
            for obj in graph.objects(class_uri, prop):
                if isinstance(obj, rdflib.Literal):
                    text = str(obj).strip()
                    if text:
                        synonyms.add(text.lower())

        # Parents via rdfs:subClassOf - skip blank-node targets (those
        # are restriction descriptions, not class-to-class is_a edges).
        parents: list[str] = []
        for parent_uri in graph.objects(class_uri, RDFS.subClassOf):
            if isinstance(parent_uri, rdflib.URIRef):
                parent_id = id_extractor(parent_uri)
                if parent_id:
                    parents.append(parent_id)

        # L7 xrefs via `oboInOwl:hasDbXref` (the OBO-in-OWL standard
        # annotation property). Stored as literal strings carrying the
        # prefixed canonical ID (e.g. "MESH:D003920") of the equivalent
        # / related concept in another ontology. Deduped via set.
        xrefs: set[str] = set()
        for prop in xref_props:
            for obj in graph.objects(class_uri, prop):
                if isinstance(obj, rdflib.Literal):
                    text = str(obj).strip()
                    if text:
                        xrefs.add(text)

        definition = _first_english_literal(graph, class_uri, def_prop)

        terms.append(
            OntologyTerm(
                id=class_id,
                label=label,
                synonyms=tuple(sorted(synonyms)),
                parents=tuple(sorted(parents)),
                definition=definition,
                xrefs=tuple(sorted(xrefs)),
            )
        )

    logger.info(
        "ontology: extracted %d terms from OWL graph (prefix=%s)",
        len(terms), id_prefix,
    )
    return terms


def _owl_id_extractor(uri: Any) -> str:
    """Default ID extractor for OWL ontologies.

    Handles the OBO Foundry PURL convention
    (`http://purl.obolibrary.org/obo/OBI_0000489` -> `"OBI:0000489"`)
    and the EBI EFO convention (`http://www.ebi.ac.uk/efo/EFO_0000001`
    -> `"EFO:0000001"`). Other URIs fall through to the SKOS default:
    last path segment, untransformed.

    Both OBO conventions use `<prefix>_<numeric>` in the last segment;
    we split on `_` once and re-join with `:`. URIs without that
    pattern (e.g. `owl:Thing`, foreign vocabulary terms) return the
    raw last segment - the caller filters empty / unwanted IDs.
    """
    text = str(uri)
    if "#" in text:
        last = text.rsplit("#", 1)[1]
    else:
        last = text.rsplit("/", 1)[1]
    # OBO style: "OBI_0000489" -> "OBI:0000489".
    if "_" in last:
        prefix, _, suffix = last.partition("_")
        if prefix and suffix and suffix[0].isdigit():
            return f"{prefix}:{suffix}"
    return last


def extract_terms_skos(
    graph,  # type: ignore[no-untyped-def]
    id_prefix: str,
    *,
    label_property: str = "http://www.w3.org/2004/02/skos/core#prefLabel",
    synonym_properties: tuple[str, ...] = (
        "http://www.w3.org/2004/02/skos/core#altLabel",
    ),
    hierarchy_property: str = "http://www.w3.org/2004/02/skos/core#broader",
    definition_property: str = (
        "http://www.w3.org/2004/02/skos/core#scopeNote"
    ),
    id_extractor: Any = None,
) -> list[OntologyTerm]:
    """Extract `OntologyTerm` records from a SKOS-shaped RDF graph.

    Covers MeSH, AGROVOC, Eurovoc, GeoNames RDF, and any other SKOS
    thesaurus. Each `skos:Concept` becomes one OntologyTerm.

    Args:
      graph: an rdflib Graph already populated with the ontology.
      id_prefix: canonical ID prefix used in the output records
        (e.g. "MESH" -> "MESH:D003920"). The function extracts an ID
        from each concept's URI using `id_extractor` (or a default
        that takes the URI's last path segment).
      label_property: RDF property for the primary label. Default:
        skos:prefLabel.
      synonym_properties: RDF properties to collect as synonyms.
        Default: skos:altLabel only; pass extras for ontologies that
        use multiple synonym properties.
      hierarchy_property: RDF property for the parent relation.
        Default: skos:broader.
      definition_property: RDF property for the textual definition.
        Default: skos:scopeNote (MeSH); other ontologies use
        skos:definition.
      id_extractor: function `URIRef -> str` that produces a bare ID
        from a concept URI (e.g. "http://id.nlm.nih.gov/mesh/D003920"
        -> "D003920"). Defaults to splitting on '/' and '#' and
        taking the last segment.

    Returns a list of OntologyTerm with IDs prefixed `id_prefix:`.
    Multilingual altLabels are filtered to English-tagged or untagged
    literals only.
    """
    import rdflib
    from rdflib.namespace import RDF, SKOS

    if id_extractor is None:
        id_extractor = _default_id_extractor

    label_prop = rdflib.URIRef(label_property)
    synonym_props = [rdflib.URIRef(p) for p in synonym_properties]
    hier_prop = rdflib.URIRef(hierarchy_property)
    def_prop = rdflib.URIRef(definition_property)

    terms: list[OntologyTerm] = []
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        # Primary label - take the first English / untagged literal.
        label = _first_english_literal(graph, concept, label_prop)
        if label is None:
            continue
        concept_id = f"{id_prefix}:{id_extractor(concept)}"

        # Synonyms across all configured synonym properties, lowercased + deduped.
        synonyms: set[str] = set()
        for prop in synonym_props:
            for obj in graph.objects(concept, prop):
                if isinstance(obj, rdflib.Literal):
                    text = str(obj).strip()
                    if text:
                        synonyms.add(text.lower())

        # Parents via hierarchy property.
        parents: list[str] = []
        for parent_uri in graph.objects(concept, hier_prop):
            if isinstance(parent_uri, rdflib.URIRef):
                parents.append(f"{id_prefix}:{id_extractor(parent_uri)}")

        # Definition (optional).
        definition = _first_english_literal(graph, concept, def_prop)

        terms.append(
            OntologyTerm(
                id=concept_id,
                label=label,
                synonyms=tuple(sorted(synonyms)),
                parents=tuple(sorted(parents)),
                definition=definition,
            )
        )
    logger.info(
        "ontology: extracted %d terms from SKOS-shaped RDF graph (prefix=%s)",
        len(terms), id_prefix,
    )
    return terms


def _default_id_extractor(uri: Any) -> str:
    """Default id extractor: last path segment of the URI.

    Handles both `/`-separated and `#`-fragment URIs. Examples:
      "http://id.nlm.nih.gov/mesh/D003920" -> "D003920"
      "http://www.w3.org/2004/02/skos/core#Concept" -> "Concept"
    """
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rsplit("/", 1)[1]


def _first_english_literal(
    graph,  # type: ignore[no-untyped-def]
    subject: Any,
    predicate: Any,
) -> str | None:
    """Return the first English-tagged or untagged literal for
    `(subject, predicate, *)`, or None if none found.

    Multilingual ontologies (Eurovoc, AGROVOC, MeSH) typically tag
    labels with language codes. We pick English-tagged labels
    preferentially; untagged literals are accepted as fallback.
    """
    import rdflib

    untagged: str | None = None
    for obj in graph.objects(subject, predicate):
        if not isinstance(obj, rdflib.Literal):
            continue
        if obj.language == "en":
            return str(obj).strip()
        if obj.language is None and untagged is None:
            untagged = str(obj).strip()
    return untagged
