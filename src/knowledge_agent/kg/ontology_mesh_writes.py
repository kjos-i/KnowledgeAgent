"""L7 ontology writes for MeSH (Medical Subject Headings).

MeSH is NLM's biomedical thesaurus - the broadest controlled vocabulary
covering diseases, chemicals, anatomy, organisms, and methods.
Distributed by NLM as RDF (N-Triples), NOT as OBO, so this module uses
rdflib (via the shared `read_rdf` helper) rather than pronto.

MeSH does NOT use pure SKOS. It uses NLM's own `meshv:` vocabulary:
  - Top-level concepts are typed as `meshv:TopicalDescriptor` (and a few
    sibling types for Geographicals, CheckTags, etc.).
  - Primary labels live on `rdfs:label`.
  - Tree-hierarchy is `meshv:broaderDescriptor` (NOT skos:broader).
  - Synonyms live across `meshv:concept` -> `meshv:term` chains; this
    module walks one hop to gather Concept-level labels as synonyms.

What we write to Neo4j (multi-label):
  (:OntologyTerm:MeSHTerm {id, label, synonyms, definition})
  (:MeSHTerm)-[:MESH_BROADER]->(:MeSHTerm)

Each `id` is prefixed `MESH:` (e.g. `MESH:D003920` for "Diabetes Mellitus").

Lifecycle:
  - `is_imported(client)` checks for any :MeSHTerm node; cheap query.
  - `import_mesh(client, force=False)` downloads + parses + writes
    if not already imported. Idempotent. `force=True` re-imports
    from scratch (useful after a MeSH yearly release update).
  - `delete_imported(client)` drops all MeSH data + edges (for clean
    wipes, not normal re-ingest).

`Neo4jClient` exposes these as 1-line delegates in `kg/client.py`,
matching the existing per-domain module pattern.
"""

from __future__ import annotations

import gzip
import logging
from typing import TYPE_CHECKING, Any

from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    require_cached,
    write_ontology_terms,
)
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.schema import (
    MESH_BROADER_REL,
    MESH_TERM_LABEL,
)

if TYPE_CHECKING:
    from pathlib import Path

    import rdflib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module metadata (consumed by the future wizard).
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("medicine", "biology", "chemistry", "nutrition")
"""Domain tags this ontology serves. The wizard suggests MeSH when the
user picks any of these tags.

Kept as a standalone constant for back-compat with any code that
imports it directly. The authoritative copy lives on
`_MESH_PROVENANCE.domain_tags` below; this is a re-export."""


# L6 entity_type labels with strong MeSH term coverage. Cross-link
# helpers (kg.ontology_lifecycle.get_canonicalization_candidates)
# join this against each extractor's `emitted_labels` to surface the
# "pairs well with" line in install dialogs. GENE is included even
# though MeSH's gene/protein coverage is shallow — MeSH has some
# gene/protein descriptors but PR/MONDO/etc. are better targets when
# available; the dialog surfaces ALL ontologies a label maps to and
# lets the user decide.
_MESH_COVERS_LABELS: tuple[str, ...] = (
    "DISEASE",
    "CHEMICAL",
    "ANATOMY",
    "GENE",
)


# ---------------------------------------------------------------------------
# Download configuration. NLM publishes MeSH on a yearly cycle; this
# module needs the URL bumped when a new release arrives. The
# corresponding cache filename matches the URL's terminal filename.
# ---------------------------------------------------------------------------

MESH_YEAR = "2025"
MESH_DOWNLOAD_URL = (
    f"https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/{MESH_YEAR}/mesh{MESH_YEAR}.nt.gz"
)
MESH_CACHE_FILENAME = f"mesh{MESH_YEAR}.nt.gz"
MESH_ID_PREFIX = "MESH"
# Cached source file size. Used by `ontology_lifecycle.import_ontology_plan`
# to surface "MeSH (~115 MB download)" in the confirmation dialog when
# the ontology isn't yet imported. NLM ships gzipped N-Triples and the
# pipeline keeps it gzipped on disk (see ontology_helpers.ensure_cached's
# `iter_raw` note); the constant tracks the gzipped size.
#
# Kept as a standalone constant for back-compat with the registry's
# `"download_size_mb"` key. The authoritative copy lives on
# `_MESH_PROVENANCE.download_size_mb` below; this is a re-export.
DOWNLOAD_SIZE_MB = 115


# ---------------------------------------------------------------------------
# Provenance: structured per-ontology install-dialog metadata. The
# OntologyProvenance dataclass (from the leaf `ontology_provenance`
# module to avoid the circular-import cycle through ontology_lifecycle)
# mirrors the ModelProvenance pattern from the extractor side — full
# publisher / license / source / domain / covers-labels surface so the
# install dialog can render an informed pre-install summary instead of
# a bare "(~115 MB)" line.
# ---------------------------------------------------------------------------

_MESH_PROVENANCE = OntologyProvenance(
    ontology_name="mesh",
    full_name="Medical Subject Headings",
    publisher="U.S. National Library of Medicine (NLM)",
    license="NLM Terms (free use with attribution; no endorsement claim)",
    source_url="https://www.nlm.nih.gov/mesh/meshhome.html",
    download_url=MESH_DOWNLOAD_URL,
    file_format="N-Triples (gzipped)",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=30000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_MESH_COVERS_LABELS,
    description=(
        "NLM's biomedical thesaurus — the broadest controlled "
        "vocabulary in biomedicine. Topical Descriptors (~30K) "
        "cover diseases, chemicals, anatomy, organisms, and "
        "methods. Gene/protein coverage is shallow; use PR or "
        "MONDO for those when available."
    ),
    heavy_warning=None,
)

# RDF namespace URIs MeSH uses.
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_MESHV = "http://id.nlm.nih.gov/mesh/vocab#"

# The descriptor types we extract. MeSH has several Descriptor classes;
# TopicalDescriptor covers the vast majority (diseases, chemicals,
# anatomy, methods). Extend this tuple to pull in GeographicalDescriptor,
# PublicationType, etc. if future corpora need them.
_DESCRIPTOR_TYPES: tuple[str, ...] = (f"{_MESHV}TopicalDescriptor",)


# ---------------------------------------------------------------------------
# Public API: import + delete.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when MeSH data is already present in Neo4j.

    Checks for the existence of at least one `:MeSHTerm` node. Cheap
    (label index lookup). Cypher failures propagate to the caller
    (typed-errors contract); `ensure_ontology_imported` is the
    orchestrator boundary that catches per-ontology.
    """
    async with client.driver.session() as session:
        result = await session.run(f"MATCH (t:{MESH_TERM_LABEL}) RETURN count(t) > 0 AS present")
        row = await result.single()
        return bool(row and row["present"])


async def import_mesh(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write MeSH to Neo4j. Idempotent.

    Returns `True` when the import actually ran, `False` when MeSH was
    already imported (no-op). When `force=True`, deletes existing MeSH
    data first then runs the full import.

    `xrefs_mode` is forwarded to the shared `write_ontology_terms`
    helper. Default `"none"`; see that helper for the three accepted
    values and what each writes.

    Download / parse / write failures and the "extracted 0 terms"
    edge case propagate to the caller under the typed-errors contract.
    """
    if not force and await is_imported(client):
        logger.info("MeSH: already imported; use force=True to re-import")
        return False

    if force:
        logger.info("MeSH: force=True - dropping existing data before re-import")
        await delete_imported(client)

    # Ingest never auto-downloads — require the file already on disk
    # (downloaded via Library → Installs); raises OntologyNotDownloadedError
    # otherwise, caught per-ontology by ensure_ontology_imported.
    logger.info("MeSH: loading cached source %s", MESH_CACHE_FILENAME)
    path = require_cached(MESH_CACHE_FILENAME, "MeSH")
    terms = _read_and_extract(path)

    if not terms:
        raise RuntimeError("MeSH: extracted 0 terms - unexpected, aborting write")

    await write_terms(client, terms, xrefs_mode=xrefs_mode)
    return True


async def delete_imported(client) -> None:
    """Drop all MeSH data: every :MeSHTerm node + its :MESH_BROADER edges.

    Idempotent; safe to call when MeSH was never imported (no-op).
    Used by `await import_mesh(force=True)` and by maintenance flows that want
    to start fresh after a yearly MeSH release.

    Cypher failures propagate to the caller (typed-errors contract).
    """
    async with client.driver.session() as session:
        # DETACH DELETE handles all incoming + outgoing edges
        # (MESH_BROADER between MeSHTerms, future CANONICAL_TO
        # from :Entity nodes).
        await session.run(f"MATCH (t:{MESH_TERM_LABEL}) DETACH DELETE t")
    logger.info("MeSH: deleted all :MeSHTerm nodes + edges")


# ---------------------------------------------------------------------------
# Cypher writes.
# ---------------------------------------------------------------------------


async def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:MeSHTerm` nodes + `:MESH_BROADER` edges.

    Delegates to the shared `write_ontology_terms` helper. The helper
    is format-agnostic — MeSH was the first ontology to ship before
    the shared family existed, so this thin wrapper kept the interface
    stable while internally duplicating the helper's Cypher. The
    refactor (2026-06-25, alongside the L7 xrefs / L10 ship) collapses
    the duplication: same Cypher shape, single source of truth.

    `xrefs_mode` is forwarded to the helper for the optional L7 xref
    3rd pass. See `write_ontology_terms` for the three accepted values.

    Public for testability + direct write paths (some tests construct
    OntologyTerm lists and call this directly).
    """
    await write_ontology_terms(
        client,
        terms,
        term_label=MESH_TERM_LABEL,
        hierarchy_rel=MESH_BROADER_REL,
        ontology_name="MeSH",
        xrefs_mode=xrefs_mode,
    )


# ---------------------------------------------------------------------------
# Reading + extraction. Internal - import_mesh is the public entry.
# ---------------------------------------------------------------------------


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached MeSH file and yield OntologyTerm records.

    Uses a STREAMING N-Triples parser (rdflib's W3CNTriplesParser with a
    custom sink) rather than building a full `rdflib.Graph`. The Graph
    approach is convenient but it loads all ~30M MeSH triples as Python
    objects, costing 10-17 GB of RAM. Streaming with a per-subject
    accumulator keeps memory to ~100 MB - the sink only retains the
    handful of triples we actually use (rdf:type, rdfs:label,
    meshv:broaderDescriptor, meshv:concept) and discards the rest.

    Detects the on-disk format by inspecting the first two bytes
    (gzip magic = `\\x1f\\x8b`) rather than trusting the filename
    extension. Some HTTP clients auto-decompress `Content-Encoding:
    gzip` responses, so a file named `mesh.nt.gz` may actually hold
    plain N-Triples on disk - the magic-bytes check handles both.
    """
    from rdflib.plugins.parsers.ntriples import W3CNTriplesParser

    with path.open("rb") as fh:
        magic = fh.read(2)
    is_gzipped = magic == b"\x1f\x8b"

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        "MeSH: streaming N-Triples parse of %s (%.0f MB, gzipped=%s) - "
        "single-threaded pure Python; expect ~1-3 minutes silent block "
        "during parse.",
        path,
        size_mb,
        is_gzipped,
    )

    sink = _MeshStreamSink()
    parser = W3CNTriplesParser(sink=sink)
    if is_gzipped:
        with gzip.open(path, "rb") as fh:
            parser.parse(fh)
    else:
        with path.open("rb") as fh:
            parser.parse(fh)

    logger.info(
        "MeSH: parse complete - sink retained %d subject records, "
        "%d total triples. Building descriptors...",
        len(sink.records),
        sink.triple_count,
    )
    terms = _build_terms_from_sink(sink)
    logger.info(
        "MeSH: extraction complete - %d descriptors ready to write",
        len(terms),
    )
    return terms


# ---------------------------------------------------------------------------
# Streaming N-Triples sink + OntologyTerm builder.
# ---------------------------------------------------------------------------


class _MeshStreamSink:
    """rdflib W3CNTriplesParser sink that accumulates only MeSH-relevant
    triples per subject.

    Why this instead of `rdflib.Graph`: a full Graph for ~30M MeSH
    triples consumes 10-17 GB of RAM (each triple is 3 Python objects
    with verbose internal representation). This sink keeps only the
    fields we actually consume downstream (rdf:type, rdfs:label,
    meshv:broaderDescriptor, meshv:concept, skos:closeMatch / exactMatch)
    and stores them as strings + small tuples. Memory: ~100 MB regardless
    of MeSH size.

    The sink object IS the parsed result - callers iterate
    `sink.records` after `parser.parse(...)` finishes.
    """

    # Property URIs we keep. Anything else is silently dropped.
    _RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    _RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
    _MESHV_BROADER = f"{_MESHV}broaderDescriptor"
    _MESHV_CONCEPT = f"{_MESHV}concept"
    # L7 xrefs: SKOS predicates MeSH uses to point at equivalent /
    # close-match concepts in OTHER vocabularies (LCSH, EuroVoc, etc.).
    # Stored verbatim as the source URI - resolves to a shipped
    # :OntologyTerm only when our prefix conventions happen to match;
    # otherwise stays in `dangling_xrefs` for future backfill.
    _SKOS_CLOSE_MATCH = "http://www.w3.org/2004/02/skos/core#closeMatch"
    _SKOS_EXACT_MATCH = "http://www.w3.org/2004/02/skos/core#exactMatch"
    _RELEVANT_PROPERTIES = frozenset(
        {
            _RDF_TYPE,
            _RDFS_LABEL,
            _MESHV_BROADER,
            _MESHV_CONCEPT,
            _SKOS_CLOSE_MATCH,
            _SKOS_EXACT_MATCH,
        }
    )

    def __init__(self) -> None:
        # subject URI string -> dict with keys:
        #   "types"    : list[str]
        #   "labels"   : list[tuple[str, str | None]]   (text, lang)
        #   "broader"  : list[str]
        #   "concepts" : list[str]
        #   "xrefs"    : list[str]                      (verbatim URIs)
        self.records: dict[str, dict[str, Any]] = {}
        # Counter exposed for log output only - cheap to maintain.
        self.triple_count: int = 0

    def triple(self, s: Any, p: Any, o: Any) -> None:
        """Called per parsed triple. We filter on property URI and only
        accumulate the relevant fields, storing string values to keep the
        sink lightweight."""
        self.triple_count += 1
        p_str = str(p)
        if p_str not in self._RELEVANT_PROPERTIES:
            return

        s_str = str(s)
        record = self.records.get(s_str)
        if record is None:
            record = {
                "types": [],
                "labels": [],
                "broader": [],
                "concepts": [],
                "xrefs": [],
            }
            self.records[s_str] = record

        if p_str == self._RDF_TYPE:
            record["types"].append(str(o))
        elif p_str == self._RDFS_LABEL:
            # Preserve language tag so we can prefer English labels later.
            # `o.language` exists on rdflib.Literal; safe to access.
            lang = getattr(o, "language", None)
            record["labels"].append((str(o), lang))
        elif p_str == self._MESHV_BROADER:
            record["broader"].append(str(o))
        elif p_str == self._MESHV_CONCEPT:
            record["concepts"].append(str(o))
        elif p_str in (self._SKOS_CLOSE_MATCH, self._SKOS_EXACT_MATCH):
            # Object is a URI pointing at the equivalent concept in
            # another vocabulary. Stored verbatim.
            record["xrefs"].append(str(o))


def _build_terms_from_sink(sink: _MeshStreamSink) -> list[OntologyTerm]:
    """Walk the sink's accumulated records and build OntologyTerm records.

    Same semantics as `_extract_descriptors` (the Graph-based path used
    in tests) but operating on the sink's dict-of-strings representation
    instead of an rdflib.Graph.
    """
    descriptor_type = f"{_MESHV}TopicalDescriptor"
    terms: list[OntologyTerm] = []

    for subject_uri, record in sink.records.items():
        if descriptor_type not in record["types"]:
            continue

        label = _pick_english_or_untagged(record["labels"])
        if label is None:
            continue

        descriptor_id = f"{MESH_ID_PREFIX}:{_local_name(subject_uri)}"
        primary_lower = label.lower()

        # Synonyms via Descriptor -> Concept -> rdfs:label hop.
        synonyms: set[str] = set()
        for concept_uri in record["concepts"]:
            concept_record = sink.records.get(concept_uri)
            if concept_record is None:
                continue
            concept_label = _pick_english_or_untagged(concept_record["labels"])
            if concept_label and concept_label.lower() != primary_lower:
                synonyms.add(concept_label.lower())

        parents = sorted(f"{MESH_ID_PREFIX}:{_local_name(p)}" for p in record["broader"])

        # L7 xrefs: stored verbatim as the URI strings the sink
        # captured. Most MeSH xref targets are LCSH / EuroVoc URIs
        # that won't match our prefixed canonical IDs - they stay
        # in `dangling_xrefs` until a future LCSH/EuroVoc ship.
        xrefs = tuple(sorted(set(record.get("xrefs", []))))

        terms.append(
            OntologyTerm(
                id=descriptor_id,
                label=label,
                synonyms=tuple(sorted(synonyms)),
                parents=tuple(parents),
                definition=None,
                xrefs=xrefs,
            )
        )

    return terms


def _pick_english_or_untagged(
    labels: list[tuple[str, str | None]],
) -> str | None:
    """Pick the first English-tagged label; fall back to first untagged.
    Returns None when no usable label is present."""
    untagged: str | None = None
    for text, lang in labels:
        if lang == "en":
            return text.strip()
        if lang is None and untagged is None:
            untagged = text.strip()
    return untagged


def _extract_descriptors(graph: rdflib.Graph) -> list[OntologyTerm]:
    """Walk the MeSH graph and build OntologyTerm records.

    Per Descriptor:
      - id = MESH:<bare-id> from the URI's last path segment.
      - label = rdfs:label (English-tagged, original casing).
      - synonyms = lowercased rdfs:labels of all linked meshv:concept
        targets (Concept-level alternative names).
      - parents = MeSH ids of meshv:broaderDescriptor targets.
      - definition = None (MeSH's scope notes live on Concepts, and
        gathering them is more work than the current value justifies;
        deferred).
    """
    import rdflib

    rdf_type = rdflib.URIRef(_RDF_TYPE)
    rdfs_label = rdflib.URIRef(_RDFS_LABEL)
    meshv_broader = rdflib.URIRef(f"{_MESHV}broaderDescriptor")
    meshv_concept = rdflib.URIRef(f"{_MESHV}concept")
    skos_close_match = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#closeMatch")
    skos_exact_match = rdflib.URIRef("http://www.w3.org/2004/02/skos/core#exactMatch")
    descriptor_types = [rdflib.URIRef(t) for t in _DESCRIPTOR_TYPES]

    # Gather all Descriptor URIs across the configured descriptor types.
    descriptor_uris: list[rdflib.URIRef] = []
    for dtype in descriptor_types:
        for subj in graph.subjects(rdf_type, dtype):
            if isinstance(subj, rdflib.URIRef):
                descriptor_uris.append(subj)

    terms: list[OntologyTerm] = []
    for descriptor_uri in descriptor_uris:
        label = _english_or_untagged_literal(graph, descriptor_uri, rdfs_label)
        if label is None:
            continue
        descriptor_id = f"{MESH_ID_PREFIX}:{_local_name(descriptor_uri)}"

        synonyms = _collect_concept_synonyms(graph, descriptor_uri, meshv_concept, rdfs_label)

        parents: list[str] = []
        for parent_uri in graph.objects(descriptor_uri, meshv_broader):
            if isinstance(parent_uri, rdflib.URIRef):
                parents.append(f"{MESH_ID_PREFIX}:{_local_name(parent_uri)}")

        # L7 xrefs via SKOS closeMatch / exactMatch (object URIs of
        # equivalent / closely-related concepts in OTHER vocabularies).
        # Stored verbatim; resolution to shipped :OntologyTerm IDs
        # happens at write time (most MeSH xref targets are LCSH /
        # EuroVoc URIs that won't match shipped IDs - those stay in
        # `dangling_xrefs` until those ontologies ship).
        xrefs: set[str] = set()
        for xref_uri in graph.objects(descriptor_uri, skos_close_match):
            xrefs.add(str(xref_uri))
        for xref_uri in graph.objects(descriptor_uri, skos_exact_match):
            xrefs.add(str(xref_uri))

        terms.append(
            OntologyTerm(
                id=descriptor_id,
                label=label,
                synonyms=tuple(sorted(synonyms)),
                parents=tuple(sorted(parents)),
                definition=None,
                xrefs=tuple(sorted(xrefs)),
            )
        )

    logger.info("MeSH: extracted %d descriptors", len(terms))
    return terms


def _collect_concept_synonyms(
    graph: rdflib.Graph,
    descriptor_uri: Any,
    meshv_concept: Any,
    rdfs_label: Any,
) -> set[str]:
    """Walk Descriptor -> Concept -> rdfs:label; gather lowercased labels.

    Each MeSH Descriptor links to one or more Concepts via
    `meshv:concept`. Each Concept has its own `rdfs:label`. Those
    Concept labels are alternate names for the Descriptor - useful as
    synonyms during canonical linking. The preferred Concept's label
    is usually identical to the Descriptor's label, so we keep only
    the non-matching ones to avoid trivial duplication.
    """
    import rdflib

    primary = _english_or_untagged_literal(graph, descriptor_uri, rdfs_label)
    primary_lower = primary.lower() if primary else ""

    synonyms: set[str] = set()
    for concept_uri in graph.objects(descriptor_uri, meshv_concept):
        if not isinstance(concept_uri, rdflib.URIRef):
            continue
        concept_label = _english_or_untagged_literal(graph, concept_uri, rdfs_label)
        if not concept_label:
            continue
        if concept_label.lower() != primary_lower:
            synonyms.add(concept_label.lower())
    return synonyms


def _english_or_untagged_literal(graph: rdflib.Graph, subject: Any, predicate: Any) -> str | None:
    """First English-tagged literal, or first untagged literal as
    fallback, for `(subject, predicate, *)`. Returns None when no
    matching literal is present."""
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


def _local_name(uri: Any) -> str:
    """Last path / fragment segment of a URI - the bare ID.

    Example: `http://id.nlm.nih.gov/mesh/D003920` -> `D003920`.
    Used to strip the MeSH namespace prefix and produce the bare
    identifier that combines with `MESH:` to form our canonical ID.
    """
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[1]
    return text.rsplit("/", 1)[1]
