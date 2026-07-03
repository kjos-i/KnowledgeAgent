"""L7 ontology writes for OBI (Ontology for Biomedical Investigations).

OBI is the canonical ontology for experimental methods + assay
protocols + biomedical investigation concepts (planned processes,
material entities, devices, study designs). OWL 2 DL, heavy
axiomatisation; we materialise only the basic class hierarchy +
labels + synonyms + definitions - the DL features (restrictions,
equivalence axioms, property chains) we leave at the OWL layer.
Maintained by the OBI consortium under the OBO Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:OBITerm {id, label, synonyms, definition})
  (:OBITerm)-[:OBI_IS_A]->(:OBITerm)

**Distribution + reader path.** OBI has no OBO release - only OWL/RDF.
We download `obi.owl` (the obo PURL serves it gzipped on the wire even
though the file is already gzipped; `ensure_cached` + `read_rdf`
handle the double-encoding via `iter_raw` + magic-bytes detection).
The reader pipeline is `read_rdf(path, format="xml")` + the rdflib-
based `extract_terms_owl`, NOT pronto's OBO reader. Reason: pronto
silently drops `oboInOwl:hasExactSynonym` etc. from OWL files (verified
against OBI: 1377 synonym annotations in source -> 0 extracted via
pronto). For canonical-matching quality, we need the synonym surface.

OBI IDs follow the OBO PURL convention - URI segment `OBI_0000489`
becomes our prefixed ID `OBI:0000489` via the default OWL id_extractor.
Foreign-import classes (BFO, IAO, CHMO that OBI imports) keep their
native prefix (`BFO:0000015`, etc.) - same behaviour as the pronto
modules, and correct for canonical matching since a paper mentioning
"process" should canonicalise to BFO:0000015.

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py` (despite their name, these helpers are format-
agnostic - they take a `read_and_extract` callback and write
`list[OntologyTerm]` to Neo4j; the source format is the caller's
business).
"""

from __future__ import annotations

import logging
from pathlib import Path

from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    ensure_cached,
    extract_terms_owl,
    delete_ontology_terms,
    import_ontology_data,
    is_ontology_imported,
    write_ontology_terms,
    read_rdf,
)
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.schema import OBI_IS_A_REL, OBI_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("experimental-methods",)
"""Domain tags this ontology serves. OBI is THE experimental-methods
vocabulary for biomedical research - the wizard suggests it for any
experimental-methods corpus that references assays, protocols, study
designs, or laboratory equipment."""

# Standard `obi.owl` distribution at the obo PURL. Source is OWL/RDF-XML;
# served gzipped on the wire. ~10 MB decompressed, ~5K OBI classes + a
# few thousand foreign-import classes (BFO, IAO, CHMO, etc.).
OBI_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/obi.owl"
OBI_CACHE_FILENAME = "obi.owl"
OBI_ID_PREFIX = "OBI"
DOWNLOAD_SIZE_MB = 10

_ONTOLOGY_NAME = "OBI"




# ---------------------------------------------------------------------------
# Provenance: structured per-ontology install-dialog metadata. See
# kg.ontology_provenance.OntologyProvenance for field semantics. Stored
# as a module-level constant so the registry can wire it through to the
# install dialog without an extra factory hop.
# ---------------------------------------------------------------------------

# L6 entity_type labels with strong coverage by this ontology. Joined
# against each extractor's `emitted_labels` by the cross-link helpers
# in kg.ontology_lifecycle to drive the install dialog's "pairs well
# with" surface.
_OBI_COVERS_LABELS: tuple[str, ...] = ()

_OBI_PROVENANCE = OntologyProvenance(
    ontology_name="obi",
    full_name="Ontology for Biomedical Investigations",
    publisher="OBI Consortium",
    license="CC BY 4.0",
    source_url="http://obi-ontology.org/",
    download_url=OBI_DOWNLOAD_URL,
    file_format="OWL/RDF-XML",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=3000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_OBI_COVERS_LABELS,
    description=(
                "Experimental methods, assay protocols, study designs, lab "
        "equipment, planned biomedical processes. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared write-helpers.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:OBITerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=OBI_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_obi(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the OBI ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=OBI_DOWNLOAD_URL,
        cache_filename=OBI_CACHE_FILENAME,
        term_label=OBI_TERM_LABEL,
        hierarchy_rel=OBI_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :OBITerm node + its :OBI_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=OBI_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:OBITerm` nodes + `:OBI_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=OBI_TERM_LABEL,
        hierarchy_rel=OBI_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached OBI OWL file via rdflib and extract terms.

    Uses `read_rdf(format="xml")` (gzip-transparent) + the rdflib-
    based `extract_terms_owl`, NOT pronto - see module docstring.
    """
    graph = read_rdf(path, format="xml")
    return extract_terms_owl(graph, id_prefix=OBI_ID_PREFIX)
