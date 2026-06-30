"""L7 ontology writes for EFO (Experimental Factor Ontology).

EFO is the canonical ontology for experimental factors across EBI
studies - diseases, cell types, cell lines, anatomy, traits, study
designs, measurement methods. Integrates upstream OBO/OWL imports
(UBERON, CL, MONDO, ChEBI, ORDO) and adds EBI-specific concepts.
Maintained by the EBI; Apache 2.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:EFOTerm {id, label, synonyms, definition})
  (:EFOTerm)-[:EFO_IS_A]->(:EFOTerm)

**Distribution + reader path.** EFO has no OBO release - only OWL/RDF.
Same as OBI: served gzipped on the wire, parsed via rdflib +
`extract_terms_owl` to preserve the synonym surface that pronto's
OWL reader drops. EFO is materially synonym-rich (trait names,
disease aliases, anatomical alternatives) so synonym preservation
matters for canonical matching quality.

**URI convention.** EFO uses its own EBI namespace
`http://www.ebi.ac.uk/efo/EFO_<id>` (NOT the obo PURL). Foreign-import
classes keep their native obo PURL convention
(`http://purl.obolibrary.org/obo/UBERON_<id>` -> `UBERON:<id>`). The
default `_owl_id_extractor` handles both via the same `<prefix>_<num>`
last-segment pattern.

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py` (despite the name, format-agnostic).
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
from knowledge_agent.kg.schema import EFO_IS_A_REL, EFO_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("experimental-methods",)
"""Domain tags this ontology serves. EFO is the EBI's integrative
experimental-factors vocabulary - the wizard suggests it alongside
OBI for experimental-methods corpora that work with EBI study data."""

# Standard `efo.owl` distribution at the EBI host. OWL/RDF-XML;
# gzipped on the wire. ~100 MB decompressed, ~30K classes (EFO + imports).
EFO_DOWNLOAD_URL = "http://www.ebi.ac.uk/efo/efo.owl"
EFO_CACHE_FILENAME = "efo.owl"
EFO_ID_PREFIX = "EFO"
DOWNLOAD_SIZE_MB = 100

_ONTOLOGY_NAME = "EFO"




# ---------------------------------------------------------------------------
# Provenance: structured per-ontology install-dialog metadata. See
# kg.ontology_provenance.OntologyProvenance for field semantics. Stored
# as a module-level constant so the registry can wire it through to the
# install dialog without an extra factory hop.
# ---------------------------------------------------------------------------

# L6a entity_type labels with strong coverage by this ontology. Joined
# against each extractor's `emitted_labels` by the cross-link helpers
# in kg.ontology_lifecycle to drive the install dialog's "pairs well
# with" surface.
_EFO_COVERS_LABELS: tuple[str, ...] = ("DISEASE", "CELL_LINE")

_EFO_PROVENANCE = OntologyProvenance(
    ontology_name="efo",
    full_name="Experimental Factor Ontology",
    publisher="EMBL-EBI",
    license="Apache-2.0",
    source_url="https://www.ebi.ac.uk/efo/",
    download_url=EFO_DOWNLOAD_URL,
    file_format="OWL/RDF-XML",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=22000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_EFO_COVERS_LABELS,
    description=(
                "EBI's integrative experimental-factors vocabulary: diseases, "
        "cell types, cell lines, traits, measurement methods. Strong "
        "cell-line coverage (HeLa, K562, etc.). "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared write-helpers.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:EFOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=EFO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_efo(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the EFO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=EFO_DOWNLOAD_URL,
        cache_filename=EFO_CACHE_FILENAME,
        term_label=EFO_TERM_LABEL,
        hierarchy_rel=EFO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :EFOTerm node + its :EFO_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=EFO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:EFOTerm` nodes + `:EFO_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=EFO_TERM_LABEL,
        hierarchy_rel=EFO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached EFO OWL file via rdflib and extract terms."""
    graph = read_rdf(path, format="xml")
    return extract_terms_owl(graph, id_prefix=EFO_ID_PREFIX)
