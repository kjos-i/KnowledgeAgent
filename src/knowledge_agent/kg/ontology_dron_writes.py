"""L7 ontology writes for DRON (Drug Ontology).

DRON is the canonical OWL-based drug ontology - drug products,
ingredients, dose forms, manufacturers, packaging. Built on RxNorm
with a realist upper-level structure (anchored in BFO), and so adds
ontological discipline RxNorm's flat RxCUI model lacks. Maintained
by Bill Hogan's group; OBO Foundry, CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:DRONTerm {id, label, synonyms, definition})
  (:DRONTerm)-[:DRON_IS_A]->(:DRONTerm)

**Distribution + reader path.** DRON has no OBO release - only OWL/RDF.
Same OWL pipeline as OBI and EFO: served gzipped on the wire, parsed
via rdflib + `extract_terms_owl` to preserve synonym surface. Drug
synonyms (brand names, alternative chemical names) are essential for
canonical matching, so synonym preservation is even more critical
than for OBI.

**Memory caveat (read before enabling in production).** DRON is the
second-largest ontology on the menu after NCBITaxon: ~700K classes,
~220 MB OWL source. Loading the full graph into rdflib eats several
GB of RAM during parse. Recommended posture:
  - Run the first import on a machine with >= 16 GB free RAM.
  - The L7 linking-pass index also scales linearly with term count;
    budget memory accordingly.
  - Streaming OWL/RDF reader is the future fix if rdflib's full-Graph
    approach OOMs at scale; mirrors the streaming N-Triples sink we
    built for MeSH (and the future-work flag on NCBITaxon).

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
from knowledge_agent.kg.schema import (
    DRON_IS_A_REL,
    DRON_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("pharmacology",)
"""Domain tags this ontology serves. DRON is the canonical drug-with-
ontological-discipline vocabulary - the wizard suggests it for any
pharmacology corpus that references drug products, ingredients, or
dose forms with structural granularity beyond a flat RxCUI list."""

# Standard `dron.owl` distribution at the obo PURL. Source is OWL/RDF-XML;
# served gzipped on the wire. ~220 MB decompressed, ~700K classes. See
# memory caveat in module docstring.
DRON_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/dron.owl"
DRON_CACHE_FILENAME = "dron.owl"
DRON_ID_PREFIX = "DRON"
DOWNLOAD_SIZE_MB = 220

_ONTOLOGY_NAME = "DRON"




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
_DRON_COVERS_LABELS: tuple[str, ...] = ("CHEMICAL",)

_DRON_PROVENANCE = OntologyProvenance(
    ontology_name="dron",
    full_name="Drug Ontology",
    publisher="University at Buffalo (Hogan + Tomovic et al.)",
    license="CC BY 3.0",
    source_url="https://github.com/ufbmi/dron",
    download_url=DRON_DOWNLOAD_URL,
    file_format="OWL/RDF-XML",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=700000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_DRON_COVERS_LABELS,
    description=(
                "Drug products, ingredients, dose forms, manufacturers, "
        "packaging — built on RxNorm with realist upper-level "
        "structure. "
    ),
    heavy_warning=(
        "~700K classes / ~220 MB OWL / several GB RAM at "
        "import. "
    ),
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared write-helpers.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:DRONTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=DRON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_dron(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the DRON ontology to Neo4j. Idempotent.

    See module docstring for the memory caveat at parse time (DRON is
    the second-largest ontology on the menu).
    """
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=DRON_DOWNLOAD_URL,
        cache_filename=DRON_CACHE_FILENAME,
        term_label=DRON_TERM_LABEL,
        hierarchy_rel=DRON_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :DRONTerm node + its :DRON_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=DRON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:DRONTerm` nodes + `:DRON_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=DRON_TERM_LABEL,
        hierarchy_rel=DRON_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached DRON OWL file via rdflib and extract terms."""
    graph = read_rdf(path, format="xml")
    return extract_terms_owl(graph, id_prefix=DRON_ID_PREFIX)
