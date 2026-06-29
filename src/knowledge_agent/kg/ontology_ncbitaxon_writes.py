"""L7 ontology writes for NCBITaxon (NCBI Taxonomy).

NCBITaxon is the canonical taxonomy for biological organisms - every
species, genus, family, etc. that has a sequenced specimen in NCBI's
databases. The largest ontology on the menu by a wide margin: ~2.74M
classes, ~440 MB OBO source. Maintained by the OBO Foundry as a
mirror of NCBI's authoritative taxonomy; CC0 (public domain).

What we write to Neo4j (multi-label):
  (:OntologyTerm:NCBITaxonTerm {id, label, synonyms, definition})
  (:NCBITaxonTerm)-[:NCBITAXON_IS_A]->(:NCBITaxonTerm)

NCBITaxon IDs come with the canonical "NCBITaxon:" prefix already
("NCBITaxon:9606" for Homo sapiens), so we store IDs verbatim.

**Memory caveat (read before enabling in production).** Pronto loads
the full OBO file into memory as Python objects. At 2.74M classes,
that is several GB of resident memory during the import - far more
than any other ontology in the menu. Recommended posture:
  - Run the first import on a machine with >= 16 GB free RAM.
  - Once imported, the Neo4j-resident copy is the working set; the
    in-memory pronto Ontology object is released after `write_terms`.
  - The L7 linking-pass index also scales linearly (2.74M term entries
    in `index: dict[str, list[str]]`) - budget memory accordingly.
  - If pronto OOMs at scale, a streaming OBO reader is the future
    fix (mirrors the streaming N-Triples sink we built for MeSH);
    not built today because the simpler pattern works on the
    machines we expect to run on.

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py` (same shape as the other OBO modules - the
memory risk is at the helper layer, not in this module's surface).
"""

from __future__ import annotations

import logging
from pathlib import Path

from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    ensure_cached,
    extract_terms_obo,
    delete_ontology_terms,
    import_ontology_data,
    is_ontology_imported,
    write_ontology_terms,
    read_obo,
)
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.schema import (
    NCBITAXON_IS_A_REL,
    NCBITAXON_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("biology",)
"""Domain tags this ontology serves. NCBITaxon is the canonical
organism taxonomy - the wizard suggests it for any biology corpus
that names species, genera, or higher taxa."""

# Standard `ncbitaxon.obo` distribution. ~2.74M classes, ~440 MB OBO
# source. See module docstring for memory caveats.
NCBITAXON_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/ncbitaxon.obo"
NCBITAXON_CACHE_FILENAME = "ncbitaxon.obo"
NCBITAXON_ID_PREFIX = "NCBITaxon"
DOWNLOAD_SIZE_MB = 440

_ONTOLOGY_NAME = "NCBITaxon"




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
_NCBITAXON_COVERS_LABELS: tuple[str, ...] = ("SPECIES",)

_NCBITAXON_PROVENANCE = OntologyProvenance(
    ontology_name="ncbitaxon",
    full_name="NCBI Taxonomy",
    publisher="NCBI (NIH National Center for Biotechnology Information)",
    license="Public Domain (NCBI policy)",
    source_url="https://www.ncbi.nlm.nih.gov/Taxonomy/",
    download_url=NCBITAXON_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=2740000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_NCBITAXON_COVERS_LABELS,
    description=(
                "Biological organism taxonomy (species, genus, family, ...) "
        "up through Life. The canonical NCBI taxonomy mirror. "
    ),
    heavy_warning=(
        "~2.74M classes / ~440 MB OBO / several GB RAM at "
        "import — recommend at least 16 GB free RAM. "
    ),
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


def is_imported(client) -> bool:
    """True when at least one `:NCBITaxonTerm` node exists in Neo4j."""
    return is_ontology_imported(
        client,
        term_label=NCBITAXON_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


def import_ncbitaxon(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write NCBITaxon to Neo4j. Idempotent.

    See module docstring for the memory caveat at parse time.
    """
    return import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=NCBITAXON_DOWNLOAD_URL,
        cache_filename=NCBITAXON_CACHE_FILENAME,
        term_label=NCBITAXON_TERM_LABEL,
        hierarchy_rel=NCBITAXON_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


def delete_imported(client) -> None:
    """DETACH DELETE every :NCBITaxonTerm node + its :NCBITAXON_IS_A edges."""
    delete_ontology_terms(
        client,
        term_label=NCBITAXON_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:NCBITaxonTerm` nodes + `:NCBITAXON_IS_A` edges."""
    write_ontology_terms(
        client, terms,
        term_label=NCBITAXON_TERM_LABEL,
        hierarchy_rel=NCBITAXON_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached NCBITaxon OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=NCBITAXON_ID_PREFIX)
