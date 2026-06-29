"""L7 ontology writes for MONDO (Mondo Disease Ontology).

MONDO is the canonical disease ontology - integrates DOID, OMIM,
Orphanet, EFO disease branch, NCIT diseases, ICD-11 via curated 1:1
equivalence axioms validated by OWL reasoning. ~61K disease classes
spanning rare disease, infectious, neoplastic, mental, metabolic, etc.

Maintained by the Monarch Initiative as part of the OBO Foundry.
Distributed in OBO + OWL + JSON; we use `mondo.obo` via pronto for
parity with the other pronto-based modules.

What we write to Neo4j (multi-label):
  (:OntologyTerm:MONDOTerm {id, label, synonyms, definition})
  (:MONDOTerm)-[:MONDO_IS_A]->(:MONDOTerm)

MONDO IDs come with their canonical prefix already ("MONDO:0005148").

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py`.

Note: MONDO carries rich xrefs to other disease vocabularies (DOID,
OMIM, ICD-11, etc.) in the source data. Today we discard those at
import time - promoting them to graph edges is the
[[deferred-cross-ontology-xrefs]] work item.
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
    MONDO_IS_A_REL,
    MONDO_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("medicine",)
"""Domain tags this ontology serves. MONDO is THE disease vocabulary
for biomedical research - the wizard suggests it for any medicine
corpus that references diseases."""

# Full `mondo.obo` distribution is the standard release. ~61K classes,
# ~150 MB OBO source.
MONDO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/mondo.obo"
MONDO_CACHE_FILENAME = "mondo.obo"
MONDO_ID_PREFIX = "MONDO"
DOWNLOAD_SIZE_MB = 150

_ONTOLOGY_NAME = "MONDO"




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
_MONDO_COVERS_LABELS: tuple[str, ...] = ("DISEASE",)

_MONDO_PROVENANCE = OntologyProvenance(
    ontology_name="mondo",
    full_name="Mondo Disease Ontology",
    publisher="Monarch Initiative",
    license="CC0 1.0",
    source_url="https://mondo.monarchinitiative.org/",
    download_url=MONDO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=24000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_MONDO_COVERS_LABELS,
    description=(
                "Integrated disease ontology across DOID, OMIM, Orphanet, EFO "
        "disease branch, NCIT diseases, and ICD-11. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


def is_imported(client) -> bool:
    """True when at least one `:MONDOTerm` node exists in Neo4j."""
    return is_ontology_imported(
        client, term_label=MONDO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


def import_mondo(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the MONDO ontology to Neo4j. Idempotent."""
    return import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=MONDO_DOWNLOAD_URL,
        cache_filename=MONDO_CACHE_FILENAME,
        term_label=MONDO_TERM_LABEL,
        hierarchy_rel=MONDO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


def delete_imported(client) -> None:
    """DETACH DELETE every :MONDOTerm node + its :MONDO_IS_A edges."""
    delete_ontology_terms(
        client, term_label=MONDO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:MONDOTerm` nodes + `:MONDO_IS_A` edges."""
    write_ontology_terms(
        client, terms,
        term_label=MONDO_TERM_LABEL,
        hierarchy_rel=MONDO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached MONDO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=MONDO_ID_PREFIX)
