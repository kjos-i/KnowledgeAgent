"""L7 ontology writes for ECO (Evidence & Conclusion Ontology).

ECO is the canonical ontology for evidence codes used to qualify
annotations - "based on experimental evidence", "inferred from
sequence orthology", "author statement", etc. Maintained by the
ECO consortium under the OBO Foundry; CC0 (public domain).

What we write to Neo4j (multi-label):
  (:OntologyTerm:ECOTerm {id, label, synonyms, definition})
  (:ECOTerm)-[:ECO_IS_A]->(:ECOTerm)

ECO IDs come with their canonical prefix already ("ECO:0000006"),
so we store IDs verbatim - same pattern as GO/HPO/UBERON/MONDO/ChEBI.

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    delete_ontology_terms,
    ensure_cached,
    extract_terms_obo,
    import_ontology_data,
    is_ontology_imported,
    read_obo,
    write_ontology_terms,
)
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.schema import ECO_IS_A_REL, ECO_TERM_LABEL

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("experimental-methods",)
"""Domain tags this ontology serves. ECO classifies the evidence
behind annotations across experimental-methods corpora."""

# Standard `eco.obo` distribution. ~3K terms, ~5 MB OBO source - small
# compared to the disease/anatomy ontologies.
ECO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/eco.obo"
ECO_CACHE_FILENAME = "eco.obo"
ECO_ID_PREFIX = "ECO"
DOWNLOAD_SIZE_MB = 5

_ONTOLOGY_NAME = "ECO"


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
_ECO_COVERS_LABELS: tuple[str, ...] = ()

_ECO_PROVENANCE = OntologyProvenance(
    ontology_name="eco",
    full_name="Evidence & Conclusion Ontology",
    publisher="Evidence Codes Group (Marcus Chibucos et al.)",
    license="CC0 1.0",
    source_url="https://evidenceontology.org/",
    download_url=ECO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=3000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_ECO_COVERS_LABELS,
    description=(
        "Evidence codes that qualify annotations (experimental "
        "evidence, sequence orthology, author statement). Useful as "
        "provenance metadata, NOT for entity canonicalisation — no "
        "shipped extractor emits matching labels. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:ECOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client,
        term_label=ECO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def import_eco(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the ECO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=ECO_DOWNLOAD_URL,
        cache_filename=ECO_CACHE_FILENAME,
        term_label=ECO_TERM_LABEL,
        hierarchy_rel=ECO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :ECOTerm node + its :ECO_IS_A edges."""
    await delete_ontology_terms(
        client,
        term_label=ECO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:ECOTerm` nodes + `:ECO_IS_A` edges."""
    await write_ontology_terms(
        client,
        terms,
        term_label=ECO_TERM_LABEL,
        hierarchy_rel=ECO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached ECO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=ECO_ID_PREFIX)
