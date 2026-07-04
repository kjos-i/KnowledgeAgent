"""L7 ontology writes for ENVO (Environment Ontology).

ENVO is the canonical ontology for environmental concepts - biomes
(rainforest, coral reef, ...), environmental features (river,
volcano, ...), environmental materials (soil, seawater, ...), and
exposures (pollutants, climate variables). Maintained by the OBO
Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:ENVOTerm {id, label, synonyms, definition})
  (:ENVOTerm)-[:ENVO_IS_A]->(:ENVOTerm)

ENVO IDs come with their canonical prefix already ("ENVO:00000111"),
so we store IDs verbatim.

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
from knowledge_agent.kg.schema import ENVO_IS_A_REL, ENVO_TERM_LABEL

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("environment",)
"""Domain tags this ontology serves. ENVO is THE environment
vocabulary - the wizard suggests it for any environment corpus that
references biomes, environmental features, or exposures."""

# Standard `envo.obo` distribution. ~7K terms, ~20 MB OBO source.
ENVO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/envo.obo"
ENVO_CACHE_FILENAME = "envo.obo"
ENVO_ID_PREFIX = "ENVO"
DOWNLOAD_SIZE_MB = 20

_ONTOLOGY_NAME = "ENVO"


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
_ENVO_COVERS_LABELS: tuple[str, ...] = ()

_ENVO_PROVENANCE = OntologyProvenance(
    ontology_name="envo",
    full_name="Environment Ontology",
    publisher="Pier Luigi Buttigieg + ENVO group",
    license="CC BY 3.0",
    source_url="http://environmentontology.org/",
    download_url=ENVO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=9000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_ENVO_COVERS_LABELS,
    description=(
        "Biomes, environmental features, materials, and exposures. "
        "Useful for environmental / ecology corpora; no shipped "
        "extractor emits matching labels today. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:ENVOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client,
        term_label=ENVO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def import_envo(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the ENVO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=ENVO_DOWNLOAD_URL,
        cache_filename=ENVO_CACHE_FILENAME,
        term_label=ENVO_TERM_LABEL,
        hierarchy_rel=ENVO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :ENVOTerm node + its :ENVO_IS_A edges."""
    await delete_ontology_terms(
        client,
        term_label=ENVO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:ENVOTerm` nodes + `:ENVO_IS_A` edges."""
    await write_ontology_terms(
        client,
        terms,
        term_label=ENVO_TERM_LABEL,
        hierarchy_rel=ENVO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached ENVO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=ENVO_ID_PREFIX)
