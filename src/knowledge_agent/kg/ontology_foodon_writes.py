"""L7 ontology writes for FOODON (Food Ontology).

FOODON is the canonical ontology for foods, food products, and
dietary components - covers raw ingredients, prepared dishes,
processing methods, and food-related concepts. Maintained by the
OBO Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:FOODONTerm {id, label, synonyms, definition})
  (:FOODONTerm)-[:FOODON_IS_A]->(:FOODONTerm)

FOODON IDs come with their canonical prefix already
("FOODON:03301720"), so we store IDs verbatim.

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py`.
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
    FOODON_IS_A_REL,
    FOODON_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("nutrition",)
"""Domain tags this ontology serves. FOODON is THE foods vocabulary -
the wizard suggests it for any nutrition corpus that references
foods, ingredients, or food products."""

# Standard `foodon.obo` distribution. ~10K terms, ~30 MB OBO source.
FOODON_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/foodon.obo"
FOODON_CACHE_FILENAME = "foodon.obo"
FOODON_ID_PREFIX = "FOODON"
DOWNLOAD_SIZE_MB = 30

_ONTOLOGY_NAME = "FOODON"




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
_FOODON_COVERS_LABELS: tuple[str, ...] = ()

_FOODON_PROVENANCE = OntologyProvenance(
    ontology_name="foodon",
    full_name="Food Ontology",
    publisher="FoodOn Project (Damion Dooley et al.)",
    license="CC BY 3.0",
    source_url="https://foodon.org/",
    download_url=FOODON_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=10000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_FOODON_COVERS_LABELS,
    description=(
                "Foods, food products, dietary components, and processing "
        "methods. Useful for nutrition corpora; no shipped extractor "
        "emits food-as-entity labels today. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:FOODONTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=FOODON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_foodon(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the FOODON ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=FOODON_DOWNLOAD_URL,
        cache_filename=FOODON_CACHE_FILENAME,
        term_label=FOODON_TERM_LABEL,
        hierarchy_rel=FOODON_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :FOODONTerm node + its :FOODON_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=FOODON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:FOODONTerm` nodes + `:FOODON_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=FOODON_TERM_LABEL,
        hierarchy_rel=FOODON_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached FOODON OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=FOODON_ID_PREFIX)
