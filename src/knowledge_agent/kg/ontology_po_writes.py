"""L7 ontology writes for PO (Plant Ontology).

PO is the canonical anatomy + developmental-stage ontology for plants
- the "what part of the plant?" and "what life-stage?" vocabulary
(root, leaf, flower, seedling, anthesis, etc.). Fills UBERON's plant
gap (UBERON skews vertebrate). Maintained by the OBO Foundry; CC BY.

What we write to Neo4j (multi-label):
  (:OntologyTerm:POTerm {id, label, synonyms, definition})
  (:POTerm)-[:PO_IS_A]->(:POTerm)

PO IDs come with their canonical prefix already ("PO:0009001"), so
we store IDs verbatim.

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
from knowledge_agent.kg.schema import PO_IS_A_REL, PO_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("biology",)
"""Domain tags this ontology serves. PO is the canonical plant
anatomy + developmental stage vocabulary - the wizard suggests it
for biology corpora that reference plants (fills UBERON's vertebrate-
skewed coverage gap)."""

# Standard `po.obo` distribution. ~1.6K terms, ~5 MB OBO source.
PO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/po.obo"
PO_CACHE_FILENAME = "po.obo"
PO_ID_PREFIX = "PO"
DOWNLOAD_SIZE_MB = 5

_ONTOLOGY_NAME = "PO"




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
_PO_COVERS_LABELS: tuple[str, ...] = ("ANATOMY",)

_PO_PROVENANCE = OntologyProvenance(
    ontology_name="po",
    full_name="Plant Ontology",
    publisher="Plant Ontology Consortium",
    license="CC BY 4.0",
    source_url="http://browser.planteome.org/",
    download_url=PO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=3000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_PO_COVERS_LABELS,
    description=(
                "Plant anatomy and developmental stages (root, leaf, flower, "
        "anthesis). Fills UBERON's vertebrate-skewed gap for plant "
        "biology corpora. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:POTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=PO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_po(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the PO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=PO_DOWNLOAD_URL,
        cache_filename=PO_CACHE_FILENAME,
        term_label=PO_TERM_LABEL,
        hierarchy_rel=PO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :POTerm node + its :PO_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=PO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:POTerm` nodes + `:PO_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=PO_TERM_LABEL,
        hierarchy_rel=PO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached PO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=PO_ID_PREFIX)
