"""L7 ontology writes for SO (Sequence Ontology).

SO is the canonical ontology for sequence features and variants -
gene, promoter, intron, CDS, exon, regulatory region, variant types
(SNV, indel, CNV, structural variant), and the relationships between
them. Maintained by the OBO Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:SOTerm {id, label, synonyms, definition})
  (:SOTerm)-[:SO_IS_A]->(:SOTerm)

SO IDs come with their canonical prefix already ("SO:0000704"), so
we store IDs verbatim - same pattern as GO/HPO/UBERON/MONDO/ChEBI/ECO.

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
from knowledge_agent.kg.schema import SO_IS_A_REL, SO_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("genomics",)
"""Domain tags this ontology serves. SO is the canonical vocabulary
for sequence features + variants - the wizard suggests it for any
genomics corpus that references gene structure, regulatory elements,
or variant types."""

# Standard `so.obo` distribution. ~2.7K terms, ~10 MB OBO source.
SO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/so.obo"
SO_CACHE_FILENAME = "so.obo"
SO_ID_PREFIX = "SO"
DOWNLOAD_SIZE_MB = 10

_ONTOLOGY_NAME = "SO"




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
_SO_COVERS_LABELS: tuple[str, ...] = ("GENE",)

_SO_PROVENANCE = OntologyProvenance(
    ontology_name="so",
    full_name="Sequence Ontology",
    publisher="Sequence Ontology Project (Karen Eilbeck et al.)",
    license="CC BY 4.0",
    source_url="http://sequenceontology.org/",
    download_url=SO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=2500,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_SO_COVERS_LABELS,
    description=(
                "Sequence features and variants: gene, promoter, intron, CDS, "
        "regulatory region, SNV, indel, structural variant. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:SOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=SO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_so(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the SO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=SO_DOWNLOAD_URL,
        cache_filename=SO_CACHE_FILENAME,
        term_label=SO_TERM_LABEL,
        hierarchy_rel=SO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :SOTerm node + its :SO_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=SO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:SOTerm` nodes + `:SO_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=SO_TERM_LABEL,
        hierarchy_rel=SO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached SO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=SO_ID_PREFIX)
