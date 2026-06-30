"""L7 ontology writes for PR (Protein Ontology).

PR is the canonical ontology for proteins - covers protein classes,
isoforms, complexes, and ortholog edges that UniProt's flat
"one entry per protein" model collapses. Maintained by the OBO
Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:PRTerm {id, label, synonyms, definition})
  (:PRTerm)-[:PR_IS_A]->(:PRTerm)

PR IDs come with their canonical prefix already ("PR:000000001"), so
we store IDs verbatim - same pattern as GO/HPO/UBERON/MONDO/ChEBI/
ECO/SO.

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
from knowledge_agent.kg.schema import PR_IS_A_REL, PR_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("proteins",)
"""Domain tags this ontology serves. PR adds protein-class /
isoform / complex granularity that UniProt collapses; the wizard
suggests it for proteins corpora that need that structure."""

# Standard `pr.obo` distribution. ~285K terms across all organisms,
# ~70 MB OBO source.
PR_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/pr.obo"
PR_CACHE_FILENAME = "pr.obo"
PR_ID_PREFIX = "PR"
DOWNLOAD_SIZE_MB = 70

_ONTOLOGY_NAME = "PR"




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
_PR_COVERS_LABELS: tuple[str, ...] = ("PROTEIN",)

_PR_PROVENANCE = OntologyProvenance(
    ontology_name="pr",
    full_name="Protein Ontology",
    publisher="Protein Ontology Consortium (Cathy Wu et al.)",
    license="CC BY 4.0",
    source_url="https://proconsortium.org/",
    download_url=PR_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=330000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_PR_COVERS_LABELS,
    description=(
                "Protein classes, isoforms, complexes, and ortholog "
        "relationships — finer granularity than UniProt's flat model. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:PRTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=PR_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_pr(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the PR ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=PR_DOWNLOAD_URL,
        cache_filename=PR_CACHE_FILENAME,
        term_label=PR_TERM_LABEL,
        hierarchy_rel=PR_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :PRTerm node + its :PR_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=PR_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:PRTerm` nodes + `:PR_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=PR_TERM_LABEL,
        hierarchy_rel=PR_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached PR OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=PR_ID_PREFIX)
