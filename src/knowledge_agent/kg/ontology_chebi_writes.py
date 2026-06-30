"""L7 ontology writes for ChEBI (Chemical Entities of Biological Interest).

ChEBI is the canonical chemistry ontology - chemical compounds with
biological relevance (drugs, metabolites, signaling molecules,
nutrients, etc.). Maintained by the EBI. ~140K chemicals at the full
release, ~30K at the LITE variant we use.

Distributed by the OBO Foundry as OBO + OWL + JSON in three variants:
FULL / CORE / LITE. We use the **LITE variant** in OBO format - it
drops chemical structure details (SMILES, InChI, formulas) that we
don't use for label-based canonicalisation, and is ~13× smaller than
FULL (~51 MB LITE OBO vs 248 MB FULL OBO).

What we write to Neo4j (multi-label):
  (:OntologyTerm:ChEBITerm {id, label, synonyms, definition})
  (:ChEBITerm)-[:CHEBI_IS_A]->(:ChEBITerm)

ChEBI IDs come with their canonical prefix already ("CHEBI:17234").

Lifecycle delegates to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py`.

Note: ChEBI carries rich xrefs to other resources (RHEA reactions,
KEGG compounds, UniProt-bound ligands, etc.) in the source data.
Today we discard those at import time - promoting them to graph edges
is the [[deferred-cross-ontology-xrefs]] work item.
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
    CHEBI_IS_A_REL,
    CHEBI_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("chemistry", "biochemistry")
"""Domain tags this ontology serves. ChEBI is "Chemical Entities of
Biological Interest" - primarily chemistry, with biochemistry as a
secondary tag matching its biological-relevance filter."""

# LITE variant in OBO format - drops chemical structure detail (SMILES,
# InChI) we don't use, ~13x smaller than FULL. ~30K terms, ~51 MB OBO source.
CHEBI_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/chebi/chebi_lite.obo"
CHEBI_CACHE_FILENAME = "chebi_lite.obo"
CHEBI_ID_PREFIX = "CHEBI"
DOWNLOAD_SIZE_MB = 51

_ONTOLOGY_NAME = "ChEBI"




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
_CHEBI_COVERS_LABELS: tuple[str, ...] = ("CHEMICAL",)

_CHEBI_PROVENANCE = OntologyProvenance(
    ontology_name="chebi",
    full_name="Chemical Entities of Biological Interest (LITE)",
    publisher="EMBL-EBI (Chemistry team)",
    license="CC BY 4.0",
    source_url="https://www.ebi.ac.uk/chebi/",
    download_url=CHEBI_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=160000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_CHEBI_COVERS_LABELS,
    description=(
                "Small molecules of biological relevance: drugs, metabolites, "
        "signaling molecules, nutrients. The LITE variant skips "
        "chemical structures to keep the file small. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:ChEBITerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=CHEBI_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_chebi(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the ChEBI ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=CHEBI_DOWNLOAD_URL,
        cache_filename=CHEBI_CACHE_FILENAME,
        term_label=CHEBI_TERM_LABEL,
        hierarchy_rel=CHEBI_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :ChEBITerm node + its :CHEBI_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=CHEBI_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:ChEBITerm` nodes + `:CHEBI_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=CHEBI_TERM_LABEL,
        hierarchy_rel=CHEBI_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached ChEBI LITE OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=CHEBI_ID_PREFIX)
