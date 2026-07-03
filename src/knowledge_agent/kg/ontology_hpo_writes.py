"""L7 ontology writes for HPO (Human Phenotype Ontology).

HPO is the canonical ontology for human phenotypic abnormalities -
clinically observable features of disease (symptoms, signs, lab
abnormalities, behavioural traits). Distributed by the OBO Foundry
as OBO + OWL + JSON; we use the standard `hp.obo` distribution via
the shared `pronto` reader.

HPO covers more than just "symptoms" - it's broader phenotypic
abnormalities (about 16K terms). SYMP (Symptom Ontology, dropped from
our menu 2026-06-23) is a narrower subset; HPO subsumes its useful
coverage and adds clinical depth.

What we write to Neo4j (multi-label):
  (:OntologyTerm:HPOTerm {id, label, synonyms, definition})
  (:HPOTerm)-[:HPO_IS_A]->(:HPOTerm)

HPO IDs come with their canonical prefix already ("HP:0000118"), so we
store IDs verbatim - same as GO, no rewriting like we did for MeSH
(which only had bare D-numbers in its URIs).

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
from knowledge_agent.kg.schema import HPO_IS_A_REL, HPO_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching - see ontology_go_writes.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("medicine",)
"""Domain tags this ontology serves. HPO is the canonical phenotype
vocabulary for biomedical research - the wizard suggests it for any
corpus with `medicine` in its domain set."""

# The standard `hp.obo` distribution is the main HPO release (no LITE
# variant - HPO is already focused). ~16K terms, ~30 MB OBO source.
HPO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/hp.obo"
HPO_CACHE_FILENAME = "hp.obo"
HPO_ID_PREFIX = "HP"
DOWNLOAD_SIZE_MB = 30

_ONTOLOGY_NAME = "HPO"




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
_HPO_COVERS_LABELS: tuple[str, ...] = ("DISEASE",)

_HPO_PROVENANCE = OntologyProvenance(
    ontology_name="hpo",
    full_name="Human Phenotype Ontology",
    publisher="Monarch Initiative + Jackson Laboratory",
    license="HPO License (free for any use; attribution required)",
    source_url="https://hpo.jax.org/",
    download_url=HPO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=17000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_HPO_COVERS_LABELS,
    description=(
                "Clinical features, signs, symptoms, lab abnormalities, "
        "behavioural traits. Backbone of clinical genetics and "
        "rare-disease research. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:HPOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=HPO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_hpo(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the HPO ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=HPO_DOWNLOAD_URL,
        cache_filename=HPO_CACHE_FILENAME,
        term_label=HPO_TERM_LABEL,
        hierarchy_rel=HPO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :HPOTerm node + its :HPO_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=HPO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:HPOTerm` nodes + `:HPO_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=HPO_TERM_LABEL,
        hierarchy_rel=HPO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached HPO OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=HPO_ID_PREFIX)
