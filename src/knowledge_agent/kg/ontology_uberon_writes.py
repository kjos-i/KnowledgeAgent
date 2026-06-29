"""L7 ontology writes for UBERON (Uber Anatomy Ontology).

UBERON is the canonical multi-species anatomy ontology - tissues,
organs, anatomical regions, body parts across metazoans (with the
strongest coverage on vertebrates / mammalians). Distributed by the
OBO Foundry; we use the `uberon/basic.obo` subset which excludes all
relationship types other than is_a, part_of, and develops_from -
the most useful and cleanest subset for label/synonym canonicalisation.

What we write to Neo4j (multi-label):
  (:OntologyTerm:UBERONTerm {id, label, synonyms, definition})
  (:UBERONTerm)-[:UBERON_IS_A]->(:UBERONTerm)

We materialise only the is_a hierarchy (NOT part_of or develops_from)
to match the existing per-ontology pattern. part_of / develops_from
cross-references can be added later via the
[[deferred-cross-ontology-xrefs]] work item.

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
    UBERON_IS_A_REL,
    UBERON_TERM_LABEL,
)

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("biology",)
"""Domain tags this ontology serves. UBERON covers cross-species
anatomy - the wizard suggests it for any biology corpus that
references organs / tissues / anatomical regions."""

# "uberon/basic.obo" subset excludes relationship types beyond is_a /
# part_of / develops_from - cleanest subset for label-based
# canonicalisation. ~14K terms, ~50 MB OBO source.
UBERON_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/uberon/basic.obo"
UBERON_CACHE_FILENAME = "uberon-basic.obo"
UBERON_ID_PREFIX = "UBERON"
DOWNLOAD_SIZE_MB = 50

_ONTOLOGY_NAME = "UBERON"




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
_UBERON_COVERS_LABELS: tuple[str, ...] = ("ANATOMY",)

_UBERON_PROVENANCE = OntologyProvenance(
    ontology_name="uberon",
    full_name="Uber Anatomy Ontology",
    publisher="OBO Foundry (Chris Mungall et al.)",
    license="CC BY 3.0",
    source_url="http://uberon.org/",
    download_url=UBERON_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=15000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_UBERON_COVERS_LABELS,
    description=(
                "Multi-species anatomy: tissues, organs, body parts. The "
        "cross-species anatomical bridge. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


def is_imported(client) -> bool:
    """True when at least one `:UBERONTerm` node exists in Neo4j."""
    return is_ontology_imported(
        client, term_label=UBERON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


def import_uberon(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the UBERON ontology to Neo4j. Idempotent."""
    return import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=UBERON_DOWNLOAD_URL,
        cache_filename=UBERON_CACHE_FILENAME,
        term_label=UBERON_TERM_LABEL,
        hierarchy_rel=UBERON_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


def delete_imported(client) -> None:
    """DETACH DELETE every :UBERONTerm node + its :UBERON_IS_A edges."""
    delete_ontology_terms(
        client, term_label=UBERON_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:UBERONTerm` nodes + `:UBERON_IS_A` edges."""
    write_ontology_terms(
        client, terms,
        term_label=UBERON_TERM_LABEL,
        hierarchy_rel=UBERON_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached UBERON OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=UBERON_ID_PREFIX)
