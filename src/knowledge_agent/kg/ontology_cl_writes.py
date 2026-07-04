"""L7 ontology writes for CL (Cell Ontology).

CL is the canonical ontology for cell types across organisms - the
"what kind of cell is this?" vocabulary. Heavy use in single-cell
transcriptomics, immunology, and developmental biology. Imports
UBERON for anatomical context and GO for biological-process / cell-
component anchors. Maintained by the OBO Foundry; CC BY 4.0.

What we write to Neo4j (multi-label):
  (:OntologyTerm:CLTerm {id, label, synonyms, definition})
  (:CLTerm)-[:CL_IS_A]->(:CLTerm)

CL IDs come with their canonical prefix already ("CL:0000540"), so
we store IDs verbatim. We materialise only the within-CL is_a
hierarchy here; cross-ontology references (CL -> UBERON, CL -> GO)
are the [[deferred-cross-ontology-xrefs]] work item.

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
from knowledge_agent.kg.schema import CL_IS_A_REL, CL_TERM_LABEL

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("cell biology",)
"""Domain tags this ontology serves. CL is THE cell-type vocabulary
for cell biology - the wizard suggests it for any cell-biology
corpus that names neurons, lymphocytes, epithelial cells, etc."""

# Standard `cl.obo` distribution. ~2.5K classes, ~20 MB OBO source.
CL_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/cl.obo"
CL_CACHE_FILENAME = "cl.obo"
CL_ID_PREFIX = "CL"
DOWNLOAD_SIZE_MB = 20

_ONTOLOGY_NAME = "CL"


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
_CL_COVERS_LABELS: tuple[str, ...] = ("CELL_TYPE",)

_CL_PROVENANCE = OntologyProvenance(
    ontology_name="cl",
    full_name="Cell Ontology",
    publisher="OBO Foundry (Alexander Diehl et al.)",
    license="CC BY 4.0",
    source_url="http://obophenotype.github.io/cell-ontology/",
    download_url=CL_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=12000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_CL_COVERS_LABELS,
    description=(
        "Cell types across organisms — neurons, lymphocytes, "
        "epithelial cells, and the rest of the canonical cytology. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:CLTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client,
        term_label=CL_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def import_cl(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the CL ontology to Neo4j. Idempotent."""
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=CL_DOWNLOAD_URL,
        cache_filename=CL_CACHE_FILENAME,
        term_label=CL_TERM_LABEL,
        hierarchy_rel=CL_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :CLTerm node + its :CL_IS_A edges."""
    await delete_ontology_terms(
        client,
        term_label=CL_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:CLTerm` nodes + `:CL_IS_A` edges."""
    await write_ontology_terms(
        client,
        terms,
        term_label=CL_TERM_LABEL,
        hierarchy_rel=CL_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached CL OBO file via pronto and extract terms."""
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=CL_ID_PREFIX)
