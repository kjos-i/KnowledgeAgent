"""L7 ontology writes for GO (Gene Ontology).

GO is the canonical ontology for cell biology + biochemistry:
biological processes, molecular functions, and cellular components.
Distributed by the OBO Foundry as OBO + OWL + OBO-JSON; we use the
"go-basic" subset (the standard non-import variant) via the shared
pronto reader.

What we write to Neo4j (multi-label):
  (:OntologyTerm:GOTerm {id, label, synonyms, definition})
  (:GOTerm)-[:GO_IS_A]->(:GOTerm)

GO terms come with their canonical prefix already in the id ("GO:0008150"),
so we store IDs verbatim - no rewriting like we did for MeSH (which only
had bare D-numbers in its URIs).

Lifecycle delegates entirely to the shared `write_ontology_terms` family helpers in
`ontology_helpers.py` (refactored 2026-06-23 alongside HPO/UBERON/
MONDO/ChEBI). The only per-ontology surface is the constants block
below + the thin function wrappers.

Re-exports `ensure_cached` from `ontology_helpers` so existing test
fixtures that patch `<this_module>.ensure_cached` keep working without
having to know the helper indirection.
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
from knowledge_agent.kg.schema import GO_IS_A_REL, GO_TERM_LABEL

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# `ensure_cached` re-exported above so tests can still patch
# `ontology_go_writes.ensure_cached`. The shared `import_ontology_data` uses
# the helper's own `ensure_cached`, NOT this re-export — so for
# strict invocation-tracking tests, patch
# `ontology_helpers.ensure_cached` instead.
_ = ensure_cached


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("biology", "chemistry", "proteins")
"""Domain tags this ontology serves. The wizard suggests GO for cell
biology / biochemistry corpora; biochemistry is the strongest fit."""

# The "go-basic" subset is the standard non-import variant (filters out
# external ontologies' imports) — the recommended GO download for most
# users. ~50K terms, ~200 MB OBO source (plain text, NOT gzipped).
GO_DOWNLOAD_URL = "http://purl.obolibrary.org/obo/go/go-basic.obo"
GO_CACHE_FILENAME = "go-basic.obo"
GO_ID_PREFIX = "GO"
# Surfaced by `ontology_lifecycle.import_ontology_plan` ("GO (~200 MB
# download)" in the confirmation dialog).
DOWNLOAD_SIZE_MB = 200

# Internal identifier used for log messages in the shared helpers.
_ONTOLOGY_NAME = "GO"


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
_GO_COVERS_LABELS: tuple[str, ...] = ("CHEMICAL", "PROTEIN")

_GO_PROVENANCE = OntologyProvenance(
    ontology_name="go",
    full_name="Gene Ontology",
    publisher="Gene Ontology Consortium",
    license="CC BY 4.0",
    source_url="http://geneontology.org/",
    download_url=GO_DOWNLOAD_URL,
    file_format="OBO",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=50000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_GO_COVERS_LABELS,
    description=(
        "Biological processes, molecular functions, and cellular "
        "components. The canonical functional annotation vocabulary. "
    ),
    heavy_warning=None,
)

# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared pronto orchestration.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:GOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client,
        term_label=GO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def import_go(
    client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download + parse + write the GO ontology to Neo4j. Idempotent.

    Returns `True` when the import actually ran, `False` when GO was
    already imported (no-op). When `force=True`, deletes existing GO
    data first then runs the full import. Download / parse / write
    failures propagate to the caller under the typed-errors contract.
    """
    return await import_ontology_data(
        client,
        ontology_name=_ONTOLOGY_NAME,
        url=GO_DOWNLOAD_URL,
        cache_filename=GO_CACHE_FILENAME,
        term_label=GO_TERM_LABEL,
        hierarchy_rel=GO_IS_A_REL,
        read_and_extract=_read_and_extract,
        force=force,
        xrefs_mode=xrefs_mode,
    )


async def delete_imported(client) -> None:
    """DETACH DELETE every :GOTerm node + its :GO_IS_A edges.

    Idempotent. Used by `await import_go(force=True)` and by maintenance flows.
    """
    await delete_ontology_terms(
        client,
        term_label=GO_TERM_LABEL,
        ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(
    client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:GOTerm` nodes + `:GO_IS_A` edges.

    Public for testability + direct write paths (some tests construct
    OntologyTerm lists and call write_terms directly).
    """
    await write_ontology_terms(
        client,
        terms,
        term_label=GO_TERM_LABEL,
        hierarchy_rel=GO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


# ---------------------------------------------------------------------------
# Reading + extraction. Kept on the per-ontology module so tests can
# patch `<module>._read_and_extract` to inject canned term lists.
# ---------------------------------------------------------------------------


def _read_and_extract(path: Path) -> list[OntologyTerm]:
    """Parse the cached GO OBO file via pronto and extract terms.

    Reads via `read_obo` (lazy pronto import), then delegates to the
    shared `extract_terms_obo` since GO follows the standard OBO
    Foundry layout: id with prefix, name, synonyms, is_a parents,
    optional definition.
    """
    ontology = read_obo(path)
    return extract_terms_obo(ontology, id_prefix=GO_ID_PREFIX)
