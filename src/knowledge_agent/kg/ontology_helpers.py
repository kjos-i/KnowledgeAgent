"""Shared utilities for L7 ontology imports — leaf module.

Three-file split (refactor 2026-06-29):
  - `ontology_helpers.py` (THIS FILE) — leaf shared by all three
    families: `OntologyTerm` dataclass + download/cache utilities.
    Slimmed from its historical role (~930 lines) to ~140 lines.
  - `ontology_pronto.py` — pronto-based OBO reading + extraction.
  - `ontology_rdf.py` — rdflib-based RDF/OWL/SKOS reading + extraction.
  - `ontology_writes.py` — Neo4j Cypher writes (format-agnostic).

For backward compatibility, this module RE-EXPORTS every public symbol
from the three new files so the 19 per-ontology importers + the
test suite don't need their imports rewritten.

Two leaf concerns owned directly by this file:

  1. **`OntologyTerm` dataclass.** The contract every reader produces
     and every writer consumes. Frozen + hashable so terms can be
     deduped via set membership if needed.

  2. **Download + cache.** Source ontology files (often hundreds of
     MB) live in a local cache directory so subsequent ingestions
     don't re-download. `ensure_cached(url, filename)` is the single
     entry point; it handles atomic writes (download to .tmp then
     rename) so an interrupted download never leaves a corrupted
     file in place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowledge_agent import _http_client
from knowledge_agent.config import get_settings

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared dataclass: the contract every reader produces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OntologyTerm:
    """One canonical concept from an imported ontology.

    The reader functions (`extract_terms_obo`, `extract_terms_skos`,
    `extract_terms_owl`) yield these; the per-ontology write modules
    consume them and produce Cypher MERGE statements. Frozen + hashable
    so terms can be deduped via set membership if needed.

    All fields are normalised:
      - `id`        prefixed canonical identifier (e.g. "MESH:D003920",
                    "GO:0008150"). The prefix is the ontology's standard
                    namespace.
      - `label`     the primary label, preserved with original casing.
      - `synonyms`  lowercased - downstream matching is case-insensitive
                    against the lowercased `key` on :Entity nodes, so we
                    pre-lowercase to avoid repeated work at link time.
      - `parents`   ids of broader/superclass terms (same prefix convention).
      - `definition` optional human-readable description; useful for LLM
                    schema-as-prompt and for diagnostic queries.
      - `xrefs`     L7 cross-ontology references. Prefixed canonical IDs
                    of equivalent / closely-related terms in OTHER
                    ontologies (e.g. ("MESH:D003920", "DOID:9352") on a
                    MONDO term). Stored verbatim as the source declares
                    them - no normalisation. When the `xrefs` layer is
                    in "use" mode these strings are written as
                    `:<X>_XREF` edges to existing :OntologyTerm nodes;
                    in any non-"none" mode they're stored as a
                    `dangling_xrefs` property on the source term for
                    later backfill. Defaults `()` so pre-xrefs-ship
                    code paths and tests work unchanged.
    """

    id: str
    label: str
    synonyms: tuple[str, ...]
    parents: tuple[str, ...]
    definition: str | None
    xrefs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Downloads directory + download utilities.
# ---------------------------------------------------------------------------


def get_downloads_dir() -> Path:
    """Return the ontology downloads directory, creating it if missing.

    Resolves from `Settings.ontology_downloads_dir` (defaults to
    `~/.research-literature-agent/ontology-downloads/`). The directory
    is created on first call — safe to call repeatedly.
    """
    downloads = get_settings().ontology_downloads_dir
    downloads.mkdir(parents=True, exist_ok=True)
    return downloads


async def ensure_cached(url: str, filename: str, *, force: bool = False) -> Path:
    """Download `url` into the cache as `filename` if not already present.

    Returns the cached file's local path. Idempotent: if the file exists
    and `force=False`, returns the existing path without re-downloading.

    Atomic writes: downloads to `<filename>.tmp` and renames on success
    so an interrupted download never leaves a partial file at the final
    path. The .tmp is cleaned up on failure.

    `force=True` forces a fresh download (overwrites the cached copy).
    Use when re-fetching after a known ontology release update.

    Streams via the central HTTP client. NOT retried (see `_http_client`
    module docstring — partial-download replay would corrupt the cache
    file). Raises `httpx.HTTPError` on network failure; the caller
    (per-ontology write module) decides whether to surface the error.
    """
    downloads = get_downloads_dir()
    dest = downloads / filename
    if dest.exists() and not force:
        logger.info("ontology download hit: %s", dest)
        return dest

    # `filename` may include nested directory components for ontologies that
    # ship as many files (e.g. FIBO's modular layout: "fibo/FND/.../foo.rdf").
    # Make the parents idempotently so the atomic rename below succeeds.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info("ontology: downloading %s -> %s", url, dest)
    try:
        async with _http_client.stream(url, timeout=None) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                # aiter_raw (NOT aiter_bytes) so httpx does NOT auto-
                # decompress `Content-Encoding: gzip` responses. We want
                # the bytes exactly as served - matters for sources like
                # NLM's MeSH which sets Content-Encoding: gzip on
                # already-gzipped files. With aiter_bytes the cached
                # file would be ~18x larger AND wouldn't match its .gz
                # filename.
                async for chunk in response.aiter_raw(chunk_size=65536):
                    fh.write(chunk)
        tmp.replace(dest)
    except Exception:
        # Tidy up partial file so a retry starts clean.
        if tmp.exists():
            tmp.unlink()
        raise

    logger.info("ontology: downloaded %d bytes to %s", dest.stat().st_size, dest)
    return dest


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports.
#
# Every public symbol that used to live here is re-exported from the
# three new modules so the 19 per-ontology importers + the existing
# test suite don't need import rewrites. New code should import
# directly from the canonical module; tests + legacy code keep working
# against `knowledge_agent.kg.ontology_helpers`.
#
# Placed at the bottom of the file (NOT the top) so the dataclass +
# cache definitions are independent of the imports below — avoids
# circular-import surprises since the new modules each import
# `OntologyTerm` from this file.
# ---------------------------------------------------------------------------


# Pronto OBO family.
from knowledge_agent.kg.ontology_pronto import (  # noqa: E402
    extract_terms_obo,
    read_obo,
)

# rdflib RDF / OWL / SKOS family.
from knowledge_agent.kg.ontology_rdf import (  # noqa: E402
    _default_id_extractor,
    _first_english_literal,
    _owl_id_extractor,
    extract_terms_owl,
    extract_terms_skos,
    read_rdf,
)

# Neo4j writes family.
from knowledge_agent.kg.ontology_writes import (  # noqa: E402
    _validate_xrefs_mode,
    _xref_rel_from_term_label,
    delete_ontology_terms,
    import_ontology_data,
    is_ontology_imported,
    write_ontology_terms,
)

__all__ = [
    # Leaf concerns owned by this file
    "OntologyTerm",
    # Internal helpers exposed for tests that patch them by path.
    "_default_id_extractor",
    "_first_english_literal",
    "_owl_id_extractor",
    "_validate_xrefs_mode",
    "_xref_rel_from_term_label",
    # Writes family (re-exported from ontology_writes)
    "delete_ontology_terms",
    "ensure_cached",
    # Pronto family (re-exported from ontology_pronto)
    "extract_terms_obo",
    # rdflib family (re-exported from ontology_rdf)
    "extract_terms_owl",
    "extract_terms_skos",
    "get_downloads_dir",
    "import_ontology_data",
    "is_ontology_imported",
    "read_obo",
    "read_rdf",
    "write_ontology_terms",
]
