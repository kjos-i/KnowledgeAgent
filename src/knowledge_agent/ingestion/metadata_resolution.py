"""OpenAlex metadata resolution for one document.

Extracted from `pipeline.py` (refactor 2026-06-29) — coherent
concern with its own external API, retry posture, and state machine.
Keeping it in pipeline.py made the orchestrator ~1500 lines; moving
it here drops pipeline.py to ~1200 and gives metadata its own
testable surface.

Two public entry points used by `pipeline.ingest_document` AND by
`bulk_ops.backfill_metadata`:

  - `lookup_known_doi(doc_id, doi)` — user supplied a DOI (typed in
    the GUI Edit dialog). No fallback. If THIS DOI doesn't resolve,
    return failure rather than silently re-extracting a different
    DOI from chunk text.
  - `resolve_openalex(doc_id, skip_manual=False)` — full resolution
    with chunk-text fallback. Tries the stored DOI first; falls back
    to extracting DOI candidates from chunks via `resolve_metadata`.
    Used by the "Retry pending metadata" bulk op.

Plus the shared application step + work-to-columns mapper:

  - `_apply_resolved_work(doc_id, work, sub_label)` — patches
    LanceDB doc-cols + surgically rewrites KG L1-L4 (citations,
    authorships, venue, topics) when the doc is a Paper.
  - `_doc_metadata_fields_from_work(work)` — single source of truth
    for the work-dict → LanceDB-column mapping. Used by ingest
    (`_build_lance_rows`) AND by `_apply_resolved_work` so the two
    paths stay in lockstep.

The metadata state machine (`metadata_status` field on each row):

  - `"enriched"` — DOI resolved against OpenAlex, full metadata.
  - `"pending"` — DOI candidate found but didn't resolve; eligible
    for `resolve_openalex` retry.
  - `"baseline"` — no DOI candidate found at all (non-paper inputs).
  - `"manual"` — user-edited via the Metadata view. Protected from
    `resolve_openalex(skip_manual=True)`.

The four-value enum lives in `search/schema.py::METADATA_STATUS_VALUES`.
"""

from __future__ import annotations

import logging
from typing import Any

from knowledge_agent.ingestion.metadata import resolve_doi, resolve_metadata
from knowledge_agent.ingestion.parse import ParsedChunk
from knowledge_agent.kg.client import get_kg_client
from knowledge_agent.kg.openalex_writes import (
    _clean_doi_for_storage,
    _extract_openalex_id,
)
from knowledge_agent.kg.schema import PAPER_LABEL
from knowledge_agent.search.client import get_search_client

logger = logging.getLogger(__name__)


_AUTHORS_DISPLAY_MAX = 3
"""Cap for the human-readable author display string. After this
many names, the rest are summarised as `'et al.'`."""


# ---------------------------------------------------------------------------
# Public ops: OpenAlex resolution for one document.
# ---------------------------------------------------------------------------


async def lookup_known_doi(doc_id: str, doi: str) -> dict[str, Any]:
    """Hit OpenAlex with a user-supplied DOI; NO chunk-extraction fallback.

    Per-doc partial op. Variant of `resolve_openalex` used when the caller
    KNOWS the DOI (typed in the GUI Edit dialog) and wants OpenAlex
    lookup-only - if this DOI doesn't resolve, return failure rather
    than silently re-extracting a different DOI from chunks (which could
    silently overwrite the user's typed value).

    Behavior:
      - Calls `resolve_doi(doi)` directly. No fallback.
      - On success: same as `resolve_openalex` (patch LanceDB doc-cols +
        surgical KG L1-L4 rewrite for Paper).
      - On failure: LanceDB and KG state unchanged.

    No `skip_manual` flag - clicking the row Edit dialog's "Look up DOI
    online" checkbox is the explicit consent.

    Returns dict:
      - "work_resolved" (bool)
      - "metadata_patched" (bool)
      - "kg_l1_l4_ok" (bool)
      - "new_status" (str | None)
    """
    if not doi:
        logger.warning("lookup_known_doi (%s): empty DOI; aborting", doc_id)
        return {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "lookup_known_doi (%s): LanceDB read failed: %r",
            doc_id,
            exc,
        )
        return {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }
    if not chunk_rows:
        logger.warning(
            "lookup_known_doi (%s): no chunks in LanceDB; aborting",
            doc_id,
        )
        return {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    sub_label = chunk_rows[0].get("sub_label")

    try:
        work = await resolve_doi(doi)
    except Exception as exc:
        logger.warning("lookup_known_doi (%s): resolve_doi %r failed: %r", doc_id, doi, exc)
        return {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }
    if work is None:
        logger.info(
            "lookup_known_doi (%s): DOI %r did not resolve; state unchanged",
            doc_id,
            doi,
        )
        return {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    applied = await _apply_resolved_work(doc_id, work, sub_label)
    return {"work_resolved": True, **applied}


async def resolve_openalex(doc_id: str, *, skip_manual: bool = False) -> dict[str, Any]:
    """Retry OpenAlex resolution for one doc; patch LanceDB + KG L1-L4.

    Per-doc partial op. Use when:
      - `metadata_status` is "pending" (the original ingest tried but
        OpenAlex didn't have it).
      - You want to refresh stale OpenAlex metadata.
      - User edited metadata manually and then wants to re-resolve
        (must pass `skip_manual=False` explicitly).

    Preconditions:
      - Doc's chunks must already be in LanceDB; this op reads them,
        never re-parses the source file.

    Behavior:
      - If `skip_manual=True` AND stored `metadata_status == "manual"`,
        the op is a no-op (protects user-curated metadata).
      - Tries the stored DOI first (fast path). Falls back to extracting
        DOI candidates from chunk text + `resolve_metadata` if no DOI
        is stored.
      - On a successful resolve:
        - Patches LanceDB doc-cols via `search.update_doc_metadata`.
        - For `sub_label == "Paper"`: surgically wipes L1-L4 edges via
          `kg.delete_doc_l1_l4_edges` (preserves `:PART_OF` to `:Chunks`)
          and writes fresh citations / authorships / venue / topics.
        - Sets `metadata_status` to "enriched".
      - On failure: LanceDB and KG state unchanged.

    Returns dict:
      - "skipped" (bool): True if `skip_manual` short-circuited.
      - "work_resolved" (bool): whether OpenAlex returned a work.
      - "metadata_patched" (bool): whether LanceDB was patched.
      - "kg_l1_l4_ok" (bool): whether KG L1-L4 rewrite succeeded
        (False when `sub_label != "Paper"` since L1-L4 doesn't apply).
      - "new_status" (str | None): new `metadata_status` if changed.
    """
    search_client = get_search_client()
    try:
        chunk_rows = await search_client.get_chunks_by_doc_id(doc_id)
    except Exception as exc:
        logger.warning(
            "resolve_openalex (%s): LanceDB read failed: %r",
            doc_id,
            exc,
        )
        return {
            "skipped": False,
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }
    if not chunk_rows:
        logger.warning(
            "resolve_openalex (%s): no chunks in LanceDB; aborting",
            doc_id,
        )
        return {
            "skipped": False,
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    first_row = chunk_rows[0]
    if skip_manual and first_row.get("metadata_status") == "manual":
        logger.info(
            "resolve_openalex (%s): skip_manual=True + status=manual; skip",
            doc_id,
        )
        return {
            "skipped": True,
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    sub_label = first_row.get("sub_label")
    stored_doi = first_row.get("doi")

    # Two-step resolution: stored DOI first (fast), chunk-text extraction
    # fallback when no DOI is stored OR the stored DOI didn't resolve.
    # resolve_doi raises on API failure; resolve_metadata catches per-
    # candidate. Wrap the stored-DOI fast path so a transient outage on
    # that lookup falls through to the chunk-text scan.
    work: dict[str, Any] | None = None
    if stored_doi:
        try:
            work = await resolve_doi(stored_doi)
        except Exception as exc:
            logger.warning(
                "resolve_openalex (%s): stored-DOI %r lookup failed: %r; "
                "falling back to chunk-text scan",
                doc_id,
                stored_doi,
                exc,
            )
    if work is None:
        chunks = [
            ParsedChunk(
                chunk_index=r["chunk_index"],
                text=r["text"],
                section=r.get("section"),
                page=r.get("page"),
                content_type=r.get("content_type", "text"),
            )
            for r in chunk_rows
        ]
        work = await resolve_metadata(chunks)

    if work is None:
        logger.info(
            "resolve_openalex (%s): no work resolved; state unchanged",
            doc_id,
        )
        return {
            "skipped": False,
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        }

    applied = await _apply_resolved_work(doc_id, work, sub_label)
    return {"skipped": False, "work_resolved": True, **applied}


# ---------------------------------------------------------------------------
# Shared application + mapping.
# ---------------------------------------------------------------------------


async def _apply_resolved_work(
    doc_id: str, work: dict[str, Any], sub_label: str | None
) -> dict[str, Any]:
    """Apply a freshly-resolved OpenAlex work to LanceDB + KG L1-L4.

    Shared post-resolution step for `resolve_openalex` and
    `lookup_known_doi`:
      - Patches LanceDB doc-cols via `update_doc_metadata` (single source
        of truth for the work->columns mapping via `_doc_metadata_fields_from_work`).
      - For `sub_label == "Paper"`: surgically wipes L1-L4 edges (preserves
        `:PART_OF` to `:Chunks`) and writes citations / authorships /
        venue / topics.
      - For non-Paper sub_labels: skips KG writes (L1-L4 doesn't apply);
        LanceDB doc-cols still patched.

    Returns dict with "metadata_patched" (bool), "kg_l1_l4_ok" (bool),
    "new_status" (str). Caller merges this into its result dict.
    """
    search_client = get_search_client()
    new_fields = _doc_metadata_fields_from_work(work)
    new_fields["metadata_status"] = "enriched"
    try:
        await search_client.update_doc_metadata(doc_id, new_fields)
        metadata_patched = True
    except Exception as exc:
        logger.warning(
            "_apply_resolved_work (%s): update_doc_metadata failed: %r",
            doc_id,
            exc,
        )
        metadata_patched = False

    kg_l1_l4_ok = False
    if sub_label == PAPER_LABEL:
        kg_client = get_kg_client()
        # L1-L4 rewrite is a single coherent stage from the user's
        # perspective. One try/except covers the whole sequence
        # including ensure_constraints (which now propagates Cypher
        # errors under the typed-errors contract); if any step fails,
        # the whole stage is False.
        try:
            await kg_client.ensure_constraints()
            await kg_client.delete_doc_l1_l4_edges(doc_id)
            await kg_client.write_citations(doc_id, work)
            await kg_client.write_authorships(doc_id, work)
            await kg_client.write_venue(doc_id, work)
            await kg_client.write_topics(doc_id, work)
            kg_l1_l4_ok = True
        except Exception as exc:
            logger.warning(
                "_apply_resolved_work (%s): L1-L4 rewrite failed: %r",
                doc_id,
                exc,
            )

    return {
        "metadata_patched": metadata_patched,
        "kg_l1_l4_ok": kg_l1_l4_ok,
        "new_status": "enriched",
    }


def _doc_metadata_fields_from_work(
    work: dict[str, Any] | None,
) -> dict[str, Any]:
    """Map an OpenAlex `work` dict (or None) to LanceDB doc-level columns.

    Single source of truth for the work->columns derivation, used by
    `_build_lance_rows` during ingest AND by `resolve_openalex` during
    metadata refresh. Returned dict carries the eight OpenAlex-derived
    fields; the caller adds `metadata_status` separately because that
    string is decided by the call-site (enriched/pending/baseline).
    """
    if work is None:
        return {
            "title": None,
            "year": None,
            "doi": None,
            "openalex_id": None,
            "venue": None,
            "source_url": None,
            "authors_display": None,
            "language": None,
        }
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}
    return {
        "title": work.get("title") or work.get("display_name"),
        "year": work.get("publication_year"),
        "doi": _clean_doi_for_storage(work.get("doi")),
        "openalex_id": _extract_openalex_id(work.get("id")),
        "venue": source.get("display_name"),
        "source_url": primary_location.get("landing_page_url"),
        "authors_display": _build_authors_display(work.get("authorships") or []),
        "language": work.get("language"),
    }


def _build_authors_display(authorships: list[dict[str, Any]]) -> str | None:
    """'Smith, Jones, Lee, et al.' - human-readable author list cache.

    Caps at the first `_AUTHORS_DISPLAY_MAX` names; appends 'et al.' when
    more authorships exist. Returns None when no resolvable names.
    """
    names: list[str] = []
    for au in authorships[:_AUTHORS_DISPLAY_MAX]:
        author = au.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
    if not names:
        return None
    display = ", ".join(names)
    if len(authorships) > _AUTHORS_DISPLAY_MAX:
        display += ", et al."
    return display
