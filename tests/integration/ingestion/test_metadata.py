"""Integration tests for `ingestion/metadata` — DOI extraction +
OpenAlex resolution against the real OpenAlex API.

Exercises:
  - `extract_doi_candidates(chunks)` — pure-function but runs on real
    parser output (depends on docling chunking + DOI regex)
  - `resolve_doi(doi)` — HTTP call to OpenAlex
  - `resolve_metadata(chunks)` — composes the two above

Network-dependent. Uses a known stable DOI (a published paper) so the
resolution is deterministic across runs.

**Cost: NO LLM cost.** OpenAlex is queried unauthenticated (free).
Network required.

Manual interactive counterpart: `scripts/smoke_metadata.py`.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge_agent.ingestion.metadata import (
    extract_doi_candidates,
    resolve_doi,
    resolve_metadata,
)
from knowledge_agent.ingestion.parse import parse_document

pytestmark = pytest.mark.integration

# A stable, well-indexed DOI — the einstein 1905 relativity paper.
# Used for the resolve_doi test so the assertion doesn't drift with
# OpenAlex enrichment over time.
KNOWN_DOI = "10.1002/andp.19053221004"


def test_extract_doi_candidates_finds_doi_in_real_pdf_chunks(
    sample_pdf: Path,
) -> None:
    """A real research PDF has a DOI in its first few chunks
    (typically on the title / abstract page). extract_doi_candidates
    finds it."""
    chunks = parse_document(sample_pdf)
    candidates = extract_doi_candidates(chunks)
    assert len(candidates) > 0
    # All candidates lowercase, no trailing punctuation.
    for c in candidates:
        assert c == c.lower()
        assert not c.endswith((".", ",", ";", ")"))


def test_extract_doi_candidates_returns_empty_for_no_doi_chunks() -> None:
    """Plain text with no DOI returns an empty list (not None,
    not a raise)."""
    from knowledge_agent.ingestion.parse import ParsedChunk

    chunks = [
        ParsedChunk(chunk_index=0, text="Hello, no DOI here."),
        ParsedChunk(chunk_index=1, text="Just text."),
    ]
    assert extract_doi_candidates(chunks) == []


async def test_resolve_doi_returns_work_dict_for_known_doi() -> None:
    """A real OpenAlex lookup of a known-good DOI returns the work
    JSON with at least `id`, `doi`, `title` keys."""
    work = await resolve_doi(KNOWN_DOI)
    assert work is not None
    assert "id" in work
    assert "doi" in work
    assert work["doi"].lower().endswith(KNOWN_DOI.lower())


async def test_resolve_doi_returns_none_for_unknown_doi() -> None:
    """A clearly bogus DOI returns None — the 404 path is fail-soft."""
    work = await resolve_doi("10.99999/this-definitely-does-not-exist-9999")
    assert work is None


async def test_resolve_metadata_full_path_on_real_pdf(sample_pdf: Path) -> None:
    """End-to-end: parse the real PDF, extract DOI candidates, resolve
    the first that hits OpenAlex. Should return a work dict when the
    sample PDF carries a real DOI."""
    chunks = parse_document(sample_pdf)
    work = await resolve_metadata(chunks)
    assert work is not None
    assert "id" in work
    # The work should have a title — a real OpenAlex record.
    assert work.get("title")
