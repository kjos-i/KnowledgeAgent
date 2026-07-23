"""Smoke test for ingestion.metadata - parse one PDF, extract DOI, resolve.

Picks the first .pdf file (alphabetically) from `test_documents/`, parses
it with docling, extracts DOI candidates from the first chunks, then
resolves the first candidate via the OpenAlex API. Prints a short
summary of the work record.

Hits the real OpenAlex API, so the machine needs internet. Set
`OPENALEX_MAILTO=you@example.com` in the shared .env for the polite pool.

Run from the project root:
    python scripts/smoke_metadata.py

Automated counterpart (for regression catching, no inspection):
  tests/integration/ingestion/test_metadata.py  (extract_doi_candidates
                                                 against real PDF +
                                                 empty-input; resolve_doi
                                                 known-DOI + 404 path;
                                                 resolve_metadata full
                                                 path)
Run via `pytest -m integration tests/integration/ingestion/`.
"""

import asyncio
from pathlib import Path
from typing import Any

from knowledge_agent.corpus_config import CorpusConfig
from knowledge_agent.ingestion.metadata import (
    extract_doi_candidates,
    resolve_doi,
)
from knowledge_agent.ingestion.parse import parse_document

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


def _summarize_work(work: dict[str, Any]) -> None:
    """Print the most useful fields from an OpenAlex work record."""
    print(f"  title:           {work.get('title')!r}")
    print(f"  doi:             {work.get('doi')}")
    print(f"  openalex_id:     {work.get('id')}")
    print(f"  publication_year: {work.get('publication_year')}")
    print(f"  type:            {work.get('type')}")

    venue = (work.get("primary_location") or {}).get("source") or {}
    print(f"  venue:           {venue.get('display_name')!r}")

    authorships = work.get("authorships") or []
    print(f"  n authors:       {len(authorships)}")
    for au in authorships[:3]:
        author = au.get("author") or {}
        print(
            f"    - {author.get('display_name')!r} "
            f"(position={au.get('author_position')!r}, "
            f"corresponding={au.get('is_corresponding')})"
        )
    if len(authorships) > 3:
        print(f"    ... and {len(authorships) - 3} more")

    refs = work.get("referenced_works") or []
    print(f"  n referenced_works: {len(refs)}")


async def main() -> None:
    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return

    target = pdfs[0]
    print(f"Parsing: {target.name}")
    chunks = parse_document(target, CorpusConfig())
    print(f"Total chunks: {len(chunks)}")
    print()

    candidates = extract_doi_candidates(chunks)
    print(f"DOI candidates ({len(candidates)}): {candidates}")
    print()

    if not candidates:
        print("No DOI candidates found in first 3 chunks; aborting.")
        return

    doi = candidates[0]
    print(f"Resolving {doi!r} against OpenAlex...")
    work = await resolve_doi(doi)
    if work is None:
        print("  -> None (DOI not in OpenAlex or network error)")
        return

    print()
    print("OpenAlex work record:")
    _summarize_work(work)


if __name__ == "__main__":
    asyncio.run(main())
