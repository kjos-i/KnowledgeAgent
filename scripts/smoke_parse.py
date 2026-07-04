"""Smoke test for ingestion.parse - runs docling on one real PDF.

Picks the first .pdf file (alphabetically) from `test_documents/`, parses
it, and prints a short summary: total chunks, a peek at the first few,
and the set of sections found.

Run from the project root:
    python scripts/smoke_parse.py

First run is slow (docling initialises tokenizers / models on first use);
subsequent runs in the same process would be fast (singleton cache).

Automated counterpart (for regression catching, no inspection):
  tests/integration/ingestion/test_parse.py  (docling against PDF +
                                              JATS-XML + DOCX; chunk
                                              field invariants;
                                              unsupported-ext raise)
Run via `pytest -m integration tests/integration/ingestion/`.
"""

from pathlib import Path

from knowledge_agent.ingestion.parse import parse_document

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


def main() -> None:
    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return

    target = pdfs[0]
    print(f"Parsing: {target.name}")
    print("(first run downloads docling model artifacts - can take a minute)")
    print()

    chunks = parse_document(target)
    print(f"Total chunks: {len(chunks)}")
    print()

    print("First 3 chunks:")
    for c in chunks[:3]:
        text_preview = c.text[:120].replace("\n", " ")
        print(
            f"  [{c.chunk_index}] section={c.section!r} page={c.page} "
            f"content_type={c.content_type!r}"
        )
        print(f"      text: {text_preview!r}...")
    print()

    sections = sorted({c.section for c in chunks if c.section})
    print(f"Unique sections found ({len(sections)}):")
    for s in sections:
        print(f"  - {s}")


if __name__ == "__main__":
    main()
