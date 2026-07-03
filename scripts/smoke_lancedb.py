"""Smoke test for LanceDB chunk writes (with manual-inspection pause).

Constructs three synthetic chunk dicts matching the schema, applies the
schema (ensure_schema), writes them, then reads them back to verify.

Embedding vectors are zeros - stand-ins so the schema's fixed-size float32
list accepts them. Real embeddings come from Voyage at ingestion time.

Run from the project root:
    python scripts/smoke_lancedb.py            # normal per-doc cleanup
    python scripts/smoke_lancedb.py --reset    # drop LanceDB table first
                                               # (use after schema changes)

Automated counterpart (for regression catching, no inspection):
  tests/integration/search/test_search_client.py  (schema + writes +
                                                   reads + delete +
                                                   list + update +
                                                   index)
Run via `pytest -m integration tests/integration/search/`.

Lifecycle (mirrors `smoke_kg_l1_l5.py` / `smoke_pipeline.py`):
  1. Clear any leftover smoke chunks from previous runs. With `--reset`:
     also drop the entire LanceDB chunks table (recreated by
     `ensure_schema` at write time).
  2. Write the synthetic chunks.
  3. Pause - you inspect LanceDB.
  4. Press Enter to clean up, Ctrl+C to keep the chunks.
"""

import argparse
import asyncio
from datetime import datetime

# Switch the process to the smoke-test LanceDB path (lancedb_test) BEFORE
# any import that triggers `get_settings()`. Mirrors the other smokes so
# this script writes synthetic chunks to the test store, never the real
# corpus's `./lance_data`.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.search.client import get_search_client  # noqa: E402
from knowledge_agent.search.schema import CHUNKS_TABLE  # noqa: E402

SYNTHETIC_DOC_ID = "smoke-doc-001"


def _synthetic_chunk(index: int, text: str) -> dict:
    """Build one chunk dict matching the schema. Embedding is a zero vector."""
    dims = get_settings().embedding_dims
    return {
        "chunk_id": f"{SYNTHETIC_DOC_ID}#{index}",
        "doc_id": SYNTHETIC_DOC_ID,
        "chunk_index": index,
        "section": "Introduction",
        "page": 1,
        "char_start": index * 500,
        "char_end": index * 500 + len(text),
        "content_type": "text",
        "image_ref": None,
        "text": text,
        "embedding": [0.0] * dims,
        "main_label": "Document",
        "sub_label": "Paper",
        "doi": "10.9990/smoke-test-1",
        "openalex_id": "W9990000001",
        "title": "A Synthetic Paper",
        "year": 2026,
        "authors_display": "Smoketest, J., Synthetic, B., Stub, C.",
        "venue": "Journal of Smoke Tests",
        "source_url": None,
        "metadata_status": "enriched",
        "language": "en",
        "source_path": "/tmp/smoke-synthetic.pdf",
        "ingested_at": datetime.now(),
    }


SYNTHETIC_CHUNKS = [
    _synthetic_chunk(0, "First chunk: introductory paragraph."),
    _synthetic_chunk(1, "Second chunk: methods overview."),
    _synthetic_chunk(2, "Third chunk: results discussion."),
]


async def _cleanup_synthetic_chunks(client) -> None:
    """Remove every chunk row written by this smoke test."""
    try:
        await client.delete_chunks_by_doc_id(SYNTHETIC_DOC_ID)
    except Exception as exc:
        print(f"(cleanup error, may be benign: {exc})")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LanceDB chunks-table smoke test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Drop the LanceDB chunks table entirely before writing. Use "
            "after a schema change so the table is recreated with current "
            "columns. Without this flag the smoke does per-doc cleanup only."
        ),
    )
    args = parser.parse_args()

    client = get_search_client()
    settings = get_settings()
    print(f"LanceDB path: {settings.lancedb_path}")
    print(f"Embedding dim: {settings.embedding_dims}")

    if args.reset:
        print("--reset: dropping LanceDB chunks table (schema-fresh start)...")
        await client.drop_chunks_table()

    print("Applying schema...")
    await client.ensure_schema()
    print("  ensure_schema -> ok")

    print(f"Clearing any leftover smoke chunks for doc_id={SYNTHETIC_DOC_ID!r}...")
    _cleanup_synthetic_chunks(client)

    print(f"Writing {len(SYNTHETIC_CHUNKS)} chunks...")
    await client.write_chunks(SYNTHETIC_CHUNKS)
    print(f"  write_chunks -> ok ({len(SYNTHETIC_CHUNKS)} chunks)")

    table = client.conn.open_table(CHUNKS_TABLE)
    total = table.count_rows()
    print()
    print(f"Total rows in {CHUNKS_TABLE!r}: {total}")
    print()
    print(f"Chunks for doc_id={SYNTHETIC_DOC_ID!r}:")
    rows = [
        r
        for r in table.to_arrow().to_pylist()
        if r["doc_id"] == SYNTHETIC_DOC_ID
    ]
    for r in rows:
        print(
            f"  {r['chunk_id']}: section={r['section']!r}, "
            f"text={r['text']!r}"
        )
    print()

    try:
        input("Press Enter to delete the smoke chunks, Ctrl+C to keep them. ")
    except KeyboardInterrupt:
        print()
        print("Keeping smoke chunks. Re-run this script to clean them up later.")
        await client.close()
        return

    print("Deleting smoke chunks...")
    _cleanup_synthetic_chunks(client)
    print("Done.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
