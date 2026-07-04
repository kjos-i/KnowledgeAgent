"""Integration tests for `search/client.py` — the LanceDB store.

Exercises schema + write + read paths against a real LanceDB at the
`.env.test` `LANCEDB_PATH`. Search (hybrid / fts / vector) is left for
the agent-side integration tests; this file pins the storage contract.

LanceDB is embedded — no server / port. The "connection" is a handle
to an on-disk directory. The `clean_lance` fixture drops the chunks
table before each test so every case starts schema-fresh.

Synthetic chunk rows use a zero embedding vector (the schema needs the
fixed-size float32 list; real embeddings come from Voyage at ingest
time). All other fields match `search/schema.py:chunks_schema()`.

Manual interactive counterpart: `scripts/smoke_lancedb.py`.

Requires the test LanceDB path from `.env.test`. Skipped by default;
opt in via `pytest -m integration`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from knowledge_agent.config import get_settings
from knowledge_agent.search.schema import CHUNKS_TABLE

pytestmark = pytest.mark.integration

SYNTHETIC_DOC_ID = "integ-doc-lance-001"


def _synthetic_chunk(index: int, text: str) -> dict[str, Any]:
    """One schema-conformant chunk row. Embedding is zeros (stand-in
    for the Voyage vector; the schema needs a fixed-size float32 list)."""
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
        "doi": "10.9990/integration-test-1",
        "openalex_id": "W9990000001",
        "title": "A Synthetic Paper",
        "year": 2026,
        "authors_display": "Integration, J., Synthetic, B.",
        "venue": "Journal of Integration Tests",
        "source_url": None,
        "metadata_status": "enriched",
        "language": "en",
        "source_path": "/tmp/integration-synthetic.pdf",
        "ingested_at": datetime.now(),
    }


async def test_ensure_schema_creates_table(lance_client: Any, clean_lance: None) -> None:
    """ensure_schema creates the chunks table when it doesn't exist."""
    assert CHUNKS_TABLE not in lance_client.conn.table_names()

    # success: returns None (typed-errors contract)
    assert await lance_client.ensure_schema() is None
    assert CHUNKS_TABLE in lance_client.conn.table_names()


async def test_ensure_schema_is_idempotent(lance_client: Any, clean_lance: None) -> None:
    """Two ensure_schema calls succeed; the second is a no-op."""
    await lance_client.ensure_schema()
    # success: returns None (typed-errors contract)
    assert await lance_client.ensure_schema() is None
    assert CHUNKS_TABLE in lance_client.conn.table_names()


async def test_drop_chunks_table_is_idempotent_on_missing(
    lance_client: Any, clean_lance: None
) -> None:
    """drop_chunks_table on a missing table is a no-op."""
    # clean_lance already dropped; second drop is a no-op.
    # success: returns None (typed-errors contract)
    assert await lance_client.drop_chunks_table() is None


async def test_write_chunks_appends_rows(lance_client: Any, clean_lance: None) -> None:
    """write_chunks adds the rows to the table; row count reflects
    the append."""
    await lance_client.ensure_schema()
    chunks = [
        _synthetic_chunk(0, "Intro paragraph."),
        _synthetic_chunk(1, "Methods overview."),
        _synthetic_chunk(2, "Results discussion."),
    ]

    # success: returns None (typed-errors contract)
    assert await lance_client.write_chunks(chunks) is None

    table = lance_client.conn.open_table(CHUNKS_TABLE)
    assert table.count_rows() == 3


async def test_write_chunks_empty_batch_succeeds_without_append(
    lance_client: Any, clean_lance: None
) -> None:
    """Empty chunk list is the documented success no-op."""
    await lance_client.ensure_schema()
    # success: returns None (typed-errors contract)
    assert await lance_client.write_chunks([]) is None

    table = lance_client.conn.open_table(CHUNKS_TABLE)
    assert table.count_rows() == 0


async def test_get_chunks_by_doc_id_returns_only_matching_rows(
    lance_client: Any, clean_lance: None
) -> None:
    """get_chunks_by_doc_id returns rows for ONLY the requested
    doc_id, sorted by chunk_index."""
    await lance_client.ensure_schema()
    chunks = [
        _synthetic_chunk(0, "Intro."),
        _synthetic_chunk(1, "Methods."),
    ]
    # Add a row for a different doc_id to make sure the filter holds.
    other = _synthetic_chunk(0, "Other doc intro.")
    other["doc_id"] = "integ-doc-lance-OTHER"
    other["chunk_id"] = "integ-doc-lance-OTHER#0"
    chunks.append(other)
    await lance_client.write_chunks(chunks)

    rows = await lance_client.get_chunks_by_doc_id(SYNTHETIC_DOC_ID)
    assert len(rows) == 2
    assert all(r["doc_id"] == SYNTHETIC_DOC_ID for r in rows)


async def test_delete_chunks_by_doc_id_removes_only_target_doc(
    lance_client: Any, clean_lance: None
) -> None:
    """delete_chunks_by_doc_id wipes rows for the named doc and
    preserves rows for other docs."""
    await lance_client.ensure_schema()
    chunks = [
        _synthetic_chunk(0, "Intro."),
        _synthetic_chunk(1, "Methods."),
    ]
    other = _synthetic_chunk(0, "Other.")
    other["doc_id"] = "integ-doc-lance-OTHER"
    other["chunk_id"] = "integ-doc-lance-OTHER#0"
    chunks.append(other)
    await lance_client.write_chunks(chunks)
    assert lance_client.conn.open_table(CHUNKS_TABLE).count_rows() == 3

    # success: returns None (typed-errors contract)
    assert await lance_client.delete_chunks_by_doc_id(SYNTHETIC_DOC_ID) is None

    table = lance_client.conn.open_table(CHUNKS_TABLE)
    assert table.count_rows() == 1
    remaining = await lance_client.get_chunks_by_doc_id("integ-doc-lance-OTHER")
    assert len(remaining) == 1


async def test_delete_chunks_on_missing_table_is_noop_success(
    lance_client: Any, clean_lance: None
) -> None:
    """delete_chunks_by_doc_id treats missing-table as no-op success
    (documented contract — fresh corpus / smoke clear)."""
    # No ensure_schema call — table doesn't exist.
    # success: returns None (typed-errors contract)
    assert await lance_client.delete_chunks_by_doc_id(SYNTHETIC_DOC_ID) is None


async def test_list_indexed_docs_returns_one_row_per_doc_id(
    lance_client: Any, clean_lance: None
) -> None:
    """list_indexed_docs aggregates chunks to one entry per distinct
    doc_id, with a chunk count."""
    await lance_client.ensure_schema()
    # Write 3 chunks for SYNTHETIC_DOC_ID and 2 for OTHER.
    chunks = [
        _synthetic_chunk(0, "A0."),
        _synthetic_chunk(1, "A1."),
        _synthetic_chunk(2, "A2."),
    ]
    for i in range(2):
        other = _synthetic_chunk(i, f"B{i}.")
        other["doc_id"] = "integ-doc-lance-B"
        other["chunk_id"] = f"integ-doc-lance-B#{i}"
        chunks.append(other)
    await lance_client.write_chunks(chunks)

    docs = await lance_client.list_indexed_docs()
    by_doc_id = {d["doc_id"]: d for d in docs}
    assert SYNTHETIC_DOC_ID in by_doc_id
    assert "integ-doc-lance-B" in by_doc_id
    assert by_doc_id[SYNTHETIC_DOC_ID]["n_chunks"] == 3
    assert by_doc_id["integ-doc-lance-B"]["n_chunks"] == 2


async def test_update_doc_metadata_rewrites_doc_level_fields(
    lance_client: Any, clean_lance: None
) -> None:
    """update_doc_metadata changes the doc-level columns (title /
    year / venue / etc.) on every chunk row for the doc."""
    await lance_client.ensure_schema()
    await lance_client.write_chunks(
        [
            _synthetic_chunk(0, "Original title chunk."),
            _synthetic_chunk(1, "Same doc, second chunk."),
        ]
    )

    new_fields = {
        "title": "Updated Title",
        "year": 2027,
        "venue": "Updated Venue",
        "metadata_status": "enriched",
    }
    # success: returns None (typed-errors contract)
    assert await lance_client.update_doc_metadata(SYNTHETIC_DOC_ID, new_fields) is None

    rows = await lance_client.get_chunks_by_doc_id(SYNTHETIC_DOC_ID)
    assert all(r["title"] == "Updated Title" for r in rows)
    assert all(r["year"] == 2027 for r in rows)
    assert all(r["venue"] == "Updated Venue" for r in rows)


async def test_ensure_indexes_below_threshold_returns_true_without_failure(
    lance_client: Any, clean_lance: None
) -> None:
    """ensure_indexes on a small table (< min_rows_for_vector_index)
    successfully short-circuits — FTS is always created; vector index
    waits for the row threshold. Either way the call returns None."""
    await lance_client.ensure_schema()
    # Write just a few chunks — below the IVF_PQ training threshold.
    await lance_client.write_chunks(
        [
            _synthetic_chunk(0, "Tiny corpus."),
            _synthetic_chunk(1, "Tiny corpus 2."),
        ]
    )

    # success: returns None (typed-errors contract)
    assert await lance_client.ensure_indexes() is None
