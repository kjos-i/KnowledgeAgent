"""Tests for the async siblings in `pipeline.py`.

The siblings are thin `asyncio.to_thread` wrappers around the sync
versions. These tests just confirm each wrapper:
  - is actually awaitable (not a typo `def` instead of `async def`)
  - delegates to its sync counterpart with the same arguments
  - returns the sync counterpart's return value verbatim

Per the no-test-driven-quality-compromise rule, we patch the sync
target in the pipeline module's namespace (where `asyncio.to_thread`
will look it up at call time) — NOT a renamed alias on the wrapper.

Day 7 of the async refactor rewrites `aingest_document` with native
await + per-chunk asyncio.gather; THAT change will need richer tests
(parallelism, semaphore bound, ordering). For now the contract is
purely "wraps the sync version" and these tests cover it.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from knowledge_agent.ingestion import pipeline


# ---------------------------------------------------------------------------
# delete_doc / re_embed — single doc_id arg
# ---------------------------------------------------------------------------


async def test_adelete_doc_delegates_to_delete_doc():
    with patch.object(pipeline, "delete_doc", return_value=True) as mock_fn:
        result = await pipeline.adelete_doc("doc-123")
    mock_fn.assert_called_once_with("doc-123")
    assert result is True


async def test_are_embed_delegates_to_re_embed():
    sentinel = {"doc_id": "doc-456", "n_chunks": 7}
    with patch.object(pipeline, "re_embed", return_value=sentinel) as mock_fn:
        result = await pipeline.are_embed("doc-456")
    mock_fn.assert_called_once_with("doc-456")
    assert result == sentinel


# ---------------------------------------------------------------------------
# backfill_* — (doc_id, config) signature
# ---------------------------------------------------------------------------


async def test_abackfill_chunks_delegates_to_backfill_chunks():
    config = MagicMock()
    sentinel = {"ok": True}
    with patch.object(pipeline, "backfill_chunks", return_value=sentinel) as mock_fn:
        result = await pipeline.abackfill_chunks("doc-1", config)
    mock_fn.assert_called_once_with("doc-1", config)
    assert result == sentinel


async def test_abackfill_entities_delegates_to_backfill_entities():
    config = MagicMock()
    sentinel = {"ok": True, "n_entities": 3}
    with patch.object(pipeline, "backfill_entities", return_value=sentinel) as mock_fn:
        result = await pipeline.abackfill_entities("doc-2", config)
    mock_fn.assert_called_once_with("doc-2", config)
    assert result == sentinel


async def test_abackfill_ontology_delegates_to_backfill_ontology():
    config = MagicMock()
    sentinel = {"mesh": {"linked": 2}}
    with patch.object(pipeline, "backfill_ontology", return_value=sentinel) as mock_fn:
        result = await pipeline.abackfill_ontology("doc-3", config)
    mock_fn.assert_called_once_with("doc-3", config)
    assert result == sentinel


async def test_abackfill_triples_delegates_to_backfill_triples():
    config = MagicMock()
    sentinel = {"ok": True, "n_triples": 5}
    with patch.object(pipeline, "backfill_triples", return_value=sentinel) as mock_fn:
        result = await pipeline.abackfill_triples("doc-4", config)
    mock_fn.assert_called_once_with("doc-4", config)
    assert result == sentinel


async def test_abackfill_cross_doc_delegates_to_backfill_cross_doc():
    config = MagicMock()
    sentinel = {"ok": True, "n_edges": 1}
    with patch.object(pipeline, "backfill_cross_doc", return_value=sentinel) as mock_fn:
        result = await pipeline.abackfill_cross_doc("doc-5", config)
    mock_fn.assert_called_once_with("doc-5", config)
    assert result == sentinel


async def test_abackfill_cross_doc_xrefs_delegates_to_backfill_cross_doc_xrefs():
    config = MagicMock()
    sentinel = {"ok": True, "n_edges": 2}
    with patch.object(
        pipeline, "backfill_cross_doc_xrefs", return_value=sentinel
    ) as mock_fn:
        result = await pipeline.abackfill_cross_doc_xrefs("doc-6", config)
    mock_fn.assert_called_once_with("doc-6", config)
    assert result == sentinel


# ---------------------------------------------------------------------------
# ingest_document — NOT a thin sibling anymore.
#
# Day 7b of the async refactor rewrote `aingest_document` as a native
# async implementation with parallel per-chunk fan-out (asyncio.gather +
# Semaphore for L6a + L8 extraction, gather for L1-L4 OpenAlex writes,
# gather for L9 + L10 cross-doc, parallel delete-stale). It NO LONGER
# wraps the sync `ingest_document` via asyncio.to_thread.
#
# Full coverage lives in the integration tests
# (tests/integration/ingestion/test_pipeline.py) which run the real
# parser + real KG/Lance + real LLM/embedder and verify end-to-end.
# Unit testing the native version would require mocking ~12 dependencies
# and would essentially re-implement the function's structure inside the
# test, which is brittle. The validation guard below + the integration
# tests are the right shape of coverage.
# ---------------------------------------------------------------------------


async def test_aingest_document_is_a_coroutine_function():
    """Sanity guard: name + shape don't drift accidentally."""
    import inspect
    assert inspect.iscoroutinefunction(pipeline.aingest_document)


async def test_aingest_document_validates_main_label():
    """Validation happens before any side effects (parser, KG, Lance)."""
    config = MagicMock()
    config.layers.entities = False  # avoid the entity_types pre-validation
    import pytest
    with pytest.raises(ValueError, match="main_label"):
        await pipeline.aingest_document(
            Path("test.pdf"), config, "NotAValidLabel", None
        )
