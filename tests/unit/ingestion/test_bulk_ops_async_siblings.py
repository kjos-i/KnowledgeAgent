"""Tests for the async siblings in `bulk_ops.py`.

Each sibling is a thin `asyncio.to_thread` wrapper around the sync
version. One parametrized test verifies each (sync_name, async_name,
args, kwargs) tuple in the SIBLINGS manifest:

  - The async wrapper exists at the expected name (no typo).
  - It awaits + delegates to the sync function in the bulk_ops module
    namespace (so `asyncio.to_thread` resolves the patched target).
  - It forwards positional + keyword args verbatim.
  - It returns the sync function's return value unchanged.

Per the no-test-driven-quality-compromise rule, we patch the sync
target in the bulk_ops module's namespace where `asyncio.to_thread`
looks it up — NOT a renamed alias on the wrapper.

Days 5-7 of the async refactor will rewrite some siblings with
native asyncio.gather + semaphore-bounded fan-out; THOSE changes
will need richer per-sibling tests. For now the contract is purely
"wraps the sync version" and this manifest covers it.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.ingestion import bulk_ops


# (sync_name, async_name, args, kwargs)
SIBLINGS: list[tuple[str, str, tuple, dict]] = [
    # --- delete_doc pair ---
    ("delete_doc_plan", "adelete_doc_plan", ("doc-1",), {}),
    ("delete_doc_execute", "adelete_doc_execute", (MagicMock(),), {}),
    # --- bulk_resolve_openalex pair ---
    (
        "bulk_resolve_openalex_plan",
        "abulk_resolve_openalex_plan",
        (),
        {"skip_manual": True},
    ),
    (
        "bulk_resolve_openalex_execute",
        "abulk_resolve_openalex_execute",
        (MagicMock(),),
        {},
    ),
    # --- bulk_re_embed pair ---
    ("bulk_re_embed_plan", "abulk_re_embed_plan", (), {}),
    ("bulk_re_embed_execute", "abulk_re_embed_execute", (MagicMock(),), {}),
    # --- 5 bulk_backfill_X pairs (chunks, entities, ontology, triples, cross_doc) ---
    ("bulk_backfill_chunks_plan", "abulk_backfill_chunks_plan", (), {}),
    (
        "bulk_backfill_chunks_execute",
        "abulk_backfill_chunks_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    ("bulk_backfill_entities_plan", "abulk_backfill_entities_plan", (), {}),
    (
        "bulk_backfill_entities_execute",
        "abulk_backfill_entities_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    ("bulk_backfill_ontology_plan", "abulk_backfill_ontology_plan", (), {}),
    (
        "bulk_backfill_ontology_execute",
        "abulk_backfill_ontology_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    ("bulk_backfill_triples_plan", "abulk_backfill_triples_plan", (), {}),
    (
        "bulk_backfill_triples_execute",
        "abulk_backfill_triples_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    (
        "bulk_backfill_cross_doc_plan",
        "abulk_backfill_cross_doc_plan",
        (),
        {},
    ),
    (
        "bulk_backfill_cross_doc_execute",
        "abulk_backfill_cross_doc_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    # --- folder ingestion pairs ---
    (
        "ingest_folder_plan",
        "aingest_folder_plan",
        (Path("/tmp/x"), "Document", "Paper"),
        {},
    ),
    (
        "ingest_folder_execute",
        "aingest_folder_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    (
        "add_plan",
        "aadd_plan",
        (Path("/tmp/y"), "Document", None),
        {},
    ),
    ("add_execute", "aadd_execute", (MagicMock(), MagicMock()), {}),
    # --- sync_plan / sync_execute: name collision forces async_ prefix ---
    (
        "sync_plan",
        "async_plan",
        (Path("/tmp/z"), "Document", "Note"),
        {},
    ),
    ("sync_execute", "async_execute", (MagicMock(), MagicMock()), {}),
    # --- xref ops ---
    ("backfill_xrefs_plan", "abackfill_xrefs_plan", (MagicMock(),), {}),
    (
        "backfill_xrefs_execute",
        "abackfill_xrefs_execute",
        (MagicMock(), MagicMock()),
        {},
    ),
    (
        "recompute_cross_doc_xrefs_plan",
        "arecompute_cross_doc_xrefs_plan",
        (MagicMock(),),
        {},
    ),
    (
        "recompute_cross_doc_xrefs_execute",
        "arecompute_cross_doc_xrefs_execute",
        (MagicMock(),),
        {},
    ),
    ("clear_xref_edges_plan", "aclear_xref_edges_plan", ("mesh",), {}),
    (
        "clear_xref_edges_execute",
        "aclear_xref_edges_execute",
        (MagicMock(),),
        {},
    ),
]


@pytest.mark.parametrize(
    "sync_name,async_name,args,kwargs",
    SIBLINGS,
    ids=[entry[1] for entry in SIBLINGS],
)
async def test_async_sibling_delegates_to_sync(
    sync_name: str,
    async_name: str,
    args: tuple,
    kwargs: dict,
):
    """Async sibling forwards args, awaits sync, returns its result.

    Sentinel-based: the patched sync function returns a fresh
    MagicMock instance; the test asserts identity (`is`) to confirm
    the wrapper passes the return value through without wrapping.
    """
    sentinel = MagicMock()
    with patch.object(bulk_ops, sync_name, return_value=sentinel) as mock_fn:
        async_fn = getattr(bulk_ops, async_name)
        result = await async_fn(*args, **kwargs)

    mock_fn.assert_called_once_with(*args, **kwargs)
    assert result is sentinel


def test_sibling_manifest_covers_all_async_siblings():
    """Sanity guard: every `a*` / `async_*` callable in bulk_ops is in SIBLINGS.

    If someone adds a new async sibling to `bulk_ops.py` without
    updating SIBLINGS, this test fails — pointing them at the
    omission.
    """
    import inspect

    found: set[str] = set()
    for name in dir(bulk_ops):
        if not (name.startswith("a") or name.startswith("async_")):
            continue
        obj = getattr(bulk_ops, name)
        if not inspect.iscoroutinefunction(obj):
            continue
        # Exclude imports / re-exports — only the module's own siblings.
        if getattr(obj, "__module__", "") != bulk_ops.__name__:
            continue
        found.add(name)

    manifest_names = {entry[1] for entry in SIBLINGS}
    missing = found - manifest_names
    assert not missing, (
        f"async siblings present in bulk_ops but not in SIBLINGS: {missing}"
    )
