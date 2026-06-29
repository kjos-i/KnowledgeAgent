"""Tests for `search/client.py`'s async siblings + the helper.

Mirrors `tests/unit/kg/test_kg_client_async_siblings.py`. The helper
in search/client.py is a duplicate of kg/client.py's helper (both
vanish at Day 8) — same tests apply on this side.

Two layers of coverage:

1. **Helper behaviour**: `_async_sibling_via_thread(sync_fn)` returns
   an async wrapper that, when awaited, calls `sync_fn` in a worker
   thread and returns its result. Verified with a synthetic class so
   no LanceDB dependency is needed.

2. **Coverage on LanceClient**: every public sync method has a
   matching `a*` sibling, and each sibling is an actual coroutine
   function. The manifest sanity guard catches the case where a
   new sync method is added without a corresponding sibling
   assignment.
"""

import inspect
from unittest.mock import MagicMock, patch

from knowledge_agent.search.client import (
    LanceClient,
    _async_sibling_via_thread,
)


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------


class _Synthetic:
    """Stand-in class so the helper tests don't pull in LanceDB."""

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        return [f"{query}#{i}" for i in range(top_k)]


_Synthetic.aretrieve = _async_sibling_via_thread(_Synthetic.retrieve)


async def test_helper_returns_coroutine_function():
    assert inspect.iscoroutinefunction(_Synthetic.aretrieve)


async def test_helper_forwards_positional_and_keyword_args():
    foo = _Synthetic()
    result = await foo.aretrieve("hello", top_k=3)
    assert result == ["hello#0", "hello#1", "hello#2"]


async def test_helper_propagates_sync_exceptions():
    class _Boom:
        def crash(self):
            raise RuntimeError("nope")

    _Boom.acrash = _async_sibling_via_thread(_Boom.crash)
    boom = _Boom()
    try:
        await boom.acrash()
    except RuntimeError as exc:
        assert str(exc) == "nope"
    else:
        raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------------------
# Coverage: every public sync method has a matching async sibling
# ---------------------------------------------------------------------------


_INTENTIONALLY_SYNC_ONLY: set[str] = {
    # `conn` is a property returning the (currently sync) DBConnection.
    # Day 7 of the async refactor swaps it for an async connection.
    "conn",
}


def _iter_public_sync_methods(cls) -> list[str]:
    """Return the names of public sync methods declared on `cls`."""
    own = vars(cls)
    out = []
    for name, attr in own.items():
        if name.startswith("_"):
            continue
        if name.startswith("a"):
            continue
        if name in _INTENTIONALLY_SYNC_ONLY:
            continue
        if isinstance(attr, property):
            continue
        if not callable(attr):
            continue
        if inspect.iscoroutinefunction(attr):
            continue
        out.append(name)
    return out


def test_every_public_sync_method_has_async_sibling():
    """Every public sync method on LanceClient has a matching `a*` sibling."""
    missing: list[str] = []
    for sync_name in _iter_public_sync_methods(LanceClient):
        async_name = "a" + sync_name
        async_attr = vars(LanceClient).get(async_name)
        if async_attr is None:
            missing.append(sync_name)
            continue
        if not inspect.iscoroutinefunction(async_attr):
            missing.append(
                f"{sync_name} (sibling {async_name} exists but isn't async)"
            )
    assert not missing, (
        "Sync methods on LanceClient without a matching async sibling: "
        f"{missing}"
    )


def test_every_async_sibling_has_matching_sync_method():
    orphans: list[str] = []
    for name, attr in vars(LanceClient).items():
        if not name.startswith("a"):
            continue
        if not inspect.iscoroutinefunction(attr):
            continue
        sync_name = name[1:]
        if sync_name not in vars(LanceClient):
            orphans.append(name)
    assert not orphans, (
        f"Async siblings without a matching sync method: {orphans}"
    )


# ---------------------------------------------------------------------------
# Spot-checks: a few siblings actually delegate through `patch.object`
# ---------------------------------------------------------------------------


async def test_aretrieve_delegates_to_retrieve():
    """End-to-end on the agent's main read entry point."""
    sentinel = MagicMock()
    with patch.object(
        LanceClient, "retrieve", return_value=sentinel
    ) as mock_fn:
        client = LanceClient.__new__(LanceClient)  # skip __init__
        result = await client.aretrieve("query text", top_k=5)
    mock_fn.assert_called_once_with("query text", top_k=5)
    assert result is sentinel


async def test_awrite_chunks_delegates_to_write_chunks():
    sentinel = MagicMock()
    with patch.object(
        LanceClient, "write_chunks", return_value=sentinel
    ) as mock_fn:
        client = LanceClient.__new__(LanceClient)
        result = await client.awrite_chunks([{"chunk_id": "c1"}])
    mock_fn.assert_called_once_with([{"chunk_id": "c1"}])
    assert result is sentinel


async def test_alist_indexed_docs_delegates_to_list_indexed_docs():
    sentinel = [{"doc_id": "d1"}]
    with patch.object(
        LanceClient, "list_indexed_docs", return_value=sentinel
    ) as mock_fn:
        client = LanceClient.__new__(LanceClient)
        result = await client.alist_indexed_docs()
    mock_fn.assert_called_once_with()
    assert result is sentinel
