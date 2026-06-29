"""Tests for `kg/client.py`'s async siblings + the helper that builds them.

Two layers of coverage:

1. **Helper behaviour**: `_async_sibling_via_thread(sync_fn)` returns an
   async wrapper that, when awaited, calls `sync_fn` in a worker thread
   and returns its result. Verified with a synthetic class so no Neo4j
   dependency is needed.

2. **Coverage on Neo4jClient**: every public sync method has a matching
   `a*` sibling, and each sibling is an actual coroutine function.
   The manifest sanity guard catches the case where a new sync method
   is added to Neo4jClient without a corresponding sibling assignment.

Per the no-test-driven-quality-compromise rule, this file tests the
HELPER and the CONTRACT (every public method has a sibling). Per-sibling
delegation tests are skipped because they would test stdlib + helper
behaviour we've already covered — testing 75 nearly-identical thin
wrappers is noise.

The siblings get richer per-call tests once Day 7 rewrites them with
native async (drops the to_thread wrap) and once Day 8 renames them.
For now, helper + coverage = sufficient.
"""

import inspect
from unittest.mock import MagicMock, patch

from knowledge_agent.kg.client import (
    Neo4jClient,
    _async_sibling_via_thread,
)


# ---------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------


class _Synthetic:
    """Stand-in class so the helper tests don't pull in Neo4j."""

    def write(self, x: int, y: int = 10) -> int:
        return x + y


_Synthetic.awrite = _async_sibling_via_thread(_Synthetic.write)


async def test_helper_returns_coroutine_function():
    """The helper's output is itself a coroutine function."""
    assert inspect.iscoroutinefunction(_Synthetic.awrite)


async def test_helper_forwards_positional_args_and_returns_result():
    foo = _Synthetic()
    result = await foo.awrite(5)
    assert result == 15


async def test_helper_forwards_keyword_args():
    foo = _Synthetic()
    result = await foo.awrite(5, y=20)
    assert result == 25


async def test_helper_propagates_sync_exceptions():
    """If the sync method raises, the async sibling raises the same exception."""
    class _Boom:
        def crash(self):
            raise ValueError("nope")

    _Boom.acrash = _async_sibling_via_thread(_Boom.crash)
    boom = _Boom()
    try:
        await boom.acrash()
    except ValueError as exc:
        assert str(exc) == "nope"
    else:
        raise AssertionError("expected ValueError")


async def test_helper_names_the_sibling_with_a_prefix():
    """Generated wrapper takes the `a` + sync name."""
    assert _Synthetic.awrite.__name__ == "awrite"
    assert "write" in (_Synthetic.awrite.__doc__ or "")


# ---------------------------------------------------------------------------
# Coverage: every public sync method has a matching async sibling
# ---------------------------------------------------------------------------


# Methods that intentionally do NOT need an async sibling. Keep this
# list small + justified — every entry is documented in-place.
_INTENTIONALLY_SYNC_ONLY: set[str] = {
    # `driver` is a property returning the (currently sync) Driver
    # instance. Day 7 of the async refactor swaps it for AsyncDriver.
    "driver",
}


def _iter_public_sync_methods(cls) -> list[str]:
    """Return the names of public sync methods declared on `cls`.

    Excludes:
      - `_` prefixed (private)
      - `a` prefixed (already an async sibling — by construction)
      - entries in `_INTENTIONALLY_SYNC_ONLY`
      - properties (descriptors don't fit the pattern)
      - inherited-from-object methods
    """
    own = vars(cls)
    out = []
    for name, attr in own.items():
        if name.startswith("_"):
            continue
        if name.startswith("a"):
            # Already a sibling (or an `async_` method) — skip.
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
    """Every public sync method on Neo4jClient has a matching `a*` sibling.

    Catches the case where a new sync method is added without a
    matching `_async_sibling_via_thread(...)` assignment.
    """
    missing: list[str] = []
    for sync_name in _iter_public_sync_methods(Neo4jClient):
        async_name = "a" + sync_name
        async_attr = vars(Neo4jClient).get(async_name)
        if async_attr is None:
            missing.append(sync_name)
            continue
        if not inspect.iscoroutinefunction(async_attr):
            missing.append(
                f"{sync_name} (sibling {async_name} exists but isn't async)"
            )
    assert not missing, (
        "Sync methods on Neo4jClient without a matching async sibling: "
        f"{missing}"
    )


def test_every_async_sibling_has_matching_sync_method():
    """Reverse direction: no orphan `a*` sibling pointing at a missing sync method."""
    orphans: list[str] = []
    for name, attr in vars(Neo4jClient).items():
        if not name.startswith("a"):
            continue
        if not inspect.iscoroutinefunction(attr):
            continue
        sync_name = name[1:]
        if sync_name not in vars(Neo4jClient):
            orphans.append(name)
    assert not orphans, (
        f"Async siblings without a matching sync method: {orphans}"
    )


# ---------------------------------------------------------------------------
# Spot-check: a handful of named siblings actually delegate
# ---------------------------------------------------------------------------


async def test_awrite_chunks_delegates_to_write_chunks():
    """Concrete end-to-end on the most-used write sibling.

    `patch.object(Neo4jClient, "write_chunks", ...)` replaces the
    attribute on the class with a MagicMock. MagicMock is not a
    descriptor, so `instance.write_chunks` returns the mock directly
    (no implicit `self` binding). The sibling calls it via
    `getattr(self, name)`, which therefore receives only the user-
    supplied args — `self` is not forwarded into the mock's call.
    """
    sentinel = MagicMock()
    with patch.object(
        Neo4jClient, "write_chunks", return_value=sentinel
    ) as mock_fn:
        client = Neo4jClient.__new__(Neo4jClient)  # skip __init__
        result = await client.awrite_chunks(
            "doc-1", [], "Document", "Paper",
        )
    mock_fn.assert_called_once_with("doc-1", [], "Document", "Paper")
    assert result is sentinel


async def test_aread_query_delegates_to_read_query():
    """Same shape on the read path."""
    sentinel = [{"row": 1}]
    with patch.object(
        Neo4jClient, "read_query", return_value=sentinel
    ) as mock_fn:
        client = Neo4jClient.__new__(Neo4jClient)
        result = await client.aread_query("MATCH (n) RETURN n", limit=10)
    mock_fn.assert_called_once_with("MATCH (n) RETURN n", limit=10)
    assert result is sentinel
