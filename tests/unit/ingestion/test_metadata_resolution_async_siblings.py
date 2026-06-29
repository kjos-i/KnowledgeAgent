"""Tests for the async siblings in `metadata_resolution.py`.

Same shape as the bulk_ops / pipeline async-sibling tests: parametrize
across (sync_name, async_name, args, kwargs) and verify each wrapper
delegates correctly. Two siblings: `alookup_known_doi` and
`aresolve_openalex`.
"""

from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.ingestion import metadata_resolution


SIBLINGS: list[tuple[str, str, tuple, dict]] = [
    (
        "lookup_known_doi",
        "alookup_known_doi",
        ("doc-1", "10.1234/abc"),
        {},
    ),
    (
        "resolve_openalex",
        "aresolve_openalex",
        ("doc-2",),
        {"skip_manual": True},
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
    """Async sibling forwards args, awaits sync, returns its result."""
    sentinel = MagicMock()
    with patch.object(
        metadata_resolution, sync_name, return_value=sentinel
    ) as mock_fn:
        async_fn = getattr(metadata_resolution, async_name)
        result = await async_fn(*args, **kwargs)

    mock_fn.assert_called_once_with(*args, **kwargs)
    assert result is sentinel


def test_sibling_manifest_covers_all_async_siblings():
    """Sanity guard: every `a*` callable in metadata_resolution is in SIBLINGS."""
    import inspect

    found: set[str] = set()
    for name in dir(metadata_resolution):
        if not name.startswith("a"):
            continue
        obj = getattr(metadata_resolution, name)
        if not inspect.iscoroutinefunction(obj):
            continue
        if (
            getattr(obj, "__module__", "")
            != metadata_resolution.__name__
        ):
            continue
        found.add(name)

    manifest_names = {entry[1] for entry in SIBLINGS}
    missing = found - manifest_names
    assert not missing, (
        f"async siblings present in metadata_resolution but not in "
        f"SIBLINGS: {missing}"
    )
