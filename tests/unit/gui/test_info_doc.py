"""Tests for the shared `InfoDoc` markdown-doc renderer (views/info_doc).

This is the one widget every static-doc surface (Metrics Guide + all Info tabs)
renders through, so the anchor-split / keyed-section / in-page-scroll behaviour is
tested here once.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from knowledge_agent.gui.views.info_doc import InfoDoc, _split_anchored, docs_dir

if TYPE_CHECKING:
    from pathlib import Path


def test_all_shipped_info_docs_exist():
    """Every Info surface's doc ships under knowledge_agent/docs/ (global tab +
    the three contextual sub-tabs)."""
    expected = {
        "info_about.md",
        "info_layout.md",
        "info_getting_started.md",
        "info_storage.md",
        "info_search.md",
        "info_library.md",
        "info_evaluation.md",
    }
    present = {p.name for p in docs_dir().glob("*.md")}
    assert expected <= present, expected - present


def test_split_anchored_strips_tags_and_keys_sections():
    text = 'intro\n<a id="a-one"></a>\n### One\nbody\n<a id="a-two"></a>\n### Two\nb2'
    segs = _split_anchored(text)
    assert segs[0] == (None, "intro\n")  # leading chunk carries no key
    assert segs[1][0] == "a-one"
    assert segs[1][1].startswith("### One")
    assert "<a id=" not in segs[1][1]  # raw anchor tag stripped from rendered text
    assert segs[2][0] == "a-two"


def test_build_missing_file_shows_hint(fake_app: MagicMock, tmp_path: Path):
    doc = InfoDoc(fake_app, tmp_path / "nope.md", missing_hint="no doc here")
    assert doc.build() is not None  # empty-state, not a crash
    assert doc._scroll is None  # never built a scroll body


def test_build_renders_keyed_anchor_sections(fake_app: MagicMock, tmp_path: Path):
    p = tmp_path / "d.md"
    p.write_text('lead\n<a id="sec"></a>\n## Sec\nbody', encoding="utf-8")
    doc = InfoDoc(fake_app, p)
    doc.build()
    assert doc._scroll is not None
    keyed = [c for c in doc._scroll.controls if getattr(c, "key", None) is not None]
    assert keyed  # the anchored section became a keyed scroll target


def test_on_tap_link_scrolls_to_anchor(fake_app: MagicMock, tmp_path: Path):
    p = tmp_path / "d.md"
    p.write_text('lead\n<a id="sec"></a>\n## Sec\nbody', encoding="utf-8")
    doc = InfoDoc(fake_app, p)
    doc.build()
    # scroll_to is a coroutine, so the handler is async and must be awaited.
    doc._scroll = MagicMock()
    doc._scroll.scroll_to = AsyncMock()
    asyncio.run(doc._on_tap_link(SimpleNamespace(data="#sec")))
    doc._scroll.scroll_to.assert_awaited_once()
    assert doc._scroll.scroll_to.await_args.kwargs["scroll_key"] == "sec"


def test_on_tap_link_opens_external_url(fake_app: MagicMock, tmp_path: Path):
    doc = InfoDoc(fake_app, tmp_path / "d.md")
    asyncio.run(doc._on_tap_link(SimpleNamespace(data="https://example.com")))
    fake_app.page.launch_url.assert_called_once_with("https://example.com")
