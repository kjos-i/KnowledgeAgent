"""Metrics Guide sub-tab — renders the packaged info_metrics.md.

The doc is split at its `<a id>` anchors so the raw tags don't render and the
glance-table links can scroll to their sections (Flet markdown has no built-in
in-page anchor jump).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from knowledge_agent.gui.evaluation.metrics_guide_tab import (
    MetricsGuideTab,
    _guide_path,
    _split_anchored,
)


def test_info_metrics_doc_ships_with_harness():
    path = _guide_path()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Evaluation Metrics Reference" in text
    assert "Knowledge Graph" in text  # KA-specific section present


def test_metrics_guide_builds(fake_app: MagicMock):
    assert MetricsGuideTab(fake_app).build() is not None


def test_split_anchored_strips_tags_and_keys_sections():
    text = (
        'intro\n<a id="answer-relevancy"></a>\n### Answer Relevancy\nbody\n'
        '<a id="faithfulness"></a>\n### Faithfulness\nb2'
    )
    segs = _split_anchored(text)
    assert segs[0] == (None, "intro\n")  # leading chunk carries no key
    assert segs[1][0] == "answer-relevancy"
    assert segs[1][1].startswith("### Answer Relevancy")
    assert "<a id=" not in segs[1][1]  # raw anchor tag stripped from rendered text
    assert segs[2][0] == "faithfulness"


def test_doc_anchors_back_every_glance_link():
    """Metrics the glance tables link to have a real <a id> anchor in the doc,
    so a tap has a section to scroll to."""
    text = _guide_path().read_text(encoding="utf-8")
    for anchor in ("answer-relevancy", "hitk", "chunk-citation-grounding", "cypher-validity"):
        assert f'<a id="{anchor}"></a>' in text  # scroll target exists
        assert f"(#{anchor})" in text  # linked from a glance table


def test_on_tap_link_scrolls_to_the_anchor_section(fake_app: MagicMock):
    # scroll_to is a coroutine, so the handler is async and must be awaited.
    tab = MetricsGuideTab(fake_app)
    tab._scroll = MagicMock()
    tab._scroll.scroll_to = AsyncMock()
    asyncio.run(tab._on_tap_link(SimpleNamespace(data="#faithfulness")))
    tab._scroll.scroll_to.assert_awaited_once()
    assert tab._scroll.scroll_to.await_args.kwargs["scroll_key"] == "faithfulness"


def test_build_renders_keyed_anchor_sections(fake_app: MagicMock):
    tab = MetricsGuideTab(fake_app)
    tab.build()
    assert tab._scroll is not None  # scrollable ListView built
    keyed = [c for c in tab._scroll.controls if getattr(c, "key", None) is not None]
    assert keyed  # at least one anchored section became a keyed scroll target
