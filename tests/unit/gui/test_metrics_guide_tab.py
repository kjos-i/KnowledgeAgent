"""Metrics Guide sub-tab — renders the packaged info_metrics.md.

The rendering itself (anchor split, keyed sections, in-page scroll) is the shared
`InfoDoc` and is tested in test_info_doc. Here we cover the doc content shipping,
that the tab builds and delegates to InfoDoc, and that every glance-table link has
a real anchor to scroll to.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from knowledge_agent.gui.evaluation.metrics_guide_tab import MetricsGuideTab, _guide_path


def test_info_metrics_doc_ships_with_harness():
    path = _guide_path()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Evaluation Metrics Reference" in text
    assert "Knowledge Graph" in text  # KA-specific section present


def test_metrics_guide_builds_and_delegates_to_infodoc(fake_app: MagicMock):
    tab = MetricsGuideTab(fake_app, coordinator=MagicMock())
    assert tab.build() is not None
    # The body is the shared InfoDoc, whose scroll body holds the keyed sections.
    assert tab.doc is not None
    assert tab.doc._scroll is not None
    keyed = [c for c in tab.doc._scroll.controls if getattr(c, "key", None) is not None]
    assert keyed  # anchored sections became keyed scroll targets


def test_doc_anchors_back_every_glance_link():
    """Metrics the glance tables link to have a real <a id> anchor in the doc,
    so a tap has a section to scroll to."""
    text = _guide_path().read_text(encoding="utf-8")
    for anchor in ("answer-relevancy", "hitk", "chunk-citation-grounding", "cypher-validity"):
        assert f'<a id="{anchor}"></a>' in text  # scroll target exists
        assert f"(#{anchor})" in text  # linked from a glance table
