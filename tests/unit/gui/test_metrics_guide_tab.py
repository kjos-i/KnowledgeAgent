"""Metrics Guide sub-tab — renders the packaged info_metrics.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.evaluation.metrics_guide_tab import MetricsGuideTab, _guide_path

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def test_info_metrics_doc_ships_with_harness():
    path = _guide_path()
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Evaluation Metrics Reference" in text
    assert "Knowledge Graph" in text  # KA-specific section present


def test_metrics_guide_builds(fake_app: MagicMock):
    assert MetricsGuideTab(fake_app).build() is not None
