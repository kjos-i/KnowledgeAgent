"""Tests for the app-wide Info tab (tabs/info_tab) — a 4 sub-tab strip, one .md
per section, rendered through the shared InfoDoc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.tabs.info_tab import _SECTIONS, InfoTab
from knowledge_agent.gui.views.info_doc import docs_dir

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def test_info_docs_ship():
    docs = docs_dir()
    for _label, filename in _SECTIONS:
        assert (docs / filename).exists(), filename


def test_info_tab_builds_all_subtabs(fake_app: MagicMock):
    ctl = InfoTab(fake_app).build()
    assert isinstance(ctl, ft.Tabs)
    # 4 shipped-doc sections + the generated Dependencies tab.
    assert ctl.length == len(_SECTIONS) + 1 == 5
    sub_bar = ctl.content.controls[0]
    labels = [t.label for t in sub_bar.tabs]
    assert labels == [
        "About",
        "Where things are",
        "Getting Started",
        "Files & storage",
        "Dependencies",
    ]


def test_dependencies_markdown_lists_packages_or_hints():
    from knowledge_agent.gui.tabs._dependencies import dependencies_markdown

    md = dependencies_markdown()
    assert md.startswith("# Dependencies")
    # Installed (the normal case) → the real deps are listed; otherwise the
    # fallback hint explains why not.
    assert ("flet" in md.lower()) or ("install the app" in md.lower())
