"""Tests for the reusable `(i)` info-icon widget + the GuiApp registry
that powers the per-tier show/hide toggles.

Pure control construction — no Flet render, no dialog clicks (those need
a live page). We check: each tier icon reflects its toggle, registers
itself under its tier, and the app registry flips one tier at a time.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui._widgets.info_icon import info_icon


def test_info_icon_standard_only_returns_one_icon(fake_app):
    fake_app.gui_config.show_info_icons = True
    registered: list[tuple[str, ft.IconButton]] = []
    fake_app.register_info_icon = lambda btn, tier="standard": registered.append((tier, btn))
    row = info_icon(fake_app, title="Router", text="what the router does")
    assert isinstance(row, ft.Row)
    assert len(row.controls) == 1  # only the standard tier has text
    assert row.controls[0].visible is True
    assert registered == [("standard", row.controls[0])]


def test_info_icon_hidden_when_toggle_off(fake_app):
    fake_app.gui_config.show_info_icons = False
    fake_app.register_info_icon = MagicMock()
    row = info_icon(fake_app, title="T", text="body")
    assert row.controls[0].visible is False


def test_info_icon_three_tiers_side_by_side(fake_app):
    """All three tier texts → three icons next to each other, each gated by
    its own flag."""
    fake_app.gui_config.show_info_icons = True
    fake_app.gui_config.show_beginner_info = False
    fake_app.gui_config.show_technical_info = True
    tiers: list[str] = []
    fake_app.register_info_icon = lambda btn, tier="standard": tiers.append(tier)
    row = info_icon(fake_app, title="X", text="std", beginner="beg", technical="tech")
    assert len(row.controls) == 3
    assert tiers == ["standard", "beginner", "technical"]  # standard first, in order
    visibility = {t: c.visible for t, c in zip(tiers, row.controls, strict=True)}
    assert visibility == {"standard": True, "beginner": False, "technical": True}


def test_info_icon_omits_tiers_without_text(fake_app):
    """A point with only technical text renders just the one technical icon."""
    fake_app.gui_config.show_technical_info = True
    fake_app.register_info_icon = MagicMock()
    row = info_icon(fake_app, title="X", technical="tech only")
    assert len(row.controls) == 1


def test_app_registry_flips_one_tier_live():
    from knowledge_agent.gui.app import GuiApp

    app = GuiApp(page=MagicMock())
    std1 = ft.IconButton(visible=True)
    std2 = ft.IconButton(visible=True)
    beg = ft.IconButton(visible=True)
    app.register_info_icon(std1, "standard")
    app.register_info_icon(std2, "standard")
    app.register_info_icon(beg, "beginner")

    app.set_info_icons_visible("standard", False)
    assert std1.visible is False and std2.visible is False
    assert beg.visible is True  # a different tier is untouched

    app.set_info_icons_visible("standard", True)
    assert std1.visible is True and std2.visible is True


def test_section_header_forwards_beginner_tier(fake_app):
    """section_header passes a beginner note through to its info icon, so a
    section header can show the green tier, not just standard + technical."""
    from knowledge_agent.gui._widgets.info_icon import section_header

    fake_app.gui_config.show_info_icons = True
    fake_app.gui_config.show_beginner_info = True
    fake_app.gui_config.show_technical_info = True
    tiers: list[str] = []
    fake_app.register_info_icon = lambda btn, tier="standard": tiers.append(tier)
    row = section_header(fake_app, "Mode", "std", beginner="beg", technical="tech")
    assert isinstance(row, ft.Row)
    # [section_title, info_icon_row]; the icon row carries all three tiers.
    assert len(row.controls) == 2
    assert tiers == ["standard", "beginner", "technical"]


def test_section_header_key_pulls_from_registry(fake_app):
    """section_header(key=...) sources its (i) text from the info_text registry
    (the single home for icon text) instead of inline strings, so the header
    still renders title + the icon row with the entry's tiers."""
    from knowledge_agent.gui._widgets.info_icon import section_header

    fake_app.gui_config.show_info_icons = True
    fake_app.gui_config.show_beginner_info = True
    fake_app.gui_config.show_technical_info = True
    tiers: list[str] = []
    fake_app.register_info_icon = lambda btn, tier="standard": tiers.append(tier)
    # A real registry key with all three tiers filled.
    row = section_header(fake_app, "LLM providers", key="installs.llm_providers")
    assert isinstance(row, ft.Row)
    assert len(row.controls) == 2  # [section_title, info row from the registry]
    assert tiers == ["standard", "beginner", "technical"]
