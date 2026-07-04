"""Tests for the reusable `(i)` info-icon widget + the GuiApp registry
that powers the global show/hide toggle.

Pure control construction — no Flet render, no dialog clicks (those need
a live page). We check: the button reflects the toggle, registers itself,
and the app registry flips every registered icon live.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui._widgets.info_icon import info_icon


def test_info_icon_returns_button_visible_when_toggle_on(fake_app):
    fake_app.gui_config.show_info_icons = True
    registered: list[ft.IconButton] = []
    fake_app.register_info_icon = registered.append
    btn = info_icon(fake_app, title="Router", text="what the router does")
    assert isinstance(btn, ft.IconButton)
    assert btn.visible is True
    assert btn in registered  # registered so the global toggle can flip it


def test_info_icon_hidden_when_toggle_off(fake_app):
    fake_app.gui_config.show_info_icons = False
    fake_app.register_info_icon = MagicMock()
    btn = info_icon(fake_app, title="T", text="body")
    assert btn.visible is False


def test_app_registry_flips_all_icons_live():
    from knowledge_agent.gui.app import GuiApp

    app = GuiApp(page=MagicMock())
    b1 = ft.IconButton(visible=True)
    b2 = ft.IconButton(visible=True)
    app.register_info_icon(b1)
    app.register_info_icon(b2)

    app.set_info_icons_visible(False)
    assert b1.visible is False
    assert b2.visible is False

    app.set_info_icons_visible(True)
    assert b1.visible is True
    assert b2.visible is True
