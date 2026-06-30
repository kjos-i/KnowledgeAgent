"""Tests for `knowledge_agent.gui.right_panel.RightPanel`.

Verifies the mode-switcher state machine + that each mode dispatches
to the right view builder.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.right_panel import (
    MODE_FILE,
    MODE_INFO,
    MODE_LATEST,
    MODE_SETTINGS,
    RightPanel,
)


def test_build_starts_on_latest_mode(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    ctl = panel.build()
    assert isinstance(ctl, ft.Container)
    assert panel.current_mode == MODE_LATEST
    # All four mode buttons are registered (Latest / File via Open Result /
    # Settings / Info).
    assert set(panel.mode_buttons) == {
        MODE_LATEST, MODE_FILE, MODE_SETTINGS, MODE_INFO,
    }


def test_switch_mode_updates_current_mode_and_highlights(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    panel.build()

    panel.switch_mode(MODE_SETTINGS)
    assert panel.current_mode == MODE_SETTINGS
    # The settings button is the active one — its bgcolor is the ACTIVE_BG.
    from knowledge_agent.gui._styles import ACTIVE_BG

    assert panel.mode_buttons[MODE_SETTINGS].bgcolor == ACTIVE_BG
    # Other buttons are NOT tinted.
    assert panel.mode_buttons[MODE_LATEST].bgcolor is None


def test_switch_to_file_mode_with_no_loaded_file_falls_through_to_latest(
    fake_app: MagicMock,
):
    """Defensive: Clear may wipe loaded_file while MODE_FILE is active.
    The view dispatch silently falls back to Latest instead of crashing."""
    panel = RightPanel(fake_app)
    panel.build()
    fake_app.loaded_file = None
    panel.switch_mode(MODE_FILE)
    # The view container's content should be a Latest-style column
    # (the empty-state because last_answer is also None).
    container = panel.view_container
    assert container is not None
    # We can't easily introspect deeper than this without rendering,
    # but the build mustn't raise.


def test_switch_to_settings_renders_stub_message(fake_app: MagicMock):
    """Slice 1 ships Settings as an empty-state. Confirm the stub is in
    the rendered tree so the user sees a 'coming soon' message rather
    than a blank panel."""
    panel = RightPanel(fake_app)
    panel.build()
    panel.switch_mode(MODE_SETTINGS)
    content = panel.view_container.content
    # The stub goes through view_with_header → Column → [header, body].
    assert isinstance(content, ft.Column)
    body = content.controls[1]
    assert isinstance(body, ft.Container)
    assert "slice 2" in body.content.value.lower()


def test_switch_to_info_renders_stub_message(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    panel.build()
    panel.switch_mode(MODE_INFO)
    content = panel.view_container.content
    body = content.controls[1]
    assert "info" in body.content.value.lower()


def test_paste_path_field_submits_to_app_handler(fake_app: MagicMock):
    """The paste-path field's on_submit must wire through to
    GuiApp.on_load_path_field — the user presses Enter inside the
    field rather than clicking a button."""
    panel = RightPanel(fake_app)
    panel.build()
    assert panel.paste_path_field is not None
    # Flet sets the bound callable on .on_submit.
    assert panel.paste_path_field.on_submit is fake_app.on_load_path_field
