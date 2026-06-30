"""Tests for `knowledge_agent.gui.chat_panel.ChatPanel`.

Static construction + state-transition tests. No real Flet rendering.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.chat_panel import ChatPanel


def test_build_returns_container_with_chat_output_and_inputs(fake_app: MagicMock):
    panel = ChatPanel(fake_app)
    ctl = panel.build()
    assert isinstance(ctl, ft.Container)
    # build() populates the late-bound attributes.
    assert panel.chat_output is not None
    assert panel.input_field is not None
    assert panel.send_button is not None
    assert panel.stop_button is not None
    assert panel.progress_ring is not None


def test_set_busy_toggles_send_stop_input_visibility(fake_app: MagicMock):
    panel = ChatPanel(fake_app)
    panel.build()

    panel.set_busy(True)
    assert panel.send_button.disabled is True
    assert panel.stop_button.disabled is False
    assert panel.input_field.disabled is True
    assert panel.progress_ring.visible is True

    panel.set_busy(False)
    assert panel.send_button.disabled is False
    assert panel.stop_button.disabled is True
    assert panel.input_field.disabled is False
    assert panel.progress_ring.visible is False


def test_append_user_clears_placeholder_on_first_message(fake_app: MagicMock):
    panel = ChatPanel(fake_app)
    panel.build()
    # Initially holds the empty-state placeholder.
    assert panel.chat_is_empty is True
    assert len(panel.chat_output.controls) == 1

    panel.append_user("first msg")
    # Placeholder is wiped + the new message appended.
    assert panel.chat_is_empty is False
    assert len(panel.chat_output.controls) == 1
    # Subsequent appends accumulate.
    panel.append_assistant("reply")
    assert len(panel.chat_output.controls) == 2


def test_reset_restores_placeholder(fake_app: MagicMock):
    """Clear button calls reset → empty state placeholder back."""
    panel = ChatPanel(fake_app)
    panel.build()
    panel.append_user("hi")
    panel.append_assistant("hello")
    panel.reset()
    assert panel.chat_is_empty is True
    assert len(panel.chat_output.controls) == 1


def test_get_input_text_strips_whitespace(fake_app: MagicMock):
    panel = ChatPanel(fake_app)
    panel.build()
    panel.input_field.value = "   what is X?   \n"
    assert panel.get_input_text() == "what is X?"


def test_clear_input_blanks_field(fake_app: MagicMock):
    panel = ChatPanel(fake_app)
    panel.build()
    panel.input_field.value = "lingering"
    panel.clear_input()
    assert panel.input_field.value == ""


def test_begin_assistant_stream_returns_updatable_body_text(
    fake_app: MagicMock,
):
    """The streaming pattern: begin_assistant_stream returns a Text,
    then update_assistant_stream overwrites its value as tokens arrive."""
    panel = ChatPanel(fake_app)
    panel.build()
    body = panel.begin_assistant_stream()
    assert isinstance(body, ft.Text)
    panel.update_assistant_stream(body, "partial answer")
    assert body.value == "partial answer"
    panel.update_assistant_stream(body, "partial answer plus more")
    assert body.value == "partial answer plus more"


def test_pop_last_removes_latest_message(fake_app: MagicMock):
    """Used to roll back a chat-router failure that left a user
    message in the column but produced no assistant reply."""
    panel = ChatPanel(fake_app)
    panel.build()
    panel.append_user("oops")
    assert len(panel.chat_output.controls) == 1
    panel.pop_last()
    assert panel.chat_output.controls == []
