"""Unit tests for the shared GUI style helpers in `gui/_styles`."""

import flet as ft

from knowledge_agent.gui._styles import (
    FIELD_LABEL_GAP,
    FIELD_LABEL_SIZE,
    labeled_field,
)


def test_labeled_field_caption_uses_shared_size() -> None:
    """Caption renders at FIELD_LABEL_SIZE (matches the Checkbox label size)."""
    row = labeled_field("Chunker strategy", ft.Dropdown())
    caption = row.controls[0]
    assert isinstance(caption, ft.Text)
    assert caption.value == "Chunker strategy:"
    assert caption.size == FIELD_LABEL_SIZE


def test_labeled_field_hugs_text_by_default() -> None:
    """No `label_width` given -> caption hugs its text (width None) and the
    gap between caption and input is the tight ~3-char spacing."""
    row = labeled_field("Chunk max tokens", ft.TextField())
    assert row.controls[0].width is None
    assert row.spacing == FIELD_LABEL_GAP


def test_labeled_field_pins_column_when_width_given() -> None:
    """`label_width` pins the caption to a fixed column for aligned groups."""
    row = labeled_field("Images scale", ft.TextField(), label_width=140)
    assert row.controls[0].width == 140


def test_labeled_field_matches_input_text_size_to_caption() -> None:
    """Text inside the field is bumped to the caption size so they read as
    one line (the user's "text inside the field == text in front" ask)."""
    field = ft.TextField()
    assert field.text_size is None
    labeled_field("Chunk max tokens", field)
    assert field.text_size == FIELD_LABEL_SIZE

    dropdown = ft.Dropdown()
    labeled_field("Chunker strategy", dropdown)
    assert dropdown.text_size == FIELD_LABEL_SIZE


def test_labeled_field_respects_explicit_input_size() -> None:
    """An input that set its own `text_size` keeps it — the helper only fills
    in the default, it doesn't override a deliberate per-field size."""
    field = ft.TextField(text_size=20)
    labeled_field("Chunk max tokens", field)
    assert field.text_size == 20


def test_labeled_field_applies_standard_border_when_unset() -> None:
    """A bare TextField/Dropdown gets the app-standard gray outline box so
    Eval-tab fields match every other tab (they were built borderless)."""
    from knowledge_agent.gui._styles import FRAME_BORDER_COLOR, PANEL_BG

    tf = ft.TextField()
    labeled_field("x", tf)
    assert tf.border == ft.InputBorder.OUTLINE
    assert tf.border_color == FRAME_BORDER_COLOR
    assert tf.bgcolor == PANEL_BG

    dd = ft.Dropdown()
    labeled_field("y", dd)
    assert dd.border == ft.InputBorder.OUTLINE
    assert dd.border_color == FRAME_BORDER_COLOR


def test_labeled_field_keeps_explicit_border() -> None:
    """A field that set its own border colour keeps it (not overridden)."""
    tf = ft.TextField(border_color=ft.Colors.RED)
    labeled_field("x", tf)
    assert tf.border_color == ft.Colors.RED


def test_labeled_field_input_fills_remaining_width() -> None:
    """The input sits in an expanding container so it fills the row."""
    field = ft.TextField()
    row = labeled_field("Chunk max tokens", field)
    container = row.controls[1]
    assert isinstance(container, ft.Container)
    assert container.expand is True
    assert container.content is field
