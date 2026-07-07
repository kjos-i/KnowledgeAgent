"""Shared visual constants used across the GUI panels and views.

Centralised so the panels + views don't drift on background / border /
highlight colours — page-wide concerns belong in one place.

Mirrors the ResearchArticlesAgent style module so the sibling apps
visually match. Bumping a value here moves both apps' look together if
we ever re-skin.
"""

import flet as ft

PANEL_BG = ft.Colors.BLACK
"""Panel background. Matches the dark Material 3 surface tone the rest
of the app expects."""

FRAME_BORDER_COLOR = ft.Colors.GREY_700
"""Border colour for panel frames, input fields, and dividers."""

ACTIVE_BG = ft.Colors.BLUE_GREY_700
"""Background tint applied to the currently-active mode button so the
view-switcher reads at a glance."""

CHAT_EMPTY_PLACEHOLDER = "Chat output will appear here."
"""Italic placeholder shown in the chat panel before the first message
arrives. Replaced wholesale once the user sends or the agent answers."""


def centered_label(text: str) -> ft.Text:
    """Button-content label that centres on both lines when it wraps.

    Flet's `Button(content="...")` wraps the string in a Text with the
    Flutter default `text_align=START`, so multi-line labels like
    "Open Result" stack left-aligned in a narrow button. Passing an
    explicit `ft.Text(text_align=CENTER)` as content keeps both lines
    centred regardless of window width.
    """
    return ft.Text(text, text_align=ft.TextAlign.CENTER)


FIELD_LABEL_SIZE = 14
"""Caption + input text size for `labeled_field` rows. Matches the Flet
default `Checkbox` label size, so form-field rows and the checkboxes that
sit beside them in a panel read at one consistent size."""

FIELD_LABEL_GAP = 18
"""Spacing (~3 characters) between a `labeled_field` caption and its input.
Small on purpose — the caption hugs its text rather than reserving a wide
column, so the value sits close to its key."""


def labeled_field(
    label: str,
    control: ft.Control,
    *,
    label_width: int | None = None,
    gap: int = FIELD_LABEL_GAP,
    trailing: ft.Control | None = None,
) -> ft.Row:
    """The app-wide `Label: [input]` form-field row.

    The caption sits left (hugging its text by default), a ~3-character gap,
    then the input filling the remaining width. The input's own text is
    bumped to `FIELD_LABEL_SIZE` too, so the caption and the value read as
    one line at one size. Give the control NO floating `label=` of its own;
    this caption replaces it.

    `label_width` pins the caption to a fixed column when a group needs its
    inputs to line up down a shared edge (pass the widest label's width);
    leave it `None` to hug the text and keep the gap tight.

    `trailing` tacks a control (e.g. a `Browse` button) onto the end of the
    row, after the input — for picker rows like `Folder: [path] [Browse]`.
    """
    # Match the input's text size to the caption (single source: FIELD_LABEL_SIZE).
    # Only touch controls that expose `text_size` and haven't set it themselves,
    # so an explicit per-field size still wins and non-text controls are skipped.
    if getattr(control, "text_size", "missing") is None:
        control.text_size = FIELD_LABEL_SIZE
    # Give the input the app-standard outlined gray box if it hasn't set its
    # own, so form fields look identical across every tab. A bare Flet
    # TextField defaults to an OUTLINE border with NO colour (invisible until
    # focus) and Dropdown to no border at all — which is why the Eval-tab
    # fields looked borderless. Only TextField/Dropdown are touched, and each
    # property is filled only when unset, so fields that styled themselves keep
    # their look.
    if isinstance(control, ft.TextField | ft.Dropdown):
        if control.border is None:
            control.border = ft.InputBorder.OUTLINE
        if control.border_color is None:
            control.border_color = FRAME_BORDER_COLOR
        if control.bgcolor is None:
            control.bgcolor = PANEL_BG
    # Multi-line inputs read better with the caption pinned to the TOP of the
    # box; single-line inputs centre the caption against the field.
    multiline = bool(getattr(control, "multiline", False))
    valign = ft.CrossAxisAlignment.START if multiline else ft.CrossAxisAlignment.CENTER
    caption: ft.Control = ft.Text(
        f"{label}:",
        size=FIELD_LABEL_SIZE,
        color=ft.Colors.GREY_300,
        width=label_width,
    )
    if multiline:
        # Nudge the caption down so it sits on the box's first text line
        # instead of hugging the very top border.
        caption = ft.Container(content=caption, padding=ft.Padding.only(top=8))
    controls: list[ft.Control] = [
        caption,
        ft.Container(content=control, expand=True),
    ]
    if trailing is not None:
        controls.append(trailing)
    return ft.Row(
        controls=controls,
        vertical_alignment=valign,
        spacing=gap,
    )
