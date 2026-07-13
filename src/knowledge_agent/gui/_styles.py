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

PANEL_BG_RAISED = ft.Colors.GREY_900
"""A slightly-lighter dark surface for a raised/nested panel — e.g. the left rail
of a two-column screen — so it lifts off the black page background. One step up
from PANEL_BG (black); the Evaluation tabs use it for their left rails."""

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

PANEL_RADIUS = 8
"""Corner radius for column-level panel frames — the left/right columns of
the two-column Library + Evaluation screens, matching the Search panels.
Inner sub-boxes (config groups, case cards) stay tighter (radius 4) so the
nesting reads as hierarchy rather than a doubled border."""


LEFT_COLUMN_WIDTH = 500
"""Fixed width for the left column of the Select + Evaluation (Run / Dataset)
two-column screens; the right column expands to fill the rest. One source so
those tabs line up at the same left-column width."""


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


def panel_box(
    content: ft.Control,
    *,
    expand: int | None = 1,
    width: int | None = None,
    padding: int = 12,
) -> ft.Container:
    """Wrap a column's content in the app-standard framed panel: thin grey
    border, dark fill, rounded corners.

    Used for the left/right columns of every two-column screen in Library +
    Evaluation so they separate the same way — one source for the frame,
    like `labeled_field` is for form rows. Pass `width` for a fixed-width
    column (then `expand` is dropped); otherwise the box flexes by `expand`.
    """
    return ft.Container(
        content=content,
        expand=None if width is not None else expand,
        width=width,
        padding=padding,
        bgcolor=PANEL_BG,
        border=ft.Border.all(1, FRAME_BORDER_COLOR),
        border_radius=PANEL_RADIUS,
    )


PANEL_TITLE_SIZE = 16
"""Size of a panel/column title — sits at the top of a `panel_box`, above a
thin rule (`ft.Divider(height=1)`), leading the panel's sections."""

SECTION_TITLE_SIZE = 14
"""Size of a section title within a panel — bold so it outranks the 14px
form-field captions below it; consecutive sections split by `section_divider`."""

SECTION_DIVIDER_THICKNESS = 2
"""Line thickness for a between-section divider — heavier than Flet's 1px
default so section breaks read clearly."""

SECTION_DIVIDER_COLOR = ft.Colors.GREY_500
"""Between-section divider colour — brighter than the dim default (and the
GREY_700 frames), so the break between sections stands out."""


def panel_title(text: str) -> ft.Text:
    """The title at the top of a `panel_box` / column — 16, bold."""
    return ft.Text(text, size=PANEL_TITLE_SIZE, weight=ft.FontWeight.BOLD)


def section_title(text: str) -> ft.Text:
    """A section heading within a panel — 14 bold. Sits above its content;
    consecutive sections are separated by `section_divider()`."""
    return ft.Text(text, size=SECTION_TITLE_SIZE, weight=ft.FontWeight.BOLD)


# The four Evaluation dashboard view tabs share one section-header style so they
# read identically: NON-bold at 16 (the tab text ceiling) with top breathing
# room. Deliberately scoped to the dashboard — the app-wide `section_title`
# stays bold. Separate sections with `section_divider()`.
DASHBOARD_HEADER_SIZE = 16


def dashboard_section_header(text: str) -> ft.Container:
    """A dashboard body section header (Run Summary / Deep Analysis / Trends /
    Metrics Guide) — non-bold, size `DASHBOARD_HEADER_SIZE`, with top padding so
    it doesn't crowd the section above."""
    return ft.Container(
        padding=ft.Padding.only(top=10),
        content=ft.Text(text, size=DASHBOARD_HEADER_SIZE),
    )


def section_divider() -> ft.Divider:
    """The separator *between* a panel's sections — thicker (2) + brighter
    (GREY_500) than Flet's dim 1px default, so section breaks stand out. For
    a subtle rule *within* a section, use a plain `ft.Divider(height=1)`.
    Fresh instance per call — a control can't be shared across the tree.
    """
    return ft.Divider(
        thickness=SECTION_DIVIDER_THICKNESS,
        color=SECTION_DIVIDER_COLOR,
    )


SUB_SECTION_TITLE_SIZE = 14
"""Size of a sub-section title — a label *within* a section: same size as the
14 form-field captions but non-bold and a brighter GREY_200, so it reads as a
heading above them (and stays lighter than the 14-bold `section_title`).
Consecutive sub-sections split by a `thin_rule()`."""


def sub_section_title(text: str, *, bold: bool = False) -> ft.Text:
    """A sub-section label within a section — 14, non-bold, GREY_200. Brighter
    than the GREY_300 field captions it heads, lighter (non-bold) than the
    14-bold `section_title`; separate consecutive sub-sections with a
    `thin_rule()`. Pass `bold=True` where a section wants its sub-headings to
    stand out (an opt-in exception, off everywhere by default)."""
    return ft.Text(
        text,
        size=SUB_SECTION_TITLE_SIZE,
        color=ft.Colors.GREY_200,
        weight=ft.FontWeight.BOLD if bold else None,
    )


def thin_rule() -> ft.Divider:
    """A subtle rule *within* a section — 1px, GREY_600. Brighter than Flet's
    near-invisible default, but clearly below the `section_divider` (2/GREY_500),
    so the line hierarchy reads: section break › sub-section rule › none.
    Used between the sub-sections of a section (fresh instance per call)."""
    return ft.Divider(height=1, thickness=1, color=ft.Colors.GREY_600)


def sub_section_header(
    text: str, *, trailing: ft.Control | None = None, bold: bool = False
) -> ft.Container:
    """A sub-section header — a `thin_rule()` then a `sub_section_title()`,
    wrapped with vertical breathing room so the sub-section isn't cramped
    against the content above it (or the next sub-section). Use in place of a
    bare `thin_rule()` + `sub_section_title()` pair. Pass `trailing` (e.g. an
    `info_icon`) to sit it next to the title; `bold=True` for a heavier title."""
    title: ft.Control = sub_section_title(text, bold=bold)
    if trailing is not None:
        title = ft.Row(
            [title, trailing],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
    return ft.Container(
        padding=ft.Padding.only(top=16, bottom=6),
        content=ft.Column(
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[thin_rule(), title],
        ),
    )


def panel_title_rule() -> ft.Divider:
    """The rule under a panel/column title — slightly thicker (3px) but dimmer
    (GREY_600) than a `section_divider` (2px / GREY_500), so a panel header
    reads as a header underline rather than another section break."""
    return ft.Divider(thickness=3, color=ft.Colors.GREY_600)
