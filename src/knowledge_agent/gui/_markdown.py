"""Shared markdown theme + renderers for every in-app `ft.Markdown` site.

`_MD_STYLE` is matched to VS Code's own preview stylesheet
(markdown-language-features/media/markdown.css): body line-height ≈ 1.57,
semibold (600) headings in a real size hierarchy, soft-white text (not stark
#fff), a bordered code panel, and a left-bar blockquote. Flet ships almost no
markdown styling by default, so every element is set explicitly.

Two entry points, sharing the one stylesheet:
  * `themed_markdown(text)` — a plain themed `ft.Markdown` widget. Use for
    answers / opened files (the common prose case). Tables render via Flet's
    built-in grid.
  * `render_markdown(text)` — also pulls GFM tables OUT of the markdown flow and
    renders them as native controls (content-sized first column, wrapping body,
    horizontal rules only), because Flet's `ft.Markdown` draws tables as a full
    equal-width grid with no knob to change either the border or the column
    widths. Use where tables matter (the Metrics Guide).

Two things Flet structurally can't match: a border UNDER h1/h2 (no per-heading
decoration; VS Code draws one) and VS Code's exact TextMate syntax colours (we
use the closest highlight.js theme, VS2015).
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

import flet as ft

if TYPE_CHECKING:
    from collections.abc import Callable

_TEXT_SIZE = 14  # body size — matches VS Code's markdown font-size (14px)
_LINE = 1.57  # VS Code body line-height (22px / 14px)
_FG = ft.Colors.with_opacity(0.86, ft.Colors.WHITE)  # soft foreground (~ #ccc)
_RULE = ft.Colors.with_opacity(0.18, ft.Colors.WHITE)  # VS Code hr / table / border
_HEAD_RULE = ft.Colors.with_opacity(0.5, ft.Colors.WHITE)  # table header underline
_SEMIBOLD = ft.FontWeight.W_600  # VS Code heading weight (not full bold)
_CODE_THEME = ft.MarkdownCodeTheme.VS2015
_INLINE_MD = re.compile(r"\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*")  # link / code / bold
# A lone `<!--section-rule-->` line renders as a heavier, brighter horizontal rule
# (a major-section divider): a thicker line at higher opacity than the thin `---`
# between individual entries. Consumed here so `ft.Markdown` never sees the token.
_SECTION_RULE_RE = re.compile(r"^[ \t]*<!--section-rule-->[ \t]*$\n?", re.MULTILINE)
_SECTION_RULE_COLOR = ft.Colors.with_opacity(0.4, ft.Colors.WHITE)  # brighter than _RULE (0.18)

_MD_STYLE = ft.MarkdownStyleSheet(
    # Body: paragraphs, links, emphasis, lists, table cells — soft foreground,
    # VS Code's roomy line-height.
    p_text_style=ft.TextStyle(size=_TEXT_SIZE, height=_LINE, color=_FG),
    a_text_style=ft.TextStyle(size=_TEXT_SIZE, color=ft.Colors.BLUE_400),
    strong_text_style=ft.TextStyle(size=_TEXT_SIZE, weight=ft.FontWeight.BOLD, color=_FG),
    em_text_style=ft.TextStyle(size=_TEXT_SIZE, italic=True, color=_FG),
    list_bullet_text_style=ft.TextStyle(size=_TEXT_SIZE, height=_LINE, color=_FG),
    table_head_text_style=ft.TextStyle(size=_TEXT_SIZE, weight=ft.FontWeight.BOLD, color=_FG),
    table_head_text_align=ft.TextAlign.LEFT,
    table_body_text_style=ft.TextStyle(size=_TEXT_SIZE, height=1.4, color=_FG),
    # Heading hierarchy — VS Code's em sizes (2 / 1.5 / 1.25 / 1 / .875 / .85 of
    # 14px), semibold, line-height 1.25, each with top breathing room.
    h1_text_style=ft.TextStyle(size=25, weight=_SEMIBOLD, height=1.25, color=_FG),
    h1_padding=ft.Padding.only(top=8, bottom=6),
    h2_text_style=ft.TextStyle(size=21, weight=_SEMIBOLD, height=1.25, color=_FG),
    h2_padding=ft.Padding.only(top=22, bottom=6),
    h3_text_style=ft.TextStyle(size=18, weight=_SEMIBOLD, height=1.25, color=_FG),
    h3_padding=ft.Padding.only(top=16, bottom=4),
    h4_text_style=ft.TextStyle(size=_TEXT_SIZE, weight=_SEMIBOLD, color=_FG),
    h4_padding=ft.Padding.only(top=14, bottom=2),
    h5_text_style=ft.TextStyle(size=13, weight=_SEMIBOLD, color=ft.Colors.GREY_300),
    h5_padding=ft.Padding.only(top=12),
    h6_text_style=ft.TextStyle(size=12, weight=_SEMIBOLD, color=ft.Colors.GREY_400),
    h6_padding=ft.Padding.only(top=12),
    # Inline code + fenced code block (VS Code pre: 1px border, radius, padding;
    # block syntax colouring = the control's code_theme).
    code_text_style=ft.TextStyle(size=14, font_family="monospace", bgcolor=ft.Colors.GREY_800),
    codeblock_decoration=ft.BoxDecoration(
        bgcolor=ft.Colors.GREY_900, border_radius=4, border=ft.Border.all(1, _RULE)
    ),
    codeblock_padding=ft.Padding.symmetric(horizontal=14, vertical=12),
    # Blockquote — a left accent bar + muted upright text, no fill (VS Code-style).
    blockquote_text_style=ft.TextStyle(size=_TEXT_SIZE, height=_LINE, color=ft.Colors.GREY_300),
    blockquote_decoration=ft.BoxDecoration(
        border=ft.Border(left=ft.BorderSide(4, ft.Colors.BLUE_GREY_400)), border_radius=2
    ),
    blockquote_padding=ft.Padding.only(left=12, right=12, top=4, bottom=4),
    # Tables (Flet's built-in ones, used by `themed_markdown`) — horizontal row
    # rules only where Flet allows it; the full native replacement is below.
    table_cells_decoration=ft.BoxDecoration(border=ft.Border(bottom=ft.BorderSide(1, _RULE))),
    table_cells_padding=ft.Padding.symmetric(horizontal=10, vertical=6),
    # `---` rule + block spacing matching VS Code's paragraph rhythm.
    horizontal_rule_decoration=ft.BoxDecoration(border=ft.Border(top=ft.BorderSide(1, _RULE))),
    block_spacing=16,
    list_indent=28,
)


# Compact heading scale for cramped surfaces — the (i) help dialogs, where the
# full h1-h3 sizes (25/21/18) look oversized next to the small dialog title.
# Only the headings shrink; body, lists, code, etc. inherit from _MD_STYLE.
_MD_STYLE_COMPACT = dataclasses.replace(
    _MD_STYLE,
    h1_text_style=ft.TextStyle(size=19, weight=_SEMIBOLD, height=1.25, color=_FG),
    h2_text_style=ft.TextStyle(size=16, weight=_SEMIBOLD, height=1.25, color=_FG),
    h3_text_style=ft.TextStyle(size=15, weight=_SEMIBOLD, height=1.3, color=_FG),
    h3_padding=ft.Padding.only(top=10, bottom=2),
)


def compact_markdown(text: str) -> ft.Markdown:
    """Themed markdown with a smaller heading scale, for the (i) help dialogs
    where the standard h1-h3 sizes look oversized in a narrow pop-up."""
    return ft.Markdown(
        text,
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        md_style_sheet=_MD_STYLE_COMPACT,
        code_theme=_CODE_THEME,
    )


def themed_markdown(text: str, *, on_tap_link: Callable | None = None) -> ft.Markdown:
    """A themed `ft.Markdown` block (VS Code-matched prose). Tables render via
    Flet's built-in grid — use `render_markdown` where content-sized native
    tables matter (e.g. the Metrics Guide)."""
    return ft.Markdown(
        text,
        selectable=True,
        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
        md_style_sheet=_MD_STYLE,
        code_theme=_CODE_THEME,
        on_tap_link=on_tap_link,
    )


def _section_rule() -> ft.Control:
    """A heavier, brighter horizontal rule for a major-section boundary: a thicker
    line at higher opacity than the thin per-entry `---`. Emitted wherever the
    source markdown carries a lone `<!--section-rule-->` line."""
    return ft.Divider(thickness=2, height=24, color=_SECTION_RULE_COLOR)


def render_markdown(text: str, *, on_tap_link: Callable | None = None) -> ft.Control:
    """Render markdown with GFM tables pulled out and drawn as native controls
    (content-sized first column, wrapping body, horizontal rules), and lone
    `<!--section-rule-->` lines drawn as heavier section dividers. Returns a
    single themed `ft.Markdown` when the text has neither."""
    parts: list[ft.Control] = []
    for block_idx, block in enumerate(_SECTION_RULE_RE.split(text)):
        if block_idx:
            parts.append(_section_rule())
        for kind, payload in _split_tables(block):
            if kind == "md":
                if str(payload).strip():
                    parts.append(themed_markdown(str(payload), on_tap_link=on_tap_link))
            else:
                header, rows = payload  # ("table", (header, rows))
                parts.append(_render_table(header, rows, on_tap_link))
    if not parts:
        return themed_markdown(text, on_tap_link=on_tap_link)
    if len(parts) == 1:
        return parts[0]
    return ft.Column(parts, spacing=10, horizontal_alignment=ft.CrossAxisAlignment.STRETCH)


# ── GFM table extraction → native controls ───────────────────────────────────


def _split_cells(row: str) -> list[str]:
    r"""One GFM row → stripped cells, dropping the outer pipes and honouring `\|`."""
    s = row.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", s)]


def _is_delimiter(row: str) -> bool:
    """True for a GFM separator row like `|---|:--:|` (each cell is dashes with
    optional alignment colons)."""
    cells = _split_cells(row)
    return bool(cells) and all(re.fullmatch(r":?-+:?", c) for c in cells)


def _split_tables(text: str) -> list[tuple[str, object]]:
    """Split a markdown chunk into ordered ``("md", text)`` / ``("table", (header,
    rows))`` parts. A table = a row followed by a delimiter row, then body rows
    until a non-row line; everything else stays markdown."""
    lines = text.split("\n")
    segments: list[tuple[str, object]] = []
    prose: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if i + 1 < n and "|" in lines[i] and lines[i].strip() and _is_delimiter(lines[i + 1]):
            if prose:
                segments.append(("md", "\n".join(prose)))
                prose = []
            header = _split_cells(lines[i])
            body: list[list[str]] = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                body.append(_split_cells(lines[j]))
                j += 1
            segments.append(("table", (header, body)))
            i = j
        else:
            prose.append(lines[i])
            i += 1
    if prose:
        segments.append(("md", "\n".join(prose)))
    return segments


def _display_text(cell: str) -> str:
    """A cell's visible text — link labels only, no code/bold markers — for width
    estimation."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cell).replace("`", "").replace("*", "")


def _first_col_width(header: list[str], rows: list[list[str]]) -> int:
    """Content-sized width for the first column: its widest display text
    (~9px/char at 14px), clamped so it hugs content without wrapping or bloating."""
    texts = [header[0], *(r[0] for r in rows if r)] if header else []
    longest = max((len(_display_text(t)) for t in texts), default=0)
    return max(90, min(380, longest * 9 + 30))


def _cell(text: str, on_tap_link: Callable | None, *, header: bool) -> ft.Control:
    """One table cell: header = bold plain text; a body cell with inline markdown
    (link/code/bold) → tight `ft.Markdown` (so links still work); a plain body
    cell → wrapping `ft.Text`."""
    if header:
        return ft.Text(_display_text(text), size=_TEXT_SIZE, weight=ft.FontWeight.BOLD, color=_FG)
    if _INLINE_MD.search(text):
        return ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            md_style_sheet=_MD_STYLE,
            on_tap_link=on_tap_link,
        )
    return ft.Text(text, size=_TEXT_SIZE, color=_FG, selectable=True)


def _table_row(
    cells: list[str], ncols: int, w1: int, on_tap_link: Callable | None, *, header: bool
) -> ft.Control:
    cells = (cells + [""] * ncols)[:ncols]  # pad / truncate to the header width
    kids = [
        ft.Container(
            content=_cell(cell, on_tap_link, header=header),
            padding=ft.Padding.only(right=16, top=7, bottom=7),
            width=w1 if idx == 0 else None,
            expand=True if idx else None,
        )
        for idx, cell in enumerate(cells)
    ]
    return ft.Container(
        content=ft.Row(kids, vertical_alignment=ft.CrossAxisAlignment.START, spacing=0),
        border=ft.Border(bottom=ft.BorderSide(1, _HEAD_RULE if header else _RULE)),
    )


def _render_table(
    header: list[str], rows: list[list[str]], on_tap_link: Callable | None
) -> ft.Control:
    """A GFM table as native controls: first column content-sized, the rest fill
    + wrap, horizontal rules only (VS Code style — no grid, no stretch)."""
    ncols = max(1, len(header))
    w1 = _first_col_width(header, rows)
    table_rows = [_table_row(header, ncols, w1, on_tap_link, header=True)]
    table_rows += [_table_row(r, ncols, w1, on_tap_link, header=False) for r in rows]
    return ft.Container(
        content=ft.Column(
            table_rows, spacing=0, horizontal_alignment=ft.CrossAxisAlignment.STRETCH
        ),
        padding=ft.Padding.only(top=4, bottom=10),
    )
