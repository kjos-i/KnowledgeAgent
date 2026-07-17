"""Shared markdown theme + GFM table renderer (`gui/_markdown.py`).

`themed_markdown` gives a plain themed widget (answers / opened files);
`render_markdown` additionally pulls GFM tables out into native controls
(content-sized columns, horizontal rules) since Flet's built-in table is an
unstyleable equal-width grid.
"""

from __future__ import annotations

import flet as ft

from knowledge_agent.gui._markdown import (
    _is_delimiter,
    _split_cells,
    _split_tables,
    render_markdown,
    themed_markdown,
)


def test_themed_markdown_carries_the_theme():
    md = themed_markdown("# Title\n\nsome **prose** with `code`.")
    assert isinstance(md, ft.Markdown)
    assert md.md_style_sheet is not None  # the shared VS Code-matched theme
    assert md.code_theme is not None  # VS2015 syntax colours


def test_split_cells_drops_outer_pipes_and_keeps_links():
    assert _split_cells("| [Hit@k](#hitk) | Did it appear? |") == [
        "[Hit@k](#hitk)",
        "Did it appear?",
    ]


def test_is_delimiter_row():
    assert _is_delimiter("|--------|-------------|")
    assert _is_delimiter("| :--- | ---: |")  # alignment colons allowed
    assert not _is_delimiter("| Metric | Description |")  # a real header row


def test_split_tables_separates_prose_and_table():
    text = "intro line\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\ntrailing"
    segs = _split_tables(text)
    assert [k for k, _ in segs] == ["md", "table", "md"]
    header, rows = segs[1][1]
    assert header == ["A", "B"]
    assert rows == [["1", "2"], ["3", "4"]]
    assert "intro line" in segs[0][1]
    assert "trailing" in segs[2][1]


def test_render_markdown_without_tables_returns_a_single_markdown():
    ctl = render_markdown("just **prose**, no table here")
    assert isinstance(ctl, ft.Markdown)


def test_render_markdown_with_table_builds_native_controls():
    """A table chunk renders as a native Container→Column (header + one row per
    body line), NOT Flet's built-in Markdown grid."""
    ctl = render_markdown("| Metric | Description |\n|---|---|\n| [A](#a) | d1 |\n| B | d2 |")
    assert not isinstance(ctl, ft.Markdown)  # native table, not a Markdown widget
    assert len(ctl.content.controls) == 3  # header + 2 body rows


def test_render_markdown_interleaves_prose_and_table():
    ctl = render_markdown("lead\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\ntail")
    assert isinstance(ctl, ft.Column)  # prose + table + prose
    assert len(ctl.controls) == 3
