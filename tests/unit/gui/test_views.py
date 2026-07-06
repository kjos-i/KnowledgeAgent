"""Tests for the slice-1 views: `LatestView`, `FileView`, and the
shared `_frame` helpers.

Static construction only — we build the Flet control trees and assert
type + structure. No `ft.run` is launched.
"""

from __future__ import annotations

import flet as ft

from knowledge_agent.gui.views._frame import (
    empty_state,
    view_header,
    view_with_header,
)
from knowledge_agent.gui.views.file_view import FileView
from knowledge_agent.gui.views.latest_view import LatestView
from knowledge_agent.models import AgentAnswer, ChunkSource

# ---- _frame helpers ----


def test_view_header_returns_column_with_title_and_divider():
    ctl = view_header("Latest Result")
    assert isinstance(ctl, ft.Column)
    # First control is the title text; second is a divider.
    assert isinstance(ctl.controls[0], ft.Text)
    assert ctl.controls[0].value == "Latest Result"
    assert isinstance(ctl.controls[1], ft.Divider)


def test_empty_state_centers_italic_message():
    ctl = empty_state("nothing yet")
    assert isinstance(ctl, ft.Container)
    inner = ctl.content
    assert isinstance(inner, ft.Text)
    assert inner.value == "nothing yet"
    assert inner.italic is True


def test_view_with_header_composes_header_plus_body():
    body = ft.Text("body")
    ctl = view_with_header("Title", body)
    assert isinstance(ctl, ft.Column)
    # Header + body (the header itself is a column with title + divider).
    assert len(ctl.controls) == 2
    assert ctl.controls[1] is body


# ---- LatestView ----


def test_latest_view_with_no_answer_shows_empty_state():
    view = LatestView(answer=None, query="")
    ctl = view.build()
    # The body cell is the Container produced by empty_state — find it
    # by walking the structure.
    assert isinstance(ctl, ft.Column)
    body = ctl.controls[1]
    assert isinstance(body, ft.Container)
    text = body.content
    assert isinstance(text, ft.Text)
    assert "Search result" in text.value


def test_latest_view_with_answer_renders_markdown():
    answer = AgentAnswer(
        answer="The answer is 42.",
        chunk_sources=[ChunkSource(chunk_id="d#0", doc_id="d")],
        kg_sources=[],
    )
    view = LatestView(answer=answer, query="what is X?")
    ctl = view.build()
    # The body is a scrollable column wrapping a Markdown control.
    body_col = ctl.controls[1]
    assert isinstance(body_col, ft.Column)
    md = body_col.controls[0]
    assert isinstance(md, ft.Markdown)
    assert "The answer is 42." in md.value
    assert "what is X?" in md.value


# ---- FileView ----


def test_file_view_renders_filename_in_header_and_content_as_markdown():
    view = FileView(name="saved.md", content="# Hello")
    ctl = view.build()
    # Header text reflects the filename.
    header_col = ctl.controls[0]
    assert isinstance(header_col, ft.Column)
    assert "saved.md" in header_col.controls[0].value
    # Body contains the markdown.
    body_col = ctl.controls[1]
    md = body_col.controls[0]
    assert isinstance(md, ft.Markdown)
    assert md.value == "# Hello"


def test_file_view_txt_renders_as_plain_text_not_markdown():
    """A .txt opens as plain monospace Text (no Markdown parsing) so it shows
    exactly its characters."""
    view = FileView(name="saved.txt", content="Question: x\n  [1] doc: d")
    ctl = view.build()
    content = ctl.controls[1].controls[0]
    assert isinstance(content, ft.Text)
    assert not isinstance(content, ft.Markdown)
    assert content.value == "Question: x\n  [1] doc: d"
