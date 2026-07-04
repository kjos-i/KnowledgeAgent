"""Tests for the Library Documents browser (`DocumentsView`).

Exercise the pure render / filter / edit-patch logic directly (no Flet
render, no real backend, no event loop). The fake app + page from
`conftest.py` supply a MagicMock `.update()`; these tests never call the
async fetch/save/resolve/delete paths (those are guarded to no-op
without a running loop).
"""

from __future__ import annotations

import flet as ft

from knowledge_agent.gui.library.documents_view import (
    DocumentsView,
    _basename,
    _fmt_date,
)


def _rows() -> list[dict]:
    return [
        {
            "doc_id": "d1", "title": "Aspirin and the heart",
            "source_path": "/papers/aspirin.pdf", "sub_label": "Paper",
            "main_label": "Document", "metadata_status": "enriched",
            "ingested_at": "2026-07-03T10:00:00", "n_chunks": 12,
            "n_figures": 3, "doi": "10.1/asp", "year": 2020,
            "authors_display": "Jane Doe", "venue": "Cardiology",
            "source_url": "https://doi.org/10.1/asp", "language": "en",
        },
        {
            "doc_id": "d2", "title": "Notes on mercury",
            "source_path": "/notes/mercury.md", "sub_label": "Note",
            "main_label": "Document", "metadata_status": "manual",
            "ingested_at": "2026-07-01T09:00:00", "n_chunks": 4,
            "n_figures": 0,
        },
    ]


def _view(fake_app, rows=None):
    dv = DocumentsView(fake_app)
    dv._loaded_rows = rows if rows is not None else _rows()
    dv._loaded_for = "corpus-x"
    return dv


def _edit_fields(**overrides) -> dict[str, ft.TextField]:
    """Build the modal's fields dict (all keys the patch collector reads)."""
    base = {
        "title": "", "authors_display": "", "year": "",
        "venue": "", "doi": "", "source_url": "", "language": "",
    }
    base.update(overrides)
    return {k: ft.TextField(value=v) for k, v in base.items()}


# ---- filter ----


def test_matches_filter_empty_needle_matches_all():
    row = {"title": "x", "source_path": "y"}
    assert DocumentsView._matches_filter(row, "") is True


def test_matches_filter_hits_title_and_source():
    row = {"title": "Aspirin", "source_path": "/p/heart.pdf"}
    assert DocumentsView._matches_filter(row, "aspirin") is True
    assert DocumentsView._matches_filter(row, "heart") is True
    assert DocumentsView._matches_filter(row, "mercury") is False


# ---- render ----


def test_render_rows_one_card_per_doc(fake_app):
    dv = _view(fake_app)
    dv._render_rows()
    assert len(dv.doc_list.controls) == 2
    assert dv.coverage_text.value == "2 documents"


def test_render_rows_filter_narrows_and_updates_coverage(fake_app):
    dv = _view(fake_app)
    dv.search_field.value = "mercury"
    dv._render_rows()
    assert len(dv.doc_list.controls) == 1
    assert dv.coverage_text.value == "1 of 2 documents"


def test_render_rows_no_match_shows_message(fake_app):
    dv = _view(fake_app)
    dv.search_field.value = "nonexistent-term"
    dv._render_rows()
    assert len(dv.doc_list.controls) == 1  # the "no match" text
    # coverage still reflects the full set
    assert dv.coverage_text.value == "0 of 2 documents"


def test_render_rows_empty_set_shows_empty_state(fake_app):
    dv = _view(fake_app, rows=[])
    dv._render_rows()
    assert len(dv.doc_list.controls) == 1
    assert dv.coverage_text.value == ""


def test_render_doc_card_returns_container(fake_app):
    dv = _view(fake_app)
    card = dv._render_doc_card(_rows()[0])
    assert isinstance(card, ft.Container)


def test_render_doc_card_has_three_action_buttons(fake_app):
    """Each card exposes Edit + Re-ingest + Delete (per-card Resolve was
    removed — single-doc resolve lives in the Edit modal's 'Look up DOI
    online' checkbox)."""
    dv = _view(fake_app)
    card = dv._render_doc_card(_rows()[0])
    action_row = card.content.controls[0]
    buttons = [c for c in action_row.controls if isinstance(c, ft.Button)]
    assert len(buttons) == 3


def test_reingest_no_source_path_sets_status(fake_app):
    dv = _view(fake_app)
    dv._open_reingest_confirm(
        {"doc_id": "d1", "title": "T", "source_path": ""}
    )
    assert "source path" in dv.op_status.value.lower()


def test_reingest_missing_file_sets_status(fake_app):
    dv = _view(fake_app)
    dv._open_reingest_confirm(
        {"doc_id": "d1", "title": "T", "source_path": "/no/such/file_xyz.pdf"}
    )
    assert "not found" in dv.op_status.value.lower()


def test_no_resolve_all_controls_in_table(fake_app):
    """Resolve all + Skip-manually-edited were relocated to the Ingest
    bulk-ops; the Documents table no longer owns them."""
    dv = DocumentsView(fake_app)
    assert not hasattr(dv, "resolve_all_button")
    assert not hasattr(dv, "skip_manual_checkbox")


# ---- edit prefill ----


def test_field_value_stringifies_int_year():
    assert DocumentsView._field_value({"year": 2020}, "year") == "2020"
    assert DocumentsView._field_value({"year": None}, "year") == ""
    assert DocumentsView._field_value({}, "title") == ""


# ---- edit patch ----


def test_collect_edit_patch_reads_all_fields(fake_app):
    dv = _view(fake_app)
    dv._active_edit = {
        "doc_id": "d1",
        "fields": _edit_fields(
            title="New title", authors_display="A; B", year="2021",
            venue="Nature", doi="10.1/x",
            source_url="https://doi.org/10.1/x", language="en",
        ),
    }
    patch = dv._collect_edit_patch()
    assert patch == {
        "title": "New title",
        "authors_display": "A; B",
        "venue": "Nature",
        "doi": "10.1/x",
        "source_url": "https://doi.org/10.1/x",
        "language": "en",
        "year": 2021,  # coerced to int
        "metadata_status": "manual",
    }


def test_collect_edit_patch_blank_strings_cleared_year_omitted(fake_app):
    dv = _view(fake_app)
    dv._active_edit = {"doc_id": "d1", "fields": _edit_fields()}  # all blank
    patch = dv._collect_edit_patch()
    assert patch == {
        "title": "", "authors_display": "", "venue": "",
        "doi": "", "source_url": "", "language": "",
        "metadata_status": "manual",
    }
    assert "year" not in patch  # blank year → left unchanged


def test_collect_edit_patch_unparseable_year_omitted(fake_app):
    dv = _view(fake_app)
    dv._active_edit = {
        "doc_id": "d1", "fields": _edit_fields(year="twenty"),
    }
    patch = dv._collect_edit_patch()
    assert "year" not in patch
    assert patch["metadata_status"] == "manual"


def test_collect_edit_patch_no_active_dialog_returns_empty(fake_app):
    dv = _view(fake_app)
    assert dv._collect_edit_patch() == {}


# ---- helpers ----


def test_basename_extracts_filename():
    assert _basename("/a/b/c.pdf") == "c.pdf"
    assert _basename("C:\\x\\y\\z.docx") == "z.docx"
    assert _basename(None) == ""


def test_fmt_date_takes_date_head():
    assert _fmt_date("2026-07-03T10:00:00") == "2026-07-03"
    assert _fmt_date(None) == ""


# ---- fetch-error control ----


def test_fetch_error_control_explains_missing_password():
    ctrl = DocumentsView._fetch_error_control(
        ValueError(
            "1 validation error for Settings\n"
            "neo4j_password\n  Field required"
        )
    )
    assert "password" in ctrl.value.lower()


def test_fetch_error_control_generic_shows_message():
    ctrl = DocumentsView._fetch_error_control(RuntimeError("disk boom"))
    assert "disk boom" in ctrl.value
