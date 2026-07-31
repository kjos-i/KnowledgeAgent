"""Tests for gui/library/documents_view — the per-document delete failure path.

Calls `_do_delete` as an unbound method against a MagicMock `self` so we don't
have to construct a full Flet view; only the failure-surfacing contract (B14)
is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.library.documents_view import DocumentsView


def _doc(doc_id: str, status: str) -> dict:
    """A minimal indexed-doc row sufficient for `_render_doc_card`."""
    return {
        "doc_id": doc_id,
        "metadata_status": status,
        "title": f"Title {doc_id}",
        "source_path": rf"C:\corpus\{doc_id}.pdf",
        "main_label": "Document",
        "sub_label": None,
        "n_chunks": 3,
        "n_figures": 0,
        "ingested_at": "2026-07-30T10:00:00",
    }


def test_matches_status_gates_known_statuses_by_the_ticked_set():
    """A canonical status shows only when its box is ticked."""
    assert DocumentsView._matches_status(_doc("d1", "enriched"), {"enriched"}) is True
    assert DocumentsView._matches_status(_doc("d2", "baseline"), {"enriched"}) is False


def test_matches_status_never_hides_unknown_or_blank():
    """A blank / unrecognised status is always shown — the filter must never
    silently hide a doc it doesn't have a checkbox for (the disappearing-docs
    guard)."""
    assert DocumentsView._matches_status(_doc("d1", "weird"), set()) is True
    assert DocumentsView._matches_status(_doc("d2", ""), set()) is True
    assert DocumentsView._matches_status({}, {"enriched"}) is True


def test_status_checkboxes_filter_the_rendered_cards(fake_app):
    """Unticking a status checkbox drops those cards from the rendered list;
    all ticked (the default) shows everything."""
    view = DocumentsView(fake_app)
    view._loaded_rows = [_doc("d1", "enriched"), _doc("d2", "baseline")]
    # Default: every status ticked -> both render.
    view._render_rows()
    assert len(view.doc_list.controls) == 2
    # Untick 'baseline' -> only the enriched doc remains.
    view.status_checks["baseline"].value = False
    view._render_rows()
    assert len(view.doc_list.controls) == 1
    assert view.coverage_text.value == "(1 of 2 documents)"
    # Untick everything -> the no-match message, not a card.
    for cb in view.status_checks.values():
        cb.value = False
    view._render_rows()
    assert len(view.doc_list.controls) == 1  # the message
    assert "No documents match the selected statuses." in view.doc_list.controls[0].value


async def test_do_delete_surfaces_exception_via_status_and_reloads():
    """B14: a delete that RAISES must surface via _set_op_status AND reload, not
    just log and return (which left the card looking deleted when it wasn't)."""
    view = MagicMock()
    view._set_op_status = MagicMock()
    view.reload = AsyncMock()
    plan = SimpleNamespace(doc_id="doc1")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.delete_doc_execute",
        AsyncMock(side_effect=RuntimeError("boom")),
    ):
        await DocumentsView._do_delete(view, plan)
    view._set_op_status.assert_called_once()
    view.reload.assert_awaited_once()


async def test_do_delete_surfaces_not_ok_result_via_status():
    """B14: a delete returning ok=False must post a status message (not silently
    reload with the doc still present and no explanation)."""
    view = MagicMock()
    view._set_op_status = MagicMock()
    view.reload = AsyncMock()
    plan = SimpleNamespace(doc_id="doc1")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.delete_doc_execute",
        AsyncMock(return_value=SimpleNamespace(ok=False)),
    ):
        await DocumentsView._do_delete(view, plan)
    view._set_op_status.assert_called_once()
    view.reload.assert_awaited_once()


async def test_save_metadata_returns_true_on_success():
    """_save_metadata reports True (and reloads) when the LanceDB write lands."""
    view = MagicMock()
    view.reload = AsyncMock()
    client = MagicMock()
    client.update_doc_metadata = AsyncMock()
    with patch(
        "knowledge_agent.search.client.get_search_client",
        return_value=client,
    ):
        ok = await DocumentsView._save_metadata(view, "d1", {"title": "X"})
    assert ok is True
    view.reload.assert_awaited_once()


async def test_save_metadata_returns_false_on_write_failure():
    """A failed metadata write reports False and does NOT reload."""
    view = MagicMock()
    view.reload = AsyncMock()
    client = MagicMock()
    client.update_doc_metadata = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "knowledge_agent.search.client.get_search_client",
        return_value=client,
    ):
        ok = await DocumentsView._save_metadata(view, "d1", {"title": "X"})
    assert ok is False
    view.reload.assert_not_awaited()


async def test_run_edit_save_success_posts_no_error_status():
    """A successful edit save posts no error status (the reload is the feedback)."""
    view = MagicMock()
    view._save_metadata = AsyncMock(return_value=True)
    view._set_button_busy = MagicMock()
    view._set_op_status = MagicMock()
    await DocumentsView._run_edit_save(view, "d1", {"title": "X"}, MagicMock())
    view._set_op_status.assert_not_called()


async def test_run_edit_save_failure_restores_button_and_posts_op_status():
    """A failed edit save restores the Edit button and reports on the op-status
    line (the same place a failed re-ingest lands)."""
    view = MagicMock()
    view._save_metadata = AsyncMock(return_value=False)
    view._set_button_busy = MagicMock()
    view._set_op_status = MagicMock()
    await DocumentsView._run_edit_save(view, "d1", {"title": "X"}, MagicMock())
    view._set_op_status.assert_called_once()
    # last _set_button_busy call restores the button (busy=False)
    assert view._set_button_busy.call_args_list[-1].args[1] is False


async def test_run_edit_save_restores_button_when_save_raises():
    """A raise from _save_metadata (e.g. the post-write list refresh failing)
    must still restore the Edit button and post a message, never leave the
    wheel spinning."""
    view = MagicMock()
    view._save_metadata = AsyncMock(side_effect=RuntimeError("render boom"))
    view._set_button_busy = MagicMock()
    view._set_op_status = MagicMock()
    await DocumentsView._run_edit_save(view, "d1", {"title": "X"}, MagicMock())
    view._set_op_status.assert_called_once()
    assert view._set_button_busy.call_args_list[-1].args[1] is False
