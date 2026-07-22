"""Tests for gui/library/documents_view — the per-document delete failure path.

Calls `_do_delete` as an unbound method against a MagicMock `self` so we don't
have to construct a full Flet view; only the failure-surfacing contract (B14)
is exercised.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.gui.library.documents_view import DocumentsView


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
