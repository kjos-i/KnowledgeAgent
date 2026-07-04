"""Tests for LibraryView — the Library sub-tab coordinator.

Focused on the cross-tab wiring (Ingest → Select auto-refresh). Uses the
shared MagicMock `fake_app`; construction of the four sub-tabs must
tolerate it (they build controls only, no DB, in __init__).
"""

from __future__ import annotations

from knowledge_agent.gui.library.library_view import LibraryView


def test_ingest_complete_wired_to_select_refresh(fake_app):
    """After a successful ingest/bulk-op, the Ingest sub-tab's completion
    callback points at the Select sub-tab's refresh (cross-tab reload)."""
    lv = LibraryView(fake_app)
    assert lv.ingest_tab.on_ingest_complete == lv.select_tab.refresh_after_ingest
