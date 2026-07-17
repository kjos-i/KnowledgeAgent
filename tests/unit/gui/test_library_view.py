"""Tests for LibraryView — the Library sub-tab coordinator.

Focused on the cross-tab wiring (Ingest → Select auto-refresh). Uses the
shared MagicMock `fake_app`; construction of the four sub-tabs must
tolerate it (they build controls only, no DB, in __init__).
"""

from __future__ import annotations

from knowledge_agent.gui.library.library_view import SUB_TAB_LABELS, LibraryView


def test_ingest_complete_wired_to_select_refresh(fake_app):
    """After a successful ingest/bulk-op, the Ingest sub-tab's completion
    callback points at the Select sub-tab's refresh (cross-tab reload)."""
    lv = LibraryView(fake_app)
    assert lv.ingest_tab.on_ingest_complete == lv.select_tab.refresh_after_ingest


def test_installs_is_no_longer_a_library_subtab():
    """Installs was promoted to a top-level tab; Library is the corpus-lifecycle
    sub-tabs plus a contextual Info tab."""
    assert "Installs" not in SUB_TAB_LABELS
    assert SUB_TAB_LABELS == ("Create New", "Ingest", "Metadata", "Info")


def test_library_view_no_longer_owns_installs(fake_app):
    """LibraryView no longer constructs the Installs sub-tab (app.py mounts it
    at the top level instead)."""
    lv = LibraryView(fake_app)
    assert not hasattr(lv, "installs_tab")
