"""Library top-tab — dataset management + ingest + documents browser.

`LibraryView` is the coordinator that composes the two-column layout:

  - LEFT: 3 sub-tabs (Select Dataset / Create New Dataset / Installs)
  - RIGHT: Documents table (read-only browser)

`tabs.library_tab.LibraryTab` dispatches its `build()` here.
"""
from knowledge_agent.gui.library.library_view import LibraryView

__all__ = ["LibraryView"]
