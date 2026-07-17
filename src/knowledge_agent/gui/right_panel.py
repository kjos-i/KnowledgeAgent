"""Right column of the Search tab — a View · Retrieval · LLM sub-tab strip.

Owns:
- The **View** sub-tab: a small history of recently-viewed results with a
  pager, plus Save / Open buttons. Each history slot is a produced answer
  (`AgentAnswer` + query) or an opened file (name + content); the newest
  five are kept, ordered by when they were shown. `GuiApp` pushes a slot
  via `push_answer` after a query and `push_file` on Open Result, and
  resets the history on Clear (`reset_history`). The pager `< n/N > ✕`
  (in the action row) walks the slots and drops the current one. History
  is in-memory only — it starts empty on every launch. Save / Open are
  plain action buttons on this tab — they act on the result, so they live
  with it.
- The **Retrieval** and **LLM** sub-tabs: the per-query search + answer-
  model settings, promoted here from Settings so they can be tuned beside
  the chat and re-run. Same `RetrievalTab` / `LlmTab` widgets as before;
  Settings no longer hosts them.

The chat stays in the LEFT column (SearchTab's split) — mounted once and
untouched no matter which sub-tab is active; only this right column swaps.
That's why the strip sits at the top of the right pane (not full-width
over the chat): the tabs genuinely govern only this column.

Settings and Info are TOP-LEVEL tabs now (not sub-tabs here) — global
config and reference that don't belong in the per-query search loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import FIELD_LABEL_SIZE, FRAME_BORDER_COLOR, centered_label
from knowledge_agent.gui.settings.llm_tab import LlmTab
from knowledge_agent.gui.settings.retrieval_tab import RetrievalTab
from knowledge_agent.gui.views.file_view import FileView
from knowledge_agent.gui.views.info_doc import InfoDoc, docs_dir
from knowledge_agent.gui.views.latest_view import LatestView

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.models import AgentAnswer


# A View history slot is one of these two kinds; `_ViewItem.kind` carries the
# value so the pager can render either an answer or a file from a snapshot.
MODE_LATEST = "latest"
MODE_FILE = "file"

# Newest-N view history kept in the pager. Five is what the user asked for —
# enough to flip back through a short burst of results, not a full log.
_MAX_HISTORY = 5

SUB_TAB_LABELS = ("View", "Retrieval", "LLMs", "Info")


@dataclass
class _ViewItem:
    """One slot in the View sub-tab's history — a produced answer or an
    opened file, snapshotted at push time so paging back stays stable even
    after later queries mutate the live `GuiApp` state."""

    kind: str  # MODE_LATEST (answer) | MODE_FILE (file)
    # MODE_LATEST payload.
    answer: AgentAnswer | None = None
    query: str = ""
    # MODE_FILE payload.
    name: str = ""
    content: str = ""


class RightPanel:
    """Right column of the Search tab — View / Retrieval / LLMs sub-tabs."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # In-memory view history + cursor. Empty on launch (no persistence);
        # `history_index` is -1 while empty, else the shown slot (0-based).
        self.history: list[_ViewItem] = []
        self.history_index: int = -1
        # Which kind the View sub-tab currently shows (answer vs file). Kept
        # in sync by `_rebuild`; starts on Latest so the empty state reads as
        # the search-result placeholder.
        self.current_mode: str = MODE_LATEST
        # Late-bound — populated by build().
        self.view_container: ft.Container | None = None
        self.path_field: ft.TextField | None = None
        self.pager_prev: ft.IconButton | None = None
        self.pager_next: ft.IconButton | None = None
        self.pager_label: ft.Text | None = None
        self.pager_close: ft.IconButton | None = None
        # The promoted per-query settings surfaces (own their own state).
        self.retrieval_tab = RetrievalTab(app)
        self.llm_tab = LlmTab(app)

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        self.view_container = ft.Container(
            content=self._build_current_view(),
            expand=True,
            padding=12,
        )
        # View sub-tab body: the result display + a pager / Save / Open row.
        view_body = ft.Column(
            controls=[self.view_container, self._view_button_row()],
            expand=True,
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        # Native Flet Tabs (secondary indicator) so the sub-strip reads as a
        # level below the primary Search / Library / … bar — the same shape
        # Library and Evaluation use. Chat is NOT in here; it's the split's
        # left pane, so switching sub-tabs never rebuilds it.
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in SUB_TAB_LABELS],
            secondary=True,
        )
        info_body = InfoDoc(
            self.app,
            docs_dir() / "info_search.md",
            missing_hint="Search help is not written yet (info_search.md).",
        ).build()
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=view_body, padding=8, expand=True),
                ft.Container(content=self.retrieval_tab.build(), padding=8, expand=True),
                ft.Container(content=self.llm_tab.build(), padding=8, expand=True),
                ft.Container(content=info_body, padding=8, expand=True),
            ],
            expand=True,
        )
        return ft.Tabs(
            length=len(SUB_TAB_LABELS),
            selected_index=0,
            content=ft.Column(controls=[sub_bar, sub_bodies], expand=True, spacing=0),
            expand=True,
        )

    def push_answer(self, answer: AgentAnswer | None, query: str) -> None:
        """Add a produced answer to the view history and jump to it.

        `GuiApp` calls this after every query so the new answer becomes the
        shown slot (and pushes the oldest out past `_MAX_HISTORY`).
        """
        self._push(_ViewItem(kind=MODE_LATEST, answer=answer, query=query))

    def push_file(self, name: str, content: str) -> None:
        """Add an opened file to the view history and jump to it.

        `GuiApp` calls this from Open Result so the file becomes the shown
        slot alongside prior answers/files in the pager.
        """
        self._push(_ViewItem(kind=MODE_FILE, name=name, content=content))

    def reset_history(self) -> None:
        """Clear the view history (Clear button). A loaded file that survives
        Clear (`keep_loaded_file_on_clear`) is re-seeded as the sole slot so
        it stays visible; otherwise the pager empties to the placeholder."""
        self.history = []
        self.history_index = -1
        loaded = self.app.loaded_file
        if loaded is not None:
            self.history.append(_ViewItem(kind=MODE_FILE, name=loaded.name, content=loaded.content))
            self.history_index = 0
        self._rebuild()

    # ----- pager handlers --------------------------------------------------

    def _on_prev(self, _e: ft.Event) -> None:
        if self.history_index > 0:
            self.history_index -= 1
            self._rebuild()

    def _on_next(self, _e: ft.Event) -> None:
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._rebuild()

    def _on_close_item(self, _e: ft.Event) -> None:
        """Drop the shown slot and land on its neighbour (the previous one,
        or the new last slot when the first was removed)."""
        if not self.history:
            return
        del self.history[self.history_index]
        if self.history_index >= len(self.history):
            self.history_index = len(self.history) - 1  # -1 when now empty
        self._rebuild()

    # ----- internals -------------------------------------------------------

    def _push(self, item: _ViewItem) -> None:
        self.history.append(item)
        if len(self.history) > _MAX_HISTORY:
            self.history = self.history[-_MAX_HISTORY:]
        self.history_index = len(self.history) - 1
        self._rebuild()

    def _rebuild(self) -> None:
        """Re-render the shown slot + refresh the pager, then repaint."""
        self.current_mode = self.history[self.history_index].kind if self.history else MODE_LATEST
        if self.view_container is not None:
            self.view_container.content = self._build_current_view()
        self._sync_pager()
        self.app.page.update()

    def _build_current_view(self) -> ft.Control:
        if not self.history:
            # Empty history → the Latest view's search-result placeholder.
            return LatestView(None, "", page=self.app.page).build()
        item = self.history[self.history_index]
        if item.kind == MODE_FILE:
            return FileView(name=item.name, content=item.content).build()
        return LatestView(item.answer, item.query, page=self.app.page).build()

    def _sync_pager(self) -> None:
        """Point the pager label + button states at the current cursor. Buttons
        stay visible-but-greyed at the ends (and when empty) rather than hiding
        — the same greyed-not-hidden convention used elsewhere in the app."""
        n = len(self.history)
        idx = self.history_index
        if self.pager_label is not None:
            self.pager_label.value = f"{idx + 1}/{n}" if n else "0/0"
        if self.pager_prev is not None:
            self.pager_prev.disabled = idx <= 0
        if self.pager_next is not None:
            self.pager_next.disabled = idx >= n - 1
        if self.pager_close is not None:
            self.pager_close.disabled = n == 0

    # ----- buttons ---------------------------------------------------------

    def _view_button_row(self) -> ft.Control:
        """Save / Open actions for the current result + a pager on the right.
        The pager walks the recently-viewed history (`< n/N > ✕`); Save / Open
        act on the shown slot. (View is the sub-tab itself, so there's no 'View
        Result' button; Settings and Info are top-level tabs, so they're gone
        from here too.)"""
        self.pager_prev = ft.IconButton(
            ft.Icons.CHEVRON_LEFT,
            icon_size=18,
            tooltip="Previous result",
            on_click=self._on_prev,
        )
        self.pager_label = ft.Text("0/0", size=12)
        self.pager_next = ft.IconButton(
            ft.Icons.CHEVRON_RIGHT,
            icon_size=18,
            tooltip="Next result",
            on_click=self._on_next,
        )
        self.pager_close = ft.IconButton(
            ft.Icons.CLOSE,
            icon_size=18,
            icon_color=ft.Colors.RED_300,
            tooltip="Remove this result from history",
            on_click=self._on_close_item,
        )
        pager = ft.Row(
            [self.pager_prev, self.pager_label, self.pager_next, self.pager_close],
            spacing=0,
            tight=True,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        # Paste-path field: type/paste a .md/.txt path + Enter to open it, an
        # alternative to the Open Result picker (`GuiApp.on_open_path`).
        self.path_field = ft.TextField(
            hint_text="Paste file path (.md, .txt)",
            dense=True,
            expand=True,
            text_size=FIELD_LABEL_SIZE,  # match the app's standard field text (14)
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            on_submit=self.app.on_open_path,
        )
        self._sync_pager()
        # Buttons stay at their natural (text) width so they aren't wider than
        # they need to be; the paste-path field takes the remaining space with
        # `expand=True`, so it's as wide as possible AND still flexes on resize.
        return ft.Row(
            controls=[
                ft.Button(
                    content=centered_label("Save Result"),
                    on_click=self.app.on_save_answer,
                ),
                ft.Button(
                    content=centered_label("Open Result"),
                    on_click=self.app.on_open_result,
                ),
                self.path_field,
                pager,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
