"""Right column of the Search tab — a View · Retrieval · LLM sub-tab strip.

Owns:
- The **View** sub-tab: the latest answer (or an opened file) + Save /
  Open buttons. Its Latest/File state is tracked by `current_mode`
  (`MODE_LATEST` / `MODE_FILE`) and swapped via `switch_mode`, which
  `GuiApp` calls after a query (refresh Latest), on Open Result (show the
  file), and on Clear. Save / Open are plain action buttons on this tab —
  they act on the result, so they live with it.
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

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import centered_label
from knowledge_agent.gui.settings.llm_tab import LlmTab
from knowledge_agent.gui.settings.retrieval_tab import RetrievalTab
from knowledge_agent.gui.views.file_view import FileView
from knowledge_agent.gui.views.latest_view import LatestView

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


# The View sub-tab shows either the latest answer or an opened file. These
# are the two states of that ONE sub-tab (Retrieval / LLM are separate
# sub-tabs, not modes) — `GuiApp` flips between them via `switch_mode`.
MODE_LATEST = "latest"
MODE_FILE = "file"

SUB_TAB_LABELS = ("View", "Retrieval", "LLMs")


class RightPanel:
    """Right column of the Search tab — View / Retrieval / LLMs sub-tabs."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Which display the View sub-tab shows: the latest answer or an
        # opened file. `GuiApp` sets this via `switch_mode`.
        self.current_mode: str = MODE_LATEST
        # Late-bound — populated by build().
        self.view_container: ft.Container | None = None
        # The promoted per-query settings surfaces (own their own state).
        self.retrieval_tab = RetrievalTab(app)
        self.llm_tab = LlmTab(app)

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        self.view_container = ft.Container(
            content=self._build_view_for_mode(self.current_mode),
            expand=True,
            padding=12,
        )
        # View sub-tab body: the result display + a Save / Open action row.
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
        sub_bodies = ft.TabBarView(
            controls=[
                ft.Container(content=view_body, padding=8, expand=True),
                ft.Container(content=self.retrieval_tab.build(), padding=8, expand=True),
                ft.Container(content=self.llm_tab.build(), padding=8, expand=True),
            ],
            expand=True,
        )
        return ft.Tabs(
            length=len(SUB_TAB_LABELS),
            selected_index=0,
            content=ft.Column(controls=[sub_bar, sub_bodies], expand=True, spacing=0),
            expand=True,
        )

    def switch_mode(self, mode: str) -> None:
        """Swap the View sub-tab between the latest answer and an opened file.

        `GuiApp` calls this after a query (`MODE_LATEST` to show the new
        answer), on Open Result (`MODE_FILE`), and on Clear. Retrieval / LLM
        are separate sub-tabs, not modes, so they're untouched here.
        """
        self.current_mode = mode
        if self.view_container is not None:
            self.view_container.content = self._build_view_for_mode(mode)
        self.app.page.update()

    # ----- view dispatch ---------------------------------------------------

    def _build_view_for_mode(self, mode: str) -> ft.Control:
        if mode == MODE_FILE and self.app.loaded_file is not None:
            return FileView(
                name=self.app.loaded_file.name,
                content=self.app.loaded_file.content,
            ).build()
        # MODE_LATEST — and MODE_FILE falls through here when the loaded file
        # was wiped (e.g. Clear with keep_loaded_file_on_clear off).
        return LatestView(
            self.app.last_answer,
            self.app.last_query or "",
            page=self.app.page,
        ).build()

    # ----- buttons ---------------------------------------------------------

    def _view_button_row(self) -> ft.Control:
        """Save / Open actions for the current result. (View is the sub-tab
        itself now, so there's no 'View Result' button; Settings and Info are
        top-level tabs, so they're gone from here too.)"""
        return ft.Row(
            controls=[
                ft.Button(
                    content=centered_label("Save Result"),
                    expand=True,
                    on_click=self.app.on_save_answer,
                ),
                ft.Button(
                    content=centered_label("Open Result"),
                    expand=True,
                    on_click=self.app.on_open_result,
                ),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
