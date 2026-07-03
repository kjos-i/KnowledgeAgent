"""Right column of the Search tab: mode-switching display area + a
single button row underneath.

Owns:
- The current mode (`MODE_LATEST` / `MODE_FILE` / `MODE_SETTINGS` /
  `MODE_INFO`).
- The view container whose content swaps as the mode changes.
- One button row mixing mode switchers and the Save Answer action:
  [View result] [Save Answer] [Open Result] [Settings] [Info].
- Mode-button highlight tracking.

Per-mode views are rebuilt on switch (lazy + keeps the controls owned
by the active view's lifecycle). The highlight on the current mode
button tracks `current_mode`.

Settings + Info are stubs in slice 1 (empty-state). Slice 2 fills
the Settings view; Info ships as static help text. Library and
Evaluation are top-level tabs, NOT right-panel modes — they need
the full window.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import (
    ACTIVE_BG,
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)
from knowledge_agent.gui.settings import SettingsView
from knowledge_agent.gui.views._frame import empty_state, view_with_header
from knowledge_agent.gui.views.file_view import FileView
from knowledge_agent.gui.views.latest_view import LatestView

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


MODE_LATEST = "latest"
MODE_FILE = "file"
MODE_SETTINGS = "settings"
MODE_INFO = "info"


class RightPanel:
    """Right column of the Search tab — mode-switching view + buttons."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.current_mode: str = MODE_LATEST
        # Late-bound — populated by build().
        self.view_container: ft.Container | None = None
        self.open_result_button: ft.Button | None = None
        self.mode_buttons: dict[str, ft.Button] = {}
        # Cached Settings view — building the full sub-tab tree (~150
        # widgets) is slow enough to feel laggy on every mode switch.
        # Cached on first MODE_SETTINGS visit, then re-displayed
        # instantly. Side benefit: sub-tab state (which sub-tab the
        # user last viewed, mid-edit field contents) survives mode
        # switches.
        self._settings_view: SettingsView | None = None
        self._settings_built: ft.Control | None = None

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        self.view_container = ft.Container(
            content=self._build_view_for_mode(self.current_mode),
            expand=True,
            padding=12,
        )

        # Single button row: mode switchers + Save Answer action mixed.
        # Order chosen so the most-used buttons (view + save) sit on
        # the left where the user reaches first.
        button_row = ft.Row(
            controls=[
                self._mode_button("View Result", MODE_LATEST),
                ft.Button(
                    content=centered_label("Save Result"), expand=True,
                    on_click=self.app.on_save_answer,
                ),
                self._open_result_button(),
                self._mode_button("Settings", MODE_SETTINGS),
                self._mode_button("Info", MODE_INFO),
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        inner = ft.Column(
            controls=[self.view_container, button_row],
            expand=True,
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self.update_button_highlights()
        return ft.Container(
            content=inner,
            padding=12,
            bgcolor=PANEL_BG,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=8,
            expand=True,
        )

    def switch_mode(self, mode: str) -> None:
        """Change mode, refresh the container content, repaint highlights."""
        self.current_mode = mode
        if self.view_container is not None:
            self.view_container.content = self._build_view_for_mode(mode)
        self.update_button_highlights()
        self.app.page.update()

    def update_button_highlights(self) -> None:
        """Tint the mode button matching the current mode."""
        for mode, btn in self.mode_buttons.items():
            btn.bgcolor = ACTIVE_BG if mode == self.current_mode else None

    # ----- view dispatch ---------------------------------------------------

    def _build_view_for_mode(self, mode: str) -> ft.Control:
        if mode == MODE_LATEST:
            return LatestView(
                self.app.last_answer, self.app.last_query or "",
                page=self.app.page,
            ).build()
        if mode == MODE_FILE:
            if self.app.loaded_file is None:
                # Loaded-file wiped (e.g. Clear with keep_loaded_file_on_clear
                # off) — silently fall through to the Latest view.
                return LatestView(
                    self.app.last_answer, self.app.last_query or "",
                ).build()
            return FileView(
                name=self.app.loaded_file.name,
                content=self.app.loaded_file.content,
            ).build()
        if mode == MODE_SETTINGS:
            if self._settings_built is None:
                self._settings_view = SettingsView(self.app)
                self._settings_built = self._settings_view.build()
            return self._settings_built
        if mode == MODE_INFO:
            return view_with_header(
                "Information",
                empty_state(
                    "Info view lands in slice 2 with the static help "
                    "text describing every setting + workflow."
                ),
            )
        return LatestView(
            self.app.last_answer, self.app.last_query or "",
        ).build()

    # ----- button builders -------------------------------------------------

    def _mode_button(self, label: str, mode: str) -> ft.Button:
        btn = ft.Button(
            content=centered_label(label), expand=True,
            on_click=lambda e, m=mode: self.switch_mode(m),
        )
        self.mode_buttons[mode] = btn
        return btn

    def _open_result_button(self) -> ft.Button:
        btn = ft.Button(
            content=centered_label("Open Result"), expand=True,
            on_click=self.app.on_open_result,
        )
        # Registered as the mode switcher for MODE_FILE so its highlight
        # tracks when the user lands on an opened file.
        self.mode_buttons[MODE_FILE] = btn
        self.open_result_button = btn
        return btn
