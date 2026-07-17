"""Log top-level tab — read the app's rotating log in-app.

Shows the resolved log path (`logging_setup.log_file_path()`) plus the tail of the
current `kagent.log` (last `_TAIL_LINES` lines), with **Refresh** (logs are live)
and **Open folder** (reveal the log dir in the OS file manager) actions. The
rotated `.gz` archives are NOT stitched in — Open folder gets you to them.

Reads the file on `build()` / Refresh, never at construct (GUI startup rule).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import deque
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import FRAME_BORDER_COLOR, centered_label
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.logging_setup import log_file_path

if TYPE_CHECKING:
    from pathlib import Path

    from knowledge_agent.gui.app import GuiApp

logger = logging.getLogger(__name__)

# How many trailing lines of the current log to show. Bounded so a large log
# doesn't flood the view; the whole file is still on disk (Open folder).
_TAIL_LINES = 500


def _read_tail(path: Path, n: int) -> tuple[str, int]:
    """Return `(last-n-lines-text, total-line-count)` for `path`.

    `("", 0)` when the file doesn't exist yet. Reads the whole (rotation-bounded)
    file once, keeping only the last `n` lines in memory."""
    if not path.is_file():
        return "", 0
    total = 0
    tail: deque[str] = deque(maxlen=n)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                total += 1
                tail.append(line)
    except OSError as exc:
        logger.warning("could not read log %s: %r", path, exc)
        return f"(could not read log: {exc})", 0
    return "".join(tail), total


def _reveal_in_file_manager(path: Path) -> None:
    """Open `path` (a directory) in the OS file manager."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # opening a dir we resolved ourselves
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        logger.warning("could not open log folder %s: %r", path, exc)


class LogTab:
    """App-wide Log tab — shows the log path + the tail of the current log."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self._path_text: ft.Text | None = None
        self._status: ft.Text | None = None
        self._body: ft.Text | None = None

    def build(self) -> ft.Control:
        self._path_text = ft.Text("", size=12, selectable=True, color=ft.Colors.GREY_400)
        self._status = ft.Text("", size=12, color=ft.Colors.GREY_500)
        self._body = ft.Text("", size=12, font_family="Consolas", selectable=True)
        self._load()  # first-show read (deferred off construct)

        actions = ft.Row(
            controls=[
                ft.Button(content=centered_label("Refresh"), on_click=self._on_refresh),
                ft.Button(content=centered_label("Open folder"), on_click=self._on_open_folder),
            ],
            spacing=8,
        )
        log_view = ft.Container(
            content=ft.Column(controls=[self._body], scroll=ft.ScrollMode.AUTO, expand=True),
            expand=True,
            padding=8,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=4,
        )
        return ft.Column(
            controls=[
                view_header("Log"),
                ft.Row(
                    controls=[
                        ft.Text("File:", size=12, weight=ft.FontWeight.BOLD),
                        self._path_text,
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                actions,
                self._status,
                log_view,
            ],
            expand=True,
            spacing=8,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def refresh(self) -> None:
        """Re-read the log (e.g. after the tab is re-selected)."""
        self._load()
        self.app.page.update()

    # ----- internals -------------------------------------------------------

    def _load(self) -> None:
        path = log_file_path()
        if self._path_text is not None:
            self._path_text.value = str(path)
        text, total = _read_tail(path, _TAIL_LINES)
        if self._body is not None:
            self._body.value = text or "(no log written yet)"
        if self._status is not None:
            if total:
                shown = min(total, _TAIL_LINES)
                self._status.value = f"Showing the last {shown} of {total} lines."
            else:
                self._status.value = "No log entries yet."

    def _on_refresh(self, _e: ft.Event) -> None:
        self.refresh()

    def _on_open_folder(self, _e: ft.Event) -> None:
        _reveal_in_file_manager(log_file_path().parent)
