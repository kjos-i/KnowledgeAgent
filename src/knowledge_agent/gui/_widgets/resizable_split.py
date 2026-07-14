"""Two-pane container with a draggable divider.

Supports both **horizontal** (first pane left, second right — divider
resizes the first pane's width) and **vertical** (first pane top,
second bottom — divider resizes the first pane's height).

Usage — horizontal:

    split = ResizableSplit(
        page=self.app.page,
        first=left_content,
        second=right_content,
        orientation="horizontal",
        initial_first_size=480,
        min_first_size=320,
        max_first_size=900,
    )
    return split.build()

Vertical works the same, just with "vertical" orientation + size
values interpreted as heights.

State (`_current_size`) lives on the instance for the session — not
persisted to disk. Persist to GuiConfig if user demand justifies it.

Close-button / collapse-to-strip pattern is deferred — see [[gui-slice3-library-design]]'s open questions for the discussion.
"""

from __future__ import annotations

from typing import Literal

import flet as ft

Orientation = Literal["horizontal", "vertical"]

# Pill-handle grip: a thin rounded capsule that brightens on hover, inset by
# `_PILL_END_GAP` px at each end so it stops around the neighbouring box's
# rounded corners (radius 8) instead of overrunning them. Tune `_PILL_END_GAP`:
# bigger = shorter pill (more corner clearance), smaller = longer (toward the
# corners). `_TRACK_COLOR` is a faint full-length seam that keeps the width grabbable.
_PILL_THICKNESS = 4
_PILL_END_GAP = 6
_PILL_COLOR = ft.Colors.GREY_500
_PILL_HOVER = ft.Colors.GREY_300
_TRACK_COLOR = ft.Colors.with_opacity(0.08, ft.Colors.WHITE)


class ResizableSplit:
    """Two children joined by a draggable divider."""

    def __init__(
        self,
        page: ft.Page,
        first: ft.Control,
        second: ft.Control,
        orientation: Orientation = "horizontal",
        initial_first_size: int = 480,
        min_first_size: int = 200,
        max_first_size: int = 1200,
        divider_thickness: int = 6,
    ) -> None:
        self.page = page
        self.first = first
        self.second = second
        self.orientation = orientation
        self._current_size = initial_first_size
        self._min_size = min_first_size
        self._max_size = max_first_size
        self._divider_thickness = divider_thickness
        # Late-bound in build().
        self._first_container: ft.Container | None = None

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        """Assemble first + divider + second into a Row or Column."""
        if self.orientation == "horizontal":
            self._first_container = ft.Container(
                content=self.first,
                width=self._current_size,
            )
            second_container = ft.Container(content=self.second, expand=True)
            divider = self._build_horizontal_divider()
            return ft.Row(
                controls=[self._first_container, divider, second_container],
                spacing=0,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        # vertical
        self._first_container = ft.Container(
            content=self.first,
            height=self._current_size,
        )
        second_container = ft.Container(content=self.second, expand=True)
        divider = self._build_vertical_divider()
        return ft.Column(
            controls=[self._first_container, divider, second_container],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    # ----- divider builders + drag handlers --------------------------------

    def _build_horizontal_divider(self) -> ft.Control:
        pill = _pill(vertical=True)
        bar = ft.Container(
            width=self._divider_thickness,
            bgcolor=_TRACK_COLOR,
            padding=ft.Padding(0, _PILL_END_GAP, 0, _PILL_END_GAP),
            content=ft.Column(
                [pill], expand=True, horizontal_alignment=ft.CrossAxisAlignment.CENTER
            ),
        )
        bar.on_hover = lambda e: self._hover_pill(pill, e)
        return ft.GestureDetector(
            content=bar,
            drag_interval=10,
            on_horizontal_drag_update=self._on_horizontal_drag,
            mouse_cursor=ft.MouseCursor.RESIZE_LEFT_RIGHT,
        )

    def _build_vertical_divider(self) -> ft.Control:
        pill = _pill(vertical=False)
        bar = ft.Container(
            height=self._divider_thickness,
            bgcolor=_TRACK_COLOR,
            padding=ft.Padding(_PILL_END_GAP, 0, _PILL_END_GAP, 0),
            content=ft.Row([pill], expand=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        )
        bar.on_hover = lambda e: self._hover_pill(pill, e)
        return ft.GestureDetector(
            content=bar,
            drag_interval=10,
            on_vertical_drag_update=self._on_vertical_drag,
            mouse_cursor=ft.MouseCursor.RESIZE_UP_DOWN,
        )

    def _hover_pill(self, pill: ft.Container, e: ft.Event) -> None:
        """Brighten the pill grip while the pointer is over the divider, so it
        reads as an interactive handle."""
        pill.bgcolor = _PILL_HOVER if e.data == "true" else _PILL_COLOR
        self.page.update()

    def _on_horizontal_drag(self, e: ft.DragUpdateEvent) -> None:
        delta = e.primary_delta or 0
        new_size = _clamp(
            int(self._current_size + delta),
            self._min_size,
            self._max_size,
        )
        if new_size == self._current_size or self._first_container is None:
            return
        self._current_size = new_size
        self._first_container.width = new_size
        self.page.update()

    def _on_vertical_drag(self, e: ft.DragUpdateEvent) -> None:
        delta = e.primary_delta or 0
        new_size = _clamp(
            int(self._current_size + delta),
            self._min_size,
            self._max_size,
        )
        if new_size == self._current_size or self._first_container is None:
            return
        self._current_size = new_size
        self._first_container.height = new_size
        self.page.update()


def _pill(*, vertical: bool) -> ft.Container:
    """A thin rounded capsule 'grip' that expands to fill the divider between the
    builders' fixed end gaps. `vertical=True` is a tall pill (left/right panes);
    False is a wide pill (top/bottom panes)."""
    return ft.Container(
        width=_PILL_THICKNESS if vertical else None,
        height=_PILL_THICKNESS if not vertical else None,
        bgcolor=_PILL_COLOR,
        border_radius=_PILL_THICKNESS,
        expand=True,
    )


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))
