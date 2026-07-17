"""Unit tests for the ResizableSplit two-pane widget.

Pure geometry / drag-handler logic. The `fake_page` fixture (a MagicMock whose
`.update()` is a no-op) stands in for `ft.Page`; drag/hover events are
lightweight `SimpleNamespace` stand-ins exposing only the attributes the
handlers read (`primary_delta`, `data`).
"""

from __future__ import annotations

from types import SimpleNamespace

import flet as ft

from knowledge_agent.gui._widgets import resizable_split as rs

# ----- _clamp ---------------------------------------------------------------


def test_clamp_within_range_returns_value():
    assert rs._clamp(50, 0, 100) == 50


def test_clamp_below_min_returns_min():
    assert rs._clamp(-5, 0, 100) == 0


def test_clamp_above_max_returns_max():
    assert rs._clamp(150, 0, 100) == 100


def test_clamp_at_bounds_inclusive():
    assert rs._clamp(0, 0, 100) == 0
    assert rs._clamp(100, 0, 100) == 100


# ----- _pill ----------------------------------------------------------------


def test_pill_vertical_is_thin_column_grip():
    pill = rs._pill(vertical=True)
    assert pill.width == rs._PILL_THICKNESS
    assert pill.height is None
    assert pill.expand is True


def test_pill_horizontal_is_thin_row_grip():
    pill = rs._pill(vertical=False)
    assert pill.height == rs._PILL_THICKNESS
    assert pill.width is None
    assert pill.expand is True


# ----- build() --------------------------------------------------------------


def _split(fake_page, orientation="horizontal", **kw):
    return rs.ResizableSplit(
        page=fake_page,
        first=ft.Text("first"),
        second=ft.Text("second"),
        orientation=orientation,
        **kw,
    )


def test_build_horizontal_is_row_with_width_sized_first_pane(fake_page):
    split = _split(fake_page, "horizontal", initial_first_size=480)
    root = split.build()
    assert isinstance(root, ft.Row)
    assert root.controls[0] is split._first_container
    assert split._first_container.width == 480
    assert isinstance(root.controls[1], ft.GestureDetector)
    assert root.controls[2].expand is True


def test_build_vertical_is_column_with_height_sized_first_pane(fake_page):
    split = _split(fake_page, "vertical", initial_first_size=300)
    root = split.build()
    assert isinstance(root, ft.Column)
    assert root.controls[0] is split._first_container
    assert split._first_container.height == 300
    assert isinstance(root.controls[1], ft.GestureDetector)


# ----- horizontal drag ------------------------------------------------------


def test_horizontal_drag_grows_first_pane_width(fake_page):
    split = _split(
        fake_page, "horizontal", initial_first_size=480, min_first_size=200, max_first_size=900
    )
    split.build()
    split._on_horizontal_drag(SimpleNamespace(primary_delta=50))
    assert split._current_size == 530
    assert split._first_container.width == 530
    fake_page.update.assert_called_once()


def test_horizontal_drag_clamps_at_max(fake_page):
    split = _split(
        fake_page, "horizontal", initial_first_size=880, min_first_size=200, max_first_size=900
    )
    split.build()
    split._on_horizontal_drag(SimpleNamespace(primary_delta=50))
    assert split._current_size == 900
    assert split._first_container.width == 900


def test_horizontal_drag_clamps_at_min(fake_page):
    split = _split(
        fake_page, "horizontal", initial_first_size=210, min_first_size=200, max_first_size=900
    )
    split.build()
    split._on_horizontal_drag(SimpleNamespace(primary_delta=-50))
    assert split._current_size == 200
    assert split._first_container.width == 200


def test_horizontal_drag_noop_when_size_unchanged(fake_page):
    split = _split(
        fake_page, "horizontal", initial_first_size=900, min_first_size=200, max_first_size=900
    )
    split.build()
    split._on_horizontal_drag(SimpleNamespace(primary_delta=50))  # already at max
    assert split._current_size == 900
    fake_page.update.assert_not_called()


def test_horizontal_drag_none_delta_is_noop(fake_page):
    split = _split(fake_page, "horizontal", initial_first_size=480)
    split.build()
    split._on_horizontal_drag(SimpleNamespace(primary_delta=None))
    assert split._current_size == 480
    fake_page.update.assert_not_called()


# ----- vertical drag --------------------------------------------------------


def test_vertical_drag_grows_first_pane_height(fake_page):
    split = _split(
        fake_page, "vertical", initial_first_size=300, min_first_size=100, max_first_size=600
    )
    split.build()
    split._on_vertical_drag(SimpleNamespace(primary_delta=40))
    assert split._current_size == 340
    assert split._first_container.height == 340
    fake_page.update.assert_called_once()


def test_vertical_drag_clamps_at_min(fake_page):
    split = _split(
        fake_page, "vertical", initial_first_size=110, min_first_size=100, max_first_size=600
    )
    split.build()
    split._on_vertical_drag(SimpleNamespace(primary_delta=-50))
    assert split._current_size == 100
    assert split._first_container.height == 100


# ----- hover ----------------------------------------------------------------


def test_hover_brightens_and_restores_pill(fake_page):
    split = _split(fake_page, "horizontal")
    pill = rs._pill(vertical=True)
    split._hover_pill(pill, SimpleNamespace(data="true"))
    assert pill.bgcolor == rs._PILL_HOVER
    split._hover_pill(pill, SimpleNamespace(data="false"))
    assert pill.bgcolor == rs._PILL_COLOR
    assert fake_page.update.call_count == 2
