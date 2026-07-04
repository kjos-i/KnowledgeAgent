"""Shared GUI test fixtures — a fake `GuiApp` + fake `ft.Page` so
panel/view tests can build their controls without launching Flet.

Flet's `Page` does NOT need to be a real one for static control
construction; `MagicMock` with a callable `.update()` is enough.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_page() -> MagicMock:
    """A MagicMock standing in for `ft.Page` — `.update()` and
    `.add()` are callable; nothing else is exercised by the panel
    tests."""
    page = MagicMock(name="ft.Page")
    return page


@pytest.fixture
def fake_app(fake_page: MagicMock) -> MagicMock:
    """A minimal GuiApp stand-in: a MagicMock pre-populated with the
    handler methods + attributes that panels read.

    Panels call back into `app.on_send`, `app.on_stop`, etc. — those
    just need to be callable; the panel tests don't exercise their
    behaviour (that lives in `test_app.py`).
    """
    app = MagicMock(name="GuiApp")
    app.page = fake_page
    app.last_answer = None
    app.last_query = None
    app.loaded_file = None
    # Persist handler attributes as plain callables so panel build()
    # methods can wire on_click bindings without surprises.
    for handler in (
        "on_send",
        "on_stop",
        "on_clear",
        "on_save_chat",
        "on_save_answer",
        "on_open_result",
    ):
        setattr(app, handler, MagicMock(name=handler))
    return app
