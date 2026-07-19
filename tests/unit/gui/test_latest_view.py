"""Tests for the Latest result view (views/latest_view).

Focus: the multimodal figure-thumbnail path. It builds real Flet controls
(`ft.Image(fit=...)`, `ft.Container(border=...)`) that must reference valid
Flet 0.85 symbols. This path only fires when a search result carries a
`content_type='figure'` chunk with an `image_ref`, so a stale Flet API here
(e.g. `ft.ImageFit` / `ft.border.all`, both removed in 0.85) stays dormant
until someone views a figure result — exactly the kind of gap a construct-it
test catches cheaply.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views.latest_view import LatestView
from knowledge_agent.models import ChunkSource

if TYPE_CHECKING:
    from unittest.mock import MagicMock


def _figure_source() -> ChunkSource:
    """A figure ChunkSource with an image_ref (the path need not exist —
    Flet Image construction is lazy and doesn't read the file)."""
    return ChunkSource(
        chunk_id="c1",
        doc_id="doc123456789abc",
        quote="A figure caption",
        content_type="figure",
        image_ref="/tmp/figure-does-not-need-to-exist.png",
        page=3,
    )


def test_build_thumbnail_uses_valid_flet_apis() -> None:
    """_build_thumbnail builds an ft.Image (fit=) wrapped in an ft.Container
    (border=). Both must use real Flet 0.85 symbols; a stale ft.ImageFit or
    ft.border.all raises AttributeError while constructing the control."""
    view = LatestView(answer=None, query="q", page=None)
    control = view._build_thumbnail(0, _figure_source())
    assert control is not None


def test_show_full_image_dialog_uses_valid_flet_apis(fake_page: MagicMock) -> None:
    """The click-to-expand modal builds another ft.Image(fit=...). Needs a
    non-None page (else it early-returns before building). Asserts the dialog
    is constructed + shown without an AttributeError from a stale Flet API."""
    view = LatestView(answer=None, query="q", page=fake_page)
    view._show_full_image_dialog(_figure_source())
    fake_page.show_dialog.assert_called_once()
