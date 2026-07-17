"""Shared renderer for shipped markdown reference docs.

One place that every static-doc surface renders through — the Metrics Guide and
all the Info tabs — so they look and behave identically. It:

1. Loads a `.md` file from disk (re-read on every `build()`, so an edited doc
   shows on the next Refresh without a restart).
2. Splits it at its `<a id="…"></a>` anchors into keyed sections. That strips the
   raw anchor tags (Flet's `ft.Markdown` would otherwise show `<a id="…">` as
   literal text) and turns each anchor into a `ScrollKey` so in-page links work.
3. Renders each section through the shared `gui/_markdown.render_markdown` (themed
   to match VS Code + GFM tables as native controls).
4. Wires `#anchor` link taps to scroll the body to that section (Flet's markdown
   has no built-in in-page anchor jump); any real URL opens in the browser.

Callers wrap the returned body in whatever frame they need (a rail, a header, a
sub-tab). This widget owns only the scrollable doc body.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._markdown import render_markdown
from knowledge_agent.gui.views._frame import empty_state

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp

# Each `<a id="X"></a>` starts a keyed section; the tag itself is removed from the
# rendered text (Flet would show it literally) and reused as the scroll key.
_ANCHOR_RE = re.compile(r'<a id="([^"]+)"></a>\s*')


def docs_dir() -> Path:
    """Directory of the shipped Info / reference markdown docs — the single place
    every Info surface (global tab + contextual sub-tabs) resolves doc paths."""
    import knowledge_agent

    return Path(knowledge_agent.__file__).parent / "docs"


def _split_anchored(text: str) -> list[tuple[str | None, str]]:
    """Split the doc into (anchor-or-None, chunk) segments at each `<a id>` tag
    (the tag is dropped). The leading text before the first anchor carries no
    key; every later segment is keyed by its anchor so a link can scroll to it."""
    parts = _ANCHOR_RE.split(text)
    segments: list[tuple[str | None, str]] = [(None, parts[0])]
    for i in range(1, len(parts), 2):
        segments.append((parts[i], parts[i + 1]))
    return segments


class InfoDoc:
    """Renders one shipped markdown doc as a scrollable, anchor-navigable body."""

    def __init__(self, app: GuiApp, path: Path, *, missing_hint: str | None = None) -> None:
        self.app = app
        self.path = path
        self.missing_hint = missing_hint
        # The scrollable doc column (scroll target). A Column — NOT a ListView:
        # `scroll_to(scroll_key=…)` is documented as ineffective for controls that
        # build items dynamically (ListView / GridView); a Column renders all its
        # children, so scroll-to-key works.
        self._scroll: ft.Column | None = None

    def build(self) -> ft.Control:
        """Build (or rebuild) the doc body from disk. Missing file → a hint."""
        if not self.path.exists():
            return empty_state(self.missing_hint or f"Doc not found at {self.path}.")
        controls: list[ft.Control] = []
        for anchor, chunk in _split_anchored(self.path.read_text(encoding="utf-8")):
            if not chunk.strip():
                continue
            # Shared renderer: themed prose + GFM tables as native controls. Links
            # are wired to the in-page anchor-scroll handler.
            content = render_markdown(chunk, on_tap_link=self._on_tap_link)
            # An anchored section becomes a keyed Container so a link can scroll to
            # it; the leading (pre-anchor) chunk is rendered as-is.
            controls.append(
                ft.Container(key=ft.ScrollKey(anchor), content=content) if anchor else content
            )
        # STRETCH so the native tables (whose body columns fill + wrap) span the
        # full width instead of collapsing to their content.
        self._scroll = ft.Column(
            controls=controls,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return self._scroll

    async def _on_tap_link(self, e: ft.Event) -> None:
        """A `#anchor` tap scrolls the doc to that section — Flet's markdown has no
        built-in in-page anchor jump, so we scroll to the section's `ScrollKey`.
        Any real URL opens in the browser.

        Async on purpose: `ScrollableControl.scroll_to` is a coroutine, so an
        un-awaited call would be silently discarded. Flet awaits async event
        handlers, so this just works."""
        href = getattr(e, "data", None) or getattr(e, "url", None) or ""
        if href.startswith("#"):
            if self._scroll is not None:
                await self._scroll.scroll_to(scroll_key=href[1:], duration=300)
        elif href:
            self.app.page.launch_url(href)
