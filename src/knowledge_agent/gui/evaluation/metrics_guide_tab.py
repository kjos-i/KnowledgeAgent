"""Evaluation → Metrics Guide sub-tab — the metrics reference.

Renders the harness's `info_metrics.md` via `ft.Markdown` so the in-app
reference never drifts from the source doc. The doc lives beside the harness
(`knowledge_agent/evaluation/info_metrics.md`) — the single source that the
CLI/docs can share too.

The doc is split at its `<a id="…"></a>` anchors so each section becomes its
own keyed control. That does two things Flet's `ft.Markdown` can't on its own:
(1) it strips the raw anchor tags — Flet would otherwise render `<a id="…">`
as literal visible text — and (2) it makes the glance-table links actually
work: an `#anchor` tap scrolls the body to the matching `ScrollKey` (Flet's
markdown has no built-in in-page anchor navigation). A `MarkdownStyleSheet`
bumps the body text one size up from Flet's small default. The `.md` keeps the
anchor tags so it still jumps when viewed on GitHub / a normal markdown viewer.

Has a left rail matching the other Evaluation tabs (Refresh reloads the doc +
a short summary), so the four tabs share the same two-column layout.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.evaluation._dashboard_rail import DashboardRail
from knowledge_agent.gui.views._frame import empty_state, view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

# Body text one size up from Flet markdown's small default (still inside the
# tab's 12–16 range), applied to paragraphs, links, lists, emphasis, and table
# cells so the whole doc reads at one consistent, legible size.
_GUIDE_TEXT_SIZE = 15
_MD_STYLE = ft.MarkdownStyleSheet(
    p_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE),
    a_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE, color=ft.Colors.BLUE_400),
    strong_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE, weight=ft.FontWeight.BOLD),
    em_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE, italic=True),
    list_bullet_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE),
    table_head_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE, weight=ft.FontWeight.BOLD),
    table_body_text_style=ft.TextStyle(size=_GUIDE_TEXT_SIZE),
)

# Each `<a id="X"></a>` starts a keyed section; the tag itself is removed from
# the rendered text (Flet would show it literally) and reused as the scroll key.
_ANCHOR_RE = re.compile(r'<a id="([^"]+)"></a>\s*')


def _guide_path() -> Path:
    import knowledge_agent.evaluation as ka_eval

    return Path(ka_eval.__file__).parent / "info_metrics.md"


def _split_anchored(text: str) -> list[tuple[str | None, str]]:
    """Split the doc into (anchor-or-None, chunk) segments at each `<a id>` tag
    (the tag is dropped). The leading text before the first anchor carries no
    key; every later segment is keyed by its anchor so a link can scroll to it."""
    parts = _ANCHOR_RE.split(text)
    segments: list[tuple[str | None, str]] = [(None, parts[0])]
    for i in range(1, len(parts), 2):
        segments.append((parts[i], parts[i + 1]))
    return segments


class MetricsGuideTab:
    """Static metrics-reference sub-tab (renders info_metrics.md)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.rail: DashboardRail | None = None
        self.body: ft.Container | None = None
        # The scrollable doc column (scroll target). A Column — NOT a ListView:
        # `scroll_to(scroll_key=…)` is documented as ineffective for controls that
        # build items dynamically (ListView / GridView); a Column renders all its
        # children, so scroll-to-key works.
        self._scroll: ft.Column | None = None

    def build(self) -> ft.Control:
        # The SAME shared rail as the other three view tabs (its selection isn't
        # used here — the guide is static — but the column stays consistent, and
        # its Refresh reloads the doc via `on_change`).
        self.rail = DashboardRail(self.app, self.coordinator, on_change=self._reload_guide)
        rail_ctl = self.rail.build()
        self.body = ft.Container(content=self._guide_body(), expand=True)
        return ft.Row(
            [
                rail_ctl,
                ft.Column(
                    [view_header("Metrics Guide"), self.body],
                    expand=True,
                    spacing=8,
                ),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def refresh(self) -> None:
        """Sync the shared rail (runs + selection). The guide body is static;
        it reloads via the rail's Refresh (`_reload_guide`)."""
        if self.rail is not None:
            self.rail.refresh()

    def _guide_body(self) -> ft.Control:
        path = _guide_path()
        if not path.exists():
            return empty_state(f"Metrics guide not found at {path}.")
        controls: list[ft.Control] = []
        for anchor, chunk in _split_anchored(path.read_text(encoding="utf-8")):
            if not chunk.strip():
                continue
            md = ft.Markdown(
                chunk,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                md_style_sheet=_MD_STYLE,
                on_tap_link=self._on_tap_link,
            )
            # An anchored section becomes a keyed Container so a link can scroll
            # to it; the leading (pre-anchor) chunk is rendered as-is.
            controls.append(ft.Container(key=ft.ScrollKey(anchor), content=md) if anchor else md)
        self._scroll = ft.Column(controls=controls, scroll=ft.ScrollMode.AUTO, expand=True)
        return self._scroll

    async def _on_tap_link(self, e: ft.Event) -> None:
        """A glance-table `#anchor` tap scrolls the doc to that section — Flet's
        markdown has no built-in in-page anchor jump, so we scroll to the section's
        `ScrollKey`. Any real URL opens in the browser.

        Async on purpose: `ScrollableControl.scroll_to` is a coroutine, so an
        un-awaited call would be silently discarded (the original bug). Flet
        awaits async event handlers, so this just works."""
        href = getattr(e, "data", None) or getattr(e, "url", None) or ""
        if href.startswith("#"):
            if self._scroll is not None:
                await self._scroll.scroll_to(scroll_key=href[1:], duration=300)
        elif href:
            self.app.page.launch_url(href)

    def _reload_guide(self) -> None:
        """Reload the guide doc from disk (it's the single source; may have been
        edited). Fired by the shared rail's Refresh."""
        if self.body is not None:
            self.body.content = self._guide_body()
            self.app.page.update()
