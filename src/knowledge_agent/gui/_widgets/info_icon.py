"""Reusable `(i)` info icons — the house style for contextual help.

Press one and a small dialog explains the setting or feature it sits
next to. There are THREE independent tiers, each with its own on/off
toggle (Settings -> App -> App behaviour):

  - **standard** (blue `(i)`)   — the default note. `gui_config.show_info_icons`
  - **beginner** (green cap)    — gentle, plain-language help for new users.
                                  `gui_config.show_beginner_info`
  - **technical** (orange wrench) — deeper detail for power users (env vars,
                                  backend behaviour). `gui_config.show_technical_info`

A help point provides text only for the tiers it needs; each tier that
has text AND is toggled on renders as its own small icon, shown next to
the others. All tiers off -> a clean, expert UI.

ALWAYS build info icons with `info_icon(app, title=..., text=..., beginner=...,
technical=...)` — never hand-roll an icon + dialog. That keeps the look, the
dialog shape, and the global show/hide behaviour in ONE place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import flet as ft

from knowledge_agent.gui._styles import section_title

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


class _Tier(NamedTuple):
    icon: str
    color: str
    label: str  # tooltip + the caption shown at the top of the help dialog


# The three tiers, in display order (standard first). `flag` is the
# `GuiConfig` attribute that gates the tier's visibility.
_TIERS: dict[str, _Tier] = {
    "standard": _Tier(ft.Icons.INFO_OUTLINE, ft.Colors.BLUE_300, "What's this?"),
    "beginner": _Tier(ft.Icons.SCHOOL_OUTLINED, ft.Colors.GREEN_300, "Beginner help"),
    "technical": _Tier(ft.Icons.BUILD_OUTLINED, ft.Colors.ORANGE_300, "Technical detail"),
}
_TIER_FLAGS: dict[str, str] = {
    "standard": "show_info_icons",
    "beginner": "show_beginner_info",
    "technical": "show_technical_info",
}


def _tier_icon(app: GuiApp, tier: str, *, title: str, text: str) -> ft.IconButton:
    """One tier's `(i)` button — opens a help dialog, self-registers so the
    per-tier toggle can flip it live, starts hidden if its tier is off."""
    spec = _TIERS[tier]

    def _open(_e: ft.Event) -> None:
        dialog = ft.AlertDialog(
            modal=True,
            # scrollable so a long body (e.g. a combined multi-control note)
            # scrolls within the dialog instead of clipping off-screen.
            scrollable=True,
            title=ft.Text(title, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                width=440,
                content=ft.Column(
                    tight=True,
                    spacing=6,
                    controls=[
                        ft.Text(spec.label, size=12, italic=True, color=spec.color),
                        ft.Text(text, size=13, selectable=True),
                    ],
                ),
            ),
            actions=[ft.TextButton("Close", on_click=lambda _c: app.page.pop_dialog())],
        )
        app.page.show_dialog(dialog)
        app.page.update()

    button = ft.IconButton(
        icon=spec.icon,
        icon_size=16,
        icon_color=spec.color,
        tooltip=spec.label,
        on_click=_open,
        visible=getattr(app.gui_config, _TIER_FLAGS[tier]),
    )
    app.register_info_icon(button, tier=tier)
    return button


def info_icon(
    app: GuiApp,
    *,
    title: str,
    text: str | None = None,
    beginner: str | None = None,
    technical: str | None = None,
) -> ft.Control:
    """Return the help icon(s) for one point, laid out next to each other.

    Give `text` for the standard note (blue), `beginner` for the plain-language
    tier (green), and/or `technical` for the power-user tier (orange). A tier
    with no text is simply omitted, so most points still show just the one
    standard `(i)`. Each tier's icon shows only when its Settings toggle is on.
    """
    tier_text = {"standard": text, "beginner": beginner, "technical": technical}
    icons = [
        _tier_icon(app, tier, title=title, text=body) for tier, body in tier_text.items() if body
    ]
    return ft.Row(icons, spacing=2, tight=True)


@dataclass(frozen=True)
class InfoText:
    """All the text for one help point, across tiers — the shape stored in
    `info_text.INFO`. `title` is the dialog header; `standard` / `beginner` /
    `technical` are the three tier bodies (any may be None). Render one with
    `info_point(app, spec)`, or from the registry via `info_text.info(app, key)`."""

    title: str
    standard: str | None = None
    beginner: str | None = None
    technical: str | None = None


def info_point(app: GuiApp, spec: InfoText) -> ft.Control:
    """Render the (i) icon row for an `InfoText` spec — the keyed path used by
    `info_text.info(app, key)`. Same as `info_icon` with the spec spread out."""
    return info_icon(
        app,
        title=spec.title,
        text=spec.standard,
        beginner=spec.beginner,
        technical=spec.technical,
    )


def section_header(
    app: GuiApp,
    title: str,
    text: str | None = None,
    *,
    beginner: str | None = None,
    technical: str | None = None,
) -> ft.Control:
    """A bold `section_title` beside its (i) help icon(s) — the app-wide
    prose → (i) pattern for section headings. `text` is the standard note,
    `beginner` the plain-language tier, `technical` the power-user tier. With
    none of them, it's just the title."""
    controls: list[ft.Control] = [section_title(title)]
    if text is not None or beginner is not None or technical is not None:
        controls.append(
            info_icon(app, title=title, text=text, beginner=beginner, technical=technical)
        )
    return ft.Row(controls, spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def tier_icon_sample(tier: str, *, size: int = 16) -> ft.Icon:
    """A static (non-pressable) sample of a tier's (i) icon — a legend chip
    shown beside the Settings toggles so users can see what each tier looks
    like (task 51)."""
    spec = _TIERS[tier]
    return ft.Icon(spec.icon, color=spec.color, size=size)
