"""Shared visual constants used across the GUI panels and views.

Centralised so the panels + views don't drift on background / border /
highlight colours — page-wide concerns belong in one place.

Mirrors the ResearchArticlesAgent style module so the sibling apps
visually match. Bumping a value here moves both apps' look together if
we ever re-skin.
"""
import flet as ft


PANEL_BG = ft.Colors.BLACK
"""Panel background. Matches the dark Material 3 surface tone the rest
of the app expects."""

FRAME_BORDER_COLOR = ft.Colors.GREY_700
"""Border colour for panel frames, input fields, and dividers."""

ACTIVE_BG = ft.Colors.BLUE_GREY_700
"""Background tint applied to the currently-active mode button so the
view-switcher reads at a glance."""

CHAT_EMPTY_PLACEHOLDER = "Chat output will appear here."
"""Italic placeholder shown in the chat panel before the first message
arrives. Replaced wholesale once the user sends or the agent answers."""
