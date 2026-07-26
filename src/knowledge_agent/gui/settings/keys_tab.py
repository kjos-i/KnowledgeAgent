"""Settings → Keys sub-tab — provider API keys.

Five keyring-backed fields: Anthropic, OpenAI, Google, Voyage provider
keys plus an optional LangSmith key used only for tracing evaluation
runs (Evaluation → Run). The user fills only the ones they actually
need; no provider is privileged.

Neo4j password does NOT live here — it's per-corpus (stored under
`neo4j-{corpus_name}` in the keyring) and managed via the Library
tab (Create New Dataset writes it on corpus creation; the
Select Dataset corpus-switch handler pushes it to env).

Save UX:
  - On blur (matches the sibling app). Clicking in + out of a field
    without changes is a no-op.
  - Empty field is also a no-op so an accidental clear doesn't wipe a
    stored key; deletion is an explicit action via the per-key delete
    button (with a confirm), which also clears the key's env var so it
    takes effect without a restart.
  - On success, status text under the form updates AND a system
    message lands in chat so the user gets confirmation in their
    primary read-surface.

Backend cache hygiene after a save: the keys reach `os.environ` via
`apply_keys_to_env()`, then `config.reset_after_settings_change()` drops
the cached singletons (Neo4j driver, LLM client, embedder client)
that captured the OLD key at construction.

Eye toggle: Flet 0.85's built-in `can_reveal_password` doesn't render
the icon reliably under `InputBorder.OUTLINE`, so we render an
explicit suffix `IconButton` that flips `password` + the icon.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import reset_after_settings_change
from knowledge_agent.gui._styles import FRAME_BORDER_COLOR, PANEL_BG, labeled_field
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.config_store import (
    API_KEY_NAMES,
    KEYRING_TO_ENV,
    SECRET_DISPLAY_LABELS,
    ConfigError,
    apply_keys_to_env,
    get_api_key,
    set_api_key,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)

# Fixed caption width for the key rows so every input's LEFT edge lines up —
# captions vary in width ("Anthropic API key (not set)" vs "OpenAI API key
# (saved)"). Wide enough for the longest label + " (not set):".
_KEY_LABEL_WIDTH = 240


class KeysTab:
    """Provider API keys — all keyring-backed, save-on-blur."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.key_fields: dict[str, ft.TextField] = {}
        # labeled_field caption Texts (one per key) — refs kept so
        # `on_key_blur` can flip the "(saved)"/"(not set)" state instantly.
        self.key_captions: dict[str, ft.Control] = {}
        self._key_revealed: dict[str, bool] = {}
        # Per-key delete button (trash) — deletes the stored key after a
        # confirm. Emptying a field is a no-op (no accidental wipe), so
        # deletion is this explicit action.
        self.key_delete_buttons: dict[str, ft.IconButton] = {}
        self.status: ft.Text | None = None
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        for name in API_KEY_NAMES:
            self._key_revealed[name] = False
            self.key_fields[name] = ft.TextField(
                value=get_api_key(name) or "",
                password=True,
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                suffix=ft.IconButton(
                    icon=ft.Icons.VISIBILITY_OFF,
                    icon_size=16,
                    # Constrain the reveal button so it doesn't inflate the
                    # field height above the plain fields on other tabs.
                    width=28,
                    height=28,
                    padding=0,
                    tooltip="Show / hide key",
                    on_click=lambda e, n=name: self._toggle_key_reveal(n),
                ),
                on_blur=lambda e, n=name: self.on_key_blur(n),
            )
            self.key_delete_buttons[name] = ft.IconButton(
                icon=ft.Icons.DELETE_OUTLINE,
                icon_size=18,
                tooltip="Delete this key",
                on_click=lambda e, n=name: self._on_key_delete(n),
            )
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # Build the caption:field rows fresh each render, stamping the
        # current "(saved)" / "(not set)" state into each caption. Keep a ref
        # to each caption Text so `on_key_blur` can flip it to "(saved)"
        # instantly after a save (without a full re-render).
        key_rows: list[ft.Control] = []
        for name in API_KEY_NAMES:
            row = labeled_field(
                self._key_label(name),
                self.key_fields[name],
                label_width=_KEY_LABEL_WIDTH,
                trailing=self.key_delete_buttons[name],
            )
            self.key_captions[name] = row.controls[0]
            key_rows.append(row)
        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("Keys", trailing=info(self.app, "keys.overview")),
                *key_rows,
                self.status,
            ],
        )

    # ----- handlers ---------------------------------------------------------

    def _key_label(self, name: str) -> str:
        """Suffix shows whether the keyring already has the key — fixes
        the "I saved the key but the field is blank" confusion."""
        suffix = " (saved)" if get_api_key(name) else " (not set)"
        return f"{SECRET_DISPLAY_LABELS[name]}{suffix}"

    def _toggle_key_reveal(self, name: str) -> None:
        """Flip the password mask + eye icon for one key field."""
        revealed = not self._key_revealed[name]
        self._key_revealed[name] = revealed
        field = self.key_fields[name]
        field.password = not revealed
        field.suffix.icon = ft.Icons.VISIBILITY if revealed else ft.Icons.VISIBILITY_OFF
        self.app.page.update()

    def on_key_blur(self, name: str) -> None:
        """Save one key on blur — only if it actually changed.

        Pre-loaded with the keyring value, so a stray in + out without
        edits is a no-op. Empty field is also a no-op (preserves the
        existing stored key; deletion is the explicit per-key delete
        button, `_on_key_delete`).
        """
        if self.status is None:
            return
        field = self.key_fields[name]
        value = (field.value or "").strip()
        if not value:
            return
        if value == get_api_key(name):
            return  # unchanged from what's already in the keyring
        try:
            set_api_key(name, value)
        except ConfigError as exc:
            msg = f"could not save {name} key: {exc}"
            self.status.value = msg
            self.app.chat_panel.append_system(msg)
            self.app.page.update()
            return
        # Bridge the new key to env so pydantic-settings picks it up,
        # then drop cached singletons that captured the OLD key at
        # construction.
        apply_keys_to_env()
        try:
            reset_after_settings_change()
        except Exception as exc:
            # Cache-clear failure is non-fatal — the new key is stored,
            # but a restart will be needed before factories see it.
            logger.warning("reset_after_settings_change failed: %r", exc)
        caption = self.key_captions.get(name)
        if caption is not None:
            caption.value = f"{self._key_label(name)}:"
        msg = f"saved {name} key"
        self.status.value = msg
        self.app.chat_panel.append_system(msg)
        self.app.page.update()

    def _on_key_delete(self, name: str) -> None:
        """Confirm, then delete the stored key. Emptying a field is a no-op
        (no accidental wipe), so this button is the explicit delete path."""
        if self.status is None:
            return
        label = SECRET_DISPLAY_LABELS[name]
        if not get_api_key(name):
            self.status.value = f"{label} is not set."
            self.app.page.update()
            return

        def _confirm(_e: ft.Event) -> None:
            self.app.page.pop_dialog()
            self._delete_key(name)

        def _cancel(_e: ft.Event) -> None:
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Delete {label}?", weight=ft.FontWeight.BOLD),
            content=ft.Text(
                f"Remove the stored {label} from your keyring. You can re-enter it "
                "later. Other keys are unaffected."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.TextButton("Delete", on_click=_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)
        self.app.page.update()

    def _delete_key(self, name: str) -> None:
        """Delete the keyring entry, then make the delete take effect in-process.

        Clears ONLY this key's env var (option i): `apply_keys_to_env` only sets
        present keys so it wouldn't clear a removed one, and clearing every
        absent key would wipe shell-exported keys the startup bridge preserves.
        Then drop the cached singletons that captured the old key, and reset the
        field + caption.
        """
        if self.status is None:
            return
        try:
            set_api_key(name, "")  # empty value deletes the keyring entry
        except ConfigError as exc:
            msg = f"could not delete {name} key: {exc}"
            self.status.value = msg
            self.app.chat_panel.append_system(msg)
            self.app.page.update()
            return
        os.environ.pop(KEYRING_TO_ENV[name], None)
        try:
            reset_after_settings_change()
        except Exception as exc:
            logger.warning("reset_after_settings_change failed: %r", exc)
        field = self.key_fields.get(name)
        if field is not None:
            field.value = ""
            field.password = True
            self._key_revealed[name] = False
            if field.suffix is not None:
                field.suffix.icon = ft.Icons.VISIBILITY_OFF
        caption = self.key_captions.get(name)
        if caption is not None:
            caption.value = f"{self._key_label(name)}:"
        msg = f"deleted {name} key"
        self.status.value = msg
        self.app.chat_panel.append_system(msg)
        self.app.page.update()
