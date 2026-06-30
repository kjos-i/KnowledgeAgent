"""Settings → App sub-tab — behaviour toggles + DB connection + diagnostics.

Three stacked blocks (top to bottom):

  1. App behaviour — `restore_last_corpus` + `keep_loaded_file_on_clear`
     checkboxes. Mirrors ResearchArticlesAgent's "App behaviour"
     section pattern: each Checkbox auto-saves to GuiConfig on change
     with rollback on failure.

  2. Diagnostics — `debug_mode` checkbox (RA-style help blurb above it)
     PLUS KA-specific `system_status()` chips + Re-run button. The
     chips render the four ComponentStatus entries (neo4j / lancedb /
     llm_key / embed_key) from `health.system_status()`.

  3. Database connection — three EDITABLE TextFields (Neo4j URI,
     Neo4j user, LanceDB path) backed by GuiConfig. On_blur:
     persist to JSON, bridge to `os.environ` via
     `apply_connection_to_env`, then `reset_after_key_change()` to
     drop the cached AsyncDriver/LanceClient + Settings. The two
     advanced tuning knobs (pool size, acquisition timeout) stay
     read-only env-only — they're rare to change.

Save pattern (RA-mirror): each handler captures the previous value,
mutates GuiConfig, calls `save_config`, rolls back on ConfigError,
updates a shared status text.

Diagnostics fetch deferred to `build()` (first show) per the GUI
view-startup feedback rule: no work in `_create_controls`.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft
from pydantic import ValidationError

from knowledge_agent.config import get_settings, reset_after_key_change
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_connection_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.health import system_status

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


_DEBUG_BLURB = (
    "When on, the chat shows per-node progress (query built, "
    "retrieved N chunks, ...) plus the search query, retrieval mode, "
    "and per-hit titles + scores. Off = clean chat with only the "
    "essential 'answer ready' closure line."
)

_CONNECTION_BLURB = (
    "Backend connection. Edits save immediately, bridge to the "
    "running process, and re-open the cached drivers — no restart "
    "needed. Pool size + acquisition timeout are advanced tuning "
    "(set NEO4J_MAX_CONNECTION_POOL_SIZE / "
    "NEO4J_CONNECTION_ACQUISITION_TIMEOUT as env vars to change)."
)


class AppTab:
    """App behaviour + DB read-only + diagnostics."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.restore_last_corpus_checkbox: ft.Checkbox | None = None
        self.keep_loaded_checkbox: ft.Checkbox | None = None
        self.debug_mode_checkbox: ft.Checkbox | None = None
        self.chips_row: ft.Row | None = None
        self.rerun_button: ft.Button | None = None
        self.neo4j_uri_field: ft.TextField | None = None
        self.neo4j_user_field: ft.TextField | None = None
        self.lancedb_path_field: ft.TextField | None = None
        self.pool_size_text: ft.Text | None = None
        self.acq_timeout_text: ft.Text | None = None
        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        # All three blocks share one status line at the bottom — RA pattern.
        # Mirrors `settings_status` in the sibling app.
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)

        # Block 1: App behaviour.
        self.restore_last_corpus_checkbox = ft.Checkbox(
            label="Restore last used dataset on startup",
            value=self.app.gui_config.restore_last_corpus,
            on_change=self.on_restore_last_corpus_changed,
        )
        self.keep_loaded_checkbox = ft.Checkbox(
            label="Keep loaded file when clearing chat",
            value=self.app.gui_config.keep_loaded_file_on_clear,
            on_change=self.on_keep_loaded_changed,
        )

        # Block 2: Diagnostics.
        self.debug_mode_checkbox = ft.Checkbox(
            label="Show diagnostic info in chat",
            value=self.app.gui_config.debug_mode,
            on_change=self.on_debug_mode_changed,
        )
        # Chips populated on first build + Re-run press. Placeholder shows
        # before the first fetch completes so the panel isn't blank.
        self.chips_row = ft.Row(
            controls=[
                ft.Text(
                    "(running...)", italic=True, size=11,
                    color=ft.Colors.GREY_500,
                ),
            ],
            wrap=True,
            spacing=8,
        )
        self.rerun_button = ft.Button(
            content=centered_label("Re-run"),
            on_click=self.on_rerun_diagnostics,
        )

        # Block 3: DB connection — editable URI / user / path TextFields
        # backed by GuiConfig + two read-only tuning rows. Defaults are
        # filled in from GuiConfig so an empty field never reaches
        # `os.environ` (the bridge already wrote the field's value at
        # startup).
        self.neo4j_uri_field = ft.TextField(
            label="Neo4j URI",
            value=self.app.gui_config.neo4j_uri,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_neo4j_uri_blur,
        )
        self.neo4j_user_field = ft.TextField(
            label="Neo4j user",
            value=self.app.gui_config.neo4j_user,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_neo4j_user_blur,
        )
        self.lancedb_path_field = ft.TextField(
            label="LanceDB path",
            value=(
                str(self.app.gui_config.lancedb_path)
                if self.app.gui_config.lancedb_path is not None
                else ""
            ),
            hint_text="(empty = platform default)",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_lancedb_path_blur,
        )
        # Tuning knobs — read-only. Populated lazily on first build
        # since settings construction can raise on a fresh install.
        self.pool_size_text = ft.Text(
            "(loading…)", size=12,
            color=ft.Colors.WHITE, selectable=True,
        )
        self.acq_timeout_text = ft.Text(
            "(loading…)", size=12,
            color=ft.Colors.WHITE, selectable=True,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # First-show fetches per [[gui-view-startup]] — no work in
        # _create_controls; defer expensive reads to here. Check for a
        # running loop BEFORE constructing the coroutine so test env
        # (no loop) doesn't leave an unawaited-coroutine warning.
        if self._first_build:
            self._first_build = False
            self._populate_tuning_display()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                asyncio.create_task(self._refresh_diagnostics())

        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("App"),
                # ---- Block 1: App behaviour ----------------------------
                ft.Text("App behaviour", weight=ft.FontWeight.BOLD),
                self.restore_last_corpus_checkbox,
                self.keep_loaded_checkbox,
                ft.Divider(),
                # ---- Block 2: Diagnostics ------------------------------
                ft.Text("Diagnostics", weight=ft.FontWeight.BOLD),
                ft.Text(
                    _DEBUG_BLURB,
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                self.debug_mode_checkbox,
                ft.Container(height=8),
                ft.Text(
                    "System health — Neo4j, LanceDB, active provider keys:",
                    size=11, color=ft.Colors.GREY_400,
                ),
                self.chips_row,
                ft.Row(controls=[self.rerun_button]),
                ft.Divider(),
                # ---- Block 3: DB connection (editable) -----------------
                ft.Text(
                    "Database connection",
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    _CONNECTION_BLURB,
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                self.neo4j_uri_field,
                self.neo4j_user_field,
                self.lancedb_path_field,
                _kv_row("Connection pool size", self.pool_size_text),
                _kv_row(
                    "Connection acquisition timeout", self.acq_timeout_text,
                ),
                # ---- Shared status text --------------------------------
                self.status,
            ],
        )

    # ----- Block 1 + 2 (checkbox) handlers ---------------------------------

    def on_restore_last_corpus_changed(self, e: ft.Event) -> None:
        """Persist restore_last_corpus, with rollback on save failure."""
        if (
            self.status is None
            or self.restore_last_corpus_checkbox is None
        ):
            return
        previous = self.app.gui_config.restore_last_corpus
        self.app.gui_config.restore_last_corpus = bool(
            self.restore_last_corpus_checkbox.value
        )
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.restore_last_corpus = previous
            self.restore_last_corpus_checkbox.value = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.status.value = (
            "will restore the last used corpus on next startup"
            if self.app.gui_config.restore_last_corpus
            else "next startup will begin with no corpus selected"
        )
        self.app.page.update()

    def on_keep_loaded_changed(self, e: ft.Event) -> None:
        """Persist keep_loaded_file_on_clear, with rollback on save failure."""
        if self.status is None or self.keep_loaded_checkbox is None:
            return
        previous = self.app.gui_config.keep_loaded_file_on_clear
        self.app.gui_config.keep_loaded_file_on_clear = bool(
            self.keep_loaded_checkbox.value
        )
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.keep_loaded_file_on_clear = previous
            self.keep_loaded_checkbox.value = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.status.value = (
            "Clear keeps any loaded file visible"
            if self.app.gui_config.keep_loaded_file_on_clear
            else "Clear resets everything (including loaded file)"
        )
        self.app.page.update()

    def on_debug_mode_changed(self, e: ft.Event) -> None:
        """Persist debug_mode, with rollback on save failure."""
        if self.status is None or self.debug_mode_checkbox is None:
            return
        previous = self.app.gui_config.debug_mode
        self.app.gui_config.debug_mode = bool(self.debug_mode_checkbox.value)
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.debug_mode = previous
            self.debug_mode_checkbox.value = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.status.value = (
            "diagnostics on"
            if self.app.gui_config.debug_mode
            else "diagnostics off"
        )
        self.app.page.update()

    # ----- Block 2 (system_status chips) -----------------------------------

    async def on_rerun_diagnostics(self, e: ft.Event) -> None:
        await self._refresh_diagnostics()

    async def _refresh_diagnostics(self) -> None:
        """Fetch system_status(); paint chips. Never raises out."""
        if self.chips_row is None or self.rerun_button is None:
            return
        self.rerun_button.disabled = True
        self.app.page.update()
        try:
            report = await system_status()
            self.chips_row.controls = [
                _status_chip(c.name, c.ok, c.detail)
                for c in report.components
            ]
        except Exception as exc:
            logger.warning("system_status() failed: %r", exc)
            self.chips_row.controls = [
                ft.Text(
                    _missing_field_message(exc),
                    size=12, color=ft.Colors.AMBER_300, italic=True,
                ),
            ]
        finally:
            self.rerun_button.disabled = False
            self.app.page.update()

    # ----- Block 3 (editable connection + read-only tuning) ----------------

    def _populate_tuning_display(self) -> None:
        """Refresh the two read-only tuning rows from backend Settings.

        URI / user / path show whatever GuiConfig holds (no need to
        consult backend Settings — the bridge keeps them in sync).
        Pool size + acquisition timeout aren't in GuiConfig, so we go
        through `get_settings()` for them. If that raises (because
        `neo4j_password` isn't in the keyring yet), we render an
        actionable placeholder pointing the user at the Keys tab.
        """
        if self.pool_size_text is None or self.acq_timeout_text is None:
            return
        try:
            settings = get_settings()
            self.pool_size_text.value = str(
                settings.neo4j_max_connection_pool_size,
            )
            self.acq_timeout_text.value = (
                f"{settings.neo4j_connection_acquisition_timeout}s"
            )
        except Exception as exc:
            logger.warning("tuning display: settings load failed: %r", exc)
            placeholder = _missing_field_message(exc)
            self.pool_size_text.value = placeholder
            self.acq_timeout_text.value = placeholder

    def _commit_connection_change(self, label: str) -> None:
        """Persist + bridge + reset cached drivers after a connection field
        changed. Triggers a diagnostics re-run so the chip turns the new
        colour right away.
        """
        if self.status is None:
            return
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        apply_connection_to_env(self.app.gui_config)
        try:
            reset_after_key_change()
        except Exception as exc:
            # Cache-clear failure is non-fatal — change is saved + bridged.
            logger.warning("reset_after_key_change failed: %r", exc)
        self.status.value = f"saved {label}"
        # Refresh tuning row + diagnostics chips for the new connection.
        self._populate_tuning_display()
        self.app.page.update()
        asyncio.create_task(self._refresh_diagnostics())

    def on_neo4j_uri_blur(self, e: ft.Event) -> None:
        """Persist neo4j_uri on blur; snap back if pydantic-side bridge fails."""
        if self.neo4j_uri_field is None:
            return
        raw = (self.neo4j_uri_field.value or "").strip()
        if not raw:
            # Empty URI is invalid — restore the previous value so the
            # backend doesn't see "" and crash at AsyncDriver init.
            self.neo4j_uri_field.value = self.app.gui_config.neo4j_uri
            if self.status is not None:
                self.status.value = "Neo4j URI can't be empty"
            self.app.page.update()
            return
        if raw == self.app.gui_config.neo4j_uri:
            return
        previous = self.app.gui_config.neo4j_uri
        self.app.gui_config.neo4j_uri = raw
        # Inline rollback hook: if save fails, _commit will restore the
        # previous value via this closure.
        try:
            self._commit_connection_change("Neo4j URI")
        except Exception as exc:
            self.app.gui_config.neo4j_uri = previous
            self.neo4j_uri_field.value = previous
            if self.status is not None:
                self.status.value = f"could not save: {exc}"
            self.app.page.update()

    def on_neo4j_user_blur(self, e: ft.Event) -> None:
        """Persist neo4j_user on blur; empty resets to default `neo4j`."""
        if self.neo4j_user_field is None:
            return
        raw = (self.neo4j_user_field.value or "").strip() or "neo4j"
        if raw == self.app.gui_config.neo4j_user:
            self.neo4j_user_field.value = raw  # normalise (whitespace, etc.)
            self.app.page.update()
            return
        previous = self.app.gui_config.neo4j_user
        self.app.gui_config.neo4j_user = raw
        self.neo4j_user_field.value = raw
        try:
            self._commit_connection_change("Neo4j user")
        except Exception as exc:
            self.app.gui_config.neo4j_user = previous
            self.neo4j_user_field.value = previous
            if self.status is not None:
                self.status.value = f"could not save: {exc}"
            self.app.page.update()

    def on_lancedb_path_blur(self, e: ft.Event) -> None:
        """Persist lancedb_path on blur; empty = platform default."""
        if self.lancedb_path_field is None:
            return
        raw = (self.lancedb_path_field.value or "").strip()
        new_value = Path(raw) if raw else None
        if new_value == self.app.gui_config.lancedb_path:
            return
        previous = self.app.gui_config.lancedb_path
        self.app.gui_config.lancedb_path = new_value
        try:
            self._commit_connection_change("LanceDB path")
        except Exception as exc:
            self.app.gui_config.lancedb_path = previous
            self.lancedb_path_field.value = (
                str(previous) if previous is not None else ""
            )
            if self.status is not None:
                self.status.value = f"could not save: {exc}"
            self.app.page.update()


# =============================================================================
# Local widget helpers — kept private to the module since only AppTab uses them.
# =============================================================================


# Human-readable name per known required-no-default field. Used to render
# clearer "missing X — set in Y tab" messages instead of raw pydantic errors.
_FIELD_GUIDANCE: dict[str, str] = {
    "neo4j_password": "Neo4j password not set — open the Keys tab",
    "neo4j_uri": "Neo4j URI not set — type one above",
    "neo4j_user": "Neo4j user not set — type one above",
    "lancedb_path": "LanceDB path not set — type one above",
}


def _missing_field_message(exc: Exception) -> str:
    """Translate a pydantic ValidationError into a one-line actionable hint.

    For non-pydantic errors (Neo4j network down, etc.), falls back to the
    exception class name + message — short and human-readable. The raw
    `repr(exc)` is logged separately at WARNING level so debugging info
    isn't lost.
    """
    if isinstance(exc, ValidationError):
        missing = [
            str(err["loc"][-1])
            for err in exc.errors()
            if err.get("type") == "missing" and err.get("loc")
        ]
        # Prefer the first missing field with specific guidance — that's
        # the most actionable thing the user can do right now.
        for field in missing:
            if field in _FIELD_GUIDANCE:
                return _FIELD_GUIDANCE[field]
        if missing:
            return f"missing required setting(s): {', '.join(missing)}"
    return f"{type(exc).__name__}: {exc}"


def _kv_row(label: str, value: object) -> ft.Control:
    """One label-value row used in the DB block's read-only tuning lines.

    `value` is either a Control (used directly so callers can keep a
    reference for later updates) or a primitive (wrapped in a Text).
    Fixed-width label column so multiple rows align.
    """
    if isinstance(value, ft.Control):
        value_widget = value
    else:
        value_widget = ft.Text(
            str(value),
            size=12, color=ft.Colors.WHITE, selectable=True,
        )
    return ft.Row(
        controls=[
            ft.Text(
                f"{label}:",
                size=12, color=ft.Colors.GREY_400, width=220,
            ),
            value_widget,
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _status_chip(name: str, ok: bool, detail: str) -> ft.Control:
    """One health chip — pill-shaped, color-coded, detail on hover.

    Flet 0.85 attaches tooltips via the `tooltip` parameter on any
    Control (string or `Tooltip` value object), NOT by wrapping a
    `Tooltip` around children.
    """
    bgcolor = ft.Colors.GREEN_700 if ok else ft.Colors.RED_700
    return ft.Container(
        content=ft.Text(
            f"{name}: {'OK' if ok else 'FAIL'}",
            size=11,
            color=ft.Colors.WHITE,
            weight=ft.FontWeight.BOLD,
        ),
        tooltip=detail,
        bgcolor=bgcolor,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        border_radius=12,
        border=ft.Border.all(1, FRAME_BORDER_COLOR),
    )
