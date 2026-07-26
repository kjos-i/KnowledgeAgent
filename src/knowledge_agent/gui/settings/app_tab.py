"""Settings tab — behaviour toggles + DB connection + diagnostics.

Two columns (house `panel_box` style): LEFT holds Save results & chat + App
behaviour; RIGHT holds Diagnostics + Database connection. The four sections:

  0. Save results & chat — format checkboxes (md/txt/docx/json) + a
     default folder (Browse). These drive the Search tab's Save Result /
     Save Chat buttons (and the CLI shares the same backend renderers).
     Each control auto-saves to GuiConfig on change.

  1. App behaviour — `restore_last_corpus` + `keep_loaded_file_on_clear`
     checkboxes. Mirrors ResearchArticlesAgent's "App behaviour"
     section pattern: each Checkbox auto-saves to GuiConfig on change
     with rollback on failure.

  2. Diagnostics — `debug_mode` checkbox (RA-style help blurb above it)
     PLUS KA-specific `system_status()` chips + Re-run button. The
     chips render the four ComponentStatus entries (neo4j / lancedb /
     llm_key / embed_key) from `health.system_status()`.

  3. Database connection — READ-ONLY display of the active corpus's
     connection (mirrors the active `CorpusEntry` from
     `GuiConfig.corpora`). A corpus's connection is set once at Library
     → Create New Dataset and can't be edited afterwards; switch the
     active corpus via the top-bar corpus dropdown. Pool size +
     acquisition timeout also read-only — they're backend Settings
     defaults, override via env vars.

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

from knowledge_agent.artifacts import ANSWER_FORMATS, CHAT_FORMATS, FORMAT_LABELS
from knowledge_agent.config import get_settings
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    centered_label,
    panel_box,
    panel_title,
    section_divider,
)
from knowledge_agent.gui._widgets.info_icon import info_icon, section_header, tier_icon_sample
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.config_store import ConfigError, save_config
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

_SAVE_RESULTS_BLURB = (
    "Where and in which format(s) the Search tab's Save Result button writes. "
    "Check one or more (JSON = answers only). When no folder is set, the first "
    "save asks for one and remembers it here."
)

_SAVE_CHAT_BLURB = (
    "Where and in which format(s) the Search tab's Save Chat button writes — its "
    "own settings, separate from Save Result. A chat transcript has no JSON form. "
    "When no folder is set, the first save asks for one and remembers it here."
)

_CONNECTION_BLURB = (
    "Read-only display of the active corpus's connection. A corpus's Neo4j "
    "connection is set once, when you create the corpus via Library → Create "
    "New Dataset; it can't be edited afterwards. Switch the active corpus with "
    "the corpus dropdown (top-right). Pool size + acquisition timeout are "
    "advanced tuning (set NEO4J_MAX_CONNECTION_POOL_SIZE / "
    "NEO4J_CONNECTION_ACQUISITION_TIMEOUT env vars to change)."
)


class AppTab:
    """App behaviour + DB read-only + diagnostics."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Strong refs to fire-and-forget background tasks so the event
        # loop doesn't GC them mid-flight (see `_spawn` uses below).
        self._bg_tasks: set[asyncio.Task] = set()
        self.status: ft.Text | None = None
        self.save_format_checkboxes: dict[str, ft.Checkbox] = {}
        self.results_dir_text: ft.Text | None = None
        self.browse_folder_button: ft.Button | None = None
        self.chat_format_checkboxes: dict[str, ft.Checkbox] = {}
        self.chat_dir_text: ft.Text | None = None
        self.chat_browse_button: ft.Button | None = None
        self.restore_last_corpus_checkbox: ft.Checkbox | None = None
        self.keep_loaded_checkbox: ft.Checkbox | None = None
        self.debug_mode_checkbox: ft.Checkbox | None = None
        self.show_info_icons_checkbox: ft.Checkbox | None = None
        self.show_beginner_info_checkbox: ft.Checkbox | None = None
        self.show_technical_info_checkbox: ft.Checkbox | None = None
        self.chips_row: ft.Row | None = None
        self.rerun_button: ft.Button | None = None
        self.active_corpus_text: ft.Text | None = None
        self.neo4j_uri_text: ft.Text | None = None
        self.neo4j_user_text: ft.Text | None = None
        self.lancedb_path_text: ft.Text | None = None
        self.pool_size_text: ft.Text | None = None
        self.acq_timeout_text: ft.Text | None = None
        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        # All three blocks share one status line at the bottom — RA pattern.
        # Mirrors `settings_status` in the sibling app.
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        # Block 0a: Save results (formats + default folder).
        self.save_format_checkboxes = {
            fmt: ft.Checkbox(
                label=FORMAT_LABELS[fmt],
                value=fmt in self.app.gui_config.save_formats,
                on_change=self.on_save_format_changed,
            )
            for fmt in ANSWER_FORMATS
        }
        rd = self.app.gui_config.results_dir
        self.results_dir_text = ft.Text(
            str(rd) if rd else "(not set — first Save will ask)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.browse_folder_button = ft.Button(
            content=centered_label("Browse…"),
            on_click=self.on_browse_folder,
        )
        # Block 0b: Save chat — its own formats + folder. Chat has no JSON form,
        # so only chat-capable formats (CHAT_FORMATS) are offered.
        self.chat_format_checkboxes = {
            fmt: ft.Checkbox(
                label=FORMAT_LABELS[fmt],
                value=fmt in self.app.gui_config.chat_save_formats,
                on_change=self.on_chat_format_changed,
            )
            for fmt in CHAT_FORMATS
        }
        cd = self.app.gui_config.chat_dir
        self.chat_dir_text = ft.Text(
            str(cd) if cd else "(not set — first Save will ask)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.chat_browse_button = ft.Button(
            content=centered_label("Browse…"),
            on_click=self.on_chat_browse_folder,
        )

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
        self.show_info_icons_checkbox = ft.Checkbox(
            label="Show standard (i) help icons",
            value=self.app.gui_config.show_info_icons,
            on_change=self.on_show_info_icons_changed,
        )
        self.show_beginner_info_checkbox = ft.Checkbox(
            label="Show beginner help icons (green)",
            value=self.app.gui_config.show_beginner_info,
            on_change=self.on_show_beginner_info_changed,
        )
        self.show_technical_info_checkbox = ft.Checkbox(
            label="Show technical help icons (orange)",
            value=self.app.gui_config.show_technical_info,
            on_change=self.on_show_technical_info_changed,
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
                    "(running...)",
                    italic=True,
                    size=12,
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

        # Block 3: DB connection — all READ-ONLY. Post-Slice-3 the
        # active corpus's connection lives in `GuiConfig.corpora`;
        # this tab just displays
        # what's active. Edits happen via Library → Select Dataset
        # (switch active corpus) or Library → Create New Dataset (add
        # a new corpus).
        self.active_corpus_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.neo4j_uri_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.neo4j_user_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.lancedb_path_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        # Tuning knobs — also read-only (backend Settings defaults;
        # override via env vars).
        self.pool_size_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
        self.acq_timeout_text = ft.Text(
            "(loading…)",
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # First-show fetches — no work in
        # _create_controls; defer expensive reads to here. Check for a
        # running loop BEFORE constructing the coroutine so test env
        # (no loop) doesn't leave an unawaited-coroutine warning.
        if self._first_build:
            self._first_build = False
            self._populate_connection_display()
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                task = asyncio.create_task(self._refresh_diagnostics())
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

        # LEFT column: Save results & chat + App behaviour.
        left_pane = panel_box(
            ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                controls=[
                    # ---- Save results -------------------------------------
                    section_header(self.app, "Save results", _SAVE_RESULTS_BLURB),
                    ft.Row(
                        controls=[self.save_format_checkboxes[f] for f in ANSWER_FORMATS],
                        wrap=True,
                        spacing=12,
                    ),
                    _kv_row("Default folder", self.results_dir_text),
                    ft.Row(controls=[self.browse_folder_button]),
                    section_divider(),
                    # ---- Save chat ----------------------------------------
                    section_header(self.app, "Save chat", _SAVE_CHAT_BLURB),
                    ft.Row(
                        controls=[self.chat_format_checkboxes[f] for f in CHAT_FORMATS],
                        wrap=True,
                        spacing=12,
                    ),
                    _kv_row("Default folder", self.chat_dir_text),
                    ft.Row(controls=[self.chat_browse_button]),
                    section_divider(),
                    # ---- App behaviour -------------------------------------
                    section_header(self.app, "App behaviour"),
                    self.restore_last_corpus_checkbox,
                    self.keep_loaded_checkbox,
                    # Each info-tier toggle shows a live sample of the icon it
                    # controls, so the user sees what "standard / beginner /
                    # technical" help icons look like (task 51).
                    ft.Row(
                        [self.show_info_icons_checkbox, tier_icon_sample("standard")],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                    ft.Row(
                        [self.show_beginner_info_checkbox, tier_icon_sample("beginner")],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                    ft.Row(
                        [self.show_technical_info_checkbox, tier_icon_sample("technical")],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                ],
            ),
            expand=1,
        )
        # RIGHT column: Diagnostics + Database connection.
        right_pane = panel_box(
            ft.Column(
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                controls=[
                    # ---- Diagnostics ---------------------------------------
                    section_header(self.app, "Diagnostics"),
                    ft.Row(
                        [
                            self.debug_mode_checkbox,
                            info_icon(
                                self.app,
                                title="Show diagnostic info in chat",
                                text=_DEBUG_BLURB,
                            ),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=4,
                    ),
                    ft.Container(height=8),
                    ft.Row(
                        controls=[
                            ft.Text(
                                "System health — Neo4j, LanceDB, active provider keys:",
                                size=12,
                                color=ft.Colors.GREY_400,
                            ),
                            info(self.app, "diagnostics.system_health"),
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=6,
                    ),
                    self.chips_row,
                    ft.Row(controls=[self.rerun_button]),
                    section_divider(),
                    # ---- Database connection (read-only display) -----------
                    section_header(self.app, "Database connection", _CONNECTION_BLURB),
                    _kv_row("Active corpus", self.active_corpus_text),
                    _kv_row("Neo4j URI", self.neo4j_uri_text),
                    _kv_row("Neo4j user", self.neo4j_user_text),
                    _kv_row("LanceDB path", self.lancedb_path_text),
                    _kv_row("Connection pool size", self.pool_size_text),
                    _kv_row("Connection acquisition timeout", self.acq_timeout_text),
                ],
            ),
            expand=1,
        )
        # Column titles sit ABOVE each box (the house two-column layout), aligned
        # to the columns via matching expand + horizontal padding.
        column_titles = ft.Row(
            spacing=12,
            controls=[
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12),
                    content=panel_title("General"),
                ),
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12),
                    content=panel_title("System"),
                ),
            ],
        )
        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[
                view_header("Settings"),
                column_titles,
                ft.Row(
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                    spacing=12,
                    controls=[left_pane, right_pane],
                ),
                # Shared status line, full width below the two columns.
                self.status,
            ],
        )

    # ----- Block 1 + 2 (checkbox) handlers ---------------------------------

    def on_restore_last_corpus_changed(self, e: ft.Event) -> None:
        """Persist restore_last_corpus, with rollback on save failure."""
        if self.status is None or self.restore_last_corpus_checkbox is None:
            return
        previous = self.app.gui_config.restore_last_corpus
        self.app.gui_config.restore_last_corpus = bool(self.restore_last_corpus_checkbox.value)
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

    def _on_info_tier_changed(
        self, tier: str, flag: str, checkbox: ft.Checkbox | None, label: str
    ) -> None:
        """Persist one `show_*_info` flag + flip that tier's (i) icons live.
        Shared by the standard / beginner / technical toggles."""
        if self.status is None or checkbox is None:
            return
        previous = getattr(self.app.gui_config, flag)
        setattr(self.app.gui_config, flag, bool(checkbox.value))
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            setattr(self.app.gui_config, flag, previous)
            checkbox.value = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        shown = getattr(self.app.gui_config, flag)
        self.status.value = f"{label} icons {'shown' if shown else 'hidden'}"
        self.app.set_info_icons_visible(tier, shown)

    def on_show_info_icons_changed(self, e: ft.Event) -> None:
        self._on_info_tier_changed(
            "standard", "show_info_icons", self.show_info_icons_checkbox, "standard help"
        )

    def on_show_beginner_info_changed(self, e: ft.Event) -> None:
        self._on_info_tier_changed(
            "beginner", "show_beginner_info", self.show_beginner_info_checkbox, "beginner help"
        )

    def on_show_technical_info_changed(self, e: ft.Event) -> None:
        self._on_info_tier_changed(
            "technical", "show_technical_info", self.show_technical_info_checkbox, "technical help"
        )

    def on_keep_loaded_changed(self, e: ft.Event) -> None:
        """Persist keep_loaded_file_on_clear, with rollback on save failure."""
        if self.status is None or self.keep_loaded_checkbox is None:
            return
        previous = self.app.gui_config.keep_loaded_file_on_clear
        self.app.gui_config.keep_loaded_file_on_clear = bool(self.keep_loaded_checkbox.value)
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
            "diagnostics on" if self.app.gui_config.debug_mode else "diagnostics off"
        )
        self.app.page.update()

    # ----- Block 0 (save results & chat) handlers --------------------------

    def on_save_format_changed(self, e: ft.Event) -> None:
        """Persist the checked save formats; keep at least one; rollback on fail."""
        if self.status is None:
            return
        previous = list(self.app.gui_config.save_formats)
        chosen = [f for f in ANSWER_FORMATS if self.save_format_checkboxes[f].value]
        if not chosen:
            # Never allow zero formats — restore the prior selection.
            for fmt, cb in self.save_format_checkboxes.items():
                cb.value = fmt in previous
            self.status.value = "keep at least one save format"
            self.app.page.update()
            return
        self.app.gui_config.save_formats = chosen
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.save_formats = previous
            for fmt, cb in self.save_format_checkboxes.items():
                cb.value = fmt in previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.status.value = f"save formats: {', '.join(chosen)}"
        self.app.page.update()

    async def on_browse_folder(self, e: ft.Event) -> None:
        """Pick + persist the default save folder for results."""
        if self.status is None or self.results_dir_text is None:
            return
        rd = self.app.gui_config.results_dir
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title="Default folder for saved results",
                initial_directory=str(rd) if rd else None,
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            self.status.value = f"folder picker error: {exc}"
            self.app.page.update()
            return
        if not chosen:
            return
        previous = self.app.gui_config.results_dir
        self.app.gui_config.results_dir = Path(chosen)
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.results_dir = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.results_dir_text.value = chosen
        self.status.value = "default results folder set"
        self.app.page.update()

    def on_chat_format_changed(self, e: ft.Event) -> None:
        """Persist the checked chat-save formats; keep at least one; rollback on fail."""
        if self.status is None:
            return
        previous = list(self.app.gui_config.chat_save_formats)
        chosen = [f for f in CHAT_FORMATS if self.chat_format_checkboxes[f].value]
        if not chosen:
            for fmt, cb in self.chat_format_checkboxes.items():
                cb.value = fmt in previous
            self.status.value = "keep at least one chat-save format"
            self.app.page.update()
            return
        self.app.gui_config.chat_save_formats = chosen
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.chat_save_formats = previous
            for fmt, cb in self.chat_format_checkboxes.items():
                cb.value = fmt in previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.status.value = f"chat-save formats: {', '.join(chosen)}"
        self.app.page.update()

    async def on_chat_browse_folder(self, e: ft.Event) -> None:
        """Pick + persist the default save folder for chats."""
        if self.status is None or self.chat_dir_text is None:
            return
        cd = self.app.gui_config.chat_dir
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title="Default folder for saved chats",
                initial_directory=str(cd) if cd else None,
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            self.status.value = f"folder picker error: {exc}"
            self.app.page.update()
            return
        if not chosen:
            return
        previous = self.app.gui_config.chat_dir
        self.app.gui_config.chat_dir = Path(chosen)
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.app.gui_config.chat_dir = previous
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        self.chat_dir_text.value = chosen
        self.status.value = "default chat folder set"
        self.app.page.update()

    # ----- Block 2 (system_status chips) -----------------------------------

    async def on_rerun_diagnostics(self, e: ft.Event) -> None:
        await self._refresh_diagnostics()

    async def _refresh_diagnostics(self) -> None:
        """Fetch system_status(); paint chips, or the config hint when the
        report can't be built yet (no active corpus). `system_status()`
        never raises — a config failure comes back as `report.config_error`,
        so there's no separate catch here (one common catch lives there)."""
        if self.chips_row is None or self.rerun_button is None:
            return
        self.rerun_button.disabled = True
        self.app.page.update()
        try:
            report = await system_status()
            if report.config_error:
                self.chips_row.controls = [
                    ft.Text(
                        report.config_error,
                        size=12,
                        color=ft.Colors.AMBER_300,
                        italic=True,
                    ),
                ]
            else:
                self.chips_row.controls = [
                    _status_chip(c.name, c.ok, c.detail) for c in report.components
                ]
        finally:
            self.rerun_button.disabled = False
            self.app.page.update()

    # ----- Block 3 (editable connection + read-only tuning) ----------------

    def _populate_connection_display(self) -> None:
        """Refresh the read-only connection + tuning rows.

        Active corpus / URI / user / path come from GuiConfig (mirror
        of the active `CorpusEntry`). Pool size + acquisition timeout
        come from backend Settings; if that raises (bridge hasn't
        seeded env yet on a fresh install), we render an actionable
        placeholder.
        """
        # ---- Corpus + connection mirrors from GuiConfig ---------------
        cfg = self.app.gui_config
        if self.active_corpus_text is not None:
            self.active_corpus_text.value = (
                cfg.active_corpus_name or "(no corpus — create one in Library)"
            )
        if self.neo4j_uri_text is not None:
            self.neo4j_uri_text.value = cfg.neo4j_uri
        if self.neo4j_user_text is not None:
            self.neo4j_user_text.value = cfg.neo4j_user
        if self.lancedb_path_text is not None:
            self.lancedb_path_text.value = (
                str(cfg.lancedb_path)
                if cfg.lancedb_path is not None
                else "(default under user_data_dir)"
            )
        # ---- Tuning knobs -----------------------------------------------
        # Always show a value. When Settings() constructs cleanly, we
        # show the LIVE value (which may reflect env-var overrides).
        # When it raises (typically because NEO4J_PASSWORD isn't set
        # on a fresh install), we fall back to the pydantic-declared
        # Field defaults so the row shows "100 (default)" instead of
        # a stale "password not set" message that doesn't apply here.
        if self.pool_size_text is None or self.acq_timeout_text is None:
            return
        try:
            settings = get_settings()
            self.pool_size_text.value = str(
                settings.neo4j_max_connection_pool_size,
            )
            self.acq_timeout_text.value = f"{settings.neo4j_connection_acquisition_timeout}s"
        except Exception as exc:
            logger.warning(
                "connection display: settings load failed; falling back to Field defaults: %r",
                exc,
            )
            from knowledge_agent.config import Settings

            pool_default = Settings.model_fields["neo4j_max_connection_pool_size"].default
            acq_default = Settings.model_fields["neo4j_connection_acquisition_timeout"].default
            self.pool_size_text.value = f"{pool_default} (default)"
            self.acq_timeout_text.value = f"{acq_default}s (default)"

    # DB connection is display-only post-Slice-3. Editing happens via
    # Library → Select Dataset (switch corpus) or Create New Dataset
    # (register a new one). The old on_blur handlers + commit helper
    # for URI / user / path were removed with the read-only pivot.


# =============================================================================
# Local widget helpers — kept private to the module since only AppTab uses them.
# =============================================================================


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
            size=12,
            color=ft.Colors.WHITE,
            selectable=True,
        )
    return ft.Row(
        controls=[
            ft.Text(
                f"{label}:",
                size=12,
                color=ft.Colors.GREY_400,
                width=220,
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
            size=12,
            color=ft.Colors.WHITE,
            weight=ft.FontWeight.BOLD,
        ),
        tooltip=detail,
        bgcolor=bgcolor,
        padding=ft.Padding.symmetric(horizontal=10, vertical=4),
        border_radius=12,
        border=ft.Border.all(1, FRAME_BORDER_COLOR),
    )
