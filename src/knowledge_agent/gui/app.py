"""Desktop GUI for the Knowledge Agent — coordinator.

Owns session state (`messages`, `last_answer`, `last_query`,
`loaded_file`, `busy`, `_send_task`) and the cross-cutting handlers
(Send → chat-router → graph, Stop, Clear, Save chat, Save Result,
Open Result). Page layout is top-level `ft.Tabs`:

    [ Search ] [ Library ] [ Evaluation ] [ Installs ] [ Keys ] [ Settings ] [ Info ]

Search composes `ChatPanel` (left, always mounted) + `RightPanel`
(right), where the right column is a View / Retrieval / LLM sub-tab
strip — only that column swaps; the chat stays put.

Installs, Settings, Keys, and Info are their own top-level tabs (global
machine-level install surface / app-level config / API keys / reference).
They were promoted out of Search's right panel so they're one click away
and the right panel stays focused on the per-query search loop. Settings +
Keys were split from a former single "Settings" sub-tab shell (2026-07-14).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flet as ft
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from knowledge_agent.artifacts import (
    CHAT_FORMATS,
    SaveError,
    save_answer,
    save_chat,
)
from knowledge_agent.config import disable_env_file, get_settings, reset_after_key_change
from knowledge_agent.corpus_config import CorpusConfig, load_corpus_config
from knowledge_agent.evaluation.models import RetrievalSettings
from knowledge_agent.gui._widgets.retrieval_form import (
    query_mode_to_knobs,
    store_forced_by_mode,
)
from knowledge_agent.gui.chat_panel import ChatPanel
from knowledge_agent.gui.chat_router import (
    CHAT_SYSTEM_PROMPT,
    ChatTurnOutput,
    get_chat_router,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    GuiConfig,
    apply_active_corpus_embedding_to_env,
    apply_active_corpus_password_to_env,
    apply_connection_to_env,
    apply_embedding_to_env,
    apply_keys_to_env,
    apply_llm_to_env,
    apply_ontology_downloads_dir_to_env,
    apply_retrieval_to_env,
    get_api_key,
    load_config,
    save_config,
    switch_active_corpus,
)
from knowledge_agent.gui.library.installs import InstallsTab
from knowledge_agent.gui.right_panel import RightPanel
from knowledge_agent.gui.settings import AppTab, KeysTab
from knowledge_agent.gui.tabs.evaluation_tab import EvaluationTab
from knowledge_agent.gui.tabs.info_tab import InfoTab
from knowledge_agent.gui.tabs.library_tab import LibraryTab
from knowledge_agent.gui.tabs.log_tab import LogTab
from knowledge_agent.gui.tabs.search_tab import SearchTab

if TYPE_CHECKING:
    from knowledge_agent.models import AgentAnswer

logger = logging.getLogger(__name__)


WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
WINDOW_MIN_WIDTH = 1000
WINDOW_MIN_HEIGHT = 640


@dataclass
class _LoadedFile:
    """A saved `.md` answer opened via Open Result / paste-path."""

    name: str
    content: str


@dataclass
class GuiApp:
    """Owns the GUI's session state + cross-cutting handlers."""

    page: ft.Page
    messages: list[BaseMessage] = field(default_factory=list)
    last_answer: AgentAnswer | None = None
    last_query: str | None = None
    # For the eval "capture from search / chat" flow: the chat router's distilled
    # query for the last conversational send (None in direct modes), and a
    # snapshot of the retrieval knobs that send ran under. They let a captured
    # case pin what the search ACTUALLY used (not form defaults) and, for chat,
    # carry the router's distilled question. See gui/evaluation/dataset_tab.
    last_search_query: str | None = None
    last_retrieval: RetrievalSettings | None = None
    gui_config: GuiConfig = field(default_factory=GuiConfig)
    busy: bool = False
    loaded_file: _LoadedFile | None = None

    # Late-bound (built in build()).
    chat_panel: ChatPanel = field(init=False)
    right_panel: RightPanel = field(init=False)
    search_tab: SearchTab = field(init=False)
    library_tab: LibraryTab = field(init=False)
    evaluation_tab: EvaluationTab = field(init=False)
    installs_tab: InstallsTab = field(init=False)
    # Settings top tab shows app-level config (AppTab); Keys is its own tab.
    app_tab: AppTab = field(init=False)
    keys_tab: KeysTab = field(init=False)
    info_tab: InfoTab = field(init=False)
    log_tab: LogTab = field(init=False)
    file_picker: ft.FilePicker = field(init=False)
    _send_task: asyncio.Task | None = field(default=None, init=False)
    _info_icons: dict[str, list[ft.IconButton]] = field(default_factory=dict, init=False)
    # Global corpus selector on the top tab row (built in build()).
    corpus_dropdown: ft.Dropdown | None = field(default=None, init=False)
    manage_button: ft.IconButton | None = field(default=None, init=False)

    # ----- diagnostic chatter ----------------------------------------------

    def _diag(self, msg: str) -> None:
        """Append a system message to chat only when `debug_mode` is on.

        Used for per-node progress + extra detail (search query,
        retrieval hits). Off by default — essential closure messages
        (answer ready, errors) still go through `append_system`
        unconditionally.
        """
        if self.gui_config.debug_mode:
            self.chat_panel.append_system(msg)

    # ----- info-icon registry ----------------------------------------------

    def register_info_icon(self, button: ft.IconButton, tier: str = "standard") -> None:
        """Track an `(i)` help icon under its tier so `set_info_icons_visible`
        can flip that tier live."""
        self._info_icons.setdefault(tier, []).append(button)

    def set_info_icons_visible(self, tier: str, visible: bool) -> None:
        """Show/hide every registered (i) icon of one tier at once (live)."""
        for button in self._info_icons.get(tier, []):
            button.visible = visible
        self.page.update()

    # ----- global corpus selection -----------------------------------------

    def select_corpus(self, name: str) -> None:
        """The one entry point for switching the app-wide active corpus.

        Every selector — the top-bar dropdown, the Manage dialog, the
        Library picker — routes here. Applies the switch (config + env, via
        `switch_active_corpus`), clears cached backend clients, then
        refreshes every tab and re-syncs the top-bar selector. A hard
        failure leaves state unchanged and only bounces the selector back.
        """
        if not name or name == self.gui_config.active_corpus_name:
            return
        # A corpus switch mid-search would swap the backend clients out from
        # under the in-flight retrieval: `switch_active_corpus` + the
        # `reset_after_key_change` below rebind config/env and clear the cached
        # clients, so a query that started against corpus A would finish reading
        # corpus B's data (a silently mixed answer). Refuse while a search is
        # live; bounce the selector back to the real active corpus.
        if self._send_task is not None and not self._send_task.done():
            self.chat_panel.append_system(
                "a search is running — stop it (or let it finish) before switching corpus."
            )
            self._sync_corpus_selector()
            self.page.update()
            return
        outcome = switch_active_corpus(self.gui_config, name)
        if not outcome.ok:
            if outcome.message:
                self.chat_panel.append_system(outcome.message)
            self._sync_corpus_selector()  # bounce selector back to the real active
            self.page.update()
            return
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed after corpus switch: %r", exc)
        if outcome.message:
            self.chat_panel.append_system(outcome.message)
        self.refresh_after_corpus_change()
        self.page.update()

    def refresh_after_corpus_change(self) -> None:
        """Broadcast an app-wide corpus switch to every surface that shows
        or depends on the active corpus (best-effort per surface, so one
        failing tab never blocks the rest)."""
        self._sync_corpus_selector()
        try:
            view = self.library_tab.view
            view.select_tab.refresh_after_switch()
            view.refresh_ingest()
        except Exception as exc:
            logger.warning("library refresh after corpus switch failed: %r", exc)
        try:
            self.evaluation_tab.view.run_tab.refresh_active_corpus()
        except Exception as exc:
            logger.warning("eval refresh after corpus switch failed: %r", exc)

    def _build_corpus_selector(self) -> tuple[ft.Dropdown, ft.IconButton]:
        """Build the global corpus dropdown + `⚙ Manage` button for the top
        tab row. Options / value are filled by `_sync_corpus_selector`.

        NOTE: Flet 0.85's `Dropdown` takes `on_select` (NOT `on_change`) and
        rejects `hint_text` / `text_size` — mirror the other Dropdowns in the
        app (e.g. the config editor's sub-label dropdown). Extracted so
        `test_app` can construct it and catch a bad kwarg without launching
        Flet (build() itself is not unit-tested)."""
        # No floating `label` — the "Selected corpus:" caption sits inline to
        # the LEFT of the dropdown (added in build()'s top row).
        dropdown = ft.Dropdown(
            width=240,
            on_select=self._on_corpus_dropdown_changed,
        )
        manage = ft.IconButton(
            icon=ft.Icons.SETTINGS,
            tooltip="Selected corpus",
            on_click=self._open_manage_dialog,
        )
        return dropdown, manage

    def _sync_corpus_selector(self) -> None:
        """Rebuild the top-bar dropdown's options + value from gui_config."""
        if self.corpus_dropdown is None:
            return
        self.corpus_dropdown.options = [
            ft.DropdownOption(key=c.name, text=c.name) for c in self.gui_config.corpora
        ]
        self.corpus_dropdown.value = self.gui_config.active_corpus_name

    def _on_corpus_dropdown_changed(self, e: ft.Event) -> None:
        if self.corpus_dropdown is not None and self.corpus_dropdown.value:
            self.select_corpus(self.corpus_dropdown.value)

    def _open_manage_dialog(self, e: ft.Event) -> None:
        """Open the 'Selected corpus' dialog: the active corpus's read-only
        card (notes #34 — paths, layers, extractor, chunking, **embedder +
        LLM**) plus Rename / Relocate / Remove / Refresh.

        The actions reach across to the Library → Select handlers (single
        source of truth — they act on the active corpus, not the Library
        picker). Each closes this dialog first so its own prompt / picker
        isn't stacked underneath.
        """
        from knowledge_agent.gui.library.corpus_card import build_corpus_card

        select_tab = self.library_tab.view.select_tab

        def _act(handler):
            async def _run(ev: ft.Event) -> None:
                self.page.pop_dialog()  # close this dialog before the action's own dialog
                result = handler(ev)
                if asyncio.iscoroutine(result):  # on_relocate_clicked is async
                    await result

            return _run

        has_active = bool(self.gui_config.active_corpus_name)
        buttons = ft.Row(
            spacing=8,
            wrap=True,
            controls=[
                ft.Button(
                    content=ft.Text("Rename"),
                    on_click=_act(select_tab.on_rename_clicked),
                    disabled=not has_active,
                ),
                ft.Button(
                    content=ft.Text("Relocate"),
                    on_click=_act(select_tab.on_relocate_clicked),
                    disabled=not has_active,
                ),
                ft.Button(
                    content=ft.Text("Remove"),
                    on_click=_act(select_tab.on_remove_clicked),
                    disabled=not has_active,
                ),
                ft.Button(
                    content=ft.Text("Refresh"),
                    on_click=_act(select_tab.on_refresh_clicked),
                ),
            ],
        )
        dialog = ft.AlertDialog(
            modal=True,
            # Title carries an always-visible Close (X): the read-only card can
            # be tall, so the bottom `actions` Close could scroll out of reach —
            # this X in the fixed title bar is always clickable.
            title=ft.Row(
                controls=[
                    ft.Text("Selected corpus"),
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        tooltip="Close",
                        on_click=lambda _c: self.page.pop_dialog(),
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            # Content is a scrolling Column (NOT wrapped in a height-less
            # Container) so Material bounds it to the viewport and it scrolls
            # internally, keeping the title + actions on-screen. Matches the
            # documents_view Edit dialog, which is why that one closes fine.
            content=ft.Column(
                width=520,
                tight=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                controls=[
                    build_corpus_card(self),
                    ft.Divider(),
                    buttons,
                ],
            ),
            actions=[ft.TextButton("Close", on_click=lambda _c: self.page.pop_dialog())],
        )
        self.page.show_dialog(dialog)
        self.page.update()

    # ----- API-key preflight ------------------------------------------------

    def _missing_active_provider_key(self) -> str | None:
        """Return the env var name of a missing key the active LLM or
        embedder provider needs, or None when both are configured.

        Local providers (Ollama, HuggingFace) don't need API keys; they
        report no missing key here. The key is checked via `get_api_key`
        (keyring) OR the env var (shell export / `.env`).
        """
        import os

        settings = get_settings()
        # LLM provider.
        llm_provider = settings.llm_provider
        llm_env = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(llm_provider)
        if llm_env and not (os.getenv(llm_env) or get_api_key(llm_provider)):
            return llm_env
        # Embedder provider.
        embed_provider = settings.embedding_provider
        embed_env = {
            "voyage": "VOYAGE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }.get(embed_provider)
        if embed_env and not (os.getenv(embed_env) or get_api_key(embed_provider)):
            return embed_env
        return None

    def _load_corpus_config(self) -> CorpusConfig | None:
        """Load the corpus.toml the agent needs to build prompts.

        Resolution order:
          1. `gui_config.corpus_config_path` if set + file exists AND
             `gui_config.restore_last_corpus` is True. When the toggle
             is off, the stored path is skipped — user starts without a
             corpus unless CWD has one.
          2. `corpus.toml` in the current working directory.
          3. None → caller surfaces a user-facing banner.
        """
        candidates: list[Path] = []
        if self.gui_config.restore_last_corpus and self.gui_config.corpus_config_path is not None:
            candidates.append(self.gui_config.corpus_config_path)
        candidates.append(Path.cwd() / "corpus.toml")
        for p in candidates:
            if p.is_file():
                try:
                    return load_corpus_config(p)
                except Exception as exc:
                    logger.warning(
                        "load_corpus_config(%s) failed: %r",
                        p,
                        exc,
                    )
        return None

    # ----- Send + chat-router + graph --------------------------------------

    def _retrieval_snapshot(self) -> RetrievalSettings:
        """The retrieval knobs the current send runs under, as a per-case
        `RetrievalSettings`.

        The graph reads these from the active GuiConfig (bridged to Settings),
        so the snapshot mirrors GuiConfig — every knob pinned, none left to
        drift — so the eval "capture from search / chat" flow reproduces the
        exact search instead of falling back to form defaults."""
        cfg = self.gui_config
        knobs = query_mode_to_knobs(cfg.input_mode)
        return RetrievalSettings(
            retrieval_mode=store_forced_by_mode(cfg.input_mode) or cfg.retrieval_mode,
            lancedb_search_mode=cfg.lancedb_search_mode,
            top_k=cfg.top_k,
            skip_query_builder=bool(knobs.get("skip_query_builder", False)),
            direct_retrieval=cfg.direct_retrieve,
            num_candidates=cfg.num_candidates,
            rrf_rank_constant=cfg.rrf_rank_constant,
            mmr_lambda=cfg.mmr_lambda,
            use_mmr=cfg.use_mmr,
            kg_max_rows=cfg.kg_max_rows,
        )

    def _invoke_state_for_input_mode(
        self,
        input_mode: str,
        text: str,
        corpus_config: CorpusConfig,
        search_query: str | None,
    ) -> dict[str, Any]:
        """Build the graph invoke-state for the selected input mode.

        Query modes map to graph knobs via the shared `query_mode_to_knobs`, so
        chat and the eval form can't disagree on what a mode means. The store
        (`retrieval_mode`) is the user's own choice for every mode EXCEPT
        direct_cypher, which pins neo4j (`store_forced_by_mode`) since Cypher
        only runs on the graph.

        - conversational: the router's distilled `search_query` (fallback to raw
          text); query-builder runs.
        - refined: raw text; query-builder runs.
        - direct_query: raw text verbatim; skips the query-builder.
        - direct_cypher: raw text run as user Cypher; store pinned to neo4j.
        """
        # conversational is the only mode whose query is the router's distilled
        # text; every other mode searches the raw text (direct_cypher runs it as
        # Cypher). skip_query_builder / user_cypher come from the shared mapping.
        query = (search_query or "").strip() or text if input_mode == "conversational" else text
        return {
            "corpus_config": corpus_config,
            "top_k": self.gui_config.top_k,
            "direct_retrieval": self.gui_config.direct_retrieve,
            "query": query,
            "retrieval_mode": store_forced_by_mode(input_mode) or self.gui_config.retrieval_mode,
            **query_mode_to_knobs(input_mode, cypher_text=text),
        }

    async def on_send(self, e: ft.Event) -> None:
        """Submit the input — chat-router decides whether to clarify or retrieve."""
        if self.busy:
            return
        text = self.chat_panel.get_input_text()
        if not text:
            return

        missing_env = self._missing_active_provider_key()
        if missing_env:
            self.chat_panel.append_system(
                f"missing API key: set {missing_env} in Settings before querying."
            )
            return

        corpus_config = self._load_corpus_config()
        if corpus_config is None:
            self.chat_panel.append_system(
                "no corpus.toml loaded — set `corpus_config_path` in "
                "Settings or place a `corpus.toml` in the working "
                "directory."
            )
            return

        self.busy = True
        self.chat_panel.set_busy(True)
        self._send_task = asyncio.current_task()
        self.chat_panel.clear_input()
        self.messages.append(HumanMessage(content=text))
        self.chat_panel.append_user(text)

        try:
            input_mode = self.gui_config.input_mode
            search_query: str | None = None
            if input_mode == "conversational":
                # Run the chat router: it decides when to retrieve + distils
                # the query. Direct modes skip it (you / a supervisor drive).
                try:
                    router = get_chat_router(
                        self.gui_config.chat_router_model,
                        self.gui_config.chat_router_temperature,
                    )
                except Exception as exc:
                    self.chat_panel.append_system(
                        f"could not initialize chat router (check provider config): {exc}"
                    )
                    self.messages.pop()
                    return

                try:
                    output: ChatTurnOutput = await router.ainvoke(
                        [
                            SystemMessage(content=CHAT_SYSTEM_PROMPT),
                            *self.messages,
                        ]
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("chat-router failed: %r", exc)
                    self.chat_panel.append_system(f"chat-router error: {exc}")
                    self.messages.pop()
                    return

                self.messages.append(AIMessage(content=output.response))
                self.chat_panel.append_assistant(output.response)
                if not output.ready_to_retrieve:
                    return
                search_query = output.search_query

            invoke_state = self._invoke_state_for_input_mode(
                input_mode, text, corpus_config, search_query
            )
            self._diag(
                f"retrieving (input_mode={input_mode}, "
                f"mode={invoke_state.get('retrieval_mode')}, "
                f"query={invoke_state['query']!r}) ..."
            )
            try:
                # Lazy-loaded off the UI thread (kept out of startup). Usually a
                # cache hit — build() pre-warms it in the background.
                graph = await asyncio.to_thread(_import_graph)
                final_state = await graph.ainvoke(invoke_state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("graph load/invoke raised: %r", exc)
                self.chat_panel.append_system(f"retrieval failed: {exc}")
                return

            answer: AgentAnswer | None = final_state.get("final_answer")
            if answer is None:
                self.chat_panel.append_system("no answer produced")
                return

            self.last_answer = answer
            self.last_query = text
            # For the eval capture flow: the router's distilled query (chat only)
            # + the exact retrieval knobs this send ran under.
            self.last_search_query = search_query
            self.last_retrieval = self._retrieval_snapshot()
            n_chunk = len(answer.chunk_sources)
            n_kg = len(answer.kg_sources)
            if self.gui_config.direct_retrieve:
                history_note = (
                    f"(Retrieved {n_chunk} raw chunk + {n_kg} KG sources; synthesizer skipped.)"
                )
                status = (
                    f"direct retrieve — {n_chunk} raw chunk + {n_kg} KG "
                    f"sources (synthesizer skipped). See display panel."
                )
            else:
                history_note = f"(Answered from {n_chunk} chunk + {n_kg} KG sources.)"
                status = f"answer ready — {n_chunk} chunk + {n_kg} KG sources. See display panel."
            self.messages.append(AIMessage(content=history_note))
            self.chat_panel.append_system(status)
            # Push the new answer into the view history and jump to it.
            self.right_panel.push_answer(answer, text)
        except asyncio.CancelledError:
            self.chat_panel.append_system("search cancelled")
        finally:
            self.busy = False
            self.chat_panel.set_busy(False)
            self._send_task = None

    def on_stop(self, e: ft.Event) -> None:
        """Cancel the in-flight task, if any."""
        task = self._send_task
        if task is not None and not task.done():
            task.cancel()

    # ----- Clear / Save ----------------------------------------------------

    def on_clear(self, e: ft.Event) -> None:
        self.messages.clear()
        self.last_answer = None
        self.last_query = None
        self.chat_panel.reset()
        # Per `keep_loaded_file_on_clear`: when off, also wipe the loaded file
        # so it isn't re-seeded into the reset pager below.
        if not self.gui_config.keep_loaded_file_on_clear:
            self.loaded_file = None
        # Reset the view-history pager; a surviving loaded file is re-seeded as
        # its sole slot so it stays visible (matches keep_loaded_file_on_clear).
        self.right_panel.reset_history()
        self.page.update()

    async def _resolve_save_dir(self, *, chat: bool = False) -> tuple[Path | None, bool]:
        """Resolve the folder a Save action writes to.

        Returns `(folder, asked)`. Save Result uses `results_dir`; Save Chat uses
        its own `chat_dir`. When that folder is set it's used silently
        (`asked=False`); when none is set, the OS picker opens once, the choice is
        remembered on the matching config field, and `asked=True` so the caller
        can point the user at Settings. `folder` is None on cancel / picker error.
        """
        attr = "chat_dir" if chat else "results_dir"
        existing = getattr(self.gui_config, attr)
        if existing is not None:
            return existing, False
        try:
            chosen = await self.file_picker.get_directory_path(
                dialog_title="Choose a folder to save the chat to"
                if chat
                else "Choose a folder to save to",
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            self.chat_panel.append_system(f"folder picker error: {exc}")
            return None, False
        if not chosen:
            return None, False
        path = Path(chosen)
        # Remember it so subsequent saves skip the picker.
        setattr(self.gui_config, attr, path)
        try:
            save_config(self.gui_config)
        except ConfigError as exc:
            # Non-fatal — this save still proceeds to `path`; it just won't be
            # remembered until the folder is set again / in Settings.
            logger.warning("could not persist %s: %r", attr, exc)
            setattr(self.gui_config, attr, None)
        return path, True

    def _settings_hint(self, section: str) -> None:
        """Point the user at the Settings section — shown after a Save that had
        to ask for a folder (i.e. the first save, before a default is set)."""
        self.chat_panel.append_system(
            f"tip: choose formats and a default folder in Settings → {section}"
        )

    async def on_save_answer(self, e: ft.Event) -> None:
        if self.last_answer is None or self.last_query is None:
            self.chat_panel.append_system("nothing to save yet — ask a question first")
            return
        target, asked = await self._resolve_save_dir()
        if target is None:
            return
        formats = self.gui_config.save_formats or ["md"]
        try:
            paths = save_answer(self.last_answer, self.last_query, target, formats)
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        for path in paths:
            self.chat_panel.append_system(f"saved: {path}")
        if asked:
            self._settings_hint("Save results")

    async def on_save_chat(self, e: ft.Event) -> None:
        if not self.messages:
            self.chat_panel.append_system("nothing to save — chat is empty")
            return
        target, asked = await self._resolve_save_dir(chat=True)
        if target is None:
            return
        # A chat transcript has no JSON form — keep only chat-capable formats
        # (fall back to md if the user selected json-only).
        chosen_formats = self.gui_config.chat_save_formats or ["md"]
        formats = [f for f in chosen_formats if f in CHAT_FORMATS] or ["md"]
        try:
            paths = save_chat(self.messages, self.last_query, target, formats)
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        for path in paths:
            self.chat_panel.append_system(f"saved: {path}")
        if asked:
            self._settings_hint("Save chat")

    # ----- Open Result / paste path ----------------------------------------

    async def on_open_result(self, e: ft.Event) -> None:
        """Open the file picker, load the chosen `.md` / `.txt` into the File view."""
        try:
            files = await self.file_picker.pick_files(
                dialog_title="Open saved answer",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["md", "txt"],
            )
        except Exception as exc:
            logger.warning("file picker failed: %r", exc)
            self.chat_panel.append_system(f"file picker error: {exc}")
            return
        if not files:
            return
        first = files[0]
        if first.path is None:
            logger.warning("picker returned a file with no path")
            return
        self._load_file_into_view(Path(first.path))

    def on_open_path(self, e: ft.Event) -> None:
        """Load the `.md` / `.txt` file whose path was typed into the paste-path
        field beside Open Result (Enter submits). Empty → no-op; a wrong
        extension or missing path → a chat message so the mistake is visible."""
        raw = (e.control.value or "").strip().strip('"')
        if not raw:
            return
        path = Path(raw)
        if path.suffix.lower() not in (".md", ".txt"):
            self.chat_panel.append_system(
                f"open a .md or .txt file (got {path.suffix or 'no extension'})"
            )
            return
        if not path.is_file():
            self.chat_panel.append_system(f"no file at: {raw}")
            return
        self._load_file_into_view(path)
        e.control.value = ""  # clear the field on a successful load
        self.page.update()

    def _load_file_into_view(self, path: Path) -> None:
        """Read a `.md` / `.txt` file into the File view — shared by the Open
        Result picker and the paste-path field. Read errors surface in the chat
        panel; on success the file is pushed as the newest history slot."""
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %r", path, exc)
            self.chat_panel.append_system(f"could not read file: {exc}")
            return
        self.loaded_file = _LoadedFile(name=path.name, content=content)
        self.right_panel.push_file(path.name, content)

    # ----- page assembly ----------------------------------------------------

    def build(self) -> None:
        self.page.title = "Knowledge Agent"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 12
        self.page.bgcolor = ft.Colors.BLACK
        self.page.window.width = WINDOW_WIDTH
        self.page.window.height = WINDOW_HEIGHT
        self.page.window.min_width = WINDOW_MIN_WIDTH
        self.page.window.min_height = WINDOW_MIN_HEIGHT

        # Cost-safety: forbid the agent config layer from reading the
        # developer's .env in this process — secrets MUST come from
        # the keyring or a shell export.
        disable_env_file()
        self.gui_config = load_config()
        apply_keys_to_env()
        apply_connection_to_env(self.gui_config)
        # Bridge the active corpus's Neo4j password (keyring -> env) so
        # backend reads work on launch, not only after a manual corpus
        # re-select. No-op (pops the var) when the active corpus has no
        # stored password.
        apply_active_corpus_password_to_env(self.gui_config)
        apply_retrieval_to_env(self.gui_config)
        apply_llm_to_env(self.gui_config)
        apply_embedding_to_env(self.gui_config)
        # Override the global embedder with the ACTIVE corpus's own
        # (corpus.toml) — the embedder is per-corpus (LanceDB pins the
        # vector dim at ingest). Must run AFTER apply_embedding_to_env.
        apply_active_corpus_embedding_to_env(self.gui_config)
        apply_ontology_downloads_dir_to_env(self.gui_config)
        get_settings.cache_clear()

        # Register FilePicker as a service (Flet 1.0+ API).
        self.file_picker = ft.FilePicker()

        # Build panels + tab modules.
        self.chat_panel = ChatPanel(self)
        self.right_panel = RightPanel(self)
        self.search_tab = SearchTab(self)
        self.library_tab = LibraryTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.installs_tab = InstallsTab(self)
        self.app_tab = AppTab(self)
        self.keys_tab = KeysTab(self)
        self.info_tab = InfoTab(self)
        self.log_tab = LogTab(self)

        # Native Flet `Tabs` (Flutter-style: TabBar + TabBarView aligned
        # by index through a shared `length`). The M3 chrome enforces a
        # ~46 px floor on Tab.height — accepted as the right cost for
        # native accessibility / keyboard nav / indicator animation.
        tab_bar = ft.TabBar(
            tabs=[
                ft.Tab(label="Search"),
                ft.Tab(label="Library"),
                ft.Tab(label="Evaluation"),
                ft.Tab(label="Installs"),
                ft.Tab(label="Keys"),
                ft.Tab(label="Settings"),
                ft.Tab(label="Log"),
                ft.Tab(label="Info"),
            ],
        )
        tab_bodies = ft.TabBarView(
            controls=[
                ft.Container(
                    content=self.search_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.library_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.evaluation_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.installs_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.keys_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.app_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.log_tab.build(),
                    padding=8,
                    expand=True,
                ),
                ft.Container(
                    content=self.info_tab.build(),
                    padding=8,
                    expand=True,
                ),
            ],
            expand=True,
        )
        # Global corpus selector on the SAME line as the tabs — the active
        # corpus is app-wide, so its selector belongs in global chrome, not
        # buried in Library → Select. A spacer pushes it to the right;
        # `⚙ Manage` opens the corpus-management dialog. Both the dropdown
        # and the dialog route through `select_corpus` (single source).
        self.corpus_dropdown, self.manage_button = self._build_corpus_selector()
        self._sync_corpus_selector()
        top_row = ft.Row(
            controls=[
                tab_bar,
                ft.Container(expand=True),  # spacer → right-align the selector
                ft.Text("Selected corpus:", size=14, weight=ft.FontWeight.BOLD),
                self.corpus_dropdown,
                self.manage_button,
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=8,
        )
        tabs = ft.Tabs(
            length=8,
            selected_index=0,
            content=ft.Column(
                controls=[top_row, tab_bodies],
                expand=True,
                spacing=0,
            ),
            expand=True,
        )

        self.page.add(tabs)

        # Register FilePicker as a service AFTER the page has content
        # so the service registry's internal update can push through
        # to the frontend correctly. Flet 0.85's public path is
        # `page._services.register_service(picker)`.
        self.page._services.register_service(self.file_picker)  # type: ignore[attr-defined]

        # Pre-warm the agent graph off the UI thread so the first query doesn't
        # pay the ~3s backend import (langgraph + lancedb + neo4j). Startup no
        # longer imports it; on_send loads it lazily — usually a cache hit by the
        # time the user sends.
        threading.Thread(target=_prewarm_graph, daemon=True).start()


def _import_graph() -> Any:
    """Import + return the agent graph. Heavy (~3s: langgraph + lancedb + neo4j),
    so it is kept out of GUI startup and loaded lazily on the first query. Reads
    the attribute fresh from its source module so tests can patch
    `knowledge_agent.graph.graph`."""
    from knowledge_agent.graph import graph

    return graph


def _prewarm_graph() -> None:
    """Best-effort background import (see build()) so the first query is a cache
    hit. Errors are swallowed here — the real load in on_send surfaces them."""
    with contextlib.suppress(Exception):  # best-effort; on_send re-raises for real
        _import_graph()


def _page_factory(page: ft.Page) -> None:
    GuiApp(page=page).build()


def main() -> None:
    """Entry point for `python -m knowledge_agent.gui` and the console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ft.run(_page_factory)


if __name__ == "__main__":
    main()
