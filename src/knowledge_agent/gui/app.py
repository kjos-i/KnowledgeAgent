"""Desktop GUI for the Knowledge Agent — coordinator.

Owns session state (`messages`, `last_answer`, `last_query`,
`loaded_file`, `busy`, `_send_task`) and the cross-cutting handlers
(Send → chat-router → graph, Stop, Clear, Save chat, Save Result,
Open Result). Page layout is top-level `ft.Tabs` with 3 entries:

    [ Search ] [ Library ] [ Evaluation ]

Search composes `ChatPanel` (left) + `RightPanel` (right, mode-
switching: Latest / File / Settings / Info). Library and Evaluation
are top-level tabs because they need the full window for dense
data tables — that's the architectural reason they aren't right-
panel modes of Search.

Settings + Info live as right-panel modes of Search so the user can
read help / view current settings WHILE composing a query in the
chat. See `right_panel.py` for the mode list.
"""

from __future__ import annotations

import asyncio
import logging
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

from knowledge_agent.config import disable_env_file, get_settings
from knowledge_agent.graph import graph
from knowledge_agent.gui.artifacts import (
    SaveError,
    save_answer,
    save_chat,
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
)
from knowledge_agent.gui.right_panel import (
    MODE_FILE,
    MODE_LATEST,
    RightPanel,
)
from knowledge_agent.gui.tabs.evaluation_tab import EvaluationTab
from knowledge_agent.gui.tabs.library_tab import LibraryTab
from knowledge_agent.gui.tabs.search_tab import SearchTab
from knowledge_agent.kg.corpus_config import CorpusConfig, load_corpus_config

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
    gui_config: GuiConfig = field(default_factory=GuiConfig)
    busy: bool = False
    loaded_file: _LoadedFile | None = None

    # Late-bound (built in build()).
    chat_panel: ChatPanel = field(init=False)
    right_panel: RightPanel = field(init=False)
    search_tab: SearchTab = field(init=False)
    library_tab: LibraryTab = field(init=False)
    evaluation_tab: EvaluationTab = field(init=False)
    file_picker: ft.FilePicker = field(init=False)
    _send_task: asyncio.Task | None = field(default=None, init=False)
    _info_icons: list[ft.IconButton] = field(default_factory=list, init=False)

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

    def register_info_icon(self, button: ft.IconButton) -> None:
        """Track an `(i)` help icon so `set_info_icons_visible` can flip it."""
        self._info_icons.append(button)

    def set_info_icons_visible(self, visible: bool) -> None:
        """Show/hide every registered (i) help icon at once (live)."""
        for button in self._info_icons:
            button.visible = visible
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

    def _invoke_state_for_input_mode(
        self,
        input_mode: str,
        text: str,
        corpus_config: CorpusConfig,
        search_query: str | None,
    ) -> dict[str, Any]:
        """Build the graph invoke-state for the selected input mode.

        - conversational: the router's distilled `search_query` (fallback to
          the raw text); the query-builder runs; retrieval mode = user's
          choice.
        - direct_query: raw text straight to vector/hybrid (skip the
          query-builder); mode forced to lancedb_only.
        - direct_cypher: raw text run as user Cypher against the KG; mode
          forced to neo4j_only.
        """
        base: dict[str, Any] = {
            "corpus_config": corpus_config,
            "top_k": self.gui_config.top_k,
            "direct_retrieval": self.gui_config.direct_retrieve,
        }
        if input_mode == "direct_query":
            return {
                **base,
                "query": text,
                "skip_query_builder": True,
                "retrieval_mode": "lancedb_only",
            }
        if input_mode == "direct_cypher":
            return {
                **base,
                "query": text,
                "user_cypher": text,
                "retrieval_mode": "neo4j_only",
            }
        return {
            **base,
            "query": (search_query or "").strip() or text,
            "skip_query_builder": False,
            "retrieval_mode": self.gui_config.retrieval_mode,
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
                final_state = await graph.ainvoke(invoke_state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("graph.ainvoke raised: %r", exc)
                self.chat_panel.append_system(f"retrieval failed: {exc}")
                return

            answer: AgentAnswer | None = final_state.get("final_answer")
            if answer is None:
                self.chat_panel.append_system("no answer produced")
                return

            self.last_answer = answer
            self.last_query = text
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
            # Refresh the right panel so the new answer shows in Latest.
            if self.right_panel.current_mode == MODE_LATEST:
                self.right_panel.switch_mode(MODE_LATEST)
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
        # Per `keep_loaded_file_on_clear`: when off, also wipe the
        # loaded file and pull the display back to Latest view.
        if not self.gui_config.keep_loaded_file_on_clear:
            self.loaded_file = None
            if self.right_panel.current_mode != MODE_LATEST:
                self.right_panel.switch_mode(MODE_LATEST)
                self.page.update()
                return
        if self.right_panel.current_mode == MODE_LATEST:
            self.right_panel.switch_mode(MODE_LATEST)
        self.page.update()

    async def _prompt_save_directory(self, dialog_title: str) -> Path | None:
        """Open the OS folder picker; remember the choice.

        Returns the chosen path (and persists it as `gui_config.results_dir`
        so the next press opens at the same place), or None on cancel.

        `results_dir` is no longer a configurable default — it's hidden
        last-used state behind the Save buttons. Settings has no form
        entry for it.
        """
        initial = (
            str(self.gui_config.results_dir) if self.gui_config.results_dir is not None else None
        )
        try:
            chosen = await self.file_picker.get_directory_path(
                dialog_title=dialog_title,
                initial_directory=initial,
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            self.chat_panel.append_system(f"folder picker error: {exc}")
            return None
        if not chosen:
            return None
        path = Path(chosen)
        # Remember the picked folder so the next press opens here.
        previous = self.gui_config.results_dir
        self.gui_config.results_dir = path
        try:
            save_config(self.gui_config)
        except ConfigError as exc:
            # Save failure is non-fatal — the picker still works next time,
            # but the picker won't restore this folder until restart.
            logger.warning("could not persist results_dir: %r", exc)
            self.gui_config.results_dir = previous
        return path

    async def on_save_answer(self, e: ft.Event) -> None:
        if self.last_answer is None or self.last_query is None:
            self.chat_panel.append_system("nothing to save yet — ask a question first")
            return
        target = await self._prompt_save_directory("Save result to folder")
        if target is None:
            return
        try:
            md_path, json_path = save_answer(
                self.last_answer,
                self.last_query,
                target,
            )
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        self.chat_panel.append_system(f"saved: {md_path}")
        self.chat_panel.append_system(f"saved: {json_path}")

    async def on_save_chat(self, e: ft.Event) -> None:
        if not self.messages:
            self.chat_panel.append_system("nothing to save — chat is empty")
            return
        target = await self._prompt_save_directory("Save chat to folder")
        if target is None:
            return
        try:
            path = save_chat(
                self.messages,
                self.last_query,
                target,
            )
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        self.chat_panel.append_system(f"saved: {path}")

    # ----- Open Result / paste path ----------------------------------------

    async def on_open_result(self, e: ft.Event) -> None:
        """Open the file picker, load the chosen `.md` into the File view."""
        try:
            files = await self.file_picker.pick_files(
                dialog_title="Open saved answer",
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=["md"],
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
        try:
            content = Path(first.path).read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read %s: %r", first.path, exc)
            self.chat_panel.append_system(f"could not read file: {exc}")
            return
        self.loaded_file = _LoadedFile(name=first.name, content=content)
        self.right_panel.switch_mode(MODE_FILE)

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

        # Native Flet `Tabs` (Flutter-style: TabBar + TabBarView aligned
        # by index through a shared `length`). The M3 chrome enforces a
        # ~46 px floor on Tab.height — accepted as the right cost for
        # native accessibility / keyboard nav / indicator animation.
        tab_bar = ft.TabBar(
            tabs=[
                ft.Tab(label="Search"),
                ft.Tab(label="Library"),
                ft.Tab(label="Evaluation"),
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
            ],
            expand=True,
        )
        tabs = ft.Tabs(
            length=3,
            selected_index=0,
            content=ft.Column(
                controls=[tab_bar, tab_bodies],
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
