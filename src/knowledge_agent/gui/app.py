"""Desktop GUI for the Knowledge Agent — coordinator.

Owns session state (`messages`, `last_answer`, `last_query`,
`loaded_file`, `busy`, `_send_task`) and the cross-cutting handlers
(Send → chat-router → graph, Stop, Clear, Save Answer, Save Chat,
Open Result, paste path). Page layout is top-level `ft.Tabs` with
3 entries:

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
from typing import Any

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
    KEYRING_TO_ENV,
    GuiConfig,
    active_results_dir,
    apply_keys_to_env,
    get_api_key,
    load_config,
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
        if embed_env and not (
            os.getenv(embed_env) or get_api_key(embed_provider)
        ):
            return embed_env
        return None

    def _load_corpus_config(self) -> CorpusConfig | None:
        """Load the corpus.toml the agent needs to build prompts.

        Resolution order:
          1. `gui_config.corpus_config_path` if set + the file exists.
          2. `corpus.toml` in the current working directory.
          3. None → caller surfaces a user-facing banner.
        """
        candidates: list[Path] = []
        if self.gui_config.corpus_config_path is not None:
            candidates.append(self.gui_config.corpus_config_path)
        candidates.append(Path.cwd() / "corpus.toml")
        for p in candidates:
            if p.is_file():
                try:
                    return load_corpus_config(p)
                except Exception as exc:
                    logger.warning(
                        "load_corpus_config(%s) failed: %r", p, exc,
                    )
        return None

    # ----- Send + chat-router + graph --------------------------------------

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
                f"missing API key: set {missing_env} in Settings before "
                "querying."
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
            # Direct-retrieve branch: bypass chat router + synthesizer
            # entirely (future slice will route this differently).
            if self.gui_config.direct_retrieve:
                self.chat_panel.append_system(
                    "direct_retrieve toggle is enabled but the bypass "
                    "path isn't wired yet — falling through to the "
                    "normal graph (slice 2 wires it)."
                )

            try:
                router = get_chat_router(
                    temperature=self.gui_config.chat_router_temperature,
                )
            except Exception as exc:
                self.chat_panel.append_system(
                    f"could not initialize chat router "
                    f"(check provider config): {exc}"
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

            mode = self.gui_config.retrieval_mode
            self._diag(f"retrieving (mode={mode}) ...")
            invoke_state: dict[str, Any] = {
                "query": text,
                "corpus_config": corpus_config,
                "top_k": self.gui_config.top_k,
                "skip_query_builder": self.gui_config.skip_query_builder,
                "retrieval_mode": mode,
            }
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
            self.messages.append(
                AIMessage(
                    content=(
                        f"(Answered from {len(answer.chunk_sources)} chunk "
                        f"+ {len(answer.kg_sources)} KG sources.)"
                    )
                )
            )
            self.chat_panel.append_system(
                f"answer ready — {len(answer.chunk_sources)} chunk "
                f"+ {len(answer.kg_sources)} KG sources. See display panel."
            )
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

    def _results_dir(self) -> Path:
        return active_results_dir(self.gui_config)

    def on_save_answer(self, e: ft.Event) -> None:
        if self.last_answer is None or self.last_query is None:
            self.chat_panel.append_system(
                "nothing to save yet — ask a question first"
            )
            return
        try:
            md_path, json_path = save_answer(
                self.last_answer, self.last_query, self._results_dir(),
            )
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        self.chat_panel.append_system(f"saved: {md_path}")
        self.chat_panel.append_system(f"saved: {json_path}")

    def on_save_chat(self, e: ft.Event) -> None:
        if not self.messages:
            self.chat_panel.append_system(
                "nothing to save — chat is empty"
            )
            return
        try:
            path = save_chat(
                self.messages, self.last_query, self._results_dir(),
            )
        except SaveError as exc:
            self.chat_panel.append_system(f"could not save: {exc}")
            return
        self.chat_panel.append_system(f"saved: {path}")

    # ----- Open Result / paste path ----------------------------------------

    async def on_open_result(self, e: ft.Event) -> None:
        """Open the file picker, load the chosen `.md` into the File view."""
        try:
            files = await self.file_picker.pick_files_async(
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

    def on_load_path_field(self, e: ft.Event) -> None:
        """Load the .md path the user typed/pasted into the path field."""
        field_ctl = self.right_panel.paste_path_field
        raw = (field_ctl.value or "").strip() if field_ctl is not None else ""
        if not raw:
            return
        path_str = raw.strip('"').strip("'")
        path = Path(path_str)
        if not path.exists():
            self.chat_panel.append_system(f"path does not exist: {path}")
            return
        if not path.is_file():
            self.chat_panel.append_system(f"not a regular file: {path}")
            return
        if path.suffix.lower() != ".md":
            self.chat_panel.append_system(
                f"only .md files supported; got: {path.suffix}"
            )
            return
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            self.chat_panel.append_system(f"could not read file: {exc}")
            return
        self.loaded_file = _LoadedFile(name=path.name, content=content)
        if field_ctl is not None:
            field_ctl.value = ""
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
        get_settings.cache_clear()

        # Register FilePicker as a service (Flet 1.0+ API).
        self.file_picker = ft.FilePicker()

        # Build panels + tab modules.
        self.chat_panel = ChatPanel(self)
        self.right_panel = RightPanel(self)
        self.search_tab = SearchTab(self)
        self.library_tab = LibraryTab(self)
        self.evaluation_tab = EvaluationTab(self)

        tabs = ft.Tabs(
            selected_index=0,
            expand=True,
            tabs=[
                ft.Tab(
                    text="Search",
                    content=ft.Container(
                        content=self.search_tab.build(),
                        padding=8,
                        expand=True,
                    ),
                ),
                ft.Tab(
                    text="Library",
                    content=ft.Container(
                        content=self.library_tab.build(),
                        padding=8,
                        expand=True,
                    ),
                ),
                ft.Tab(
                    text="Evaluation",
                    content=ft.Container(
                        content=self.evaluation_tab.build(),
                        padding=8,
                        expand=True,
                    ),
                ),
            ],
        )

        self.page.add(tabs)

        # FilePicker is a service in Flet 1.0+ — append AFTER the page
        # has content so the service registry is ready.
        try:
            self.page.services.append(self.file_picker)
        except AttributeError:
            # Older Flet (pre-1.0): fall back to the private registry.
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
