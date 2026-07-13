"""Library → Installs sub-tab — global install surface.

Flat section layout mirroring `gui/settings/llm_tab.py` (bold header +
description + rows + Divider between sections). No ExpansionTiles —
the user needs to see every install target at a glance to decide
what's next.

Three sections, plus a global-config header:

  0. **Global config** — `ontology_downloads_dir` path (read-only for
     now; editable landing in Slice 4). Downloaded ontology source
     files live here; shared across every corpus.
  1. **Ontologies** — 18 rows. Status = disk state (whether the
     source file has been downloaded to `ontology_downloads_dir`),
     with size in bytes when present. Neo4j node writes are NOT
     managed here — they happen automatically during ingest when a
     corpus with the ontology enabled runs. One button axis:
       - `Download`/`Delete download` — wraps
         `download_ontology_download_execute` /
         `delete_ontology_download_execute` (disk only).
  2. **Entity extractors** — 4 rows. Status = compound (pip package +
     pinned weights on disk). Two independent axes:
       - `Install`/`Uninstall` — pip package (wraps
         `install_extractor_execute` / `uninstall_extractor_execute`).
       - `Download weights`/`Delete download` — HF-cached model files
         (wraps `download_extractor_weights_execute` /
         `delete_extractor_weights_execute`).
     Disk size for downloaded weights shown in the status chip.
  3. **Parsers** — 2 rows. Pip install/uninstall wired to
     `install_parser_extra_execute` / `uninstall_parser_extra_execute`.
     Whisper weights auto-download inside Docling on first ingest use
     — disclosed in the row text rather than fought upstream.

Button label flip: like the LLM tab, `Install` becomes `Uninstall`
(and `Download` becomes `Delete download`) based on live state so the
user gets one button per action, not two mutually-exclusive buttons
stacked.

Startup rule (per `gui-view-startup` feedback memory): no work in
`_create_controls`; all lifecycle queries defer to the first `build()`
call. Ontology state is a disk probe (via
`ontology_lifecycle.is_ontology_downloaded` / `get_ontology_download_bytes`),
so the tab renders synchronously — no background task needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.embedder_lifecycle import (
    EMBEDDER_PROVIDER_REGISTRY,
    install_embedder_provider_execute,
    install_embedder_provider_plan,
    uninstall_embedder_provider_execute,
    uninstall_embedder_provider_plan,
)
from knowledge_agent.entity_extractors.extractor_lifecycle import (
    EXTRACTOR_REGISTRY,
    delete_extractor_weights_execute,
    delete_extractor_weights_plan,
    download_extractor_weights_execute,
    download_extractor_weights_plan,
    get_weights_downloaded_bytes,
    install_extractor_execute,
    install_extractor_plan,
    uninstall_extractor_execute,
    uninstall_extractor_plan,
)
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
    labeled_field,
    section_divider,
)
from knowledge_agent.gui._widgets.info_icon import section_header
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_ontology_downloads_dir_to_env,
    save_config,
)
from knowledge_agent.gui.library.corpus_config_editor import _ONTOLOGY_DISPLAY
from knowledge_agent.gui.views._frame import view_with_header
from knowledge_agent.ingestion.parser_lifecycle import (
    PARSER_LIFECYCLE_REGISTRY,
    install_parser_extra_execute,
    install_parser_extra_plan,
    uninstall_parser_extra_execute,
    uninstall_parser_extra_plan,
)
from knowledge_agent.kg.ontology_lifecycle import (
    _safe_downloads_dir,
    delete_ontology_download_execute,
    delete_ontology_download_plan,
    download_ontology_download_execute,
    download_ontology_download_plan,
    get_ontology_download_bytes,
    is_ontology_downloaded,
)
from knowledge_agent.kg.ontology_linking import ONTOLOGY_REGISTRY
from knowledge_agent.llm_lifecycle import (
    LLM_PROVIDER_REGISTRY,
    install_llm_provider_execute,
    install_llm_provider_plan,
    uninstall_llm_provider_execute,
    uninstall_llm_provider_plan,
)

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


_ONTOLOGY_ORDER: tuple[str, ...] = tuple(_ONTOLOGY_DISPLAY.keys())
_EXTRACTOR_ORDER: tuple[str, ...] = tuple(EXTRACTOR_REGISTRY.keys())
_PARSER_ORDER: tuple[str, ...] = tuple(PARSER_LIFECYCLE_REGISTRY.keys())
_EMBEDDER_PROVIDER_ORDER: tuple[str, ...] = ("voyage", "openai", "google", "huggingface")
_LLM_PROVIDER_ORDER: tuple[str, ...] = ("anthropic", "openai", "google", "ollama")


def _fmt_bytes(n: int) -> str:
    """Human-readable disk size. Truncates instead of rounding — 245 MB
    reads cleaner than 245.371... MB and the extra precision isn't
    useful at Installs-tab granularity."""
    if n <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            if unit == "B":
                return f"{n} {unit}"
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n} PB"


def _source_link_button(url: str) -> ft.IconButton:
    """Small open-in-new icon linking to an install target's source /
    homepage page. Declarative `url=` (same mechanism as the LangSmith
    links) — the OS browser opens it; no async handler needed."""
    return ft.IconButton(
        icon=ft.Icons.OPEN_IN_NEW,
        icon_size=16,
        icon_color=ft.Colors.BLUE_300,
        width=40,
        url=url,
        tooltip=f"Open source page — {url}",
    )


def _resolve_hf_hub_cache_dir() -> Path | None:
    """Return the huggingface_hub cache dir (where extractor weights land).

    Reuses `huggingface_hub.constants.HF_HUB_CACHE` so we never disagree
    with the library. Honours `HF_HOME` / `HF_HUB_CACHE` env overrides.
    None when `huggingface_hub` isn't installed — the row still renders
    with an explanatory placeholder.
    """
    try:
        from huggingface_hub import constants as _hf_constants

        return Path(_hf_constants.HF_HUB_CACHE)
    except ImportError:
        return None


def _resolve_ollama_models_dir() -> Path:
    """Return the directory where Ollama stores pulled models.

    Ollama's precedence:
      1. `OLLAMA_MODELS` env var (explicit override)
      2. `~/.ollama/models` (universal default across Windows/macOS/Linux
         for the user-space install path — the deb / systemd install
         additionally uses `/usr/share/ollama/.ollama/models` but the GUI
         is a user-space process so the home path is the useful one).

    Always returns a Path even when the directory doesn't exist yet —
    Ollama creates it on first pull.
    """
    import os

    env_override = os.environ.get("OLLAMA_MODELS")
    if env_override:
        return Path(env_override)
    return Path.home() / ".ollama" / "models"


class InstallsTab:
    """Global install surface — ontologies, extractors, parsers, providers.

    Provider *installs* (embedder / LLM pip adapters) live here — the single
    machine-level install surface. The active-provider CHOICE + model stay in
    their own tabs (Embedding settings / Search → LLM), which read install
    state to offer only installed providers.
    """

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        # Strong refs to in-flight background tasks (folder picker) so they
        # aren't GC'd mid-run; they self-discard on completion.
        self._bg_tasks: set[asyncio.Task] = set()

        # Ontology row widgets. One button axis: Download ↔ Delete
        # download (disk only; Neo4j node writes remain ingest's job).
        self.ontology_status_texts: dict[str, ft.Text] = {}
        self.ontology_download_buttons: dict[str, ft.Button] = {}
        self.ontology_delete_buttons: dict[str, ft.Button] = {}

        # Extractor row widgets. Two independent button axes:
        # pip (install/uninstall) + weights (download/delete). Buttons
        # are created twice per row (one per state) and the flip is done
        # by toggling `.visible`, mirroring the LLM tab's pattern.
        self.extractor_status_texts: dict[str, ft.Text] = {}
        self.extractor_install_buttons: dict[str, ft.Button] = {}
        self.extractor_uninstall_buttons: dict[str, ft.Button] = {}
        self.extractor_download_buttons: dict[str, ft.Button] = {}
        self.extractor_delete_buttons: dict[str, ft.Button] = {}

        # Parser row widgets (Install / Uninstall pip extras).
        self.parser_status_texts: dict[str, ft.Text] = {}
        self.parser_install_buttons: dict[str, ft.Button] = {}
        self.parser_uninstall_buttons: dict[str, ft.Button] = {}

        # Embedding-provider row widgets (Install / Uninstall the pip
        # adapter). The active-provider choice + model live in the Embedding
        # settings tab; only the install lives here.
        self.embedder_provider_status_texts: dict[str, ft.Text] = {}
        self.embedder_provider_install_buttons: dict[str, ft.Button] = {}
        self.embedder_provider_uninstall_buttons: dict[str, ft.Button] = {}

        # LLM-provider row widgets — same shape as the embedder rows. The
        # active-provider choice + Ollama daemon status stay in the LLM tab.
        self.llm_provider_status_texts: dict[str, ft.Text] = {}
        self.llm_provider_install_buttons: dict[str, ft.Button] = {}
        self.llm_provider_uninstall_buttons: dict[str, ft.Button] = {}

        # ontology_downloads_dir — EDITABLE TextField backed by
        # `GuiConfig.ontology_downloads_dir` (JSON persistence via
        # save_config; env bridge via apply_ontology_downloads_dir_to_env).
        # Blank = fall back to backend Settings default.
        self.downloads_dir_field: ft.TextField | None = None
        # Effective ontology downloads dir — read-only echo of where files
        # ACTUALLY land (the override when set, the backend default when the
        # field is blank), so a blank field isn't an invisible location.
        self.effective_downloads_display: ft.Text | None = None
        # HF hub + Ollama models dirs — read-only. Third-party libraries
        # own these locations; GUI just displays.
        self.hf_hub_display: ft.Text | None = None
        self.ollama_models_display: ft.Text | None = None

        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        """Widgets only. No lifecycle queries (see startup rules)."""
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        # Editable ontology_downloads_dir field. Initial value populated
        # from GuiConfig on first build. Empty = fall back to Settings
        # default (bridge deletes the env var in that case).
        stored = getattr(self.app.gui_config, "ontology_downloads_dir", None)
        self.downloads_dir_field = ft.TextField(
            value="" if stored is None else str(stored),
            hint_text="Leave blank to use the backend default",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_downloads_dir_blur,
        )
        self.downloads_dir_browse_button = ft.Button(
            content=centered_label("Browse"),
            on_click=self._on_downloads_dir_browse_clicked,
        )
        self.effective_downloads_display = ft.Text(
            "(checking…)",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.hf_hub_display = ft.Text(
            "(checking…)",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.ollama_models_display = ft.Text(
            "(checking…)",
            size=12,
            color=ft.Colors.GREY_300,
        )

        # Ontology: 2 buttons per row (Download / Delete download),
        # flipped by visibility. Node writes remain ingest's job — no
        # "Install" button here.
        for name in _ONTOLOGY_ORDER:
            self.ontology_status_texts[name] = ft.Text(
                "(checking…)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
            self.ontology_download_buttons[name] = ft.Button(
                content=centered_label("Download"),
                on_click=lambda e, n=name: self._on_ontology_download(n),
            )
            self.ontology_delete_buttons[name] = ft.Button(
                content=centered_label("Delete download"),
                on_click=lambda e, n=name: self._on_ontology_delete(n),
            )

        # Extractor: 4 buttons per row (install/uninstall + download/delete),
        # flipped by visibility based on compound state.
        for name in _EXTRACTOR_ORDER:
            self.extractor_status_texts[name] = ft.Text(
                "(checking…)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
            self.extractor_install_buttons[name] = ft.Button(
                content=centered_label("Install"),
                on_click=lambda e, n=name: self._on_extractor_install(n),
            )
            self.extractor_uninstall_buttons[name] = ft.Button(
                content=centered_label("Uninstall"),
                on_click=lambda e, n=name: self._on_extractor_uninstall(n),
            )
            self.extractor_download_buttons[name] = ft.Button(
                content=centered_label("Download weights"),
                on_click=lambda e, n=name: self._on_extractor_download(n),
            )
            self.extractor_delete_buttons[name] = ft.Button(
                content=centered_label("Delete download"),
                on_click=lambda e, n=name: self._on_extractor_delete(n),
            )

        # Parser: 2 buttons (Install / Uninstall) — pip extras via
        # install_parser_extra_* (wired below).
        for name in _PARSER_ORDER:
            self.parser_status_texts[name] = ft.Text(
                "(checking…)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
            self.parser_install_buttons[name] = ft.Button(
                content=centered_label("Install"),
                on_click=lambda e, n=name: self._on_parser_install(n),
            )
            self.parser_uninstall_buttons[name] = ft.Button(
                content=centered_label("Uninstall"),
                on_click=lambda e, n=name: self._on_parser_uninstall(n),
            )

        # Embedding providers: 2 buttons (Install / Uninstall pip adapter),
        # flipped by visibility. Same shared backend the Embedding tab used.
        for name in _EMBEDDER_PROVIDER_ORDER:
            self.embedder_provider_status_texts[name] = ft.Text(
                "(checking…)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
            self.embedder_provider_install_buttons[name] = ft.Button(
                content=centered_label("Install"),
                on_click=lambda e, n=name: self._on_embedder_provider_install(n),
            )
            self.embedder_provider_uninstall_buttons[name] = ft.Button(
                content=centered_label("Uninstall"),
                on_click=lambda e, n=name: self._on_embedder_provider_uninstall(n),
            )

        for name in _LLM_PROVIDER_ORDER:
            self.llm_provider_status_texts[name] = ft.Text(
                "(checking…)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
            self.llm_provider_install_buttons[name] = ft.Button(
                content=centered_label("Install"),
                on_click=lambda e, n=name: self._on_llm_provider_install(n),
            )
            self.llm_provider_uninstall_buttons[name] = ft.Button(
                content=centered_label("Uninstall"),
                on_click=lambda e, n=name: self._on_llm_provider_uninstall(n),
            )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        if self._first_build:
            self._first_build = False
            self._sync_downloads_dir()
            self._sync_extractor_state()
            self._sync_parser_state()
            self._sync_embedder_provider_state()
            self._sync_llm_provider_state()
            # Ontology state is a disk probe now (no Neo4j) — cheap +
            # sync. The former async Neo4j probe moved into the Ingest
            # tab's bulk_ops surface where node writes belong.
            self._sync_ontology_state()

        controls: list[ft.Control] = [
            # ---- Global paths --------------------------------------
            section_header(
                self.app,
                "Global download locations",
                "Where each install target keeps its files. The first is "
                "editable — the other two are managed by their libraries "
                "(HF Hub / Ollama) and shown for reference.",
            ),
            # ontology_downloads_dir — editable + Browse
            labeled_field(
                "ontology_downloads_dir",
                self.downloads_dir_field,
                trailing=self.downloads_dir_browse_button,
            ),
            # Effective ontology location — read-only echo (shows the backend
            # default when the field above is blank).
            ft.Row(
                spacing=6,
                controls=[
                    ft.Text(
                        "Effective location:",
                        size=12,
                        color=ft.Colors.GREY_400,
                        width=280,
                    ),
                    self.effective_downloads_display,
                ],
            ),
            # HF hub cache dir — read-only display
            ft.Row(
                spacing=6,
                controls=[
                    ft.Text(
                        "HF Hub cache (extractor weights):",
                        size=12,
                        color=ft.Colors.GREY_400,
                        width=280,
                    ),
                    self.hf_hub_display,
                ],
            ),
            # Ollama models dir — read-only display
            ft.Row(
                spacing=6,
                controls=[
                    ft.Text(
                        "Ollama models (local LLMs):",
                        size=12,
                        color=ft.Colors.GREY_400,
                        width=280,
                    ),
                    self.ollama_models_display,
                ],
            ),
            section_divider(),
        ]

        # ---- Ontologies (18) ---------------------------------------
        controls.append(
            section_header(
                self.app,
                "Ontologies (18)",
                "Download the source file(s) to disk here. Neo4j term nodes "
                "are written during ingest — not by these buttons — so "
                "downloading is safe (no schema change).",
            )
        )
        for name in _ONTOLOGY_ORDER:
            ont_prov = ONTOLOGY_REGISTRY.get(name, {}).get("provenance")
            controls.append(
                self._simple_row(
                    _ONTOLOGY_DISPLAY[name],
                    self.ontology_status_texts[name],
                    (
                        self.ontology_download_buttons[name],
                        self.ontology_delete_buttons[name],
                    ),
                    source_url=ont_prov.source_url if ont_prov else None,
                )
            )
        controls.append(section_divider())

        # ---- Entity extractors (4) ---------------------------------
        controls.append(
            section_header(
                self.app,
                "Entity extractors (4)",
                "L6 adapters. LLM is bundled. GLiNER / GLiNER-BioMed / HunFlair2 "
                "need BOTH the pip extras (adapter library) AND their pinned "
                "model weights downloaded. No auto-download at first inference — "
                "extraction raises if weights are missing.",
            )
        )
        for name in _EXTRACTOR_ORDER:
            display = EXTRACTOR_REGISTRY[name]["display_name"]
            ext_prov = EXTRACTOR_REGISTRY[name].get("provenance")
            controls.append(
                self._simple_row(
                    display,
                    self.extractor_status_texts[name],
                    (
                        self.extractor_install_buttons[name],
                        self.extractor_uninstall_buttons[name],
                        self.extractor_download_buttons[name],
                        self.extractor_delete_buttons[name],
                    ),
                    source_url=ext_prov.source_url if ext_prov else None,
                )
            )
        controls.append(section_divider())

        # ---- Parsers (2) -------------------------------------------
        controls.append(
            section_header(
                self.app,
                "Parsers (2)",
                "Optional docling extras. ASR (audio + video transcription): pip "
                "install of the extra pulls openai-whisper + bundled ffmpeg. "
                "Whisper model weights (~1.5 GB) are downloaded by Docling on "
                "first ingest use — not managed here. AST-aware code parsing "
                "ships its tree-sitter grammars inside the pip wheel.",
            )
        )
        for name in _PARSER_ORDER:
            display = PARSER_LIFECYCLE_REGISTRY[name]["display_name"]
            # Parsers install via pip, so the download source IS the PyPI
            # project of the extra's primary package — derived from the
            # registry's `library_packages`, no separate URL to maintain.
            pkgs = PARSER_LIFECYCLE_REGISTRY[name].get("library_packages") or ()
            parser_source_url = f"https://pypi.org/project/{pkgs[0]}/" if pkgs else None
            controls.append(
                self._simple_row(
                    display,
                    self.parser_status_texts[name],
                    (
                        self.parser_install_buttons[name],
                        self.parser_uninstall_buttons[name],
                    ),
                    source_url=parser_source_url,
                )
            )
        controls.append(section_divider())

        # ---- Embedding providers (4) -------------------------------
        controls.append(
            section_header(
                self.app,
                "Embedding providers (4)",
                "Install / Uninstall each embedder's pip adapter (confirm "
                "dialog; a restart is needed after). Choose the ACTIVE embedder "
                "+ model in the Embedding settings, not here. The active "
                "embedder can't be uninstalled — switch it there first.",
            )
        )
        for name in _EMBEDDER_PROVIDER_ORDER:
            controls.append(
                self._simple_row(
                    EMBEDDER_PROVIDER_REGISTRY[name]["display_name"],
                    self.embedder_provider_status_texts[name],
                    (
                        self.embedder_provider_install_buttons[name],
                        self.embedder_provider_uninstall_buttons[name],
                    ),
                )
            )
        controls.append(section_divider())

        # ---- LLM providers (4) -------------------------------------
        controls.append(
            section_header(
                self.app,
                "LLM providers (4)",
                "Install / Uninstall each LLM adapter's pip package (confirm "
                "dialog; a restart is needed after). Choose the ACTIVE LLM + "
                "per-node models in Search → LLM, not here (that's also where "
                "the Ollama daemon status lives). The active LLM can't be "
                "uninstalled — switch it there first.",
            )
        )
        for name in _LLM_PROVIDER_ORDER:
            controls.append(
                self._simple_row(
                    LLM_PROVIDER_REGISTRY[name]["display_name"],
                    self.llm_provider_status_texts[name],
                    (
                        self.llm_provider_install_buttons[name],
                        self.llm_provider_uninstall_buttons[name],
                    ),
                )
            )
        controls.append(self.status)
        # `expand=True` on the inner Column is REQUIRED for scrolling to
        # work — without it the Column shrinks to intrinsic content
        # height, so once the 18-row Ontologies section fills the visible
        # area the Extractor + Parser sections are pushed off-screen and
        # unreachable via scroll. Matches the LLM tab's Column setup.
        return view_with_header(
            "Installs",
            ft.Column(
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=10,
                controls=controls,
            ),
        )

    def _simple_row(
        self,
        display_name: str,
        status_text: ft.Text,
        buttons: tuple[ft.Button, ...],
        *,
        source_url: str | None = None,
    ) -> ft.Control:
        """One install row: name + status + optional source-link icon + N
        buttons (flipped by visibility). When `source_url` is None a fixed
        spacer holds the icon's slot so the action buttons stay aligned
        across rows that do and don't have a link."""
        link_slot: ft.Control = (
            _source_link_button(source_url) if source_url else ft.Container(width=40)
        )
        return ft.Row(
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                ft.Text(display_name, size=12, width=280),
                ft.Container(content=status_text, expand=True),
                link_slot,
                *buttons,
            ],
        )

    # ----- state sync ------------------------------------------------------

    def _sync_downloads_dir(self) -> None:
        """Populate the HF hub + Ollama read-only displays.

        Editable `downloads_dir_field` is populated on `_create_controls`
        from `GuiConfig` — no live-Settings read needed here (would
        overwrite user edits mid-session). The two library-owned paths
        below reflect where those libraries put files regardless of our
        settings.
        """
        if self.effective_downloads_display is not None:
            effective = _safe_downloads_dir()
            self.effective_downloads_display.value = (
                str(effective) if effective is not None else "(could not resolve)"
            )
        if self.hf_hub_display is not None:
            hf = _resolve_hf_hub_cache_dir()
            if hf is None:
                self.hf_hub_display.value = "(huggingface_hub not installed)"
            else:
                self.hf_hub_display.value = str(hf)
        if self.ollama_models_display is not None:
            self.ollama_models_display.value = str(_resolve_ollama_models_dir())

    def _spawn(self, coro) -> None:
        """Fire-and-forget a coroutine while holding a strong reference
        until it completes (avoids the task being garbage-collected)."""
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _on_downloads_dir_browse_clicked(self, e: ft.Event) -> None:
        """Open a native folder picker for `ontology_downloads_dir`.

        Guarded off without a running loop so unit tests can call it
        without leaving an un-awaited coroutine."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spawn(self._pick_downloads_dir(e))

    async def _pick_downloads_dir(self, e: ft.Event) -> None:
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title="Pick the ontology downloads folder",
            )
        except Exception as exc:
            self._set_status(f"folder picker error: {exc}", ok=False)
            return
        if chosen and self.downloads_dir_field is not None:
            self.downloads_dir_field.value = chosen
            # Reuse the blur persist path (validate → save → env bridge →
            # re-probe). It reads the field value and ignores the event.
            self._on_downloads_dir_blur(e)

    def _on_downloads_dir_blur(self, e: ft.Event) -> None:
        """Persist `ontology_downloads_dir` to GuiConfig on blur.

        Empty string → None → env-var deleted → backend Settings default
        wins. Non-empty → validated as `Path` → written to GuiConfig +
        env var. `reset_after_key_change()` clears the Settings cache
        so `get_settings()` picks up the change without a restart.

        Persisted via `save_config` (JSON at `<config_dir>/settings.json`).
        The GUI NEVER writes to `.env` — that file is developer-only.
        """
        if self.downloads_dir_field is None:
            return
        raw = (self.downloads_dir_field.value or "").strip()
        new_value: Path | None = None if not raw else Path(raw)
        current = self.app.gui_config.ontology_downloads_dir
        if new_value == current:
            return
        previous = current
        self.app.gui_config.ontology_downloads_dir = new_value
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            # Rollback: field + config.
            self.app.gui_config.ontology_downloads_dir = previous
            self.downloads_dir_field.value = "" if previous is None else str(previous)
            self._set_status(
                f"Could not save ontology_downloads_dir: {exc}",
                ok=False,
            )
            return
        apply_ontology_downloads_dir_to_env(self.app.gui_config)
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)
        if new_value is None:
            self._set_status("ontology_downloads_dir reset — using backend default.")
        else:
            self._set_status(f"Saved ontology_downloads_dir: {new_value}")
        # Re-run the ontology disk probes against the new path so the
        # rows update immediately, and refresh the effective-location echo.
        self._sync_downloads_dir()
        self._sync_ontology_state()
        self._safe_update()

    def _sync_extractor_state(self) -> None:
        """Compound state (pip + weights) drives status chip + button flip."""
        for name in _EXTRACTOR_ORDER:
            entry = EXTRACTOR_REGISTRY[name]
            bundled = bool(entry.get("bundled"))
            pip_installed = self._safe_bool(entry.get("is_installed_fn"))
            provenance = entry.get("provenance")
            has_weights_concept = provenance is not None
            weights_bytes = get_weights_downloaded_bytes(name)
            weights_present = weights_bytes > 0

            status_text = self.extractor_status_texts[name]
            install_btn = self.extractor_install_buttons[name]
            uninstall_btn = self.extractor_uninstall_buttons[name]
            download_btn = self.extractor_download_buttons[name]
            delete_btn = self.extractor_delete_buttons[name]

            # Status chip.
            if bundled:
                status_text.value = "bundled (always available)"
                status_text.color = ft.Colors.GREY_400
            elif pip_installed and (not has_weights_concept or weights_present):
                size_clause = f" ({_fmt_bytes(weights_bytes)})" if has_weights_concept else ""
                status_text.value = f"✓ ready — pip + weights{size_clause}"
                status_text.color = ft.Colors.GREEN_300
            elif pip_installed and not weights_present:
                status_text.value = "○ pip installed; weights not downloaded"
                status_text.color = ft.Colors.AMBER_300
            elif not pip_installed and weights_present:
                status_text.value = (
                    f"○ weights downloaded ({_fmt_bytes(weights_bytes)}); pip not installed"
                )
                status_text.color = ft.Colors.AMBER_300
            else:
                status_text.value = "○ not installed (pip + weights both needed)"
                status_text.color = ft.Colors.GREY_400

            # Buttons. Bundled hides all four (nothing to do).
            if bundled:
                for b in (install_btn, uninstall_btn, download_btn, delete_btn):
                    b.visible = False
                continue

            install_btn.visible = not pip_installed
            uninstall_btn.visible = pip_installed

            if has_weights_concept:
                download_btn.visible = not weights_present
                delete_btn.visible = weights_present
            else:
                download_btn.visible = False
                delete_btn.visible = False

    def _sync_parser_state(self) -> None:
        """Simple pip-only state (Docling handles its own weights)."""
        for name in _PARSER_ORDER:
            entry = PARSER_LIFECYCLE_REGISTRY[name]
            installed = self._safe_bool(entry.get("is_installed_fn"))
            status_text = self.parser_status_texts[name]
            install_btn = self.parser_install_buttons[name]
            uninstall_btn = self.parser_uninstall_buttons[name]

            if installed:
                # For the ASR parser, surface Docling's autodownload
                # behavior in-line so the user isn't surprised by a
                # 1.5 GB fetch on their first ingest.
                if name == "asr":
                    status_text.value = (
                        "✓ installed — Docling downloads Whisper (~1.5 GB) on first ingest"
                    )
                else:
                    status_text.value = "✓ installed"
                status_text.color = ft.Colors.GREEN_300
            else:
                status_text.value = "○ not installed"
                status_text.color = ft.Colors.GREY_400
            install_btn.visible = not installed
            uninstall_btn.visible = installed

    def _sync_embedder_provider_state(self) -> None:
        """Install-state (pip adapter) per embedding provider. The active
        embedder's Uninstall is disabled — switch it in the Embedding settings
        first (same install-here / choose-there split as ontologies +
        extractors)."""
        active = self.app.gui_config.embedding_provider
        for name in _EMBEDDER_PROVIDER_ORDER:
            entry = EMBEDDER_PROVIDER_REGISTRY[name]
            installed = self._safe_bool(entry.get("is_installed_fn"))
            status_text = self.embedder_provider_status_texts[name]
            install_btn = self.embedder_provider_install_buttons[name]
            uninstall_btn = self.embedder_provider_uninstall_buttons[name]
            if installed:
                status_text.value = "✓ installed"
                status_text.color = ft.Colors.GREEN_300
            else:
                status_text.value = "○ not installed"
                status_text.color = ft.Colors.GREY_400
            install_btn.visible = not installed
            uninstall_btn.visible = installed
            if installed and name == active:
                uninstall_btn.disabled = True
                uninstall_btn.tooltip = (
                    "Active embedder can't be uninstalled — switch it in the "
                    "Embedding settings first."
                )
            else:
                uninstall_btn.disabled = False
                uninstall_btn.tooltip = None

    def _sync_llm_provider_state(self) -> None:
        """Install-state (pip adapter) per LLM provider. The active LLM's
        Uninstall is disabled — switch it in Search → LLM first. Ollama's
        DAEMON reachability is a runtime concern shown in the LLM tab, not
        here — this row is only 'is the adapter pip-installed'."""
        active = self.app.gui_config.llm_provider
        for name in _LLM_PROVIDER_ORDER:
            entry = LLM_PROVIDER_REGISTRY[name]
            installed = self._safe_bool(entry.get("is_installed_fn"))
            status_text = self.llm_provider_status_texts[name]
            install_btn = self.llm_provider_install_buttons[name]
            uninstall_btn = self.llm_provider_uninstall_buttons[name]
            if installed:
                status_text.value = "✓ installed"
                status_text.color = ft.Colors.GREEN_300
            else:
                status_text.value = "○ not installed"
                status_text.color = ft.Colors.GREY_400
            install_btn.visible = not installed
            uninstall_btn.visible = installed
            if installed and name == active:
                uninstall_btn.disabled = True
                uninstall_btn.tooltip = (
                    "Active LLM can't be uninstalled — switch it in Search → LLM first."
                )
            else:
                uninstall_btn.disabled = False
                uninstall_btn.tooltip = None

    def _sync_ontology_state(self) -> None:
        """Read on-disk state for each ontology and update its row.

        Pure disk read — no Neo4j probe. Fast + synchronous. Ingest
        owns the Neo4j-write path; whether an ontology's *nodes* live
        in the graph is a separate story surfaced in the Ingest tab's
        bulk_ops. Here we tell the user whether the source *file* is
        on disk (and how big) so they can decide to download or wipe.
        """
        for name in _ONTOLOGY_ORDER:
            entry = ONTOLOGY_REGISTRY.get(name)
            if entry is None:
                self.ontology_status_texts[name].value = "(not registered)"
                continue
            self._update_ontology_row(
                name,
                downloaded=is_ontology_downloaded(name),
                on_disk_bytes=get_ontology_download_bytes(name),
            )

    def _update_ontology_row(
        self,
        name: str,
        *,
        downloaded: bool,
        on_disk_bytes: int,
    ) -> None:
        status_text = self.ontology_status_texts[name]
        download_btn = self.ontology_download_buttons[name]
        delete_btn = self.ontology_delete_buttons[name]
        if downloaded:
            status_text.value = f"✓ downloaded ({_fmt_bytes(on_disk_bytes)})"
            status_text.color = ft.Colors.GREEN_300
        else:
            status_text.value = "○ not downloaded"
            status_text.color = ft.Colors.GREY_400
        download_btn.visible = not downloaded
        delete_btn.visible = downloaded

    def _safe_bool(self, fn) -> bool:
        try:
            return bool(fn()) if callable(fn) else False
        except Exception as exc:
            logger.warning("is_installed probe failed: %r", exc)
            return False

    def _safe_update(self) -> None:
        with contextlib.suppress(Exception):
            self.app.page.update()

    # ----- handlers --------------------------------------------------------

    def _set_status(self, msg: str, *, ok: bool = True) -> None:
        if self.status is None:
            return
        self.status.value = msg
        self.status.color = ft.Colors.GREY_400 if ok else ft.Colors.AMBER_300
        self._safe_update()

    def _show_confirm_dialog(
        self,
        *,
        title: str,
        body: str,
        confirm_label: str,
        on_confirm,
    ) -> None:
        """Modal confirm-before-act dialog.

        Same pattern as `ingest.py._show_invalid_config_dialog`: build
        `ft.AlertDialog`, show via `page.show_dialog`, dismiss via
        `page.pop_dialog`. `on_confirm` is a plain callable (sync) —
        the button handlers pass a lambda that kicks off the actual
        async work via `asyncio.create_task` after dismissing.

        `body` is `plan.summary` — the same summary text that would
        appear in a CLI dry-run. Rendered in a scrollable Column so
        long provenance blocks (rich MeSH / FIBO summaries) still fit.
        """

        def _cancel(_ev):
            self.app.page.pop_dialog()

        def _confirm(_ev):
            self.app.page.pop_dialog()
            on_confirm()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title, weight=ft.FontWeight.BOLD),
            content=ft.Container(
                width=520,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    tight=True,
                    controls=[
                        ft.Text(body, size=12, selectable=True),
                    ],
                ),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.TextButton(confirm_label, on_click=_confirm),
            ],
        )
        self.app.page.show_dialog(dialog)

    # --- Ontology (fully wired to disk-only download / delete) ---
    def _on_ontology_download(self, name: str) -> None:
        plan = download_ontology_download_plan(name)
        self._show_confirm_dialog(
            title=f"Download {name}?",
            body=plan.summary,
            confirm_label="Download",
            on_confirm=lambda: asyncio.create_task(self._run_ontology_download(name)),
        )

    async def _run_ontology_download(self, name: str) -> None:
        plan = download_ontology_download_plan(name)
        # Surface heavy_warning (e.g. NCBITaxon: "peak RAM ~4 GB during
        # parse") BEFORE the download starts so the user sees it while
        # it runs.
        warning = ""
        if plan.provenance is not None and plan.provenance.heavy_warning:
            warning = f" — WARNING: {plan.provenance.heavy_warning}"
        target_mb = plan.download_size_mb
        self._set_status(f"Downloading {name!r} source file (~{target_mb} MB){warning}…")

        # Progress polling (same shape as extractor download): poll
        # disk-size every 2s while the fetch runs in its worker thread.
        async def _poll() -> None:
            try:
                while True:
                    await asyncio.sleep(2)
                    on_disk = get_ontology_download_bytes(name)
                    if on_disk > 0:
                        mb = on_disk / (1024 * 1024)
                        self._set_status(
                            f"Downloading {name!r}: {mb:.0f} MB / ~{target_mb} MB{warning}"
                        )
            except asyncio.CancelledError:
                pass

        poll_task = asyncio.create_task(_poll())
        try:
            result = await download_ontology_download_execute(plan)
        finally:
            poll_task.cancel()
        if not result.download_ok:
            self._set_status(
                f"Download {name!r} failed: {result.download_error}",
                ok=False,
            )
            return
        self._set_status(
            f"Downloaded {name!r} ({_fmt_bytes(result.on_disk_bytes)} "
            f"on disk). Neo4j nodes will be written on next ingest."
        )
        self._sync_ontology_state()
        self._safe_update()

    def _on_ontology_delete(self, name: str) -> None:
        plan = delete_ontology_download_plan(name)
        self._show_confirm_dialog(
            title=f"Delete {name} download?",
            body=plan.summary,
            confirm_label="Delete download",
            on_confirm=lambda: asyncio.create_task(self._run_ontology_delete(name)),
        )

    async def _run_ontology_delete(self, name: str) -> None:
        plan = delete_ontology_download_plan(name)
        if not plan.is_downloaded:
            self._set_status(f"{name!r} has nothing to delete.")
            return
        self._set_status(f"Deleting {name!r} download ({_fmt_bytes(plan.on_disk_bytes)})…")
        result = await delete_ontology_download_execute(plan)
        if not result.delete_ok:
            self._set_status(
                f"Delete {name!r} download failed: {result.delete_error}",
                ok=False,
            )
            return
        self._set_status(
            f"Freed {_fmt_bytes(result.freed_bytes)} by deleting "
            f"{name!r} download. Existing Neo4j nodes (if any) untouched."
        )
        self._sync_ontology_state()
        self._safe_update()

    # --- Extractor (fully wired to the new backend) ---
    def _on_extractor_install(self, name: str) -> None:
        plan = install_extractor_plan(name)
        self._show_confirm_dialog(
            title=f"Install {name}?",
            body=plan.summary,
            confirm_label="Install",
            on_confirm=lambda: asyncio.create_task(self._run_extractor_install(name)),
        )

    async def _run_extractor_install(self, name: str) -> None:
        self._set_status(f"Installing {name!r} pip package…")
        plan = install_extractor_plan(name)
        result = await install_extractor_execute(plan)
        if not result.install_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(
                f"Install {name!r} failed: {tail}",
                ok=False,
            )
            return
        if result.restart_required:
            self._set_status(
                f"Installed {name!r}. Restart the app for the new package to take effect."
            )
        else:
            self._set_status(f"{name!r} was already installed.")
        self._sync_extractor_state()
        self._safe_update()

    def _on_extractor_uninstall(self, name: str) -> None:
        plan = uninstall_extractor_plan(name)
        self._show_confirm_dialog(
            title=f"Uninstall {name}?",
            body=plan.summary,
            confirm_label="Uninstall",
            on_confirm=lambda: asyncio.create_task(self._run_extractor_uninstall(name)),
        )

    async def _run_extractor_uninstall(self, name: str) -> None:
        self._set_status(f"Uninstalling {name!r} pip package…")
        plan = uninstall_extractor_plan(name)
        result = await uninstall_extractor_execute(plan)
        if not result.uninstall_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(
                f"Uninstall {name!r} failed: {tail}",
                ok=False,
            )
            return
        self._set_status(f"Uninstalled {name!r}. Restart the app to fully release the module.")
        self._sync_extractor_state()
        self._safe_update()

    def _on_extractor_download(self, name: str) -> None:
        plan = download_extractor_weights_plan(name)
        self._show_confirm_dialog(
            title=f"Download {name} weights?",
            body=plan.summary,
            confirm_label="Download weights",
            on_confirm=lambda: asyncio.create_task(self._run_extractor_download(name)),
        )

    async def _run_extractor_download(self, name: str) -> None:
        plan = download_extractor_weights_plan(name)
        if plan.provenance is None:
            self._set_status(f"{name!r} has no downloadable weights.")
            return
        target_mb = plan.provenance.download_size_mb
        self._set_status(f"Downloading {plan.provenance.model_name} (~{target_mb} MB)…")

        # Progress polling: while the HF snapshot download runs in its
        # worker thread, poll disk-size every 2s so the user sees
        # incremental progress in the status line. Cancelled cleanly
        # when the download finishes (finally block).
        async def _poll() -> None:
            try:
                while True:
                    await asyncio.sleep(2)
                    on_disk = get_weights_downloaded_bytes(name)
                    if on_disk > 0:
                        mb = on_disk / (1024 * 1024)
                        self._set_status(
                            f"Downloading {plan.provenance.model_name}: "
                            f"{mb:.0f} MB / ~{target_mb} MB"
                        )
            except asyncio.CancelledError:
                pass

        poll_task = asyncio.create_task(_poll())
        try:
            result = await download_extractor_weights_execute(plan)
        finally:
            poll_task.cancel()
        if not result.download_ok:
            self._set_status(
                f"Download {name!r} weights failed: {result.download_error}",
                ok=False,
            )
            return
        self._set_status(
            f"Downloaded {name!r} weights ({_fmt_bytes(result.on_disk_bytes)} on disk)."
        )
        self._sync_extractor_state()
        self._safe_update()

    def _on_extractor_delete(self, name: str) -> None:
        plan = delete_extractor_weights_plan(name)
        summary = (
            f"{plan.display_name} has no weights on disk — nothing to delete."
            if not plan.is_downloaded
            else (
                f"Delete {plan.display_name} weights from disk "
                f"({_fmt_bytes(plan.on_disk_bytes)} will be freed). "
                f"The pip package stays."
            )
        )
        self._show_confirm_dialog(
            title=f"Delete {name} weights?",
            body=summary,
            confirm_label="Delete weights",
            on_confirm=lambda: asyncio.create_task(self._run_extractor_delete(name)),
        )

    async def _run_extractor_delete(self, name: str) -> None:
        plan = delete_extractor_weights_plan(name)
        if not plan.is_downloaded:
            self._set_status(f"{name!r} has no weights to delete.")
            return
        self._set_status(f"Deleting {name!r} weights ({_fmt_bytes(plan.on_disk_bytes)})…")
        result = await delete_extractor_weights_execute(plan)
        if not result.delete_ok:
            self._set_status(
                f"Delete {name!r} weights failed: {result.delete_error}",
                ok=False,
            )
            return
        self._set_status(f"Freed {_fmt_bytes(result.freed_bytes)} by deleting {name!r} weights.")
        self._sync_extractor_state()
        self._safe_update()

    # --- Parser (fully wired to install_parser_extra_execute) ---
    def _on_parser_install(self, name: str) -> None:
        plan = install_parser_extra_plan(name)
        self._show_confirm_dialog(
            title=f"Install parser extra {name}?",
            body=plan.summary,
            confirm_label="Install",
            on_confirm=lambda: asyncio.create_task(self._run_parser_install(name)),
        )

    async def _run_parser_install(self, name: str) -> None:
        self._set_status(f"Installing {name!r} parser extra…")
        plan = install_parser_extra_plan(name)
        result = await install_parser_extra_execute(plan)
        if not result.install_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(
                f"Install parser {name!r} failed: {tail}",
                ok=False,
            )
            return
        if result.restart_required:
            self._set_status(
                f"Installed parser {name!r}. Restart the app for the new extra to take effect."
            )
        else:
            self._set_status(f"Parser {name!r} was already installed.")
        self._sync_parser_state()
        self._safe_update()

    def _on_parser_uninstall(self, name: str) -> None:
        plan = uninstall_parser_extra_plan(name)
        self._show_confirm_dialog(
            title=f"Uninstall parser extra {name}?",
            body=plan.summary,
            confirm_label="Uninstall",
            on_confirm=lambda: asyncio.create_task(self._run_parser_uninstall(name)),
        )

    async def _run_parser_uninstall(self, name: str) -> None:
        self._set_status(f"Uninstalling {name!r} parser extra…")
        plan = uninstall_parser_extra_plan(name)
        result = await uninstall_parser_extra_execute(plan)
        if not result.uninstall_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(
                f"Uninstall parser {name!r} failed: {tail}",
                ok=False,
            )
            return
        self._set_status(
            f"Uninstalled parser {name!r}. Restart the app to fully release the module."
        )
        self._sync_parser_state()
        self._safe_update()

    # --- Embedding providers (pip adapter install / uninstall) ---
    # Same shape as the extractor / parser handlers above: no pre-check —
    # bundled / already-installed / active-uninstall are handled by button
    # visibility (+ the disabled active-Uninstall set in _sync), and the
    # execute functions are themselves no-op guarded for those cases.
    def _on_embedder_provider_install(self, name: str) -> None:
        plan = install_embedder_provider_plan(name)
        self._show_confirm_dialog(
            title=f"Install {plan.display_name}?",
            body=plan.summary,
            confirm_label="Install",
            on_confirm=lambda: asyncio.create_task(self._run_embedder_provider_install(name)),
        )

    async def _run_embedder_provider_install(self, name: str) -> None:
        self._set_status(f"Installing {name!r} embedder adapter…")
        plan = install_embedder_provider_plan(name)
        result = await install_embedder_provider_execute(plan)
        if not result.install_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(f"Install {name!r} failed: {tail}", ok=False)
            return
        if result.restart_required:
            self._set_status(f"Installed {name!r}. Restart the app for it to take effect.")
        else:
            self._set_status(f"{name!r} was already installed.")
        self._sync_embedder_provider_state()
        self._safe_update()

    def _on_embedder_provider_uninstall(self, name: str) -> None:
        plan = uninstall_embedder_provider_plan(name)
        self._show_confirm_dialog(
            title=f"Uninstall {plan.display_name}?",
            body=plan.summary,
            confirm_label="Uninstall",
            on_confirm=lambda: asyncio.create_task(self._run_embedder_provider_uninstall(name)),
        )

    async def _run_embedder_provider_uninstall(self, name: str) -> None:
        self._set_status(f"Uninstalling {name!r} embedder adapter…")
        plan = uninstall_embedder_provider_plan(name)
        result = await uninstall_embedder_provider_execute(plan)
        if not result.uninstall_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(f"Uninstall {name!r} failed: {tail}", ok=False)
            return
        self._set_status(f"Uninstalled {name!r}. Restart the app to fully release the module.")
        self._sync_embedder_provider_state()
        self._safe_update()

    # --- LLM providers (pip adapter install / uninstall) ---
    # Same no-pre-check pattern as the embedder handlers, with ONE difference:
    # the install PLAN is async (it probes the Ollama daemon), so it's fetched
    # in a task before the dialog — mirroring how the LLM tab did it. The
    # uninstall plan is sync, like every other plan.
    def _on_llm_provider_install(self, name: str) -> None:
        self._spawn(self._prompt_llm_provider_install(name))

    async def _prompt_llm_provider_install(self, name: str) -> None:
        plan = await install_llm_provider_plan(name)
        self._show_confirm_dialog(
            title=f"Install {plan.display_name}?",
            body=plan.summary,
            confirm_label="Install",
            on_confirm=lambda: asyncio.create_task(self._run_llm_provider_install(name)),
        )

    async def _run_llm_provider_install(self, name: str) -> None:
        self._set_status(f"Installing {name!r} LLM adapter…")
        plan = await install_llm_provider_plan(name)
        result = await install_llm_provider_execute(plan)
        if not result.install_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(f"Install {name!r} failed: {tail}", ok=False)
            return
        if result.restart_required:
            self._set_status(f"Installed {name!r}. Restart the app for it to take effect.")
        else:
            self._set_status(f"{name!r} was already installed.")
        self._sync_llm_provider_state()
        self._safe_update()

    def _on_llm_provider_uninstall(self, name: str) -> None:
        plan = uninstall_llm_provider_plan(name)
        self._show_confirm_dialog(
            title=f"Uninstall {plan.display_name}?",
            body=plan.summary,
            confirm_label="Uninstall",
            on_confirm=lambda: asyncio.create_task(self._run_llm_provider_uninstall(name)),
        )

    async def _run_llm_provider_uninstall(self, name: str) -> None:
        self._set_status(f"Uninstalling {name!r} LLM adapter…")
        plan = uninstall_llm_provider_plan(name)
        result = await uninstall_llm_provider_execute(plan)
        if not result.uninstall_ok:
            tail = result.pip_output[-200:] if result.pip_output else ""
            self._set_status(f"Uninstall {name!r} failed: {tail}", ok=False)
            return
        self._set_status(f"Uninstalled {name!r}. Restart the app to fully release the module.")
        self._sync_llm_provider_state()
        self._safe_update()
