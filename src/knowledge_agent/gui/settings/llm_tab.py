"""Search → LLM sub-tab — installed providers + query-time node models + rates.

Layout (top to bottom):

  1. Installed providers — read-only list of installed adapters. There is no
     single "active" provider any more; a model is chosen per node (below)
     from ANY installed provider and stored as a 'provider:model' ref.
  2. Ollama — base URL + daemon-reachable status.
  3. Per-node models + temperatures — model Dropdown + temperature
     Slider for each of: mode_classifier, query_builder,
     cypher_builder, synthesizer. Plus a separate "Chat router" row
     for the GUI-only chat-router model + temperature. Each picker offers
     every installed provider's models (see `available_models`).
  4. Per-provider rate limits + retries — 4 Optional[float] fields
     + llm_max_retries int.

Provider Install / Uninstall moved to the Installs tab (the global install
surface). This tab reads install-state (per-provider `is_installed_fn`) to
show which providers are installed and to build the per-node model-picker
options. Ollama daemon reachability is async — probed in a background task
that updates the Ollama daemon-status line (in the Ollama section) when it
completes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    labeled_field,
    section_divider,
)
from knowledge_agent.gui._widgets.info_icon import info_icon, section_header
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_llm_to_env,
    apply_voyage_rate_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.llm_factory import (
    format_model_ref,
    parse_model_ref,
    supports_temperature,
    to_model_ref,
)
from knowledge_agent.llm_lifecycle import (
    LLM_PROVIDER_REGISTRY,
    _ollama_daemon_is_reachable,
)

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


# Order in which provider rows render. Anthropic + OpenAI first
# because they're the most common starting choices; Google + Ollama
# after.
_PROVIDER_ORDER: tuple[str, ...] = ("anthropic", "openai", "google", "ollama")

# Rate limits also cover the Voyage embedder — its native (non-LangChain)
# client honours a requests/sec cap. Voyage is embedding-only, so it belongs
# in the rate-limit section but NOT in the LLM "Installed providers" list;
# hence a separate tuple that appends it to the LLM providers.
_RATE_LIMIT_PROVIDERS: tuple[str, ...] = (*_PROVIDER_ORDER, "voyage")


# Curated model menus for the per-node model dropdowns here AND the
# Evaluation Run tab's judge-model panel (which imports this). Each
# dropdown that uses it is `editable=True` so the user can override with a
# typed value when their preferred model isn't listed. (If a third consumer
# appears, promote this to a shared gui/_models.py.)
LLM_AVAILABLE_MODELS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
        "claude-fable-5",
    ),
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "google": ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"),
    "ollama": (
        "llama3.2:3b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "llama3.3:70b",
    ),
}


def available_models() -> list[tuple[str, str]]:
    """Every (provider, model) offered across INSTALLED providers — the option
    source for every model picker now that a choice carries its own provider.

    A provider's models appear only when its adapter is pip-installed (probed
    via the lifecycle `is_installed_fn`), so the pickers naturally span exactly
    the providers the user has set up. Shared by the per-node pickers here and
    (from Phase 4) the corpus extractor/triples, judge-panel, and generator
    pickers, so "pick any model from any installed provider" is one list.
    """
    out: list[tuple[str, str]] = []
    for provider in _PROVIDER_ORDER:
        entry = LLM_PROVIDER_REGISTRY[provider]
        try:
            installed = bool(entry["is_installed_fn"]())
        except Exception as exc:  # broad: a flaky probe must not break the picker
            logger.warning("is_installed_fn(%s) failed: %r", provider, exc)
            installed = False
        if installed:
            out.extend((provider, m) for m in LLM_AVAILABLE_MODELS.get(provider, ()))
    return out


def model_options() -> list[ft.DropdownOption]:
    """`available_models()` as picker options: key = the stored 'provider:model'
    ref, text = a readable 'provider — model' label."""
    return [
        ft.DropdownOption(key=format_model_ref(p, m), text=f"{p} — {m}")
        for p, m in available_models()
    ]


class LlmTab:
    """LLM provider + per-query-node tuning + rate limits."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Strong refs to fire-and-forget background tasks so the event
        # loop doesn't GC them mid-flight (see the daemon probe below).
        self._bg_tasks: set[asyncio.Task] = set()
        self.status: ft.Text | None = None

        # Installed-providers display box — filled on first show + on install
        # change (read-only; install/uninstall live in the Installs tab). There
        # is no single "active provider" any more — a model is picked per node.
        self.installed_providers_box: ft.Container | None = None

        # Adapter-installed bool per provider (from `is_installed_fn`) — read
        # to build the installed-providers display + the model-picker options.
        self._installed_state: dict[str, bool] = {}
        # Ollama daemon reachability (separate from adapter-installed) —
        # probed async + shown in the Ollama section below.
        self._ollama_daemon_ok: bool | None = None
        self.ollama_daemon_status: ft.Text | None = None

        self.ollama_url_field: ft.TextField | None = None

        # Per-node model Dropdown (editable, options driven by the
        # active provider's curated list) + temperature Slider + value
        # text. Keyed by node name (`mode_classifier`, etc.).
        self.node_model_fields: dict[str, ft.Dropdown] = {}
        self.node_temp_sliders: dict[str, ft.Slider] = {}
        self.node_temp_value_texts: dict[str, ft.Text] = {}

        # Rate-limit Optional[float] TextFields, keyed by provider.
        self.rate_limit_fields: dict[str, ft.TextField] = {}
        self.max_retries_field: ft.TextField | None = None

        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        cfg = self.app.gui_config
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        # Installed-providers display — populated on first build once we know
        # the install state.
        self.installed_providers_box = ft.Container(
            content=ft.Text(
                "(checking install state...)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            ),
        )

        # Ollama daemon-reachability status line (shown in the Ollama section
        # below). Adapter install/uninstall moved to the Installs tab.
        self.ollama_daemon_status = ft.Text(
            "(checking daemon…)",
            size=12,
            color=ft.Colors.GREY_500,
            italic=True,
        )

        # Ollama base URL.
        self.ollama_url_field = ft.TextField(
            value=cfg.ollama_base_url,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_ollama_url_blur,
        )

        # Per-node models + temperatures (4 nodes). Each model picker is
        # an `editable` Dropdown — options come from the curated
        # `LLM_AVAILABLE_MODELS[active_provider]` menu, but the user
        # can also type any model name as an override (covers off-menu
        # / future models without a code edit). on_blur fires for both
        # paths (typed text + menu pick).
        # Model pickers span EVERY installed provider — a choice is stored as a
        # 'provider:model' ref. Options need install state, so they're filled on
        # first show (`_sync_model_options`); here each starts empty with its
        # value normalized to a composite ref (a legacy bare value is wrapped
        # with the global provider so it matches an option + dispatches
        # explicitly).
        for node_name, default_temp in (
            ("mode_classifier", cfg.mode_classifier_temperature),
            ("query_builder", cfg.query_builder_temperature),
            ("cypher_builder", cfg.cypher_builder_temperature),
            ("synthesizer", cfg.synthesizer_temperature),
            # The GUI chat-router isn't a graph node, but it reuses the same
            # model-dropdown + temp-slider construction + handlers. Rendered
            # in its own "Chat router" section below (not the node loop).
            ("chat_router", cfg.chat_router_temperature),
        ):
            ref = self._node_ref(node_name)
            setattr(cfg, f"{node_name}_model", ref)  # in-memory migrate bare → composite
            self.node_model_fields[node_name] = ft.Dropdown(
                value=ref,
                options=[],
                editable=True,
                enable_filter=True,
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                on_blur=lambda e, n=node_name: self.on_model_blur(n),
            )
            self.node_temp_value_texts[node_name] = ft.Text(
                _fmt_float(default_temp),
                size=12,
                color=ft.Colors.WHITE,
                width=42,
            )
            self.node_temp_sliders[node_name] = ft.Slider(
                value=default_temp,
                min=0.0,
                max=1.0,
                divisions=20,
                on_change=lambda e, n=node_name: self._on_temp_slide(n),
                on_change_end=lambda e, n=node_name: self.on_temp_committed(n),
            )
            # Grey the slider out when the node's model doesn't accept
            # temperature (the backend omits it for those models anyway).
            self._sync_temp_enabled(node_name)

        # Rate-limit fields. Empty = None (limiter disabled). Spans the LLM
        # providers + the Voyage embedder (see _RATE_LIMIT_PROVIDERS).
        for provider in _RATE_LIMIT_PROVIDERS:
            current = getattr(cfg, f"{provider}_requests_per_second")
            self.rate_limit_fields[provider] = ft.TextField(
                value="" if current is None else str(current),
                hint_text="(empty = no limit)",
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                on_blur=lambda e, p=provider: self.on_rate_limit_blur(p),
            )

        # llm_max_retries — int.
        self.max_retries_field = ft.TextField(
            value=str(cfg.llm_max_retries),
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_max_retries_blur,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # First show: query install state (sync), kick off async Ollama
        # daemon probe. Re-rebuild the active-provider radio + provider
        # rows once we know the state.
        if self._first_build:
            self._first_build = False
            self._sync_installed_state()
            self._sync_installed_providers_display()
            self._sync_model_options()
            self._sync_ollama_daemon_status()
            # Async daemon probe — skip when there's no running loop
            # (test env). Live GUI always has one running. Loop-check
            # FIRST so we don't construct a coroutine we never await
            # (would emit a RuntimeWarning).
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                task = asyncio.create_task(self._probe_ollama_daemon())
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)

        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("LLMs"),
                # ---- Installed providers -------------------------------
                section_header(
                    self.app,
                    "Installed providers",
                    "Install / uninstall providers in the Installs tab. Pick a "
                    "model — from any installed provider — for each node below.",
                ),
                self.installed_providers_box,
                section_divider(),
                # ---- Ollama --------------------------------------------
                section_header(self.app, "Ollama"),
                labeled_field(
                    "Ollama base URL (used when Ollama is active)",
                    self.ollama_url_field,
                ),
                self.ollama_daemon_status,
                section_divider(),
                # ---- Models + temperatures -----------------------------
                section_header(
                    self.app,
                    "Select LLM models and temperatures",
                    "Query-time nodes only. Ingest-time extractor models live in the Library tab.",
                ),
                # Chat router leads the list — it's the first model to run on a
                # query (the GUI-only conversational layer above retrieval). The
                # (i) explains its role, since the node name alone doesn't.
                self._render_node_block(
                    "chat_router",
                    info=(
                        "Chat router",
                        "The conversational layer above the retrieval graph: it "
                        "reads the running conversation, decides when to search, "
                        "and distils a clean, self-contained query from what "
                        "you've said (resolving 'it' / 'those' / earlier context) "
                        "before handing off to retrieval. It's a stand-in for a "
                        "future supervisor agent, so it lives in the GUI, not the "
                        "backend graph. A cheap, fast model is usually enough — "
                        "routing is a light task.",
                    ),
                ),
                *[
                    self._render_node_block(n)
                    for n in (
                        "mode_classifier",
                        "query_builder",
                        "cypher_builder",
                        "synthesizer",
                    )
                ],
                section_divider(),
                # ---- Rate limits + retries -----------------------------
                section_header(
                    self.app,
                    "Per-provider rate limits + retries",
                    "Leave blank for no limit. Useful when your provider tier "
                    "rate-limits below your concurrent fan-out. Includes the "
                    "Voyage embedder (a global cap, applied at ingest).",
                ),
                *[
                    labeled_field(f"{p} requests/sec", self.rate_limit_fields[p])
                    for p in _RATE_LIMIT_PROVIDERS
                ],
                labeled_field("LLM max retries (1-10)", self.max_retries_field),
                # ---- Shared status text --------------------------------
                self.status,
            ],
        )

    # ----- save core --------------------------------------------------------

    def _commit(self, label: str) -> bool:
        if self.status is None:
            return False
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return False
        apply_llm_to_env(self.app.gui_config)
        # The Voyage embedder rate lives in this tab's rate-limit section too,
        # but it bridges via its own helper (not apply_llm_to_env).
        apply_voyage_rate_to_env(self.app.gui_config)
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)
        self.status.value = f"saved {label}"
        self.app.page.update()
        return True

    # ----- install state ---------------------------------------------------

    def _sync_installed_state(self) -> None:
        """Query lifecycle's per-provider is_installed_fn synchronously."""
        for provider in _PROVIDER_ORDER:
            entry = LLM_PROVIDER_REGISTRY[provider]
            try:
                self._installed_state[provider] = bool(entry["is_installed_fn"]())
            except Exception as exc:
                logger.warning(
                    "is_installed_fn(%s) failed: %r",
                    provider,
                    exc,
                )
                self._installed_state[provider] = False

    async def _probe_ollama_daemon(self) -> None:
        """Async daemon probe — 1s timeout. Updates Ollama row status."""
        try:
            self._ollama_daemon_ok = await _ollama_daemon_is_reachable()
        except Exception as exc:
            logger.warning("ollama daemon probe failed: %r", exc)
            self._ollama_daemon_ok = False
        self._sync_ollama_daemon_status()
        self.app.page.update()

    def _sync_ollama_daemon_status(self) -> None:
        """Update the Ollama daemon-reachability line (the adapter-install rows
        moved to the Installs tab; whether the DAEMON is actually running is a
        runtime concern that belongs beside the base URL)."""
        if self.ollama_daemon_status is None:
            return
        installed = self._installed_state.get("ollama", False)
        daemon = self._ollama_daemon_ok
        if not installed:
            self.ollama_daemon_status.value = (
                "Ollama adapter not installed — install it in the Installs tab."
            )
            self.ollama_daemon_status.color = ft.Colors.GREY_400
        elif daemon is True:
            self.ollama_daemon_status.value = "✓ daemon reachable"
            self.ollama_daemon_status.color = ft.Colors.GREEN_300
        elif daemon is False:
            self.ollama_daemon_status.value = (
                "○ daemon not reachable — install / start it from https://ollama.com/download"
            )
            self.ollama_daemon_status.color = ft.Colors.AMBER_300
        else:
            self.ollama_daemon_status.value = "probing daemon…"
            self.ollama_daemon_status.color = ft.Colors.GREY_400

    def _sync_installed_providers_display(self) -> None:
        """Show which providers are installed (read-only). A model is chosen per
        node below, from any installed provider — there's no single 'active'
        one. Install / uninstall live in the Installs tab."""
        if self.installed_providers_box is None:
            return
        installed = [p for p in _PROVIDER_ORDER if self._installed_state.get(p)]
        if not installed:
            self.installed_providers_box.content = ft.Text(
                "No LLM providers installed yet — install one in the Installs tab.",
                size=12,
                color=ft.Colors.AMBER_300,
                italic=True,
            )
            return
        names = ", ".join(LLM_PROVIDER_REGISTRY[p]["display_name"] for p in installed)
        self.installed_providers_box.content = ft.Text(names, size=12, color=ft.Colors.GREY_200)

    def _sync_model_options(self) -> None:
        """Fill every node picker's options from the installed providers' models
        (deferred to first show — the options need install state)."""
        options = model_options()
        for field in self.node_model_fields.values():
            field.options = list(options)

    def _node_ref(self, node_name: str) -> str:
        """The node's stored model as a composite 'provider:model' ref — a legacy
        bare value wrapped with the global provider (see `to_model_ref`)."""
        stored = getattr(self.app.gui_config, f"{node_name}_model", "") or ""
        return to_model_ref(stored, self.app.gui_config.llm_provider)

    # ----- Provider row builder --------------------------------------------

    def _render_node_block(
        self, node_name: str, *, info: tuple[str, str] | None = None
    ) -> ft.Control:
        """One node's model dropdown + temperature slider, stacked (model on
        its own line, temperature below it — matches the extractor layout).

        `info` (title, text) tacks an (i) help icon onto the model row — used
        for the chat-router node, whose role isn't obvious from its name."""
        model_trailing = (
            info_icon(self.app, title=info[0], text=info[1]) if info is not None else None
        )
        return ft.Column(
            controls=[
                labeled_field(
                    f"{node_name} model",
                    self.node_model_fields[node_name],
                    trailing=model_trailing,
                ),
                labeled_field(
                    f"{node_name} temperature",
                    self.node_temp_sliders[node_name],
                    trailing=self.node_temp_value_texts[node_name],
                ),
            ],
            spacing=4,
        )

    # ----- Handlers --------------------------------------------------------

    def on_ollama_url_blur(self, e: ft.Event) -> None:
        if self.ollama_url_field is None:
            return
        raw = (self.ollama_url_field.value or "").strip()
        if not raw:
            self.ollama_url_field.value = self.app.gui_config.ollama_base_url
            if self.status is not None:
                self.status.value = "Ollama base URL can't be empty"
            self.app.page.update()
            return
        if raw == self.app.gui_config.ollama_base_url:
            return
        previous = self.app.gui_config.ollama_base_url
        self.app.gui_config.ollama_base_url = raw
        if not self._commit("Ollama base URL"):
            self.app.gui_config.ollama_base_url = previous
            self.ollama_url_field.value = previous
            self.app.page.update()

    def on_model_blur(self, node_name: str) -> None:
        field = self.node_model_fields.get(node_name)
        if field is None or self.status is None:
            return
        raw = (field.value or "").strip()
        attr = f"{node_name}_model"
        current = getattr(self.app.gui_config, attr)
        if not raw:
            field.value = current
            self.status.value = f"{node_name} model can't be empty"
            self.app.page.update()
            return
        if raw == current:
            return
        setattr(self.app.gui_config, attr, raw)
        if not self._commit(f"{node_name} model: {raw}"):
            setattr(self.app.gui_config, attr, current)
            field.value = current
        self._sync_temp_enabled(node_name)
        self.app.page.update()

    def _sync_temp_enabled(self, node_name: str) -> None:
        """Enable/disable a node's temperature slider based on whether its
        selected model accepts a temperature. Sampling-free models (e.g.
        Opus 4.8) grey out the slider with a tooltip — the backend omits
        temperature for them regardless."""
        slider = self.node_temp_sliders.get(node_name)
        value_text = self.node_temp_value_texts.get(node_name)
        if slider is None:
            return
        provider, model = parse_model_ref(getattr(self.app.gui_config, f"{node_name}_model", ""))
        if provider is None:  # a legacy bare value → the global fallback provider
            provider = self.app.gui_config.llm_provider
        takes_temp = supports_temperature(provider, model)
        slider.disabled = not takes_temp
        slider.tooltip = None if takes_temp else f"{model} ignores temperature"
        if value_text is not None:
            value_text.color = ft.Colors.WHITE if takes_temp else ft.Colors.WHITE_38

    def _on_temp_slide(self, node_name: str) -> None:
        slider = self.node_temp_sliders.get(node_name)
        value_text = self.node_temp_value_texts.get(node_name)
        if slider is None or value_text is None:
            return
        value_text.value = _fmt_float(slider.value)
        self.app.page.update()

    def on_temp_committed(self, node_name: str) -> None:
        slider = self.node_temp_sliders.get(node_name)
        if slider is None:
            return
        attr = f"{node_name}_temperature"
        current = getattr(self.app.gui_config, attr)
        new_value = float(slider.value)
        if abs(new_value - current) < 1e-6:
            return
        setattr(self.app.gui_config, attr, new_value)
        if not self._commit(f"{node_name} temperature: {new_value:.2f}"):
            setattr(self.app.gui_config, attr, current)
            slider.value = current
            text = self.node_temp_value_texts.get(node_name)
            if text is not None:
                text.value = _fmt_float(current)
            self.app.page.update()

    def on_rate_limit_blur(self, provider: str) -> None:
        field = self.rate_limit_fields.get(provider)
        if field is None or self.status is None:
            return
        raw = (field.value or "").strip()
        attr = f"{provider}_requests_per_second"
        current = getattr(self.app.gui_config, attr)
        if not raw:
            new_value: float | None = None
        else:
            try:
                new_value = float(raw)
                if new_value <= 0:
                    raise ValueError("must be positive")
            except (ValueError, TypeError):
                field.value = "" if current is None else str(current)
                self.status.value = f"{provider} rate limit must be a positive number or empty"
                self.app.page.update()
                return
        if new_value == current:
            return
        setattr(self.app.gui_config, attr, new_value)
        if not self._commit(
            f"{provider} rate limit: {'unlimited' if new_value is None else new_value}"
        ):
            setattr(self.app.gui_config, attr, current)
            field.value = "" if current is None else str(current)
            self.app.page.update()

    def on_max_retries_blur(self, e: ft.Event) -> None:
        if self.max_retries_field is None or self.status is None:
            return
        raw = (self.max_retries_field.value or "").strip()
        current = self.app.gui_config.llm_max_retries
        try:
            new_value = max(1, min(10, int(raw)))
        except (ValueError, TypeError):
            self.max_retries_field.value = str(current)
            self.status.value = "max retries must be 1-10"
            self.app.page.update()
            return
        if new_value == current:
            self.max_retries_field.value = str(new_value)
            self.app.page.update()
            return
        self.app.gui_config.llm_max_retries = new_value
        if not self._commit(f"LLM max retries: {new_value}"):
            self.app.gui_config.llm_max_retries = current
            self.max_retries_field.value = str(current)
            self.app.page.update()


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "0.00"
    return f"{float(value):.2f}"
