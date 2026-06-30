"""Settings → LLM sub-tab — provider + query-time node models + rates.

Layout (top to bottom):

  1. Active provider — radio group, only INSTALLED providers shown.
  2. Providers list — all 4 (anthropic / openai / google / ollama)
     with install state + Install / Uninstall buttons. Buttons are
     stubs in slice 2; install lifecycle dialogs ship in slice 4.
  3. Ollama base URL — TextField (relevant when Ollama is active).
  4. Per-node models + temperatures — model TextField + temperature
     Slider for each of: mode_classifier, query_builder,
     cypher_builder, synthesizer.
  5. Per-provider rate limits + retries — 4 Optional[float] fields
     + llm_max_retries int.

Install-state queried at first show via the lifecycle's per-provider
`is_installed_fn`. Ollama daemon reachability is async — probed in a
background task that updates the Ollama row's status when it
completes.

UX rule: the currently-active provider can't be uninstalled — that
button is disabled with an explanatory tooltip. User must switch the
active provider before removing it.
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
    centered_label,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_llm_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header
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


# Curated model menus shown in the per-node model dropdowns. Inline
# constant because the data is local to this tab + small + has no
# other consumer. Each dropdown is `editable=True` so the user can
# override with a typed value when their preferred model isn't here.
LLM_AVAILABLE_MODELS: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    ),
    "openai": ("gpt-4o-mini", "gpt-4o"),
    "google": ("gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"),
    "ollama": (
        "llama3.2:3b", "qwen2.5:7b", "qwen2.5:14b", "llama3.3:70b",
    ),
}


_QUERY_NODES: tuple[str, ...] = (
    "mode_classifier", "query_builder", "cypher_builder", "synthesizer",
)


class LlmTab:
    """LLM provider + per-query-node tuning + rate limits."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None

        # Active provider radio — rebuilt on first show + on install change.
        self.active_provider_radio: ft.RadioGroup | None = None
        self.active_provider_container: ft.Container | None = None

        # Per-provider row widgets — keyed by provider name.
        self.provider_status_texts: dict[str, ft.Text] = {}
        self.provider_install_buttons: dict[str, ft.Button] = {}
        self.provider_uninstall_buttons: dict[str, ft.Button] = {}
        # Adapter-installed bool per provider (from `is_installed_fn`).
        # Ollama daemon-reachable is a separate async probe.
        self._installed_state: dict[str, bool] = {}
        self._ollama_daemon_ok: bool | None = None

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
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)

        # Active provider radio — populated on first build once we know
        # the install state.
        self.active_provider_container = ft.Container(
            content=ft.Text(
                "(checking install state...)",
                size=12, color=ft.Colors.GREY_500, italic=True,
            ),
        )

        # Provider list — one row per provider. Status text + buttons
        # are populated by _sync_provider_state() at first build.
        for provider in _PROVIDER_ORDER:
            self.provider_status_texts[provider] = ft.Text(
                "(checking…)", size=12, color=ft.Colors.GREY_500,
                italic=True,
            )
            self.provider_install_buttons[provider] = ft.Button(
                content=centered_label("Install"),
                on_click=lambda e, p=provider: self.on_install_clicked(p),
            )
            self.provider_uninstall_buttons[provider] = ft.Button(
                content=centered_label("Uninstall"),
                on_click=lambda e, p=provider: self.on_uninstall_clicked(p),
            )

        # Ollama base URL.
        self.ollama_url_field = ft.TextField(
            label="Ollama base URL (used when Ollama is active)",
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
        provider_options = [
            ft.DropdownOption(key=m, text=m)
            for m in LLM_AVAILABLE_MODELS.get(cfg.llm_provider, ())
        ]
        for node_name, default_temp in (
            ("mode_classifier", cfg.mode_classifier_temperature),
            ("query_builder", cfg.query_builder_temperature),
            ("cypher_builder", cfg.cypher_builder_temperature),
            ("synthesizer", cfg.synthesizer_temperature),
        ):
            self.node_model_fields[node_name] = ft.Dropdown(
                label=f"{node_name} model",
                value=getattr(cfg, f"{node_name}_model"),
                options=list(provider_options),
                editable=True,
                enable_filter=True,
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                on_blur=lambda e, n=node_name: self.on_model_blur(n),
            )
            self.node_temp_value_texts[node_name] = ft.Text(
                _fmt_float(default_temp), size=12,
                color=ft.Colors.WHITE, width=42,
            )
            self.node_temp_sliders[node_name] = ft.Slider(
                value=default_temp, min=0.0, max=1.0, divisions=20,
                on_change=lambda e, n=node_name: self._on_temp_slide(n),
                on_change_end=lambda e, n=node_name: self.on_temp_committed(n),
            )

        # Rate-limit fields. Empty = None (limiter disabled).
        for provider in _PROVIDER_ORDER:
            current = getattr(cfg, f"{provider}_requests_per_second")
            self.rate_limit_fields[provider] = ft.TextField(
                label=f"{provider} requests/sec",
                value="" if current is None else str(current),
                hint_text="(empty = no limit)",
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                on_blur=lambda e, p=provider: self.on_rate_limit_blur(p),
            )

        # llm_max_retries — int.
        self.max_retries_field = ft.TextField(
            label="LLM max retries (1-10)",
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
            self._sync_provider_rows()
            self._sync_active_provider_radio()
            # Async daemon probe — skip when there's no running loop
            # (test env). Live GUI always has one running. Loop-check
            # FIRST so we don't construct a coroutine we never await
            # (would emit a RuntimeWarning).
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                asyncio.create_task(self._probe_ollama_daemon())

        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("LLM"),
                # ---- Active provider -----------------------------------
                ft.Text("Active provider", weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Switching is immediate. Install / Uninstall are "
                    "separate per-provider actions below.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                self.active_provider_container,
                ft.Divider(),
                # ---- Providers list ------------------------------------
                ft.Text("Providers", weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Install / Uninstall dialogs ship in slice 4 — the "
                    "buttons stage the action for now.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                *[
                    self._render_provider_row(p) for p in _PROVIDER_ORDER
                ],
                ft.Divider(),
                # ---- Ollama base URL -----------------------------------
                ft.Text("Ollama", weight=ft.FontWeight.BOLD),
                self.ollama_url_field,
                ft.Divider(),
                # ---- Per-node models + temperatures --------------------
                ft.Text(
                    "Per-node models + temperatures",
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    "Query-time nodes only. Ingest-time extractor "
                    "models live in the Library tab.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                *[
                    self._render_node_block(n) for n in (
                        "mode_classifier", "query_builder",
                        "cypher_builder", "synthesizer",
                    )
                ],
                ft.Divider(),
                # ---- Rate limits + retries -----------------------------
                ft.Text(
                    "Per-provider rate limits + retries",
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    "Leave blank for no limit. Useful when your "
                    "provider tier rate-limits below your "
                    "concurrent fan-out.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                *[
                    self.rate_limit_fields[p] for p in _PROVIDER_ORDER
                ],
                self.max_retries_field,
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
                self._installed_state[provider] = bool(
                    entry["is_installed_fn"]()
                )
            except Exception as exc:
                logger.warning(
                    "is_installed_fn(%s) failed: %r", provider, exc,
                )
                self._installed_state[provider] = False

    async def _probe_ollama_daemon(self) -> None:
        """Async daemon probe — 1s timeout. Updates Ollama row status."""
        try:
            self._ollama_daemon_ok = await _ollama_daemon_is_reachable()
        except Exception as exc:
            logger.warning("ollama daemon probe failed: %r", exc)
            self._ollama_daemon_ok = False
        self._sync_provider_rows()
        self.app.page.update()

    def _sync_provider_rows(self) -> None:
        """Update per-provider status text + button visibility."""
        active = self.app.gui_config.llm_provider
        for provider in _PROVIDER_ORDER:
            installed = self._installed_state.get(provider, False)
            status_text = self.provider_status_texts[provider]
            install_btn = self.provider_install_buttons[provider]
            uninstall_btn = self.provider_uninstall_buttons[provider]

            if provider == "ollama":
                # Two-part status: adapter + daemon.
                daemon = self._ollama_daemon_ok
                if installed and daemon is True:
                    status_text.value = "✓ installed (daemon reachable)"
                    status_text.color = ft.Colors.GREEN_300
                elif installed and daemon is False:
                    status_text.value = (
                        "○ adapter installed; daemon not reachable"
                    )
                    status_text.color = ft.Colors.AMBER_300
                elif installed and daemon is None:
                    status_text.value = (
                        "✓ adapter installed; probing daemon…"
                    )
                    status_text.color = ft.Colors.GREY_400
                else:
                    status_text.value = (
                        "○ not installed (adapter + daemon both needed)"
                    )
                    status_text.color = ft.Colors.GREY_400
            else:
                if installed:
                    status_text.value = "✓ installed"
                    status_text.color = ft.Colors.GREEN_300
                else:
                    status_text.value = "○ not installed"
                    status_text.color = ft.Colors.GREY_400

            # Install button visible when NOT installed; Uninstall when
            # installed. Active provider's Uninstall is disabled.
            install_btn.visible = not installed
            uninstall_btn.visible = installed
            if installed and provider == active:
                uninstall_btn.disabled = True
                uninstall_btn.tooltip = (
                    "Active provider can't be uninstalled — switch "
                    "to another first."
                )
            else:
                uninstall_btn.disabled = False
                uninstall_btn.tooltip = None

    def _sync_active_provider_radio(self) -> None:
        """Rebuild the radio group from currently-installed providers."""
        if self.active_provider_container is None:
            return
        installed = [
            p for p in _PROVIDER_ORDER if self._installed_state.get(p)
        ]
        if not installed:
            self.active_provider_container.content = ft.Text(
                "No providers installed yet — pick one below and click "
                "Install when slice 4 ships the dialogs.",
                size=12, color=ft.Colors.AMBER_300, italic=True,
            )
            return
        current = self.app.gui_config.llm_provider
        # If the stored active provider isn't installed, default to the
        # first installed one — keeps the radio coherent. Don't write
        # to GuiConfig here; let the user click to confirm.
        radio_value = current if current in installed else installed[0]
        self.active_provider_radio = ft.RadioGroup(
            value=radio_value,
            on_change=self.on_active_provider_changed,
            content=ft.Row(
                controls=[
                    ft.Radio(
                        value=p,
                        label=LLM_PROVIDER_REGISTRY[p]["display_name"],
                    )
                    for p in installed
                ],
                spacing=16,
                wrap=True,
            ),
        )
        self.active_provider_container.content = self.active_provider_radio

    # ----- Provider row builder --------------------------------------------

    def _render_provider_row(self, provider: str) -> ft.Control:
        display = LLM_PROVIDER_REGISTRY[provider]["display_name"]
        return ft.Row(
            controls=[
                ft.Text(display, size=12, width=160),
                ft.Container(
                    content=self.provider_status_texts[provider],
                    expand=True,
                ),
                self.provider_install_buttons[provider],
                self.provider_uninstall_buttons[provider],
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _render_node_block(self, node_name: str) -> ft.Control:
        """One node's model TextField + temperature slider row."""
        return ft.Column(
            controls=[
                self.node_model_fields[node_name],
                ft.Row(
                    controls=[
                        ft.Text(
                            f"{node_name} temperature",
                            size=12, color=ft.Colors.GREY_300, width=220,
                        ),
                        ft.Container(
                            content=self.node_temp_sliders[node_name],
                            expand=True,
                        ),
                        self.node_temp_value_texts[node_name],
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            ],
            spacing=4,
        )

    # ----- Handlers --------------------------------------------------------

    def on_active_provider_changed(self, e: ft.Event) -> None:
        """Switch the active provider AND refresh the 4 model dropdowns.

        Per-provider model defaults come from
        `LLM_PROVIDER_REGISTRY[provider]["default_models"]` — backend's
        canonical source for "which model the lifecycle picks for this
        provider's call sites." We copy them into GuiConfig + the
        dropdown values + repopulate the dropdown options to the new
        provider's curated menu.

        Without this refresh, switching from Anthropic → OpenAI would
        leave `cypher_builder_model="claude-sonnet-4-6"` (nonsense to
        OpenAI's API) until the user noticed + hand-edited it. The
        auto-update protects new users from a broken-state misclick.
        """
        if self.active_provider_radio is None:
            return
        new_value = self.active_provider_radio.value
        if new_value == self.app.gui_config.llm_provider:
            return
        # Snapshot previous state for rollback on save failure.
        previous_provider = self.app.gui_config.llm_provider
        previous_models = {
            n: getattr(self.app.gui_config, f"{n}_model")
            for n in _QUERY_NODES
        }
        self.app.gui_config.llm_provider = new_value  # type: ignore[assignment]
        # Copy the new provider's default models into GuiConfig.
        defaults = LLM_PROVIDER_REGISTRY[new_value]["default_models"]
        for node in _QUERY_NODES:
            setattr(self.app.gui_config, f"{node}_model", defaults[node])
        if not self._commit(f"active LLM provider: {new_value}"):
            # Rollback both the provider AND the model fields.
            self.app.gui_config.llm_provider = previous_provider
            for node, prev_model in previous_models.items():
                setattr(self.app.gui_config, f"{node}_model", prev_model)
            self.active_provider_radio.value = previous_provider
            self.app.page.update()
            return
        # Repopulate each dropdown's options + reset its visible value
        # to the new provider's default.
        new_options = [
            ft.DropdownOption(key=m, text=m)
            for m in LLM_AVAILABLE_MODELS.get(new_value, ())
        ]
        for node in _QUERY_NODES:
            dropdown = self.node_model_fields.get(node)
            if dropdown is None:
                continue
            dropdown.options = list(new_options)
            dropdown.value = defaults[node]
        self._sync_provider_rows()
        self.app.page.update()

    def on_install_clicked(self, provider: str) -> None:
        """Slice 2 stub — install dialogs ship in slice 4."""
        if self.status is None:
            return
        self.status.value = (
            f"Install dialog for {provider} ships in slice 4. For now, "
            f"install manually via the lifecycle CLI."
        )
        self.app.page.update()

    def on_uninstall_clicked(self, provider: str) -> None:
        """Slice 2 stub — uninstall dialogs ship in slice 4."""
        if self.status is None:
            return
        self.status.value = (
            f"Uninstall dialog for {provider} ships in slice 4."
        )
        self.app.page.update()

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
            self.app.page.update()

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
                self.status.value = (
                    f"{provider} rate limit must be a positive number "
                    f"or empty"
                )
                self.app.page.update()
                return
        if new_value == current:
            return
        setattr(self.app.gui_config, attr, new_value)
        if not self._commit(
            f"{provider} rate limit: "
            f"{'unlimited' if new_value is None else new_value}"
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
