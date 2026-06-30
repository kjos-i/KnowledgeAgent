"""Settings → Embedding sub-tab — provider + per-provider model + rate.

Same UX pattern as the LLM sub-tab, simpler scope:

  1. Active provider — radio group, only INSTALLED providers shown.
  2. Providers list — all 4 (voyage / openai / google / huggingface)
     with install state + Install / Uninstall buttons. Buttons are
     stubs in slice 2; install lifecycle dialogs ship in slice 4.
  3. Embedding model — single editable Dropdown. Options come from
     the curated `EMBEDDING_AVAILABLE_MODELS[active_provider]` menu;
     user can type a custom model name to override.
  4. Voyage rate limit — Optional[float] (empty = no limit).

Switching the active provider in slice 2 will eventually trigger
the lifecycle's dimension guard (slice 4): if the new provider's
default dim doesn't match the LanceDB chunks-table schema, the
switch fails. Per-call dim guard isn't surfaced here yet — slice 4.

Active provider's Uninstall is disabled (same rule as LLM tab) —
switch to another first.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.embedder_lifecycle import EMBEDDER_PROVIDER_REGISTRY
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_embedding_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


_PROVIDER_ORDER: tuple[str, ...] = (
    "voyage", "openai", "google", "huggingface",
)


# Curated model menus per provider. Same inline-constant rationale as
# LLM tab: one consumer, small data, local domain. Each dropdown is
# `editable=True` so off-menu / custom models are typeable.
EMBEDDING_AVAILABLE_MODELS: dict[str, tuple[str, ...]] = {
    "voyage": (
        "voyage-multimodal-3",
        "voyage-3-large",
        "voyage-3",
        "voyage-code-3",
    ),
    "openai": (
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ),
    "google": (
        "models/text-embedding-004",
        "models/embedding-001",
    ),
    "huggingface": (
        "BAAI/bge-m3",
        "mixedbread-ai/mxbai-embed-large-v1",
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
    ),
}


# Map provider name → GuiConfig attribute holding that provider's model.
# Used by the active-provider switch to (a) restore the per-provider
# field's stored value, (b) update `embedding_model` (the mirror).
_PER_PROVIDER_MODEL_ATTR: dict[str, str] = {
    "voyage": "voyage_embedding_model",
    "openai": "openai_embedding_model",
    "google": "google_embedding_model",
    "huggingface": "hf_embedding_model",
}


class EmbeddingTab:
    """Embedding provider + per-provider model + Voyage rate limit."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None

        self.active_provider_radio: ft.RadioGroup | None = None
        self.active_provider_container: ft.Container | None = None

        self.provider_status_texts: dict[str, ft.Text] = {}
        self.provider_install_buttons: dict[str, ft.Button] = {}
        self.provider_uninstall_buttons: dict[str, ft.Button] = {}
        self._installed_state: dict[str, bool] = {}

        self.model_field: ft.Dropdown | None = None
        self.voyage_rate_field: ft.TextField | None = None

        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        cfg = self.app.gui_config
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)

        self.active_provider_container = ft.Container(
            content=ft.Text(
                "(checking install state...)",
                size=12, color=ft.Colors.GREY_500, italic=True,
            ),
        )

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

        # Single editable model dropdown — options driven by the active
        # provider's curated list; value reflects the active model.
        self.model_field = ft.Dropdown(
            label="Embedding model",
            value=cfg.embedding_model,
            options=[
                ft.DropdownOption(key=m, text=m)
                for m in EMBEDDING_AVAILABLE_MODELS.get(
                    cfg.embedding_provider, ()
                )
            ],
            editable=True,
            enable_filter=True,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_model_blur,
        )

        # Voyage rate limit — Optional[float], empty = no limit.
        self.voyage_rate_field = ft.TextField(
            label="Voyage requests/sec",
            value=(
                ""
                if cfg.voyage_requests_per_second is None
                else str(cfg.voyage_requests_per_second)
            ),
            hint_text="(empty = no limit)",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self.on_voyage_rate_blur,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        if self._first_build:
            self._first_build = False
            self._sync_installed_state()
            self._sync_provider_rows()
            self._sync_active_provider_radio()

        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("Embedding"),
                # ---- Active provider -----------------------------------
                ft.Text("Active provider", weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Switching is immediate. The dimension guard "
                    "(slice 4) will block a switch when the new "
                    "provider's dim doesn't match the LanceDB schema.",
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
                # ---- Model ---------------------------------------------
                ft.Text("Model", weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Pick from the menu or type a custom name. The "
                    "model + its dimension must match the chunks "
                    "already in your LanceDB corpus.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                self.model_field,
                ft.Divider(),
                # ---- Rate limit ----------------------------------------
                ft.Text("Rate limit", weight=ft.FontWeight.BOLD),
                ft.Text(
                    "Voyage uses its native client (not LangChain), so "
                    "its rate limit lives here separately from the LLM "
                    "tab's per-provider rates.",
                    size=11, color=ft.Colors.GREY_500, italic=True,
                ),
                self.voyage_rate_field,
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
        apply_embedding_to_env(self.app.gui_config)
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)
        self.status.value = f"saved {label}"
        self.app.page.update()
        return True

    # ----- install state ---------------------------------------------------

    def _sync_installed_state(self) -> None:
        for provider in _PROVIDER_ORDER:
            entry = EMBEDDER_PROVIDER_REGISTRY[provider]
            try:
                self._installed_state[provider] = bool(
                    entry["is_installed_fn"]()
                )
            except Exception as exc:
                logger.warning(
                    "embedder is_installed_fn(%s) failed: %r",
                    provider, exc,
                )
                self._installed_state[provider] = False

    def _sync_provider_rows(self) -> None:
        active = self.app.gui_config.embedding_provider
        for provider in _PROVIDER_ORDER:
            installed = self._installed_state.get(provider, False)
            status_text = self.provider_status_texts[provider]
            install_btn = self.provider_install_buttons[provider]
            uninstall_btn = self.provider_uninstall_buttons[provider]

            if installed:
                status_text.value = "✓ installed"
                status_text.color = ft.Colors.GREEN_300
            else:
                status_text.value = "○ not installed"
                status_text.color = ft.Colors.GREY_400

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
        current = self.app.gui_config.embedding_provider
        radio_value = current if current in installed else installed[0]
        self.active_provider_radio = ft.RadioGroup(
            value=radio_value,
            on_change=self.on_active_provider_changed,
            content=ft.Row(
                controls=[
                    ft.Radio(
                        value=p,
                        label=EMBEDDER_PROVIDER_REGISTRY[p]["display_name"],
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
        display = EMBEDDER_PROVIDER_REGISTRY[provider]["display_name"]
        return ft.Row(
            controls=[
                ft.Text(display, size=12, width=180),
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

    # ----- Handlers --------------------------------------------------------

    def on_active_provider_changed(self, e: ft.Event) -> None:
        """Switch the active embedding provider; refresh model dropdown.

        On switch, restore the per-provider model field's stored value
        as the new active `embedding_model`. Refresh the dropdown's
        options to the new provider's curated menu. Same rollback
        shape as the LLM tab.
        """
        if self.active_provider_radio is None or self.model_field is None:
            return
        new_value = self.active_provider_radio.value
        if new_value == self.app.gui_config.embedding_provider:
            return
        previous_provider = self.app.gui_config.embedding_provider
        previous_active_model = self.app.gui_config.embedding_model
        self.app.gui_config.embedding_provider = new_value  # type: ignore[assignment]
        # Mirror: copy the new provider's per-provider field into the
        # active `embedding_model` so the bridge writes the right thing.
        per_provider_attr = _PER_PROVIDER_MODEL_ATTR[new_value]
        new_model = getattr(self.app.gui_config, per_provider_attr)
        self.app.gui_config.embedding_model = new_model
        if not self._commit(f"active embedding provider: {new_value}"):
            self.app.gui_config.embedding_provider = previous_provider
            self.app.gui_config.embedding_model = previous_active_model
            self.active_provider_radio.value = previous_provider
            self.app.page.update()
            return
        # Refresh the dropdown options + value.
        self.model_field.options = [
            ft.DropdownOption(key=m, text=m)
            for m in EMBEDDING_AVAILABLE_MODELS.get(new_value, ())
        ]
        self.model_field.value = new_model
        self._sync_provider_rows()
        self.app.page.update()

    def on_install_clicked(self, provider: str) -> None:
        """Slice 2 stub — install dialogs ship in slice 4."""
        if self.status is None:
            return
        self.status.value = (
            f"Install dialog for {provider} ships in slice 4."
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

    def on_model_blur(self, e: ft.Event) -> None:
        """Persist the model field — to both the per-provider attr AND
        the active `embedding_model` mirror.
        """
        if self.model_field is None or self.status is None:
            return
        raw = (self.model_field.value or "").strip()
        if not raw:
            self.model_field.value = self.app.gui_config.embedding_model
            self.status.value = "Embedding model can't be empty"
            self.app.page.update()
            return
        if raw == self.app.gui_config.embedding_model:
            return
        previous_model = self.app.gui_config.embedding_model
        per_provider_attr = _PER_PROVIDER_MODEL_ATTR[
            self.app.gui_config.embedding_provider
        ]
        previous_per_provider = getattr(
            self.app.gui_config, per_provider_attr,
        )
        self.app.gui_config.embedding_model = raw
        setattr(self.app.gui_config, per_provider_attr, raw)
        if not self._commit(f"embedding model: {raw}"):
            self.app.gui_config.embedding_model = previous_model
            setattr(
                self.app.gui_config, per_provider_attr, previous_per_provider,
            )
            self.model_field.value = previous_model
            self.app.page.update()

    def on_voyage_rate_blur(self, e: ft.Event) -> None:
        if self.voyage_rate_field is None or self.status is None:
            return
        raw = (self.voyage_rate_field.value or "").strip()
        current = self.app.gui_config.voyage_requests_per_second
        if not raw:
            new_value: float | None = None
        else:
            try:
                new_value = float(raw)
                if new_value <= 0:
                    raise ValueError("must be positive")
            except (ValueError, TypeError):
                self.voyage_rate_field.value = (
                    "" if current is None else str(current)
                )
                self.status.value = (
                    "Voyage rate limit must be a positive number "
                    "or empty"
                )
                self.app.page.update()
                return
        if new_value == current:
            return
        self.app.gui_config.voyage_requests_per_second = new_value
        if not self._commit(
            f"Voyage rate limit: "
            f"{'unlimited' if new_value is None else new_value}"
        ):
            self.app.gui_config.voyage_requests_per_second = current
            self.voyage_rate_field.value = (
                "" if current is None else str(current)
            )
            self.app.page.update()
