"""Settings → Embedding sub-tab — active provider + model + rate.

  1. Active provider — radio group, only INSTALLED providers shown.
  2. Embedding model — single editable Dropdown. Options come from the
     curated `EMBEDDING_AVAILABLE_MODELS[active_provider]` menu; user can
     type a custom model name to override.
  3. Voyage rate limit — Optional[float] (empty = no limit).

Install / Uninstall of each provider's pip adapter lives in the **Installs**
tab now (the global install surface) — the same install-here / choose-there
split the app already uses for ontologies + extractors. This tab is the
CHOICE of which installed embedder to use, plus its model + rate.

Switching the active provider runs a dimension guard: if the corpus already
holds chunks at a different vector dimension than the new provider's default,
the switch is DESTRUCTIVE (the LanceDB chunks table pins the dim at creation)
and requires a hard confirm. The confirm dialog points the user at the
Re-embed bulk operation to rebuild the corpus under the new provider.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import flet as ft

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.embedder_lifecycle import (
    EMBEDDER_PROVIDER_REGISTRY,
    switch_embedder_plan,
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
    apply_embedding_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


_PROVIDER_ORDER: tuple[str, ...] = (
    "voyage",
    "openai",
    "google",
    "huggingface",
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

        # is_installed per provider — read to build the active-provider radio
        # (only installed providers are selectable). Install / Uninstall live
        # in the Installs tab now (the global install surface).
        self._installed_state: dict[str, bool] = {}

        self.model_field: ft.Dropdown | None = None
        self.voyage_rate_field: ft.TextField | None = None

        self._first_build = True
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        cfg = self.app.gui_config
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        self.active_provider_container = ft.Container(
            content=ft.Text(
                "(checking install state...)",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            ),
        )

        # Single editable model dropdown — options driven by the active
        # provider's curated list; value reflects the active model.
        self.model_field = ft.Dropdown(
            value=cfg.embedding_model,
            options=[
                ft.DropdownOption(key=m, text=m)
                for m in EMBEDDING_AVAILABLE_MODELS.get(cfg.embedding_provider, ())
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
            self._sync_active_provider_radio()

        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("Embedding"),
                # ---- Active provider -----------------------------------
                section_header(
                    self.app,
                    "Active provider",
                    "Switching is immediate. Install / Uninstall providers in the "
                    "Installs tab. The dimension guard blocks a switch when the "
                    "new provider's dimension doesn't match the chunks already in "
                    "your LanceDB corpus.",
                ),
                self.active_provider_container,
                section_divider(),
                # ---- Model ---------------------------------------------
                section_header(
                    self.app,
                    "Model",
                    "Pick from the menu or type a custom name. The model + its "
                    "dimension must match the chunks already in your LanceDB "
                    "corpus.",
                ),
                labeled_field("Embedding model", self.model_field),
                section_divider(),
                # ---- Rate limit ----------------------------------------
                section_header(
                    self.app,
                    "Rate limit",
                    "Voyage uses its native client (not LangChain), so its rate "
                    "limit lives here separately from the LLM tab's per-provider "
                    "rates.",
                ),
                labeled_field("Voyage requests/sec", self.voyage_rate_field),
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
                self._installed_state[provider] = bool(entry["is_installed_fn"]())
            except Exception as exc:
                logger.warning(
                    "embedder is_installed_fn(%s) failed: %r",
                    provider,
                    exc,
                )
                self._installed_state[provider] = False

    def _sync_active_provider_radio(self) -> None:
        if self.active_provider_container is None:
            return
        installed = [p for p in _PROVIDER_ORDER if self._installed_state.get(p)]
        if not installed:
            self.active_provider_container.content = ft.Text(
                "No providers installed yet — install one in the Installs tab.",
                size=12,
                color=ft.Colors.AMBER_300,
                italic=True,
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

    # ----- Handlers --------------------------------------------------------

    def on_active_provider_changed(self, e: ft.Event) -> None:
        """Switch the active embedding provider; refresh model dropdown.

        C3 dimension guard: the LanceDB chunks table pins its vector
        dimension at creation. If the corpus already holds chunks at a
        dimension that differs from the new provider's default, the
        switch is DESTRUCTIVE — the existing chunks can't be reused and
        must be re-ingested. Ask the backend (`switch_embedder_plan`)
        whether that's the case; if so, require a hard confirm before
        applying, and point the user at the Re-embed bulk operation.

        Non-destructive switches (no corpus, empty corpus, or matching
        dim) apply straight through with the same rollback shape as the
        LLM tab.
        """
        if self.active_provider_radio is None or self.model_field is None:
            return
        new_value = self.active_provider_radio.value
        if new_value == self.app.gui_config.embedding_provider:
            return
        previous_provider = self.app.gui_config.embedding_provider
        # Ask the backend whether this switch would strand existing chunks
        # at a mismatched dim. If LanceDB is unreachable the plan can't
        # read the corpus dim and reports no mismatch — fall through to a
        # plain switch (a separate LanceDB-down banner covers that case).
        try:
            plan = switch_embedder_plan(new_value)
        except Exception as exc:
            # Telemetry only; a plan failure must not block the switch.
            logger.warning(
                "switch_embedder_plan(%s) failed (%r); switching without dim guard",
                new_value,
                exc,
            )
            plan = None
        if plan is not None and plan.dim_mismatch:
            self._show_confirm(
                title="Embedding dimension change",
                body=(
                    f"{plan.summary}\n\nAfter switching, run the Re-embed bulk "
                    f"operation (Library → Bulk operations) to re-ingest the "
                    f"existing chunks under {new_value}."
                ),
                confirm_label="Switch anyway",
                on_confirm=lambda: self._apply_provider_switch(new_value),
                on_cancel=lambda: self._revert_provider_radio(previous_provider),
            )
            return
        self._apply_provider_switch(new_value)

    def _revert_provider_radio(self, provider: str) -> None:
        """Snap the active-provider radio back to `provider`.

        Used when the user cancels a destructive dim-change switch: Flet
        already moved the radio to the new value on click, so we restore
        it to match the (unchanged) committed provider.
        """
        if self.active_provider_radio is None:
            return
        self.active_provider_radio.value = provider
        self.app.page.update()

    def _apply_provider_switch(self, new_value: str) -> None:
        """Commit the active-provider switch and refresh the dropdown.

        Mirrors the new provider's per-provider model into the active
        `embedding_model`, persists, and repopulates the model options.
        Rolls back config + radio on commit failure. Shared by the plain
        path and the confirmed dim-change path.
        """
        if self.active_provider_radio is None or self.model_field is None:
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
            ft.DropdownOption(key=m, text=m) for m in EMBEDDING_AVAILABLE_MODELS.get(new_value, ())
        ]
        self.model_field.value = new_model
        self.app.page.update()

    def _set_status(self, msg: str, *, ok: bool = True) -> None:
        if self.status is None:
            return
        self.status.value = msg
        self.status.color = ft.Colors.GREY_400 if ok else ft.Colors.RED_300
        self.app.page.update()

    def _show_confirm(
        self,
        *,
        title: str,
        body: str,
        confirm_label: str,
        on_confirm: Any,
        on_cancel: Any = None,
    ) -> None:
        def _go(_e: ft.Event) -> None:
            self.app.page.pop_dialog()
            on_confirm()

        def _cancel(_e: ft.Event) -> None:
            self.app.page.pop_dialog()
            if on_cancel is not None:
                on_cancel()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(title),
            content=ft.Container(
                width=460,
                content=ft.Text(body, size=12, selectable=True),
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_cancel),
                ft.Button(content=centered_label(confirm_label), on_click=_go),
            ],
        )
        self.app.page.show_dialog(dialog)
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
        per_provider_attr = _PER_PROVIDER_MODEL_ATTR[self.app.gui_config.embedding_provider]
        previous_per_provider = getattr(
            self.app.gui_config,
            per_provider_attr,
        )
        self.app.gui_config.embedding_model = raw
        setattr(self.app.gui_config, per_provider_attr, raw)
        if not self._commit(f"embedding model: {raw}"):
            self.app.gui_config.embedding_model = previous_model
            setattr(
                self.app.gui_config,
                per_provider_attr,
                previous_per_provider,
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
                self.voyage_rate_field.value = "" if current is None else str(current)
                self.status.value = "Voyage rate limit must be a positive number or empty"
                self.app.page.update()
                return
        if new_value == current:
            return
        self.app.gui_config.voyage_requests_per_second = new_value
        if not self._commit(
            f"Voyage rate limit: {'unlimited' if new_value is None else new_value}"
        ):
            self.app.gui_config.voyage_requests_per_second = current
            self.voyage_rate_field.value = "" if current is None else str(current)
            self.app.page.update()
