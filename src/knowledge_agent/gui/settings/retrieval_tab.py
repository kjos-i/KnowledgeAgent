"""Settings → Retrieval sub-tab — search-side defaults.

Grouped into visual sections:

  1. Mode — `retrieval_mode` (6 agent-level topologies) +
     `lancedb_search_mode` (LanceDB-internal mode).
  2. Result size — `top_k` (final result count).
  3. Hybrid fusion (RRF) — `num_candidates` (pre-truncation candidate
     pool), `rrf_rank_constant`. Ordering rule: `top_k <= num_candidates`
     (mirrors `Settings._validate_retrieval_windows`).
  4. Diversity (MMR) — `mmr_lambda` (slider).
  5. KG — `kg_max_rows`.
  6. Input mode + synthesis — `input_mode` radio (Conversational /
     Direct query / Direct Cypher) + `direct_retrieve` (synthesizer
     skip). The chat-router temperature + model live on the LLM tab.

Auto-save pattern (RA-mirror): every control commits its own change.
TextFields validate-then-persist on blur with snap-back on bad input.
Sliders persist on_change_end (after the user lifts off) so we don't
fire a save for every pixel of drag. Dropdowns / Checkboxes persist
on_change.

After a successful save we bridge to env + clear cached factories so
the next search uses the new values without restart.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import reset_after_key_change
from knowledge_agent.gui._styles import FRAME_BORDER_COLOR, PANEL_BG, labeled_field
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui._widgets.retrieval_form import (
    LANCE_MODES,
    RetrievalControls,
    apply_gray_out,
    build_mmr_slider,
    build_search_mode_radios,
    mmr_help_text,
)
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_retrieval_to_env,
    save_config,
)
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp


logger = logging.getLogger(__name__)


# Display labels for the 6-mode dropdown. Maps GuiConfig literal value
# to a human-readable label. Order is the dropdown display order;
# `auto` first since it's the default and the most-used option.
_MODE_LABELS: list[tuple[str, str]] = [
    ("auto", "Auto (LLM picks)"),
    ("lancedb_only", "LanceDB only (chunks)"),
    ("neo4j_only", "Neo4j only (KG)"),
    ("lancedb_then_neo4j", "LanceDB → Neo4j (hits, then graph enrichment)"),
    ("neo4j_then_lancedb", "Neo4j → LanceDB (graph filter, then hits)"),
    ("parallel_fused", "Parallel fused (both at once, RRF-fused)"),
]


class RetrievalTab:
    """Retrieval defaults — auto-save fields with an RRF ordering check."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.top_k_field: ft.TextField | None = None
        self.mode_dropdown: ft.Dropdown | None = None
        self.lancedb_mode_radio: ft.RadioGroup | None = None
        self._lancedb_radios: list[ft.Radio] = []
        self._lancedb_mode_box: ft.Container | None = None
        self.num_candidates_field: ft.TextField | None = None
        self.rrf_constant_field: ft.TextField | None = None
        self.mmr_help_text: ft.Text | None = None
        self.use_mmr_checkbox: ft.Checkbox | None = None
        self.mmr_lambda_slider: ft.Slider | None = None
        self.mmr_lambda_value_text: ft.Text | None = None
        self.kg_max_rows_field: ft.TextField | None = None
        self.input_mode_radio: ft.RadioGroup | None = None
        self.direct_retrieve_checkbox: ft.Checkbox | None = None
        self._create_controls()
        # Initial gray-out based on the loaded mode + lancedb mode + MMR state.
        self._sync_enabled_state()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)
        cfg = self.app.gui_config

        # Mode dropdowns.
        # Flet 0.85 `Dropdown` uses `on_select` (not `on_change`); the
        # event fires after the user picks an option from the menu.
        # 6-option agent-level mode stays a dropdown — too many for radios
        # in a narrow panel. 3-option LanceDB mode uses radios (all
        # options visible at once, one click to switch).
        self.mode_dropdown = ft.Dropdown(
            value=cfg.retrieval_mode,
            options=[ft.DropdownOption(key=k, text=lbl) for k, lbl in _MODE_LABELS],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_select=self.on_mode_changed,
        )
        # Radios + slider come from the shared builders, so the eval Dataset
        # form renders the identical widgets (single source, no look-alikes).
        self._lancedb_radios, self.lancedb_mode_radio, self._lancedb_mode_box = (
            build_search_mode_radios(cfg.lancedb_search_mode, self.on_lancedb_mode_changed)
        )

        # Integer fields.
        self.top_k_field = _int_field(
            cfg.top_k,
            self.on_top_k_blur,
        )
        self.num_candidates_field = _int_field(
            cfg.num_candidates,
            self.on_num_candidates_blur,
        )
        self.rrf_constant_field = _int_field(
            cfg.rrf_rank_constant,
            self.on_rrf_constant_blur,
        )
        self.kg_max_rows_field = _int_field(
            cfg.kg_max_rows,
            self.on_kg_max_rows_blur,
        )

        # MMR enable checkbox — when off, slider+multiplier are grayed.
        # When LanceDB mode is FTS, the checkbox itself is grayed (MMR
        # is meaningless there).
        self.use_mmr_checkbox = ft.Checkbox(
            label="Apply MMR diversity re-rank",
            value=cfg.use_mmr,
            on_change=self.on_use_mmr_changed,
        )
        # Sliders for 0-1 floats. on_change updates the inline value
        # display; on_change_end persists (avoid one save per pixel).
        self.mmr_help_text = ft.Text(
            "λ = 1.0 → pure relevance; λ = 0.0 → pure diversity.",
            size=12,
            color=ft.Colors.GREY_500,
            italic=True,
        )
        self.mmr_lambda_slider, self.mmr_lambda_value_text = build_mmr_slider(
            cfg.mmr_lambda,
            on_change=self._on_mmr_lambda_slide,
            on_change_end=self.on_mmr_lambda_committed,
        )
        self.input_mode_radio = ft.RadioGroup(
            value=cfg.input_mode,
            on_change=self.on_input_mode_changed,
            content=ft.Column(
                spacing=2,
                controls=[
                    ft.Radio(
                        value="conversational",
                        label=(
                            "Conversational — the chat router refines intent "
                            "and decides when to search"
                        ),
                    ),
                    ft.Radio(
                        value="direct_query",
                        label=(
                            "Direct query — your text goes straight to "
                            "vector/hybrid search (no router)"
                        ),
                    ),
                    ft.Radio(
                        value="direct_cypher",
                        label=(
                            "Direct Cypher — run your text as raw Cypher on the knowledge graph"
                        ),
                    ),
                ],
            ),
        )

        self.direct_retrieve_checkbox = ft.Checkbox(
            label=("Direct retrieval — skip synthesizer, show raw chunks / rows"),
            value=cfg.direct_retrieve,
            on_change=self.on_direct_retrieve_changed,
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        return ft.Column(
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=10,
            controls=[
                view_header("Retrieval"),
                # ---- Mode ----------------------------------------------
                ft.Text("Mode", weight=ft.FontWeight.BOLD),
                labeled_field("Retrieval mode (agent-level)", self.mode_dropdown),
                self._lancedb_mode_box,
                ft.Divider(),
                # ---- Result size ---------------------------------------
                ft.Text("Result size", weight=ft.FontWeight.BOLD),
                labeled_field("top_k (final result count, 1–50)", self.top_k_field),
                ft.Divider(),
                # ---- Hybrid fusion (RRF) -------------------------------
                ft.Text(
                    "Hybrid fusion (RRF)",
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Text(
                    "Rule: top_k ≤ num_candidates. The form rejects edits "
                    "that break this ordering.",
                    size=12,
                    color=ft.Colors.GREY_500,
                    italic=True,
                ),
                labeled_field(
                    "num_candidates (candidate pool before top_k)",
                    self.num_candidates_field,
                ),
                labeled_field("RRF rank constant k (1/(k+rank))", self.rrf_constant_field),
                ft.Divider(),
                # ---- Diversity (MMR) -----------------------------------
                ft.Text("Diversity (MMR)", weight=ft.FontWeight.BOLD),
                self.mmr_help_text,
                self.use_mmr_checkbox,
                _slider_row(
                    "MMR λ",
                    self.mmr_lambda_slider,
                    self.mmr_lambda_value_text,
                ),
                ft.Divider(),
                # ---- Knowledge graph -----------------------------------
                ft.Text("Knowledge graph", weight=ft.FontWeight.BOLD),
                labeled_field("kg_max_rows (cap on Neo4j rows per query)", self.kg_max_rows_field),
                ft.Divider(),
                # ---- Input mode + synthesis ----------------------------
                ft.Row(
                    controls=[
                        ft.Text("Input mode", weight=ft.FontWeight.BOLD),
                        info_icon(
                            self.app,
                            title="Input mode",
                            text=(
                                "How your chat input is handled:\n\n"
                                "- Conversational: the chat router reads the "
                                "conversation, refines your intent, and "
                                "decides when to search. The normal chat "
                                "experience.\n\n"
                                "- Direct query: skips the router and the "
                                "query-builder - your exact text is the "
                                "search query, run over vector/hybrid "
                                "search.\n\n"
                                "- Direct Cypher: power users - your text is "
                                "run as raw Cypher against the Neo4j "
                                "knowledge graph (read-only queries only)."
                            ),
                            beginner=(
                                "Just leave this on 'Conversational' — you type "
                                "a question like you would to a person, and the "
                                "app figures out when and how to search for you. "
                                "The other two are shortcuts for advanced users."
                            ),
                            technical=(
                                "Conversational routes through the chat "
                                "router + query-builder LLM nodes. Direct query "
                                "bypasses both and passes your text straight to "
                                "the retriever (vector / hybrid per the LanceDB "
                                "search mode). Direct Cypher bypasses retrieval "
                                "entirely and executes your text as a read-only "
                                "Cypher statement against Neo4j (writes are "
                                "rejected by the safety guard)."
                            ),
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                self.input_mode_radio,
                self.direct_retrieve_checkbox,
                # ---- Shared status text --------------------------------
                self.status,
            ],
        )

    # ----- save core --------------------------------------------------------

    def _commit(self, label: str) -> bool:
        """Persist + bridge + reset caches. Returns True on success.

        Status text + chat append handle user feedback. Bridging
        re-runs apply_retrieval_to_env on the WHOLE GuiConfig so any
        valid in-memory edit reaches env (saves us per-field bridge
        functions).
        """
        if self.status is None:
            return False
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return False
        apply_retrieval_to_env(self.app.gui_config)
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)
        self.status.value = f"saved {label}"
        self.app.page.update()
        return True

    # ----- Dropdown handlers -----------------------------------------------

    def on_mode_changed(self, e: ft.Event) -> None:
        if self.mode_dropdown is None:
            return
        new_value = self.mode_dropdown.value
        if new_value == self.app.gui_config.retrieval_mode:
            return
        previous = self.app.gui_config.retrieval_mode
        self.app.gui_config.retrieval_mode = new_value  # type: ignore[assignment]
        if not self._commit(f"retrieval mode: {new_value}"):
            self.app.gui_config.retrieval_mode = previous
            self.mode_dropdown.value = previous
        # Mode change flips which legs run → re-gray the LanceDB / KG blocks.
        self._sync_enabled_state()
        self.app.page.update()

    def on_lancedb_mode_changed(self, e: ft.Event) -> None:
        if self.lancedb_mode_radio is None:
            return
        new_value = self.lancedb_mode_radio.value
        if new_value == self.app.gui_config.lancedb_search_mode:
            return
        previous = self.app.gui_config.lancedb_search_mode
        self.app.gui_config.lancedb_search_mode = new_value  # type: ignore[assignment]
        if not self._commit(f"LanceDB search mode: {new_value}"):
            self.app.gui_config.lancedb_search_mode = previous
            self.lancedb_mode_radio.value = previous
        # MMR enable/disable state depends on the LanceDB mode (FTS has
        # no vectors → MMR doesn't apply). Sync after every mode change.
        self._sync_enabled_state()
        self.app.page.update()

    def _sync_enabled_state(self) -> None:
        """Gray out every control the current mode/toggles can't use, via the
        shared `apply_gray_out` (the single source both this tab and the eval
        form use), then refresh the MMR help line."""
        cfg = self.app.gui_config
        apply_gray_out(
            RetrievalControls(
                lancedb_mode_box=self._lancedb_mode_box,
                lancedb_radios=self._lancedb_radios,
                lancedb_mode_control=self.lancedb_mode_radio,
                num_candidates_field=self.num_candidates_field,
                rrf_constant_field=self.rrf_constant_field,
                use_mmr_checkbox=self.use_mmr_checkbox,
                mmr_lambda_control=self.mmr_lambda_slider,
                kg_max_rows_field=self.kg_max_rows_field,
            ),
            retrieval_mode=cfg.retrieval_mode,
            lancedb_search_mode=cfg.lancedb_search_mode,
            use_mmr=cfg.use_mmr,
        )
        if self.mmr_help_text is not None:
            self.mmr_help_text.value = mmr_help_text(
                lance_on=cfg.retrieval_mode in LANCE_MODES,
                is_fts=cfg.lancedb_search_mode == "fts",
                use_mmr=cfg.use_mmr,
            )

    def on_use_mmr_changed(self, e: ft.Event) -> None:
        """Persist use_mmr; re-sync slider/multiplier gray-out."""
        if self.use_mmr_checkbox is None:
            return
        previous = self.app.gui_config.use_mmr
        self.app.gui_config.use_mmr = bool(self.use_mmr_checkbox.value)
        if not self._commit("MMR on" if self.app.gui_config.use_mmr else "MMR off"):
            self.app.gui_config.use_mmr = previous
            self.use_mmr_checkbox.value = previous
        self._sync_enabled_state()
        self.app.page.update()

    # ----- Integer TextField handlers --------------------------------------

    def on_top_k_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.top_k_field,
            new_value_parser=lambda raw: max(1, min(50, int(raw))),
            current=self.app.gui_config.top_k,
            label="top_k",
            apply=lambda v: setattr(self.app.gui_config, "top_k", v),
            validate=self._validate_window_ordering,
        )

    def on_num_candidates_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.num_candidates_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.num_candidates,
            label="num_candidates",
            apply=lambda v: setattr(self.app.gui_config, "num_candidates", v),
            validate=self._validate_window_ordering,
        )

    def on_rrf_constant_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.rrf_constant_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.rrf_rank_constant,
            label="RRF rank constant",
            apply=lambda v: setattr(
                self.app.gui_config,
                "rrf_rank_constant",
                v,
            ),
            validate=None,
        )

    def on_kg_max_rows_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.kg_max_rows_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.kg_max_rows,
            label="kg_max_rows",
            apply=lambda v: setattr(self.app.gui_config, "kg_max_rows", v),
            validate=None,
        )

    def _handle_int(
        self,
        *,
        field: ft.TextField | None,
        new_value_parser,
        current: int,
        label: str,
        apply,
        validate,
    ) -> None:
        """Shared blur logic: parse → validate → apply → commit (rollback
        on failure or invalid input).

        `new_value_parser(raw_str) -> int` clamps to the field's valid
        range. `apply(value)` writes to GuiConfig. `validate` is an
        optional callable that takes no args + returns an error
        message (str) or None — checked AFTER apply, before persist.
        """
        if field is None or self.status is None:
            return
        raw = (field.value or "").strip()
        try:
            new_value = new_value_parser(raw)
        except (ValueError, TypeError):
            field.value = str(current)
            self.status.value = f"{label} must be a positive integer"
            self.app.page.update()
            return
        if new_value == current:
            field.value = str(new_value)  # normalize
            self.app.page.update()
            return
        apply(new_value)
        if validate is not None:
            err = validate()
            if err is not None:
                apply(current)
                field.value = str(current)
                self.status.value = err
                self.app.page.update()
                return
        if not self._commit(f"{label}: {new_value}"):
            apply(current)
            field.value = str(current)
            self.app.page.update()
            return
        field.value = str(new_value)
        self.app.page.update()

    def _validate_window_ordering(self) -> str | None:
        """Enforce top_k <= num_candidates after a change.

        Self-contained — the GUI owns its own copy of the rule and never
        imports backend internals. The per-layer drift-guard tests keep this
        in sync with the `Settings` validator without coupling the two.
        """
        cfg = self.app.gui_config
        if cfg.num_candidates < cfg.top_k:
            return f"num_candidates ({cfg.num_candidates}) must be >= top_k ({cfg.top_k})"
        return None

    # ----- Slider handlers --------------------------------------------------

    def _on_mmr_lambda_slide(self, e: ft.Event) -> None:
        """Drag-in-progress: the shared builder already synced the value text, so
        just flush it to the page."""
        self.app.page.update()

    def on_mmr_lambda_committed(self, e: ft.Event) -> None:
        if self.mmr_lambda_slider is None:
            return
        new_value = float(self.mmr_lambda_slider.value)
        if abs(new_value - self.app.gui_config.mmr_lambda) < 1e-6:
            return
        previous = self.app.gui_config.mmr_lambda
        self.app.gui_config.mmr_lambda = new_value
        if not self._commit(f"MMR λ: {new_value:.2f}"):
            self.app.gui_config.mmr_lambda = previous
            self.mmr_lambda_slider.value = previous
            if self.mmr_lambda_value_text is not None:
                self.mmr_lambda_value_text.value = _fmt_float(previous)
            self.app.page.update()

    def on_input_mode_changed(self, e: ft.Event) -> None:
        """Persist the input mode (Conversational / Direct query / Direct Cypher)."""
        if self.input_mode_radio is None:
            return
        previous = self.app.gui_config.input_mode
        self.app.gui_config.input_mode = self.input_mode_radio.value
        if not self._commit(f"input mode: {self.app.gui_config.input_mode}"):
            self.app.gui_config.input_mode = previous
            self.input_mode_radio.value = previous
            self.app.page.update()

    # ----- Checkbox handlers -----------------------------------------------

    def on_direct_retrieve_changed(self, e: ft.Event) -> None:
        if self.direct_retrieve_checkbox is None:
            return
        previous = self.app.gui_config.direct_retrieve
        self.app.gui_config.direct_retrieve = bool(self.direct_retrieve_checkbox.value)
        if not self._commit(
            "direct retrieval on (synthesizer skipped)"
            if self.app.gui_config.direct_retrieve
            else "direct retrieval off"
        ):
            self.app.gui_config.direct_retrieve = previous
            self.direct_retrieve_checkbox.value = previous
            self.app.page.update()


# =============================================================================
# Module-local widget helpers.
# =============================================================================


def _int_field(
    value: int,
    on_blur,
) -> ft.TextField:
    """Standard integer-input TextField using the panel's style. The caption
    is added at the build site via `labeled_field`."""
    return ft.TextField(
        value=str(value),
        border=ft.InputBorder.OUTLINE,
        border_color=FRAME_BORDER_COLOR,
        bgcolor=PANEL_BG,
        on_blur=on_blur,
    )


def _slider_row(
    label: str,
    slider: ft.Slider,
    value_text: ft.Text,
) -> ft.Control:
    """Caption + slider (expanding) + numeric value display, in the shared
    `labeled_field` style (caption hugs its text, value trails)."""
    return labeled_field(label, slider, trailing=value_text)


def _fmt_float(value: float | None) -> str:
    """Two-decimal display for slider value (handles None defensively)."""
    if value is None:
        return "0.00"
    return f"{float(value):.2f}"
