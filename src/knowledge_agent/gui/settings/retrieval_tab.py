"""Settings → Retrieval sub-tab — search-side defaults.

11 fields total, grouped into 5 visual sections:

  1. Mode — `retrieval_mode` (6 agent-level topologies) +
     `lancedb_search_mode` (LanceDB-internal mode).
  2. Result size — `top_k` (final result count).
  3. Hybrid fusion (RRF) — `num_candidates`, `rrf_rank_constant`,
     `rrf_rank_window_size`. Ordering rule:
     `top_k <= rrf_rank_window_size <= num_candidates` (mirrors
     `Settings._validate_retrieval_windows`).
  4. Diversity (MMR) — `mmr_lambda` (slider), `mmr_candidate_multiplier`.
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

_LANCEDB_MODE_LABELS: list[tuple[str, str]] = [
    ("hybrid", "Hybrid (BM25 + vector, RRF-fused)"),
    ("fts", "FTS (BM25 only)"),
    ("vector", "Vector (kNN cosine only)"),
]


class RetrievalTab:
    """Retrieval defaults — 11 fields with auto-save + RRF ordering check."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.top_k_field: ft.TextField | None = None
        self.mode_dropdown: ft.Dropdown | None = None
        self.lancedb_mode_radio: ft.RadioGroup | None = None
        self.num_candidates_field: ft.TextField | None = None
        self.rrf_constant_field: ft.TextField | None = None
        self.rrf_window_field: ft.TextField | None = None
        self.mmr_help_text: ft.Text | None = None
        self.use_mmr_checkbox: ft.Checkbox | None = None
        self.mmr_lambda_slider: ft.Slider | None = None
        self.mmr_lambda_value_text: ft.Text | None = None
        self.mmr_multiplier_field: ft.TextField | None = None
        self.kg_max_rows_field: ft.TextField | None = None
        self.input_mode_radio: ft.RadioGroup | None = None
        self.direct_retrieve_checkbox: ft.Checkbox | None = None
        self._create_controls()
        # Initial MMR enable/disable based on the loaded lancedb mode.
        self._sync_mmr_enabled_state()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)
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
        self.lancedb_mode_radio = ft.RadioGroup(
            value=cfg.lancedb_search_mode,
            on_change=self.on_lancedb_mode_changed,
            content=ft.Row(
                controls=[
                    ft.Radio(value=k, label=lbl.split(" (")[0]) for k, lbl in _LANCEDB_MODE_LABELS
                ],
                spacing=16,
            ),
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
        self.rrf_window_field = _int_field(
            cfg.rrf_rank_window_size,
            self.on_rrf_window_blur,
        )
        self.mmr_multiplier_field = _int_field(
            cfg.mmr_candidate_multiplier,
            self.on_mmr_multiplier_blur,
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
            size=11,
            color=ft.Colors.GREY_500,
            italic=True,
        )
        self.mmr_lambda_value_text = ft.Text(
            _fmt_float(cfg.mmr_lambda),
            size=12,
            color=ft.Colors.WHITE,
            width=42,
        )
        self.mmr_lambda_slider = ft.Slider(
            value=cfg.mmr_lambda,
            min=0.0,
            max=1.0,
            divisions=20,
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
                ft.Text(
                    "LanceDB search mode (within-store):",
                    size=12,
                    color=ft.Colors.GREY_400,
                ),
                self.lancedb_mode_radio,
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
                    "Rule: top_k ≤ window ≤ num_candidates. The form "
                    "rejects edits that break this ordering.",
                    size=11,
                    color=ft.Colors.GREY_500,
                    italic=True,
                ),
                labeled_field("num_candidates (vector kNN pool size)", self.num_candidates_field),
                labeled_field("RRF rank window size", self.rrf_window_field),
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
                labeled_field("MMR candidate multiplier", self.mmr_multiplier_field),
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
        self._sync_mmr_enabled_state()
        self.app.page.update()

    def _sync_mmr_enabled_state(self) -> None:
        """Sync MMR control gray-out state — three combined cases.

        - LanceDB mode == FTS → checkbox + slider + multiplier all
          grayed (MMR can't apply to FTS).
        - LanceDB mode is hybrid/vector AND use_mmr=False → checkbox
          enabled (user can flip it on); slider + multiplier grayed
          (MMR is off, its parameters are irrelevant).
        - LanceDB mode is hybrid/vector AND use_mmr=True → all
          controls enabled.

        Help text reflects the active state.
        """
        is_fts = self.app.gui_config.lancedb_search_mode == "fts"
        use_mmr = self.app.gui_config.use_mmr
        params_disabled = is_fts or not use_mmr

        if self.use_mmr_checkbox is not None:
            self.use_mmr_checkbox.disabled = is_fts
            # Force the visible value off when FTS, regardless of stored
            # `use_mmr` — the user shouldn't see a checked box that has
            # no effect.
            self.use_mmr_checkbox.value = use_mmr and not is_fts
        if self.mmr_lambda_slider is not None:
            self.mmr_lambda_slider.disabled = params_disabled
        if self.mmr_multiplier_field is not None:
            self.mmr_multiplier_field.disabled = params_disabled
        if self.mmr_help_text is not None:
            if is_fts:
                self.mmr_help_text.value = "Disabled — FTS has no vectors to diversify."
            elif not use_mmr:
                self.mmr_help_text.value = "MMR off — enable to diversify the result pool."
            else:
                self.mmr_help_text.value = "λ = 1.0 → pure relevance; λ = 0.0 → pure diversity."

    def on_use_mmr_changed(self, e: ft.Event) -> None:
        """Persist use_mmr; re-sync slider/multiplier gray-out."""
        if self.use_mmr_checkbox is None:
            return
        previous = self.app.gui_config.use_mmr
        self.app.gui_config.use_mmr = bool(self.use_mmr_checkbox.value)
        if not self._commit("MMR on" if self.app.gui_config.use_mmr else "MMR off"):
            self.app.gui_config.use_mmr = previous
            self.use_mmr_checkbox.value = previous
        self._sync_mmr_enabled_state()
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

    def on_rrf_window_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.rrf_window_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.rrf_rank_window_size,
            label="RRF rank window size",
            apply=lambda v: setattr(
                self.app.gui_config,
                "rrf_rank_window_size",
                v,
            ),
            validate=self._validate_window_ordering,
        )

    def on_mmr_multiplier_blur(self, e: ft.Event) -> None:
        self._handle_int(
            field=self.mmr_multiplier_field,
            new_value_parser=lambda raw: max(1, int(raw)),
            current=self.app.gui_config.mmr_candidate_multiplier,
            label="MMR candidate multiplier",
            apply=lambda v: setattr(
                self.app.gui_config,
                "mmr_candidate_multiplier",
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
        """Enforce top_k <= rrf_window <= num_candidates after a change."""
        cfg = self.app.gui_config
        if cfg.rrf_rank_window_size < cfg.top_k:
            return (
                f"rrf_rank_window_size ({cfg.rrf_rank_window_size}) must be >= top_k ({cfg.top_k})"
            )
        if cfg.num_candidates < cfg.rrf_rank_window_size:
            return (
                f"num_candidates ({cfg.num_candidates}) must be >= "
                f"rrf_rank_window_size ({cfg.rrf_rank_window_size})"
            )
        return None

    # ----- Slider handlers --------------------------------------------------

    def _on_mmr_lambda_slide(self, e: ft.Event) -> None:
        """Drag-in-progress: update the inline value display only."""
        if self.mmr_lambda_slider is None or self.mmr_lambda_value_text is None:
            return
        self.mmr_lambda_value_text.value = _fmt_float(
            self.mmr_lambda_slider.value,
        )
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
