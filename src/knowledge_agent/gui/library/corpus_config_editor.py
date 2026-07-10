"""Per-corpus config editor — 8 collapsible per-layer sections.

Reads the active corpus's `corpus.toml` via `load_corpus_config`,
renders 8 collapsible sections, writes changes back on Ingest via
`try_save_and_get_error` (called by `IngestTab`).

Sections (top to bottom, mirroring the ingest pipeline order):

  1. **Labels and sub-labels** — main / sub label pickers + overwrite
     checkbox. **Per-run args**, not persisted to corpus.toml.
     Consumed by `IngestTab` via `get_ingest_args()`.
  2. **openalex_papers (L1–L4)** — bundled citation / author /
     venue / topic graph from OpenAlex. `openalex_mailto` is global.
  3. **Chunks (L5)** — `:Chunk` nodes joinable with LanceDB. Chunker
     settings (strategy, max_tokens, OCR) are global.
  4. **Entities (L6)** — chunk-level entity extraction. Layer flag
     + extractor dropdown + entity_types field. Requires chunks.
  5. **Ontology linking (L7)** — 18 ontology checkboxes + xrefs
     3-state radio. Requires the entities layer.
  6. **Triples (L8)** — LLM-extracted (subj, pred, obj). Layer flag
     only; LLM model + temperature are global.
  7. **Cross-doc (L9)** — `:RELATED_TO` edges between shared-entity
     docs. Layer flag + threshold.
  8. **Cross-doc xrefs (L10)** — `:RELATED_BY_XREF` via canonical
     concepts. Layer flag + threshold. Requires xrefs=`"use"`.

Each layer section (2–8) shows a Current / New chip line as its
subtitle — Current mirrors the last-saved corpus.toml state,
New is what would be written if the user hits Ingest now.

`allowed_types` is no longer edited here — the backend defaults to
every known sub-label, and the Labels section's sub-label dropdown
consumes it. Users who want to restrict a corpus hand-edit
corpus.toml directly.

Staging model: toggles mutate the in-memory `CorpusConfig` only.
Disk write happens once at Ingest kickoff (via
`try_save_and_get_error`), so failed / abandoned ingests don't
persist half-formed state to corpus.toml.

Empty state (no active corpus): the editor renders a hint instead
of the sections.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import flet as ft
from pydantic import ValidationError

from knowledge_agent.config import Settings, get_settings
from knowledge_agent.entity_extractors.extractor_lifecycle import (
    EXTRACTOR_REGISTRY,
    is_extractor_ready,
)
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    PANEL_BG,
    centered_label,
    labeled_field,
    section_divider,
    section_title,
    sub_section_header,
)
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui.library.create_new_dataset import _write_corpus_toml
from knowledge_agent.gui.library.session_state import load_session, update_draft
from knowledge_agent.gui.settings.llm_tab import LLM_AVAILABLE_MODELS
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    CrossDocConfig,
    CrossDocXrefsConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
    load_corpus_config,
)
from knowledge_agent.kg.schema import (
    DOCUMENT_LABEL,
    MAIN_LABELS,
    SUB_LABEL_TO_MAIN,
)
from knowledge_agent.llm_factory import supports_temperature

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.gui.app import GuiApp


# Sentinel for the sub-label dropdown's "no sub-label applied" option.
_SUB_NONE_KEY = "__none__"
_SUB_NONE_TEXT = "(none)"


logger = logging.getLogger(__name__)


# Ontology → display name for the checkboxes. Keys match the
# `LayerFlags.ontology_*` field names (without the `ontology_` prefix).
_ONTOLOGY_DISPLAY: dict[str, str] = {
    "mesh": "MeSH",
    "go": "GO",
    "hpo": "HPO",
    "uberon": "UBERON",
    "mondo": "MONDO",
    "chebi": "ChEBI",
    "eco": "ECO",
    "so": "SO",
    "pr": "PR",
    "cl": "CL",
    "po": "PO",
    "foodon": "FOODON",
    "envo": "ENVO",
    "ncbitaxon": "NCBITaxon",
    "obi": "OBI",
    "efo": "EFO",
    "dron": "DRON",
    "fibo": "FIBO",
}


class CorpusConfigEditor:
    """8-section per-layer editor for the active corpus's `corpus.toml`.

    Sections (top to bottom):
      1. Labels and sub-labels — main / sub label pickers + overwrite
         checkbox. Per-run args (not persisted to corpus.toml).
      2. openalex_papers (L1–L4)
      3. Chunks (L5)
      4. Entities (L6)
      5. Ontology linking (L7)
      6. Triples (L8)
      7. Cross-doc (L9)
      8. Cross-doc xrefs (L10)
    """

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        # In-memory config being edited + which corpus it's for. Toggles
        # mutate this in memory only — nothing writes to corpus.toml until
        # an Ingest action fires. `_baseline_config` snapshots the on-disk
        # (last-ingested) state at load time so we can detect pending
        # changes and offer a Discard button that reverts to it.
        #
        # Unsaved edits ARE persisted, though: every mutation mirrors the
        # draft to the corpus's session sidecar (see `session_state`), so
        # closing the app mid-edit and reopening the corpus restores the
        # draft — Discard button lit, pending-changes summary intact.
        self._corpus_config: CorpusConfig | None = None
        self._baseline_config: CorpusConfig | None = None
        self._loaded_for_corpus: str | None = None
        # Fired after the draft is (re)persisted, so a sibling view (the
        # Select tab's corpus card) can refresh its "changed since ingest"
        # section live. Wired by `LibraryView`.
        self.on_draft_changed: Callable[[], None] | None = None

        # Dirty-state indicator + Discard button — created in
        # `_create_controls`, mounted in `build()`.
        self.dirty_indicator: ft.Text | None = None
        self.discard_button: ft.Button | None = None

        # ----- Labels and sub-labels (per-run args) -----
        self.main_label_dropdown: ft.Dropdown | None = None
        self.sub_label_dropdown: ft.Dropdown | None = None
        self.overwrite_checkbox: ft.Checkbox | None = None

        # ----- Ontologies -----
        self.ontology_checkboxes: dict[str, ft.Checkbox] = {}
        # Per-ontology matching Dropdown (exact / fuzzy). Promoted from
        # read-only global to editable per-ontology 2026-07-02. Backend
        # already fully supports this via `OntologyConfig.matching`.
        self.ontology_matching_dropdowns: dict[str, ft.Dropdown] = {}

        # ----- Layer flags -----
        self.chunks_checkbox: ft.Checkbox | None = None
        self.openalex_checkbox: ft.Checkbox | None = None
        self.entities_checkbox: ft.Checkbox | None = None
        self.triples_checkbox: ft.Checkbox | None = None
        self.cross_doc_checkbox: ft.Checkbox | None = None
        self.cross_doc_xrefs_checkbox: ft.Checkbox | None = None
        self.xrefs_radio: ft.RadioGroup | None = None

        # ----- Entities (L6) per-corpus fields (promoted 2026-07-02) -----
        # LLM model + temperature moved from global Settings to per corpus.
        # `entity_types` is reused across extractors; the "Use custom labels"
        # checkbox derives from whether entity_types is non-empty (option A).
        # 2026-07-02: model = editable Dropdown, temperature = Slider + readout
        # (mirrors the Settings LLM tab pattern).
        self.entity_extractor_model_field: ft.Dropdown | None = None
        self.entity_extractor_temperature_slider: ft.Slider | None = None
        self.entity_extractor_temperature_readout: ft.Text | None = None
        self.entities_llm_group: ft.Container | None = None
        self.entities_gliner_group: ft.Container | None = None
        self.entities_hunflair2_group: ft.Container | None = None
        # entity_types_mode ("replace" | "add") — RadioGroup. Replaces
        # the old GLiNER-only "use custom labels" checkbox.
        self.entity_types_mode_radio: ft.RadioGroup | None = None

        # ----- Triples (L8) per-corpus fields (promoted 2026-07-02) -----
        # Same Dropdown + Slider pattern as Entities.
        self.triples_extractor_model_field: ft.Dropdown | None = None
        self.triples_extractor_temperature_slider: ft.Slider | None = None
        self.triples_extractor_temperature_readout: ft.Text | None = None

        # ----- Chunks (L5) per-corpus fields (promoted 2026-07-02) -----
        # Previously read from global Settings; now editable per corpus.
        # `min_rows_for_vector_index` intentionally stays read-only —
        # LanceDB internal, not user-facing tuning.
        self.chunker_strategy_dropdown: ft.Dropdown | None = None
        self.chunk_max_tokens_field: ft.TextField | None = None
        self.merge_peers_checkbox: ft.Checkbox | None = None
        self.enable_pdf_ocr_checkbox: ft.Checkbox | None = None
        self.enable_image_ocr_checkbox: ft.Checkbox | None = None
        self.images_scale_field: ft.TextField | None = None
        self.extract_figures_checkbox: ft.Checkbox | None = None
        self.embed_images_checkbox: ft.Checkbox | None = None
        self.min_figure_bytes_field: ft.TextField | None = None
        self.optimize_indexes_checkbox: ft.Checkbox | None = None

        # ----- Entity extraction -----
        # Multi-extractor (priority-ordered union): `_selected_extractors`
        # is the ordered source of truth (index 0 = base/primary). The
        # widget is a rebuilt Column of per-adapter rows (checkbox + name
        # + priority + up/down). `extractor_info_banner` warns from the
        # 3rd selected extractor on.
        self._selected_extractors: list[str] = ["llm"]
        self.extractor_rows_column: ft.Column | None = None
        self.extractor_info_banner: ft.Container | None = None
        self.entity_types_field: ft.TextField | None = None

        # ----- Cross-doc thresholds -----
        self.cross_doc_threshold_field: ft.TextField | None = None
        self.cross_doc_xrefs_threshold_field: ft.TextField | None = None

        # ----- Per-section subtitles -----
        # Every section gets a Text subtitle showing state at a glance
        # while collapsed. Labels section shows the current dropdown
        # selection; every other section shows Current / New for the
        # persistent config that section owns.
        self.labels_subtitle: ft.Text | None = None
        self.openalex_subtitle: ft.Text | None = None
        self.chunks_subtitle: ft.Text | None = None
        self.entities_subtitle: ft.Text | None = None
        self.ontology_subtitle: ft.Text | None = None
        self.triples_subtitle: ft.Text | None = None
        self.cross_doc_subtitle: ft.Text | None = None
        self.cross_doc_xrefs_subtitle: ft.Text | None = None

        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=12, color=ft.Colors.GREY_400)

        # Section subtitles — Text instances so the collapsed state
        # of each ExpansionTile shows what's active.
        self.labels_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.openalex_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.chunks_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.entities_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.ontology_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.triples_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.cross_doc_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.cross_doc_xrefs_subtitle = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_400,
        )

        # ----- Labels and sub-labels (per-run args) -----
        self.main_label_dropdown = ft.Dropdown(
            value=DOCUMENT_LABEL,
            options=[ft.DropdownOption(key=lbl, text=lbl) for lbl in MAIN_LABELS],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_select=self._on_main_label_changed,
        )
        self.sub_label_dropdown = ft.Dropdown(
            value=_SUB_NONE_KEY,
            options=[
                ft.DropdownOption(key=_SUB_NONE_KEY, text=_SUB_NONE_TEXT),
            ],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_select=self._on_sub_label_changed,
        )
        self.overwrite_checkbox = ft.Checkbox(
            label="Overwrite existing labels",
            value=False,
            tooltip=(
                "Only affects Sync and Re-ingest, when a file is already "
                "in the corpus. OFF (default): keep the stored labels. "
                "ON: overwrite with the values above."
            ),
            on_change=self._on_overwrite_changed,
        )

        # Dirty-state indicator + Discard button. Both hidden/disabled
        # when in-memory state matches on-disk state.
        self.dirty_indicator = ft.Text(
            "• unsaved",
            size=12,
            color=ft.Colors.AMBER_300,
            italic=True,
            visible=False,
        )
        self.discard_button = ft.Button(
            content=centered_label("Discard changes"),
            on_click=self._on_discard_clicked,
            disabled=True,
        )

        # ----- Ontologies -----
        for key, display in _ONTOLOGY_DISPLAY.items():
            self.ontology_checkboxes[key] = ft.Checkbox(
                label=display,
                value=False,
                on_change=lambda e, k=key: self._on_ontology_toggle(k),
            )
            self.ontology_matching_dropdowns[key] = ft.Dropdown(
                value="exact",
                width=110,
                options=[
                    ft.DropdownOption(key="exact", text="exact"),
                    ft.DropdownOption(key="fuzzy", text="fuzzy"),
                ],
                border=ft.InputBorder.OUTLINE,
                border_color=FRAME_BORDER_COLOR,
                bgcolor=PANEL_BG,
                tooltip=(
                    f"{display}: entity → term matching strategy. "
                    "exact = label / synonym exact match only. "
                    "fuzzy = try exact first, then morphological "
                    "variants (plural / hyphen flips, edit-distance <= 2)."
                ),
                on_select=lambda e, k=key: self._on_ontology_matching_changed(k),
            )

        # ----- Layer flags -----
        self.chunks_checkbox = ft.Checkbox(
            label="chunks (L5) — per-chunk :Chunk nodes for retrieval",
            value=True,
            on_change=lambda e: self._on_layer_toggle("chunks"),
        )
        self.openalex_checkbox = ft.Checkbox(
            label="openalex_papers — resolve DOIs via OpenAlex",
            value=False,
            on_change=lambda e: self._on_layer_toggle("openalex_papers"),
        )
        self.entities_checkbox = ft.Checkbox(
            label="entities (L6) — extract named entities per chunk",
            value=False,
            on_change=lambda e: self._on_layer_toggle("entities"),
        )
        self.triples_checkbox = ft.Checkbox(
            label="triples (L8) — extract (subj, pred, obj) per chunk",
            value=False,
            on_change=lambda e: self._on_layer_toggle("triples"),
        )
        self.cross_doc_checkbox = ft.Checkbox(
            label="cross_doc (L9) — shared-entity links across docs",
            value=False,
            on_change=lambda e: self._on_layer_toggle("cross_doc"),
        )
        self.cross_doc_xrefs_checkbox = ft.Checkbox(
            label="cross_doc_xrefs (L10) — shared-xref links across docs",
            value=False,
            on_change=lambda e: self._on_layer_toggle("cross_doc_xrefs"),
        )
        # xrefs 3-state — user-facing wording uses "materialize"
        # consistently (backend values stay "none" / "collect_only" / "use").
        self.xrefs_radio = ft.RadioGroup(
            value="none",
            on_change=self._on_xrefs_changed,
            content=ft.Column(
                controls=[
                    ft.Radio(value="none", label="Do not materialize"),
                    ft.Radio(
                        value="collect_only",
                        label="Collect only — don't materialize edges yet",
                    ),
                    ft.Radio(value="use", label="Materialize edges now"),
                ],
                spacing=4,
            ),
        )

        # ----- Entity extraction -----
        # Extractors whose pip package OR pinned weights aren't on disk
        # are grayed out (disabled=True). Ready-state text appended to
        # the label so the user knows WHY the option can't be selected:
        # "GLiNER (install + download weights in Library → Installs)".
        # `is_extractor_ready` reads the compound state (pip+weights)
        # so we don't duplicate the readiness logic here.
        # Multi-extractor selection + priority widget. The rows (one per
        # adapter: checkbox + name + priority + up/down) are rebuilt by
        # `_rebuild_extractor_widget()` from `_selected_extractors` during
        # load + on every change, so order + readiness stay in sync.
        self.extractor_rows_column = ft.Column(spacing=2)
        # Cost / diminishing-returns warning, shown from the 3rd selected
        # extractor on (inform-don't-restrict; mirrors the Installs
        # heavy-download warnings).
        self.extractor_info_banner = ft.Container(
            visible=False,
            padding=ft.Padding.symmetric(vertical=6, horizontal=8),
            border=ft.Border.all(1, ft.Colors.AMBER_700),
            border_radius=4,
            content=ft.Row(
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
                controls=[
                    ft.Icon(
                        ft.Icons.INFO_OUTLINE,
                        size=16,
                        color=ft.Colors.AMBER_400,
                    ),
                    ft.Text(
                        "Each extractor runs a full pass over every chunk "
                        "— runtime and (if the LLM is included) cost scale "
                        "linearly with each one added. A third often adds "
                        "little new coverage, especially two NERs in the "
                        "same domain. Add only if the extra recall is "
                        "worth it.",
                        size=12,
                        color=ft.Colors.AMBER_200,
                        italic=True,
                        expand=True,
                    ),
                ],
            ),
        )
        # Shared across all selected extractors: LLM as hints, the GLiNER
        # adapters as detection targets, HunFlair2 ignores it. Enabled
        # whenever any non-HunFlair2 extractor is selected (see
        # `_refresh_extractor_groups`).
        self.entity_types_field = ft.TextField(
            hint_text="e.g. GENE, DISEASE, CHEMICAL",
            text_size=14,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_entity_types_blur,
        )
        # entity_types_mode: how a non-empty types list interacts with
        # each adapter's own DEFAULT_LABELS. Replaces the old GLiNER
        # "use custom labels" checkbox.
        self.entity_types_mode_radio = ft.RadioGroup(
            value="replace",
            on_change=self._on_entity_types_mode_changed,
            content=ft.Row(
                spacing=16,
                controls=[
                    ft.Radio(value="replace", label="Replace defaults"),
                    ft.Radio(value="add", label="Add to defaults"),
                ],
            ),
        )

        # ----- Entities (L6) LLM adapter — model + temperature -----
        # LLM-group controls (visible when extractor="llm"). Editable
        # Dropdown with per-provider curated menu (typed override allowed
        # for off-menu models) + Slider + readout — mirrors the Settings
        # LLM tab pattern.
        provider_options = [
            ft.DropdownOption(key=m, text=m)
            for m in LLM_AVAILABLE_MODELS.get(self.app.gui_config.llm_provider, ())
        ]
        self.entity_extractor_model_field = ft.Dropdown(
            value="claude-haiku-4-5-20251001",
            options=list(provider_options),
            editable=True,
            enable_filter=True,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_entity_extractor_model_blur,
        )
        self.entity_extractor_temperature_readout = ft.Text(
            _fmt_float(0.0),
            size=12,
            color=ft.Colors.WHITE,
            width=42,
        )
        self.entity_extractor_temperature_slider = ft.Slider(
            value=0.0,
            min=0.0,
            max=1.0,
            divisions=20,
            on_change=self._on_entity_extractor_temperature_slide,
            on_change_end=self._on_entity_extractor_temperature_committed,
        )

        # ----- Triples (L8) LLM adapter — model + temperature -----
        self.triples_extractor_model_field = ft.Dropdown(
            value="claude-haiku-4-5-20251001",
            options=list(provider_options),
            editable=True,
            enable_filter=True,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_triples_extractor_model_blur,
        )
        self.triples_extractor_temperature_readout = ft.Text(
            _fmt_float(0.0),
            size=12,
            color=ft.Colors.WHITE,
            width=42,
        )
        self.triples_extractor_temperature_slider = ft.Slider(
            value=0.0,
            min=0.0,
            max=1.0,
            divisions=20,
            on_change=self._on_triples_extractor_temperature_slide,
            on_change_end=self._on_triples_extractor_temperature_committed,
        )
        self._sync_extractor_temp_enabled()
        # Entities per-extractor group containers — built ONCE here so
        # `_refresh_extractor_groups()` can toggle `.visible` reliably.
        # (Previously built lazily inside build(), which overwrote the
        # instance every render and lost the visibility state — bug fix
        # 2026-07-02.)
        self.entities_llm_group = ft.Container(
            visible=False,
            content=ft.Column(
                spacing=6,
                controls=[
                    sub_section_header("LLM extractor settings"),
                    # model on its own line, temperature below it.
                    labeled_field(
                        "entity_extractor_model",
                        self.entity_extractor_model_field,
                    ),
                    labeled_field(
                        "temperature",
                        self.entity_extractor_temperature_slider,
                        trailing=self.entity_extractor_temperature_readout,
                    ),
                ],
            ),
        )
        self.entities_gliner_group = ft.Container(
            visible=False,
            content=ft.Column(
                spacing=6,
                controls=[
                    sub_section_header("GLiNER extractor settings"),
                    ft.Text(
                        "Empty 'Types to extract' → the adapter's default "
                        "labels; the label sets + Replace/Add behaviour are "
                        "described in the section (i).",
                        size=12,
                        color=ft.Colors.GREY_500,
                        italic=True,
                    ),
                ],
            ),
        )
        self.entities_hunflair2_group = ft.Container(
            visible=False,
            content=ft.Column(
                spacing=4,
                controls=[
                    sub_section_header("HunFlair2 extractor settings"),
                    ft.Text(
                        "Fixed label set — 'Types to extract' has no effect "
                        "(details in the section (i)).",
                        size=12,
                        color=ft.Colors.GREY_500,
                        italic=True,
                    ),
                ],
            ),
        )

        # ----- Chunks (L5) per-corpus fields -----
        self.chunker_strategy_dropdown = ft.Dropdown(
            value="hybrid",
            options=[
                ft.DropdownOption(key="hybrid", text="hybrid"),
                ft.DropdownOption(key="hierarchical", text="hierarchical"),
            ],
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_select=self._on_chunker_strategy_changed,
        )
        self.chunk_max_tokens_field = ft.TextField(
            value="512",
            hint_text="e.g. 512 (HybridChunker only)",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_chunk_max_tokens_blur,
        )
        self.merge_peers_checkbox = ft.Checkbox(
            label="merge_peers — greedy-merge adjacent chunks",
            value=True,
            tooltip=(
                "HybridChunker only. When on, adjacent same-section "
                "chunks that both fit under chunk_max_tokens are "
                "merged into one."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "merge_peers",
            ),
        )
        self.enable_pdf_ocr_checkbox = ft.Checkbox(
            label="enable_pdf_ocr — OCR for PDF inputs",
            value=False,
            tooltip=(
                "Turn on for scanned / image-only PDFs. Adds latency "
                "for born-digital papers, so default is off."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "enable_pdf_ocr",
            ),
        )
        self.enable_image_ocr_checkbox = ft.Checkbox(
            label="enable_image_ocr — OCR for image inputs",
            value=True,
            tooltip=(
                "Standalone PNG / JPG / TIFF inputs. Without OCR they "
                "produce no searchable text; default is on."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "enable_image_ocr",
            ),
        )
        self.images_scale_field = ft.TextField(
            value="1.0",
            hint_text="e.g. 1.0 (higher = more detail, slower)",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_images_scale_blur,
        )
        self.extract_figures_checkbox = ft.Checkbox(
            label="extract_figures — save PDF/Word/PPT pictures to disk",
            value=False,
            tooltip=(
                "Multimodal: when on, embedded pictures in PDF / Word / "
                "PPT sources are extracted and saved as PNGs alongside "
                "the parsed doc, for later embedding + display. "
                "Standalone image files are always referenced in place."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "extract_figures",
            ),
        )
        self.embed_images_checkbox = ft.Checkbox(
            label="embed_images — produce figure chunks + multimodal embeddings",
            value=False,
            tooltip=(
                "Multimodal: when on, figure chunks are emitted at "
                "chunking time and sent as [caption, image] to the "
                "multimodal embedder (Voyage today). Requires a "
                "multimodal-capable embedding provider. Off = only "
                "text chunks produced, even if extract_figures=true."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "embed_images",
            ),
        )
        self.min_figure_bytes_field = ft.TextField(
            value="2048",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_min_figure_bytes_blur,
            tooltip=(
                "Multimodal: minimum PNG size in bytes for a saved "
                "figure to be kept. Files smaller than this are deleted "
                "right after being written and no figure chunk is "
                "emitted. Docling can flag tiny decorative pictures "
                "(page banners, logos, single-glyph icons) around "
                "400-1000 B; the 2048 (2 KB) default filters those "
                "without touching real diagrams. Set to 0 to disable "
                "the filter and keep every picture Docling emits. "
                "Applies only when extract_figures=true."
            ),
        )
        self.optimize_indexes_checkbox = ft.Checkbox(
            label="optimize_indexes_per_ingest — refresh LanceDB indexes",
            value=True,
            tooltip=(
                "When on, LanceDB vector + FTS indexes rebuild after "
                "every ingest. Turn off for bulk ingest sessions to "
                "defer to a single optimize at the end."
            ),
            on_change=lambda e: self._on_chunks_bool_changed(
                "optimize_indexes_per_ingest",
            ),
        )

        # ----- Cross-doc thresholds -----
        self.cross_doc_threshold_field = ft.TextField(
            value="2",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_cross_doc_threshold_blur,
        )
        self.cross_doc_xrefs_threshold_field = ft.TextField(
            value="2",
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
            on_blur=self._on_cross_doc_xrefs_threshold_blur,
        )

    # ----- public API -------------------------------------------------------

    def _section_title(self, label: str, help_text: str) -> ft.Control:
        """Section header: the label plus an `(i)` whose dialog carries the
        section's help prose.

        The descriptive text that used to sit inline under each section is
        moved into this dialog to keep the panel uncluttered; the `(i)`
        obeys the global teaching-mode toggle (Settings → App). Read-only
        VALUE displays (allowed_types, the process-wide globals) stay inline
        — only the explanatory prose moves here.
        """
        return ft.Row(
            controls=[
                section_title(label),
                info_icon(self.app, title=label, text=help_text),
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def build(self) -> ft.Control:
        """Load the active corpus's config (if a different one from the
        last render), then render the 8 per-layer sections + status."""
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return ft.Text(
                "No active corpus — pick one above (or create one) to edit its config.",
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            )
        if active_name != self._loaded_for_corpus:
            self._reload_for_active_corpus(active_name)

        if self._corpus_config is None:
            # Load failed — the status text explains why.
            return ft.Column(
                controls=[self.status] if self.status is not None else [],
                spacing=8,
            )

        # Sub-label dropdown options depend on current allowed_types +
        # the currently-picked main_label; refresh on every render so
        # switching corpora + editing allowed_types both update it.
        self._refresh_sub_label_options()

        col = ft.Column(
            spacing=8,
            controls=[
                # The panel title ("Ingestion settings" + dirty ● / Discard)
                # lives in the Ingest tab's fixed header above this pane, so
                # it stays visible while the pane scrolls.
                # ---- Ingest infrastructure (read-only, process-wide) ----
                # HTTP + concurrency knobs that affect every ingest but
                # aren't per-corpus tunable. Collapsed by default so the
                # 8 per-layer sections stay the focal point.
                self._build_ingest_infrastructure_block(),
                # ---- Section 1: Labels and sub-labels (per-run) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Labels and sub-labels",
                        "Applied to every file in the next ingest. Not "
                        "persisted to corpus.toml — pick fresh each run.\n\n"
                        "Sub-label options come from allowed_types, filtered "
                        "to the chosen Main label. If allowed_types is empty, "
                        "no sub-labels are available.\n\n"
                        "allowed_types (read-only): "
                        f"{_format_allowed_types(list(self._corpus_config.allowed_types))}"
                        ". Default = all 14 sub-labels; hand-edit corpus.toml "
                        "to restrict this corpus.",
                    ),
                    subtitle=self.labels_subtitle,
                    expanded=True,
                    controls=[
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field(
                                "Main label",
                                self.main_label_dropdown,
                            ),
                        ),
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field(
                                "Sub-label",
                                self.sub_label_dropdown,
                            ),
                        ),
                        self.overwrite_checkbox,
                    ],
                ),
                # ---- Section 2: openalex_papers (L1–L4) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "openalex_papers (L1–L4)",
                        "L1–L4 bundle: citation + author + venue + topic graph "
                        "from a single OpenAlex API lookup per doc. Turn on for "
                        "scientific papers.\n\n"
                        "Global (read-only): openalex_mailto = "
                        f"{self._read_global('openalex_mailto')}. "
                        "Edit via .env / global Settings.",
                    ),
                    subtitle=self.openalex_subtitle,
                    controls=[
                        self.openalex_checkbox,
                    ],
                ),
                # ---- Section 3: Chunks (L5) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Chunks (L5)",
                        "Per-chunk :Chunk nodes joinable with LanceDB via "
                        "`chunk_id`. Required for retrieval — effectively "
                        "always on.\n\n"
                        "min_rows_for_vector_index (shown below, read-only): "
                        "LanceDB IVF_PQ needs ~256 rows to train; below this, "
                        "brute-force scan is used. Not user-facing tuning.",
                    ),
                    subtitle=self.chunks_subtitle,
                    controls=[
                        self.chunks_checkbox,
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field(
                                "Chunker strategy", self.chunker_strategy_dropdown
                            ),
                        ),
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field("Chunk max tokens", self.chunk_max_tokens_field),
                        ),
                        self.merge_peers_checkbox,
                        ft.Row(
                            controls=[
                                self.enable_pdf_ocr_checkbox,
                                self.enable_image_ocr_checkbox,
                            ],
                            spacing=16,
                            wrap=True,
                        ),
                        # Fields use the app-wide `Label: [input]` row style;
                        # padding gives them more breathing room than the
                        # tightly-stacked checkboxes.
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field("Images scale", self.images_scale_field),
                        ),
                        ft.Row(
                            controls=[
                                self.extract_figures_checkbox,
                                self.embed_images_checkbox,
                            ],
                            spacing=16,
                            wrap=True,
                        ),
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field("Min figure bytes", self.min_figure_bytes_field),
                        ),
                        self.optimize_indexes_checkbox,
                        _globals_block(
                            [
                                (
                                    "min_rows_for_vector_index",
                                    self._read_global("min_rows_for_vector_index"),
                                ),
                            ],
                        ),
                    ],
                ),
                # ---- Section 4: Entities (L6) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Entities (L6)",
                        "Chunk-level entity extraction. Requires the chunks "
                        "layer.\n\n"
                        "Extractors run in priority order — the top one owns "
                        "overlapping spans; lower ones only add spans the "
                        "earlier ones missed. Each extractor is a full pass "
                        "over every chunk, so runtime (and LLM cost, if the "
                        "LLM is included) scales with each one added; a third "
                        "rarely adds much new coverage.\n\n"
                        "Types to extract (shared): LLM — empty = categorises "
                        "freely (open vocabulary). GLiNER / GLiNER-BioMed — "
                        "empty = the adapter's default labels (GLiNER: PERSON, "
                        "ORG, LOC, EVENT, DATE, MISC; GLiNER-BioMed: DISEASE, "
                        "CHEMICAL, GENE, PROTEIN, SPECIES, CELL_LINE, "
                        "CELL_TYPE, ANATOMY); Replace/Add controls whether "
                        "typed labels replace those defaults or extend them. "
                        "HunFlair2 — fixed 5-label set (DISEASE, CHEMICAL, "
                        "GENE, SPECIES, CELL_LINE); entity_types has no effect.",
                    ),
                    subtitle=self.entities_subtitle,
                    controls=[
                        self.entities_checkbox,
                        # --- Sub-section: Extractors (ordering explained in (i)) ---
                        sub_section_header("Extractors"),
                        self.extractor_rows_column,
                        self.extractor_info_banner,
                        # --- Sub-section: Types to extract ---
                        sub_section_header("Types to extract"),
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=self.entity_types_field,
                        ),
                        self.entity_types_mode_radio,
                        # Per-adapter settings — each group toggles `.visible`
                        # by set membership (see `_refresh_extractor_groups`)
                        # and carries its own thin-rule + sub-section title.
                        self.entities_llm_group,
                        self.entities_gliner_group,
                        self.entities_hunflair2_group,
                    ],
                ),
                # ---- Section 5: Ontology linking (L7) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Ontology linking (L7)",
                        "Canonicalise entities against curated vocabularies "
                        "(MeSH, GO, ...). Requires the entities layer. Each "
                        "ontology must first be installed via the Installs "
                        "tab; the checkboxes below only toggle whether the "
                        "already-installed ontology gets linked against for "
                        "this corpus.",
                    ),
                    subtitle=self.ontology_subtitle,
                    controls=[
                        # Nested collapsible sub-folder — 18 ontology
                        # rows with checkbox + matching Dropdown each.
                        ft.ExpansionTile(
                            title=ft.Text("Enabled ontologies", size=14),
                            controls=[
                                self._build_ontology_rows(),
                            ],
                        ),
                        # Cross-ontology equivalence edges (xrefs) sub-section.
                        sub_section_header(
                            "Cross-ontology equivalence edges (xrefs)",
                            trailing=info_icon(
                                self.app,
                                title="Cross-ontology equivalence edges (xrefs)",
                                text=(
                                    "Materialise `:<X>_XREF` edges between term nodes "
                                    "across ontologies (MeSH ↔ MONDO, MONDO ↔ ChEBI, "
                                    "...). L10 cross-doc xrefs requires this be set to "
                                    '"Materialize edges now".'
                                ),
                            ),
                        ),
                        self.xrefs_radio,
                    ],
                ),
                # ---- Section 6: Triples (L8) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Triples (L8)",
                        "LLM-extracted (subject, predicate, object) edges "
                        "between entities per chunk. Requires the entities "
                        "layer. The 15 predicate types (INHIBITS / ACTIVATES "
                        "/ ...) are fixed at the code level.",
                    ),
                    subtitle=self.triples_subtitle,
                    controls=[
                        self.triples_checkbox,
                        # model on its own line, temperature below it.
                        labeled_field(
                            "triples_extractor_model",
                            self.triples_extractor_model_field,
                        ),
                        labeled_field(
                            "temperature",
                            self.triples_extractor_temperature_slider,
                            trailing=self.triples_extractor_temperature_readout,
                        ),
                    ],
                ),
                # ---- Section 7: Cross-doc (L9) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Cross-doc (L9)",
                        ":RELATED_TO edges between docs sharing ≥ N L6 "
                        "entities. Requires the entities layer.",
                    ),
                    subtitle=self.cross_doc_subtitle,
                    controls=[
                        self.cross_doc_checkbox,
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field(
                                "Threshold (min shared entities)",
                                self.cross_doc_threshold_field,
                            ),
                        ),
                    ],
                ),
                # ---- Section 8: Cross-doc xrefs (L10) ----
                ft.ExpansionTile(
                    title=self._section_title(
                        "Cross-doc xrefs (L10)",
                        ":RELATED_BY_XREF edges between docs sharing ≥ N "
                        "canonical concepts (via xref equivalence). Requires "
                        'entities + xrefs="use".',
                    ),
                    subtitle=self.cross_doc_xrefs_subtitle,
                    controls=[
                        self.cross_doc_xrefs_checkbox,
                        ft.Container(
                            padding=ft.Padding.symmetric(vertical=6),
                            content=labeled_field(
                                "Threshold (min shared xrefs)",
                                self.cross_doc_xrefs_threshold_field,
                            ),
                        ),
                    ],
                ),
                self.status,
            ],
        )
        # One section_divider between consecutive dropdowns, always visible
        # (same 2px / GREY_500 rule as the left column). ExpansionTiles also
        # draw their OWN top/bottom divider lines when open (a Material
        # default) — silence those with a border-less shape so a section
        # break is our single intended line, not a stack of three.
        laid_out: list[ft.Control] = []
        for i, ctrl in enumerate(col.controls):
            if isinstance(ctrl, ft.ExpansionTile):
                ctrl.shape = ft.RoundedRectangleBorder(radius=0)
                ctrl.collapsed_shape = ft.RoundedRectangleBorder(radius=0)
                # Fill width + left-align (the default centres plain Text and
                # bare fields), matching the left column's STRETCH layout.
                ctrl.expanded_cross_axis_alignment = ft.CrossAxisAlignment.STRETCH
                if i and isinstance(col.controls[i - 1], ft.ExpansionTile):
                    laid_out.append(section_divider())
            laid_out.append(ctrl)
        col.controls = laid_out
        return col

    # ----- load + populate -------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load state for the active corpus if not already loaded.

        Public entry point for sibling components (e.g. `IngestTab`
        reading `allowed_types` to populate its sub-label dropdown
        before this editor's own `build()` has been called).
        """
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return
        if active_name != self._loaded_for_corpus:
            self._reload_for_active_corpus(active_name)

    def invalidate(self) -> None:
        """Drop the load cache so the next `build()` / `ensure_loaded()`
        reloads from disk.

        The cache is keyed by corpus NAME (`_loaded_for_corpus`), so a
        plain corpus SWITCH already reloads. This covers the cases the
        name-key misses: a relocate (same name, new corpus.toml path)
        and the rebuild `LibraryView.refresh_ingest` forces after any
        corpus mutation. In-memory edits aren't lost on reload — every
        edit is mirrored to the corpus's session sidecar, so the reload
        restores that corpus's own draft (one corpus's unsaved config is
        never carried onto another).
        """
        self._loaded_for_corpus = None

    def _reload_for_active_corpus(self, name: str) -> None:
        """Load the active corpus's config from its `corpus.toml` and
        snapshot it as the baseline for dirty-state comparison."""
        if self.status is None:
            return
        entry = self._find_active_entry(name)
        if entry is None:
            self._corpus_config = None
            self._baseline_config = None
            self._loaded_for_corpus = name
            self.status.value = f"active corpus {name!r} not found in registry"
            return
        try:
            self._corpus_config = load_corpus_config(entry.corpus_config_path)
        except FileNotFoundError:
            self._corpus_config = None
            self._baseline_config = None
            self._loaded_for_corpus = name
            self.status.value = f"corpus.toml not found at {entry.corpus_config_path}"
            return
        except Exception as exc:
            logger.warning("load_corpus_config failed: %r", exc)
            self._corpus_config = None
            self._baseline_config = None
            self._loaded_for_corpus = name
            self.status.value = f"corpus.toml load failed: {exc}"
            return
        # Snapshot on-disk state — `_baseline_config` stays put while
        # `_corpus_config` mutates on toggle. `is_dirty()` compares
        # the two.
        self._baseline_config = self._corpus_config.model_copy(deep=True)
        self._loaded_for_corpus = name
        # Restore any unsaved draft from the session sidecar so reopening a
        # corpus resumes mid-edit: the baseline stays the on-disk config
        # and the draft rides on top → Discard button lit + pending-changes
        # summary intact.
        self._restore_draft(entry.corpus_config_path)
        self._populate_controls()
        self._refresh_availability()
        self._refresh_dirty_indicator()
        self.status.value = ""

    def _populate_controls(self) -> None:
        """Fill every control from the currently loaded `CorpusConfig`."""
        cfg = self._corpus_config
        if cfg is None:
            return
        # Ontologies.
        for key, checkbox in self.ontology_checkboxes.items():
            checkbox.value = getattr(cfg.layers, f"ontology_{key}", False)
        # Per-ontology matching dropdowns (matching value if present,
        # default 'exact' otherwise).
        for key, dropdown in self.ontology_matching_dropdowns.items():
            ont_cfg = cfg.ontology.get(key)
            dropdown.value = ont_cfg.matching if ont_cfg is not None else "exact"
        # Layer flags.
        if self.chunks_checkbox is not None:
            self.chunks_checkbox.value = cfg.layers.chunks
        if self.openalex_checkbox is not None:
            self.openalex_checkbox.value = cfg.layers.openalex_papers
        if self.entities_checkbox is not None:
            self.entities_checkbox.value = cfg.layers.entities
        if self.triples_checkbox is not None:
            self.triples_checkbox.value = cfg.layers.triples
        if self.cross_doc_checkbox is not None:
            self.cross_doc_checkbox.value = cfg.layers.cross_doc
        if self.cross_doc_xrefs_checkbox is not None:
            self.cross_doc_xrefs_checkbox.value = cfg.layers.cross_doc_xrefs
        if self.xrefs_radio is not None:
            self.xrefs_radio.value = cfg.layers.xrefs
        # Chunks (L5) per-corpus fields.
        if self.chunker_strategy_dropdown is not None:
            self.chunker_strategy_dropdown.value = cfg.chunker_strategy
        if self.chunk_max_tokens_field is not None:
            self.chunk_max_tokens_field.value = str(cfg.chunk_max_tokens)
        if self.merge_peers_checkbox is not None:
            self.merge_peers_checkbox.value = cfg.merge_peers
        if self.enable_pdf_ocr_checkbox is not None:
            self.enable_pdf_ocr_checkbox.value = cfg.enable_pdf_ocr
        if self.enable_image_ocr_checkbox is not None:
            self.enable_image_ocr_checkbox.value = cfg.enable_image_ocr
        if self.images_scale_field is not None:
            self.images_scale_field.value = str(cfg.images_scale)
        if self.extract_figures_checkbox is not None:
            self.extract_figures_checkbox.value = cfg.extract_figures
        if self.embed_images_checkbox is not None:
            self.embed_images_checkbox.value = cfg.embed_images
        if self.min_figure_bytes_field is not None:
            self.min_figure_bytes_field.value = str(cfg.min_figure_bytes)
        if self.optimize_indexes_checkbox is not None:
            self.optimize_indexes_checkbox.value = cfg.optimize_indexes_per_ingest
        # Entities (L6) per-corpus fields.
        if self.entity_extractor_model_field is not None:
            self.entity_extractor_model_field.value = cfg.entity_extractor_model
        if self.entity_extractor_temperature_slider is not None:
            self.entity_extractor_temperature_slider.value = cfg.entity_extractor_temperature
        if self.entity_extractor_temperature_readout is not None:
            self.entity_extractor_temperature_readout.value = _fmt_float(
                cfg.entity_extractor_temperature,
            )
        # Triples (L8) per-corpus fields.
        if self.triples_extractor_model_field is not None:
            self.triples_extractor_model_field.value = cfg.triples_extractor_model
        if self.triples_extractor_temperature_slider is not None:
            self.triples_extractor_temperature_slider.value = cfg.triples_extractor_temperature
        if self.triples_extractor_temperature_readout is not None:
            self.triples_extractor_temperature_readout.value = _fmt_float(
                cfg.triples_extractor_temperature,
            )
        # Loaded models may be sampling-free — grey their temp sliders.
        self._sync_extractor_temp_enabled()
        # Entity extraction — ordered multi-extractor selection.
        self._selected_extractors = (
            list(cfg.entities.extractors) if cfg.entities is not None else ["llm"]
        )
        if self.entity_types_field is not None:
            self.entity_types_field.value = (
                ", ".join(cfg.entities.entity_types) if cfg.entities is not None else ""
            )
        if self.entity_types_mode_radio is not None:
            self.entity_types_mode_radio.value = (
                cfg.entities.entity_types_mode if cfg.entities is not None else "replace"
            )
        # Cross-doc thresholds.
        if self.cross_doc_threshold_field is not None:
            self.cross_doc_threshold_field.value = str(
                cfg.cross_doc.threshold if cfg.cross_doc is not None else 2
            )
        if self.cross_doc_xrefs_threshold_field is not None:
            self.cross_doc_xrefs_threshold_field.value = str(
                cfg.cross_doc_xrefs.threshold if cfg.cross_doc_xrefs is not None else 2
            )
        self._rebuild_extractor_widget()
        self._refresh_subtitles()
        self._refresh_availability()
        self._refresh_dirty_indicator()

    def _refresh_subtitles(self) -> None:
        """Update every section's collapsed-state subtitle.

        Sections owning persistent config (openalex / chunks / entities
        / ontology / triples / cross_doc / cross_doc_xrefs) show Current
        / New chips comparing `_baseline_config` to `_corpus_config`.
        The Labels section shows the current dropdown selection — no
        baseline to compare against.
        """
        self._refresh_labels_subtitle()
        cfg = self._corpus_config
        base = self._baseline_config
        if cfg is None or base is None:
            return

        # openalex_papers.
        if self.openalex_subtitle is not None:
            self.openalex_subtitle.value = _cur_new_bool(
                base.layers.openalex_papers,
                cfg.layers.openalex_papers,
            )
        # Chunks: layer flag + chunker choice + a "+N knob changes" hint
        # so users notice pending edits without clicking through. Only
        # the layer bool and chunker_strategy are surfaced; token / OCR
        # / merge / scale / optimize knob deltas are counted.
        if self.chunks_subtitle is not None:
            base_str = _fmt_bool(base.layers.chunks) + f" · {base.chunker_strategy}"
            cur_str = _fmt_bool(cfg.layers.chunks) + f" · {cfg.chunker_strategy}"
            knob_deltas = 0
            for name in (
                "chunk_max_tokens",
                "merge_peers",
                "enable_pdf_ocr",
                "enable_image_ocr",
                "images_scale",
                "min_figure_bytes",
                "optimize_indexes_per_ingest",
            ):
                if getattr(base, name) != getattr(cfg, name):
                    knob_deltas += 1
            if knob_deltas:
                cur_str += f" · +{knob_deltas} knob change"
                if knob_deltas > 1:
                    cur_str += "s"
            self.chunks_subtitle.value = _cur_new(base_str, cur_str)
        # Entities: on/off + extractor + "+N knob change(s)" hint when
        # LLM model / temperature are pending changes.
        if self.entities_subtitle is not None:
            base_extractor = (
                " → ".join(base.entities.extractors) if base.entities is not None else "not set"
            )
            cur_extractor = (
                " → ".join(cfg.entities.extractors) if cfg.entities is not None else "not set"
            )
            base_str = _fmt_bool(base.layers.entities) + f" · {base_extractor}"
            cur_str = _fmt_bool(cfg.layers.entities) + f" · {cur_extractor}"
            knob_deltas = 0
            for name in (
                "entity_extractor_model",
                "entity_extractor_temperature",
            ):
                if getattr(base, name) != getattr(cfg, name):
                    knob_deltas += 1
            if knob_deltas:
                cur_str += f" · +{knob_deltas} knob change"
                if knob_deltas > 1:
                    cur_str += "s"
            self.entities_subtitle.value = _cur_new(base_str, cur_str)
        # Ontology linking.
        if self.ontology_subtitle is not None:
            base_onts = [
                display
                for key, display in _ONTOLOGY_DISPLAY.items()
                if getattr(base.layers, f"ontology_{key}", False)
            ]
            cur_onts = [
                display
                for key, display in _ONTOLOGY_DISPLAY.items()
                if getattr(cfg.layers, f"ontology_{key}", False)
            ]
            # Count non-default (fuzzy) matching settings for a "+N fuzzy"
            # hint in the subtitle so it's obvious when defaults were
            # overridden.
            n_fuzzy = sum(1 for oc in cfg.ontology.values() if oc.matching == "fuzzy")
            base_str = (
                ", ".join(base_onts) if base_onts else "none"
            ) + f" · xrefs={base.layers.xrefs}"
            cur_str = (", ".join(cur_onts) if cur_onts else "none") + f" · xrefs={cfg.layers.xrefs}"
            if n_fuzzy:
                cur_str += f" · {n_fuzzy} fuzzy"
            self.ontology_subtitle.value = _cur_new(base_str, cur_str)
        # Triples: layer flag + "+N knob change(s)" hint for model / temp.
        if self.triples_subtitle is not None:
            base_str = _fmt_bool(base.layers.triples)
            cur_str = _fmt_bool(cfg.layers.triples)
            knob_deltas = 0
            for name in (
                "triples_extractor_model",
                "triples_extractor_temperature",
            ):
                if getattr(base, name) != getattr(cfg, name):
                    knob_deltas += 1
            if knob_deltas:
                cur_str += f" · +{knob_deltas} knob change"
                if knob_deltas > 1:
                    cur_str += "s"
            self.triples_subtitle.value = _cur_new(base_str, cur_str)
        # Cross-doc.
        if self.cross_doc_subtitle is not None:
            base_thr = base.cross_doc.threshold if base.cross_doc is not None else 2
            cur_thr = cfg.cross_doc.threshold if cfg.cross_doc is not None else 2
            base_str = _fmt_bool(base.layers.cross_doc) + f" · threshold={base_thr}"
            cur_str = _fmt_bool(cfg.layers.cross_doc) + f" · threshold={cur_thr}"
            self.cross_doc_subtitle.value = _cur_new(base_str, cur_str)
        # Cross-doc xrefs.
        if self.cross_doc_xrefs_subtitle is not None:
            base_thr = base.cross_doc_xrefs.threshold if base.cross_doc_xrefs is not None else 2
            cur_thr = cfg.cross_doc_xrefs.threshold if cfg.cross_doc_xrefs is not None else 2
            base_str = (
                _fmt_bool(
                    base.layers.cross_doc_xrefs,
                )
                + f" · threshold={base_thr}"
            )
            cur_str = (
                _fmt_bool(
                    cfg.layers.cross_doc_xrefs,
                )
                + f" · threshold={cur_thr}"
            )
            self.cross_doc_xrefs_subtitle.value = _cur_new(base_str, cur_str)

    def _refresh_labels_subtitle(self) -> None:
        """Update the Labels section subtitle from the dropdown values.
        Independent of _corpus_config — Labels are per-run args."""
        if self.labels_subtitle is None:
            return
        main, sub, overwrite = self._read_ingest_controls()
        sub_str = sub if sub is not None else "(none)"
        overwrite_str = "overwrite existing labels" if overwrite else "preserve existing labels"
        self.labels_subtitle.value = f"Main: {main} · Sub: {sub_str} · {overwrite_str}"

    def _read_global(self, field_name: str) -> str:
        """Read the current value of a global Settings field, or fall
        back to the Pydantic field default if `get_settings()` isn't
        available (e.g. missing API key stops construction).

        Displayed read-only in the config editor so users can see the
        globals that affect a layer without editing them here.
        """
        try:
            settings = get_settings()
            value = getattr(settings, field_name, None)
        except Exception:
            value = None
        if value is None:
            model_field = Settings.model_fields.get(field_name)
            if model_field is not None:
                value = model_field.default
        if value is None:
            return "(not set)"
        return str(value)

    def _find_active_entry(self, name: str):
        for c in self.app.gui_config.corpora:
            if c.name == name:
                return c
        return None

    # ----- save core --------------------------------------------------------

    def try_save_and_get_error(self) -> str | None:
        """Validate current in-memory state + write to disk if valid.

        Called by Ingest tab's action buttons (Ingest folder / Re-ingest /
        Sync / Ingest single file) at kickoff time — the only place
        corpus.toml gets written. Field-handler toggles are in-memory
        only.

        Returns None on success + refreshes baseline / dirty indicator.
        Returns a compact error message on validation or write failure.
        """
        if self._corpus_config is None:
            return "no corpus loaded"
        active_name = self._loaded_for_corpus
        if active_name is None:
            return "no active corpus"
        entry = self._find_active_entry(active_name)
        if entry is None:
            return f"active corpus {active_name!r} not found"
        try:
            validated = CorpusConfig.model_validate(
                self._corpus_config.model_dump(mode="json"),
            )
        except ValidationError as exc:
            return _format_validation_error(exc)
        try:
            _write_corpus_toml(entry.corpus_config_path, validated)
        except Exception as exc:
            logger.warning("corpus.toml write failed: %r", exc)
            return f"could not save corpus.toml: {exc}"
        # Successful write — the validated CorpusConfig (auto-populated
        # subsections, etc.) becomes the new baseline.
        self._corpus_config = validated
        self._baseline_config = validated.model_copy(deep=True)
        # The draft is now the saved baseline — drop the persisted draft so
        # a later reopen doesn't resurrect stale "pending" changes.
        self._clear_draft()
        self._refresh_subtitles()
        self._refresh_availability()
        self._refresh_dirty_indicator()
        self._notify_draft_changed()
        if self.status is not None:
            self.status.value = "saved"
            self.app.page.update()
        return None

    def _mutate_layers(self, **changes) -> LayerFlags:
        """Return a new `LayerFlags` with the given field overrides."""
        assert self._corpus_config is not None
        return self._corpus_config.layers.model_copy(update=changes)

    # ============ Labels section — per-run args (not persisted) =============
    #
    # `main_label` / `sub_label` / `overwrite` don't affect the CorpusConfig
    # baseline; they're consumed by IngestTab's action handlers when the
    # user clicks Ingest / Re-ingest / Sync / Ingest single file.

    def get_ingest_args(self) -> tuple[str, str | None, bool]:
        """Read the current Labels section values.

        Called by `IngestTab` when firing an ingest action. Returns
        `(main_label, sub_label, overwrite)` where sub_label is None
        when the `(none)` sentinel is picked.
        """
        return self._read_ingest_controls()

    def _read_ingest_controls(self) -> tuple[str, str | None, bool]:
        main = (
            self.main_label_dropdown.value
            if self.main_label_dropdown is not None
            else DOCUMENT_LABEL
        )
        raw_sub = (
            self.sub_label_dropdown.value if self.sub_label_dropdown is not None else _SUB_NONE_KEY
        )
        sub = None if raw_sub == _SUB_NONE_KEY else raw_sub
        overwrite = (
            bool(self.overwrite_checkbox.value) if self.overwrite_checkbox is not None else False
        )
        return main, sub, overwrite

    def _refresh_sub_label_options(self) -> None:
        """Rebuild the sub-label dropdown from allowed_types filtered by
        the current main_label selection. Always includes `(none)`."""
        if self.main_label_dropdown is None or self.sub_label_dropdown is None:
            return
        main = self.main_label_dropdown.value or DOCUMENT_LABEL
        cfg = self._corpus_config
        allowed = list(cfg.allowed_types) if cfg is not None else []
        compatible = [sub for sub in allowed if SUB_LABEL_TO_MAIN.get(sub) == main]
        new_options = [
            ft.DropdownOption(key=_SUB_NONE_KEY, text=_SUB_NONE_TEXT),
        ] + [ft.DropdownOption(key=sub, text=sub) for sub in compatible]
        self.sub_label_dropdown.options = new_options
        current = self.sub_label_dropdown.value
        valid_keys = {opt.key for opt in new_options}
        if current not in valid_keys:
            self.sub_label_dropdown.value = _SUB_NONE_KEY

    def _on_main_label_changed(self, e: ft.Event) -> None:
        self._refresh_sub_label_options()
        self._refresh_labels_subtitle()
        self.app.page.update()

    def _on_sub_label_changed(self, e: ft.Event) -> None:
        self._refresh_labels_subtitle()
        self.app.page.update()

    def _on_overwrite_changed(self, e: ft.Event) -> None:
        self._refresh_labels_subtitle()
        self.app.page.update()

    # ============ Field handlers (staging model, no disk writes) ============
    #
    # Toggles mutate `_corpus_config` in memory only. Disk write happens
    # once, at Ingest kickoff, via `try_save_and_get_error`.
    #
    # Handlers that touch fields with cross-field constraints (entities /
    # triples / cross_doc / cross_doc_xrefs / ontology_*) check
    # `_dependency_error_for` before mutating; on violation, the UI
    # control is reverted and a warning dialog is shown.

    # ----- Ontologies handler ---------------------------------------------

    def _on_ontology_toggle(self, key: str) -> None:
        if self._corpus_config is None:
            return
        checkbox = self.ontology_checkboxes.get(key)
        if checkbox is None:
            return
        field_name = f"ontology_{key}"
        current = getattr(self._corpus_config.layers, field_name, False)
        new_value = bool(checkbox.value)
        if new_value == current:
            return
        # Dependency check — turning ON when deps unmet blocks.
        err = self._dependency_error_for(field_name, new_value)
        if err is not None:
            checkbox.value = current
            self._show_dependency_dialog(err)
            self.app.page.update()
            return
        new_layers = self._mutate_layers(**{field_name: new_value})
        self._corpus_config = self._corpus_config.model_copy(
            update={"layers": new_layers},
        )
        self._after_mutation()

    # ----- Layer flag handlers --------------------------------------------

    def _on_layer_toggle(self, field_name: str) -> None:
        if self._corpus_config is None:
            return
        checkbox = {
            "chunks": self.chunks_checkbox,
            "openalex_papers": self.openalex_checkbox,
            "entities": self.entities_checkbox,
            "triples": self.triples_checkbox,
            "cross_doc": self.cross_doc_checkbox,
            "cross_doc_xrefs": self.cross_doc_xrefs_checkbox,
        }.get(field_name)
        if checkbox is None:
            return
        current = getattr(self._corpus_config.layers, field_name, False)
        new_value = bool(checkbox.value)
        if new_value == current:
            return
        err = self._dependency_error_for(field_name, new_value)
        if err is not None:
            checkbox.value = current
            self._show_dependency_dialog(err)
            self.app.page.update()
            return
        new_layers = self._mutate_layers(**{field_name: new_value})
        self._corpus_config = self._corpus_config.model_copy(
            update={"layers": new_layers},
        )
        # Turning the entities layer ON needs a valid [entities] section: the
        # model validator refuses `layers.entities=true` with no section (it
        # would have no extractor to dispatch to). Seed one from the current
        # extractor selection (defaults to ["llm"]) so the config stays valid
        # and the extractor UI shows the default — otherwise the invalid state
        # sits in the draft and only surfaces at the next ingest / bulk-op
        # pre-flight ("layers.entities=true requires an [entities] section").
        if field_name == "entities" and new_value and self._corpus_config.entities is None:
            self._commit_extractors()  # builds EntityConfig + runs _after_mutation
            return
        self._after_mutation()

    def _on_xrefs_changed(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.xrefs_radio is None:
            return
        new_value = self.xrefs_radio.value
        if new_value not in ("none", "collect_only", "use"):
            return
        if new_value == self._corpus_config.layers.xrefs:
            return
        new_layers = self._mutate_layers(xrefs=new_value)
        self._corpus_config = self._corpus_config.model_copy(
            update={"layers": new_layers},
        )
        self._after_mutation()

    # ----- Entity extraction handlers -------------------------------------

    def _rebuild_extractor_widget(self) -> None:
        """Rebuild the per-adapter selection rows from
        `_selected_extractors`, refresh the info banner + group
        visibility. Does NOT call `page.update()` — the caller does
        (via `_after_mutation` or an explicit update)."""
        if self.extractor_rows_column is not None:
            self.extractor_rows_column.controls = self._build_extractor_rows()
        if self.extractor_info_banner is not None:
            # Inform (don't restrict) from the 3rd selected extractor on.
            self.extractor_info_banner.visible = len(self._selected_extractors) >= 3
        self._refresh_extractor_groups()

    def _build_extractor_rows(self) -> list[ft.Control]:
        """One row per adapter: include checkbox + name (+ ready state)
        + priority number + up/down reorder. Selected adapters come
        first in priority order, then the rest in registry order."""
        selected = self._selected_extractors
        ordered = list(selected) + [k for k in EXTRACTOR_REGISTRY if k not in selected]
        rows: list[ft.Control] = []
        for key in ordered:
            display = EXTRACTOR_REGISTRY[key]["display_name"]
            ready = is_extractor_ready(key)
            pos = selected.index(key) if key in selected else None
            checkbox = ft.Checkbox(
                value=pos is not None,
                disabled=not ready,
                on_change=lambda e, k=key: self._on_extractor_toggle(k),
            )
            if not ready:
                name_text = f"{display} — install + download weights in Library → Installs"
            elif pos is None:
                name_text = display
            else:
                name_text = f"{pos + 1}. {display}"
            name = ft.Text(
                name_text,
                size=12,
                color=ft.Colors.WHITE if ready else ft.Colors.GREY_500,
                expand=True,
            )
            up = ft.IconButton(
                icon=ft.Icons.ARROW_UPWARD,
                icon_size=16,
                tooltip="Higher priority (owns overlaps)",
                disabled=(pos is None or pos == 0),
                on_click=lambda e, k=key: self._on_extractor_move(k, -1),
            )
            down = ft.IconButton(
                icon=ft.Icons.ARROW_DOWNWARD,
                icon_size=16,
                tooltip="Lower priority (fills gaps)",
                disabled=(pos is None or pos >= len(selected) - 1),
                on_click=lambda e, k=key: self._on_extractor_move(k, 1),
            )
            rows.append(
                ft.Row(
                    controls=[checkbox, name, up, down],
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                )
            )
        return rows

    def _commit_extractors(self) -> None:
        """Persist `_selected_extractors` (+ preserved entity_types /
        mode) onto the config, then rebuild the widget + refresh derived
        UI."""
        if self._corpus_config is None:
            return
        if not self._selected_extractors:
            # EntityConfig requires >=1; never persist empty.
            self._selected_extractors = ["llm"]
        cur = self._corpus_config.entities
        new_entities = EntityConfig(
            extractors=list(self._selected_extractors),
            entity_types=(cur.entity_types if cur is not None else []),
            entity_types_mode=(cur.entity_types_mode if cur is not None else "replace"),
        )
        self._corpus_config = self._corpus_config.model_copy(
            update={"entities": new_entities},
        )
        self._rebuild_extractor_widget()
        self._after_mutation()

    def _on_extractor_toggle(self, key: str) -> None:
        """Include/exclude an adapter. Adding appends to the end of the
        priority order (lowest priority); removing drops it. At least
        one adapter must stay selected."""
        if self._corpus_config is None:
            return
        if key in self._selected_extractors:
            if len(self._selected_extractors) == 1:
                # Can't deselect the last one — restore the tick + bail.
                self._rebuild_extractor_widget()
                self.app.page.update()
                return
            self._selected_extractors = [k for k in self._selected_extractors if k != key]
        else:
            self._selected_extractors = [*self._selected_extractors, key]
        self._commit_extractors()

    def _on_extractor_move(self, key: str, delta: int) -> None:
        """Reorder an adapter within the priority list (delta -1 = up /
        higher priority, +1 = down / lower)."""
        if key not in self._selected_extractors:
            return
        i = self._selected_extractors.index(key)
        j = i + delta
        if j < 0 or j >= len(self._selected_extractors):
            return
        lst = list(self._selected_extractors)
        lst[i], lst[j] = lst[j], lst[i]
        self._selected_extractors = lst
        self._commit_extractors()

    def _on_entity_types_mode_changed(self, e: ft.Event) -> None:
        """Replace / Add toggle for how a non-empty types list interacts
        with each adapter's DEFAULT_LABELS."""
        if self._corpus_config is None or self.entity_types_mode_radio is None:
            return
        new_mode = self.entity_types_mode_radio.value or "replace"
        cur = self._corpus_config.entities
        if cur is not None and cur.entity_types_mode == new_mode:
            return
        new_entities = EntityConfig(
            extractors=(
                list(cur.extractors) if cur is not None else list(self._selected_extractors)
            ),
            entity_types=(cur.entity_types if cur is not None else []),
            entity_types_mode=new_mode,
        )
        self._corpus_config = self._corpus_config.model_copy(
            update={"entities": new_entities},
        )
        self._after_mutation()

    def _on_entity_types_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.entity_types_field is None:
            return
        raw = self.entity_types_field.value or ""
        types = [t.strip() for t in raw.split(",") if t.strip()]
        current_entities = self._corpus_config.entities
        if current_entities is not None and current_entities.entity_types == types:
            return
        new_entities = EntityConfig(
            extractors=(
                list(current_entities.extractors)
                if current_entities is not None
                else list(self._selected_extractors)
            ),
            entity_types=types,
            entity_types_mode=(
                current_entities.entity_types_mode if current_entities is not None else "replace"
            ),
        )
        self._corpus_config = self._corpus_config.model_copy(
            update={"entities": new_entities},
        )
        self._after_mutation()

    # ----- Cross-doc threshold handlers -----------------------------------

    def _on_cross_doc_threshold_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.cross_doc_threshold_field is None:
            return
        raw = (self.cross_doc_threshold_field.value or "").strip()
        current = (
            self._corpus_config.cross_doc.threshold
            if self._corpus_config.cross_doc is not None
            else 2
        )
        try:
            new_value = max(1, int(raw))
        except ValueError:
            # Non-int input reverts silently — no scary status text.
            self.cross_doc_threshold_field.value = str(current)
            self.app.page.update()
            return
        if new_value == current:
            self.cross_doc_threshold_field.value = str(new_value)
            self.app.page.update()
            return
        new_cd = CrossDocConfig(threshold=new_value)
        self._corpus_config = self._corpus_config.model_copy(
            update={"cross_doc": new_cd},
        )
        self._after_mutation()

    def _on_cross_doc_xrefs_threshold_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.cross_doc_xrefs_threshold_field is None:
            return
        raw = (self.cross_doc_xrefs_threshold_field.value or "").strip()
        current = (
            self._corpus_config.cross_doc_xrefs.threshold
            if self._corpus_config.cross_doc_xrefs is not None
            else 2
        )
        try:
            new_value = max(1, int(raw))
        except ValueError:
            self.cross_doc_xrefs_threshold_field.value = str(current)
            self.app.page.update()
            return
        if new_value == current:
            self.cross_doc_xrefs_threshold_field.value = str(new_value)
            self.app.page.update()
            return
        new_cdx = CrossDocXrefsConfig(threshold=new_value)
        self._corpus_config = self._corpus_config.model_copy(
            update={"cross_doc_xrefs": new_cdx},
        )
        self._after_mutation()

    # ----- Ingest infrastructure block (top-level, read-only) -------------

    def _build_ingest_infrastructure_block(self) -> ft.Control:
        """Collapsible read-only block with process-wide ingest knobs.

        Shows the 3 shared HTTP knobs + 2 concurrency knobs — all
        process-wide, not per-corpus. Displayed so users can find them
        without hunting through Settings. Adjusting them still means
        editing `.env` or global Settings (surfaced here as a note).
        """
        rows: list[ft.Control] = []
        for name in (
            "http_default_timeout",
            "http_max_retries",
            "http_user_agent",
            "pipeline_max_concurrent_chunks",
            "bulk_ops_max_concurrent_docs",
        ):
            rows.append(
                ft.Row(
                    spacing=6,
                    controls=[
                        ft.Text(
                            name,
                            size=12,
                            color=ft.Colors.GREY_400,
                            width=260,
                        ),
                        ft.Text(
                            self._read_global(name),
                            size=12,
                            color=ft.Colors.GREY_300,
                        ),
                    ],
                ),
            )
        return ft.ExpansionTile(
            title=self._section_title(
                "Ingest infrastructure (read-only)",
                "Process-wide (shared across all corpora, not per-corpus). "
                "Edit via `.env` / global Settings; the Ollama daemon probe "
                "uses its own timeout=1.0 override.",
            ),
            controls=[
                ft.Container(
                    padding=ft.Padding.symmetric(
                        vertical=6,
                        horizontal=8,
                    ),
                    content=ft.Column(
                        spacing=3,
                        controls=list(rows),
                    ),
                ),
            ],
        )

    # ----- Ontology (L7) row builder + matching handler -------------------

    def _build_ontology_rows(self) -> ft.Control:
        """Two-column layout: 18 ontology rows, each `[checkbox] [matching ▾]`.

        `wrap=True` keeps the layout readable on narrow windows —
        rows flow to a second column when needed.
        """
        rows: list[ft.Control] = []
        for key in _ONTOLOGY_DISPLAY:
            checkbox = self.ontology_checkboxes[key]
            dropdown = self.ontology_matching_dropdowns[key]
            rows.append(
                ft.Row(
                    controls=[
                        ft.Container(width=190, content=checkbox),
                        dropdown,
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        # Two-column wrap so 18 rows don't stretch the section too tall.
        return ft.Row(
            controls=rows,
            wrap=True,
            spacing=16,
            run_spacing=4,
        )

    def _on_ontology_matching_changed(self, key: str) -> None:
        """Update `OntologyConfig.matching` for one ontology.

        The matching setting only takes effect when the ontology's
        layer is enabled (validator auto-populates OntologyConfig then).
        Storing the choice regardless so it's remembered if the user
        toggles the layer off and on again.
        """
        if self._corpus_config is None:
            return
        dropdown = self.ontology_matching_dropdowns.get(key)
        if dropdown is None:
            return
        new_value = dropdown.value
        if new_value not in ("exact", "fuzzy"):
            return
        current_ont_cfg = self._corpus_config.ontology.get(key)
        current_matching = current_ont_cfg.matching if current_ont_cfg is not None else "exact"
        if new_value == current_matching:
            return
        new_ontology = dict(self._corpus_config.ontology)
        new_ontology[key] = OntologyConfig(matching=new_value)
        self._corpus_config = self._corpus_config.model_copy(
            update={"ontology": new_ontology},
        )
        self._after_mutation()

    # ----- Entities (L6) per-extractor group refresh + handlers -----------
    # The 3 group Containers themselves are built in _create_controls so
    # that `_refresh_extractor_groups()` toggles `.visible` on stable
    # instances (not re-created every build()).

    def _refresh_extractor_groups(self) -> None:
        """Show each per-adapter settings group by SET MEMBERSHIP (a
        group is visible if its adapter is anywhere in the selected
        set), and reshape the shared `entity_types` field + Replace/Add
        toggle for the multi-extractor union.

        Called from `_rebuild_extractor_widget` (initial paint + every
        selection change). Enables `entity_types` whenever any adapter
        that CONSUMES it is selected (i.e. anything other than a
        HunFlair2-only selection). The Replace/Add toggle only matters
        for adapters with defaults (the GLiNERs), so it's shown when a
        GLiNER is selected.
        """
        selected = self._selected_extractors
        llm_on = "llm" in selected
        gliner_on = any(k in ("gliner", "gliner_biomed") for k in selected)
        hunflair2_on = "hunflair2" in selected
        # Does any selected adapter actually consume entity_types?
        # HunFlair2 ignores it, so a HunFlair2-only selection disables
        # the field.
        types_consumed = llm_on or gliner_on

        if self.entities_llm_group is not None:
            self.entities_llm_group.visible = llm_on
        if self.entities_gliner_group is not None:
            self.entities_gliner_group.visible = gliner_on
        if self.entities_hunflair2_group is not None:
            self.entities_hunflair2_group.visible = hunflair2_on

        if self.entity_types_field is not None:
            # HunFlair2-only selection ignores entity_types: grey the field out.
            # The "entity_types has no effect here" note in the HunFlair2 group
            # (visible whenever HunFlair2 is selected) explains why, so the
            # field no longer restates it via a dynamic label.
            self.entity_types_field.disabled = not types_consumed
        # Replace/Add only affects adapters with DEFAULT_LABELS (GLiNERs).
        if self.entity_types_mode_radio is not None:
            self.entity_types_mode_radio.visible = gliner_on

    def _sync_extractor_temp_enabled(self) -> None:
        """Grey out each extractor's temperature slider when its selected
        model doesn't accept a temperature (e.g. Opus 4.8). The backend
        omits temperature for those models regardless — this just makes the
        no-op visible. Provider is the active global LLM provider."""
        provider = self.app.gui_config.llm_provider
        cfg = self._corpus_config
        pairs = (
            (
                cfg.entity_extractor_model if cfg is not None else "",
                self.entity_extractor_temperature_slider,
                self.entity_extractor_temperature_readout,
            ),
            (
                cfg.triples_extractor_model if cfg is not None else "",
                self.triples_extractor_temperature_slider,
                self.triples_extractor_temperature_readout,
            ),
        )
        for model, slider, readout in pairs:
            if slider is None:
                continue
            takes_temp = supports_temperature(provider, model)
            slider.disabled = not takes_temp
            slider.tooltip = None if takes_temp else f"{model} ignores temperature"
            if readout is not None:
                readout.color = ft.Colors.WHITE if takes_temp else ft.Colors.WHITE_38

    def _on_entity_extractor_model_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.entity_extractor_model_field is None:
            return
        raw = (self.entity_extractor_model_field.value or "").strip()
        current = self._corpus_config.entity_extractor_model
        if not raw:
            self.entity_extractor_model_field.value = current
            self.app.page.update()
            return
        if raw == current:
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"entity_extractor_model": raw},
        )
        self._sync_extractor_temp_enabled()
        self._after_mutation()

    def _on_entity_extractor_temperature_slide(self, e: ft.Event) -> None:
        """Live-update the readout as the user drags the slider (no commit)."""
        if (
            self.entity_extractor_temperature_slider is None
            or self.entity_extractor_temperature_readout is None
        ):
            return
        self.entity_extractor_temperature_readout.value = _fmt_float(
            self.entity_extractor_temperature_slider.value,
        )
        self.app.page.update()

    def _on_entity_extractor_temperature_committed(self, e: ft.Event) -> None:
        """Commit the slider's final value on release."""
        if self._corpus_config is None or self.entity_extractor_temperature_slider is None:
            return
        new_value = float(self.entity_extractor_temperature_slider.value)
        current = self._corpus_config.entity_extractor_temperature
        if abs(new_value - current) < 1e-6:
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"entity_extractor_temperature": new_value},
        )
        self._after_mutation()

    def _on_triples_extractor_model_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.triples_extractor_model_field is None:
            return
        raw = (self.triples_extractor_model_field.value or "").strip()
        current = self._corpus_config.triples_extractor_model
        if not raw:
            self.triples_extractor_model_field.value = current
            self.app.page.update()
            return
        if raw == current:
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"triples_extractor_model": raw},
        )
        self._sync_extractor_temp_enabled()
        self._after_mutation()

    def _on_triples_extractor_temperature_slide(self, e: ft.Event) -> None:
        """Live-update the readout as the user drags the slider (no commit)."""
        if (
            self.triples_extractor_temperature_slider is None
            or self.triples_extractor_temperature_readout is None
        ):
            return
        self.triples_extractor_temperature_readout.value = _fmt_float(
            self.triples_extractor_temperature_slider.value,
        )
        self.app.page.update()

    def _on_triples_extractor_temperature_committed(self, e: ft.Event) -> None:
        """Commit the slider's final value on release."""
        if self._corpus_config is None or self.triples_extractor_temperature_slider is None:
            return
        new_value = float(self.triples_extractor_temperature_slider.value)
        current = self._corpus_config.triples_extractor_temperature
        if abs(new_value - current) < 1e-6:
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"triples_extractor_temperature": new_value},
        )
        self._after_mutation()

    # ----- Chunks (L5) per-corpus field handlers --------------------------

    def _on_chunker_strategy_changed(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.chunker_strategy_dropdown is None:
            return
        new_value = self.chunker_strategy_dropdown.value
        if new_value not in ("hybrid", "hierarchical"):
            return
        if new_value == self._corpus_config.chunker_strategy:
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"chunker_strategy": new_value},
        )
        self._after_mutation()

    def _on_chunk_max_tokens_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.chunk_max_tokens_field is None:
            return
        raw = (self.chunk_max_tokens_field.value or "").strip()
        current = self._corpus_config.chunk_max_tokens
        try:
            new_value = max(64, int(raw))
        except ValueError:
            self.chunk_max_tokens_field.value = str(current)
            self.app.page.update()
            return
        if new_value == current:
            self.chunk_max_tokens_field.value = str(new_value)
            self.app.page.update()
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"chunk_max_tokens": new_value},
        )
        self.chunk_max_tokens_field.value = str(new_value)
        self._after_mutation()

    def _on_min_figure_bytes_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.min_figure_bytes_field is None:
            return
        raw = (self.min_figure_bytes_field.value or "").strip()
        current = self._corpus_config.min_figure_bytes
        try:
            new_value = int(raw)
            if new_value < 0:
                raise ValueError("must be >= 0")
        except ValueError:
            self.min_figure_bytes_field.value = str(current)
            self.app.page.update()
            return
        if new_value == current:
            self.min_figure_bytes_field.value = str(new_value)
            self.app.page.update()
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"min_figure_bytes": new_value},
        )
        self.min_figure_bytes_field.value = str(new_value)
        self._after_mutation()

    def _on_images_scale_blur(self, e: ft.Event) -> None:
        if self._corpus_config is None or self.images_scale_field is None:
            return
        raw = (self.images_scale_field.value or "").strip()
        current = self._corpus_config.images_scale
        try:
            new_value = float(raw)
            if new_value <= 0:
                raise ValueError("must be positive")
        except ValueError:
            self.images_scale_field.value = str(current)
            self.app.page.update()
            return
        if new_value == current:
            self.images_scale_field.value = str(new_value)
            self.app.page.update()
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={"images_scale": new_value},
        )
        self.images_scale_field.value = str(new_value)
        self._after_mutation()

    def _on_chunks_bool_changed(self, field_name: str) -> None:
        """Shared handler for the 6 bool checkboxes in the Chunks
        section: merge_peers, enable_pdf_ocr, enable_image_ocr,
        extract_figures, embed_images, optimize_indexes_per_ingest."""
        if self._corpus_config is None:
            return
        checkbox = {
            "merge_peers": self.merge_peers_checkbox,
            "enable_pdf_ocr": self.enable_pdf_ocr_checkbox,
            "enable_image_ocr": self.enable_image_ocr_checkbox,
            "extract_figures": self.extract_figures_checkbox,
            "embed_images": self.embed_images_checkbox,
            "optimize_indexes_per_ingest": self.optimize_indexes_checkbox,
        }.get(field_name)
        if checkbox is None:
            return
        new_value = bool(checkbox.value)
        if new_value == getattr(self._corpus_config, field_name):
            return
        self._corpus_config = self._corpus_config.model_copy(
            update={field_name: new_value},
        )
        self._after_mutation()

    # ============ Cross-cutting helpers (staging, deps, dirty, discard) =====

    def _after_mutation(self) -> None:
        """Run after every in-memory mutation — refresh derived UI and
        mirror the draft to the session sidecar so it survives a restart."""
        self._refresh_subtitles()
        self._refresh_availability()
        self._refresh_dirty_indicator()
        self._persist_draft()
        self.app.page.update()

    # ----- draft persistence (session sidecar) ------------------------------

    def _active_corpus_toml_path(self):
        """The loaded corpus's `corpus.toml` path, or None when unresolved.

        Used to locate the session sidecar (which lives beside it)."""
        name = self._loaded_for_corpus
        if name is None:
            return None
        entry = self._find_active_entry(name)
        return entry.corpus_config_path if entry is not None else None

    def _persist_draft(self) -> None:
        """Mirror the current draft to the sidecar — or clear it when the
        draft equals the saved baseline — then notify any listener."""
        path = self._active_corpus_toml_path()
        if path is None:
            return
        if self.is_dirty() and self._corpus_config is not None:
            update_draft(path, self._corpus_config.model_dump(mode="json"))
        else:
            update_draft(path, None)
        self._notify_draft_changed()

    def _clear_draft(self) -> None:
        """Drop the persisted draft (the draft now equals the baseline)."""
        path = self._active_corpus_toml_path()
        if path is not None:
            update_draft(path, None)

    def _restore_draft(self, corpus_toml_path) -> None:
        """Override `_corpus_config` with the persisted unsaved draft, if any.

        Called right after the baseline is loaded. A draft that no longer
        validates (e.g. after a schema upgrade) is dropped and its sidecar
        entry cleared, falling back to the baseline.
        """
        session = load_session(corpus_toml_path)
        if session.draft_config is None:
            return
        try:
            self._corpus_config = CorpusConfig.model_validate(session.draft_config)
        except ValidationError:
            logger.info("discarding incompatible persisted draft at %s", corpus_toml_path)
            update_draft(corpus_toml_path, None)

    def _notify_draft_changed(self) -> None:
        """Tell the wired listener (Select card) the draft moved."""
        if self.on_draft_changed is not None:
            self.on_draft_changed()

    def is_dirty(self) -> bool:
        """True when in-memory config differs from the on-disk baseline."""
        if self._corpus_config is None or self._baseline_config is None:
            return False
        return self._corpus_config != self._baseline_config

    def _refresh_dirty_indicator(self) -> None:
        dirty = self.is_dirty()
        if self.dirty_indicator is not None:
            self.dirty_indicator.visible = dirty
        if self.discard_button is not None:
            self.discard_button.disabled = not dirty

    def _ontology_downloaded(self, key: str) -> bool:
        """Disk probe: is this ontology's source file downloaded?

        Un-downloaded ontologies can't be enabled — ingest never auto-
        downloads (the file must be fetched via Library → Installs). Local
        import keeps the view's startup light; defaults to True on any probe
        error so a transient failure doesn't wrongly block the whole list
        (the ingest path hard-guards regardless)."""
        try:
            from knowledge_agent.kg.ontology_lifecycle import is_ontology_downloaded

            return bool(is_ontology_downloaded(key))
        except Exception:
            return True

    def _refresh_availability(self) -> None:
        """Grey out controls that can't currently apply. Two kinds:

        1. Cross-field *dependency* greying on the layer TOGGLE checkboxes
           (e.g. entities requires chunks) — uses `label_style`, NOT
           `disabled=True`, so the click still fires `on_change` and we can
           show the warning dialog + revert.
        2. Layer-off greying of a section's SUB-controls when its own toggle
           is off — uses `disabled=True` (no revert needed), so it's clear
           those settings won't apply. The toggle itself stays enabled, so
           the layer can be turned back on. `entity_types_field` / its mode
           radio are intentionally left to `_refresh_extractor_groups`,
           which owns their disabled/visible state by extractor membership.
        """
        cfg = self._corpus_config
        if cfg is None:
            return
        chunks_on = cfg.layers.chunks
        entities_on = cfg.layers.entities
        xrefs_use = cfg.layers.xrefs == "use"

        def _grey(cb: ft.Checkbox | None, ok: bool, tip: str) -> None:
            if cb is None:
                return
            cb.label_style = (
                None
                if ok
                else ft.TextStyle(
                    color=ft.Colors.GREY_600,
                )
            )
            cb.tooltip = None if ok else tip

        _grey(
            self.entities_checkbox,
            chunks_on,
            "Requires the chunks layer",
        )
        _grey(
            self.triples_checkbox,
            entities_on,
            "Requires the entities layer",
        )
        _grey(
            self.cross_doc_checkbox,
            entities_on,
            "Requires the entities layer",
        )
        # Ontologies also HARD-require their source file to be downloaded —
        # ingest never auto-downloads, so an un-downloaded ontology can't be
        # enabled here at all (disabled, not just greyed). Download happens
        # only via Library → Installs.
        for key, cb in self.ontology_checkboxes.items():
            if not self._ontology_downloaded(key):
                cb.disabled = True
                cb.label_style = ft.TextStyle(color=ft.Colors.GREY_600)
                cb.tooltip = (
                    f"{_ONTOLOGY_DISPLAY[key]} is not downloaded — download it in "
                    f"Library → Installs to enable it here."
                )
            else:
                cb.disabled = False
                _grey(cb, entities_on, "Requires the entities layer")
        if self.cross_doc_xrefs_checkbox is not None:
            ok = entities_on and xrefs_use
            if not entities_on:
                tip = "Requires the entities layer"
            elif not xrefs_use:
                tip = 'Requires xrefs="use"'
            else:
                tip = ""
            _grey(self.cross_doc_xrefs_checkbox, ok, tip)

        # ---- Layer-off greying of each section's sub-controls ----
        def _set_disabled(controls: list[ft.Control | None], disabled: bool) -> None:
            for control in controls:
                if control is not None:
                    control.disabled = disabled

        _set_disabled(
            [
                self.chunker_strategy_dropdown,
                self.chunk_max_tokens_field,
                self.merge_peers_checkbox,
                self.enable_pdf_ocr_checkbox,
                self.enable_image_ocr_checkbox,
                self.images_scale_field,
                self.extract_figures_checkbox,
                self.embed_images_checkbox,
                self.min_figure_bytes_field,
                self.optimize_indexes_checkbox,
            ],
            not chunks_on,
        )
        _set_disabled(
            [
                self.extractor_rows_column,
                self.entities_llm_group,
                self.entities_gliner_group,
                self.entities_hunflair2_group,
            ],
            not entities_on,
        )
        _set_disabled(
            [self.triples_extractor_model_field, self.triples_extractor_temperature_slider],
            not cfg.layers.triples,
        )
        _set_disabled([self.cross_doc_threshold_field], not cfg.layers.cross_doc)
        _set_disabled([self.cross_doc_xrefs_threshold_field], not cfg.layers.cross_doc_xrefs)

    def _dependency_error_for(
        self,
        field_name: str,
        new_value: bool,
    ) -> str | None:
        """If turning `field_name` to `new_value` violates a cross-field
        rule, return the message to show. Turning fields OFF never
        violates."""
        if not new_value or self._corpus_config is None:
            return None
        cfg = self._corpus_config
        if field_name == "entities":
            if not cfg.layers.chunks:
                return "The entities layer requires the chunks layer. Turn chunks on first."
        elif field_name == "triples":
            if not cfg.layers.entities:
                return "The triples layer requires the entities layer. Turn entities on first."
        elif field_name == "cross_doc":
            if not cfg.layers.entities:
                return "cross_doc requires the entities layer. Turn entities on first."
        elif field_name == "cross_doc_xrefs":
            if not cfg.layers.entities:
                return "cross_doc_xrefs requires the entities layer. Turn entities on first."
            if cfg.layers.xrefs != "use":
                return "cross_doc_xrefs requires xrefs = \"use\". Set xrefs to 'use' first."
        elif field_name.startswith("ontology_") and not cfg.layers.entities:
            return "Ontology layers require the entities layer. Turn entities on first."
        return None

    def _show_dependency_dialog(self, message: str) -> None:
        """Modal warning explaining a dependency violation. One [OK]
        button — the reverted control state is already restored by
        the caller."""

        def _close(_ev):
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Requires another setting"),
            content=ft.Text(message, size=12),
            actions=[ft.TextButton("OK", on_click=_close)],
        )
        self.app.page.show_dialog(dialog)

    def _on_discard_clicked(self, e: ft.Event) -> None:
        """Revert in-memory state to the on-disk baseline + drop the draft."""
        if self._baseline_config is None or self._loaded_for_corpus is None:
            return
        self._corpus_config = self._baseline_config.model_copy(deep=True)
        self._populate_controls()
        self._clear_draft()
        self._refresh_dirty_indicator()
        self._notify_draft_changed()
        self.app.page.update()


def _fmt_float(value: float | None) -> str:
    """Format a float for the temperature-slider readout. Mirrors the
    `_fmt_float` helper in `gui.settings.llm_tab` (same "0.00" style)."""
    if value is None:
        return "0.00"
    return f"{float(value):.2f}"


def _fmt_bool(v: bool) -> str:
    return "on" if v else "off"


def _format_allowed_types(allowed: list[str]) -> str:
    """Format the corpus-wide allowed_types list for the Labels (i) dialog —
    the same "X, Y, Z (N items)" style that used to show inline."""
    if allowed:
        return f"{', '.join(allowed)} ({len(allowed)} items)"
    return "(empty)"


def _globals_block(
    pairs: list[tuple[str, str]],
    *,
    note: str | None = None,
) -> ft.Control:
    """A read-only sub-section — thin rule + sub-section title + a compact
    `field: value` block — for global settings that affect a layer. Editing
    them belongs elsewhere (Settings tab / .env / env vars); shown here so
    users know what defaults are in effect. Wrapped with vertical padding so
    the block gets breathing room top + bottom (not cramped against the
    control above or the section divider below)."""
    value_rows: list[ft.Control] = []
    for name, value in pairs:
        value_rows.append(
            ft.Row(
                spacing=6,
                controls=[
                    ft.Text(
                        name,
                        size=12,
                        color=ft.Colors.GREY_400,
                        width=220,
                    ),
                    ft.Text(value, size=12, color=ft.Colors.GREY_300),
                ],
            ),
        )
    if note is not None:
        value_rows.append(
            ft.Text(
                note,
                size=12,
                color=ft.Colors.GREY_500,
                italic=True,
            ),
        )
    return ft.Container(
        padding=ft.Padding.only(bottom=12),
        content=ft.Column(
            spacing=6,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            controls=[
                sub_section_header("Global (read-only)"),
                ft.Column(spacing=3, controls=value_rows),
            ],
        ),
    )


def _cur_new(baseline: str, current: str) -> str:
    """Format a Current/New chip line: `Current: X → New: Y` (or just
    `Current: X` when nothing pending). Plain-text — colour treatment
    could be added later via a Row of Text controls if desired."""
    if baseline == current:
        return f"Current: {baseline}"
    return f"Current: {baseline}  →  New: {current}"


def _cur_new_bool(baseline: bool, current: bool) -> str:
    """Shorthand for a bool-only field."""
    return _cur_new(_fmt_bool(baseline), _fmt_bool(current))


def _grid_of(items: list[ft.Control]) -> ft.Control:
    """Wrap a list of controls in a wrapping Row so long lists (18
    ontology checkboxes) don't force horizontal scroll on narrow
    windows."""
    return ft.Row(
        controls=items,
        wrap=True,
        spacing=12,
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Pull a compact single-line message out of a pydantic
    ValidationError (trimmed of the '[type=...]' suffix pydantic
    appends). Falls back to `str(exc)` if the error list is empty."""
    errs = exc.errors()
    if errs:
        msg = errs[0].get("msg", str(exc))
        return msg.split("[type=")[0].strip()
    return str(exc)
