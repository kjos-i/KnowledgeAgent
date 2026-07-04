"""Library → Select sub-tab — pick a corpus, view its status + documents.

Internal 2-column layout (fixed proportions, no dragger):

  Left column  (~320 px): dataset picker + Rename / Remove / Refresh.
  Right column (expand):  info card (name, URIs, paths, counts,
                          active ontologies + extractor + layer flags)
                          on top; Documents table below.

Config editing (allowed_types, ontologies, layer flags, extractor,
thresholds) lives under the sibling **Ingest** sub-tab — Select is a
read-only "browse this corpus" view.

Switch handler flow (unchanged from Phase 3):

  1. Find the target `CorpusEntry` in `GuiConfig.corpora`.
  2. Mirror its URI / user / lancedb_path / corpus_config_path to
     the top-level `GuiConfig` fields (so `apply_connection_to_env`
     keeps working).
  3. Set `active_corpus_name` = new name.
  4. Push per-corpus Neo4j password from keyring → `NEO4J_PASSWORD`.
  5. `save_config` + `apply_connection_to_env` + `reset_after_key_change`.
  6. Repopulate the info card.

Rename: updates `CorpusEntry.name`; migrates keyring entry from
`neo4j-{old}` → `neo4j-{new}`; refreshes info card if the renamed
corpus is active.

Remove: unregisters the `CorpusEntry` + deletes its keyring entry.
Does NOT delete the actual Neo4j DBMS / LanceDB folder / corpus.toml
— those are external artefacts the user manages themselves.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter
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
    apply_connection_to_env,
    get_corpus_password,
    save_config,
    set_corpus_password,
)
from knowledge_agent.gui.library.documents_view import DocumentsView
from knowledge_agent.gui.views._frame import view_header
from knowledge_agent.kg.corpus_config import CorpusConfig, load_corpus_config
from knowledge_agent.search.client import get_search_client

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.config_store import CorpusEntry


logger = logging.getLogger(__name__)


_LEFT_COLUMN_WIDTH = 420


# Layer coverage badges — key = LayerFlags field, label = short badge.
# Displayed in a Row on the info card; grey when disabled, green when on.
_LAYER_BADGES: tuple[tuple[str, str], ...] = (
    ("chunks", "L4"),
    ("openalex_papers", "OA"),
    ("entities", "L6"),
    ("triples", "L8"),
    ("cross_doc", "L9"),
    ("cross_doc_xrefs", "L10"),
)


# Same ontology keys as CorpusConfigEditor's `_ONTOLOGY_DISPLAY` —
# kept local to avoid a cross-import between sibling files.
_ONTOLOGY_KEYS: tuple[tuple[str, str], ...] = (
    ("mesh", "MeSH"), ("go", "GO"), ("hpo", "HPO"), ("uberon", "UBERON"),
    ("mondo", "MONDO"), ("chebi", "ChEBI"), ("eco", "ECO"), ("so", "SO"),
    ("pr", "PR"), ("cl", "CL"), ("po", "PO"), ("foodon", "FOODON"),
    ("envo", "ENVO"), ("ncbitaxon", "NCBITaxon"), ("obi", "OBI"),
    ("efo", "EFO"), ("dron", "DRON"), ("fibo", "FIBO"),
)


class SelectDatasetTab:
    """Corpus picker (left) + read-only info card + docs table (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        self.status: ft.Text | None = None
        self.picker_container: ft.Container | None = None
        self.picker_radio: ft.RadioGroup | None = None
        self.rename_button: ft.Button | None = None
        self.remove_button: ft.Button | None = None
        self.refresh_button: ft.Button | None = None

        # Right column — read-only info card. Every field is a Text
        # control we repopulate on switch / refresh; the container
        # wraps them so we can rebuild wholesale on empty-state.
        self.info_container: ft.Container | None = None
        self.info_name: ft.Text | None = None
        self.info_uri: ft.Text | None = None
        self.info_user: ft.Text | None = None
        self.info_lancedb: ft.Text | None = None
        self.info_toml: ft.Text | None = None
        self.info_counts: ft.Text | None = None
        self.info_layers: ft.Row | None = None
        self.info_ontologies: ft.Text | None = None
        self.info_extractor: ft.Text | None = None
        self.info_xrefs: ft.Text | None = None
        self.info_breakdown: ft.Text | None = None
        self.info_chunking: ft.Text | None = None
        self.info_ocr: ft.Text | None = None
        self.info_figures: ft.Text | None = None
        self.info_embedder: ft.Text | None = None
        self.info_llm: ft.Text | None = None
        # Strong refs to in-flight count-fetch tasks (they self-discard).
        self._bg_tasks: set[asyncio.Task] = set()

        self.documents = DocumentsView(app)
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_400)

        self.picker_container = ft.Container(
            content=ft.Text(
                "(building picker…)",
                size=12, color=ft.Colors.GREY_500, italic=True,
            ),
        )

        self.rename_button = ft.Button(
            content=centered_label("Rename"),
            on_click=self.on_rename_clicked,
        )
        self.remove_button = ft.Button(
            content=centered_label("Remove"),
            on_click=self.on_remove_clicked,
        )
        self.refresh_button = ft.Button(
            content=centered_label("Refresh"),
            on_click=self.on_refresh_clicked,
        )

        # ---- info card controls ----
        self.info_name = ft.Text(
            "", size=15, weight=ft.FontWeight.BOLD,
        )
        self.info_uri = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_user = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_lancedb = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_toml = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_counts = ft.Text(
            "docs: — · chunks: — · mentions: —",
            size=12, color=ft.Colors.GREY_400,
        )
        self.info_layers = ft.Row(spacing=6, wrap=True, controls=[])
        self.info_ontologies = ft.Text(
            "", size=12, color=ft.Colors.GREY_300,
        )
        self.info_extractor = ft.Text(
            "", size=12, color=ft.Colors.GREY_300,
        )
        self.info_xrefs = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_breakdown = ft.Text("", size=12, color=ft.Colors.GREY_400)
        self.info_chunking = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_ocr = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_figures = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_embedder = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_llm = ft.Text("", size=12, color=ft.Colors.GREY_300)

        self.info_container = ft.Container(
            padding=12,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            bgcolor=PANEL_BG,
            border_radius=4,
            content=ft.Column(
                spacing=6,
                controls=[
                    self.info_name,
                    ft.Divider(height=1),
                    _kv_row("Neo4j URI", self.info_uri),
                    _kv_row("Neo4j user", self.info_user),
                    _kv_row("LanceDB path", self.info_lancedb),
                    _kv_row("corpus.toml", self.info_toml),
                    ft.Divider(height=1),
                    _kv_row("Counts", self.info_counts),
                    _kv_row("Status", self.info_breakdown),
                    ft.Row(
                        controls=[
                            ft.Text(
                                "Layers",
                                size=12, color=ft.Colors.GREY_500,
                                width=110,
                            ),
                            self.info_layers,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    _kv_row("Ontologies", self.info_ontologies),
                    _kv_row("Extractor", self.info_extractor),
                    _kv_row("Xrefs", self.info_xrefs),
                    ft.Divider(height=1),
                    _kv_row("Chunking", self.info_chunking),
                    _kv_row("OCR", self.info_ocr),
                    _kv_row("Figures", self.info_figures),
                    ft.Divider(height=1),
                    _kv_row("Embedder", self.info_embedder),
                    _kv_row("LLM", self.info_llm),
                ],
            ),
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # Rebuild the picker + info card every time the sub-tab renders
        # — cheap, avoids stale state after sibling-tab mutations.
        self._sync_picker()
        self._populate_info_card()

        # Left column: picker + Rename/Remove/Refresh + info card + status.
        # Scrollable so the info card + picker don't get clipped on
        # short viewports.
        left_pane = ft.Container(
            width=_LEFT_COLUMN_WIDTH,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    ft.Text(
                        "Switch active corpus:",
                        size=12, color=ft.Colors.GREY_400,
                    ),
                    self.picker_container,
                    ft.Row(
                        controls=[
                            self.rename_button,
                            self.remove_button,
                            self.refresh_button,
                        ],
                        spacing=8,
                    ),
                    self.status,
                    ft.Divider(),
                    self.info_container,
                ],
            ),
        )

        # Right column: documents area (scrollable).
        right_pane = ft.Container(
            expand=True,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=10,
                controls=[self.documents.build()],
            ),
        )

        body = ft.Row(
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=12,
            controls=[left_pane, right_pane],
        )

        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[view_header("Select Dataset"), body],
        )

    # ----- picker rebuild --------------------------------------------------

    def _sync_picker(self) -> None:
        """Rebuild the picker from `GuiConfig.corpora`.

        No corpora → amber hint pointing to Create New.
        One or more → RadioGroup with the active one selected.
        """
        if self.picker_container is None:
            return
        corpora = self.app.gui_config.corpora
        if not corpora:
            self.picker_radio = None
            self.picker_container.content = ft.Text(
                "No corpora registered yet — open Create New to "
                "register your first corpus.",
                size=12, color=ft.Colors.AMBER_300, italic=True,
            )
            self._sync_button_disable_state(has_corpora=False)
            return
        active_name = (
            self.app.gui_config.active_corpus_name
            or corpora[0].name
        )
        self.picker_radio = ft.RadioGroup(
            value=active_name,
            on_change=self.on_active_changed,
            content=ft.Column(
                controls=[
                    ft.Radio(value=c.name, label=self._radio_label(c))
                    for c in corpora
                ],
                spacing=4,
            ),
        )
        self.picker_container.content = self.picker_radio
        self._sync_button_disable_state(has_corpora=True)

    def _radio_label(self, entry: CorpusEntry) -> str:
        return f"{entry.name}   ({entry.neo4j_uri})"

    def _sync_button_disable_state(self, has_corpora: bool) -> None:
        if self.rename_button is not None:
            self.rename_button.disabled = not has_corpora
        if self.remove_button is not None:
            self.remove_button.disabled = not has_corpora

    # ----- info card population -------------------------------------------

    def _populate_info_card(self) -> None:
        """Fill the info card from GuiConfig + the active corpus.toml.

        Called on: initial build, corpus switch, Refresh button, and
        after Rename / Remove.
        """
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None or not self.app.gui_config.corpora:
            self._info_empty_state()
            return
        entry = self._find_corpus(active_name)
        if entry is None:
            self._info_empty_state()
            return
        assert self.info_name is not None
        assert self.info_uri is not None
        assert self.info_user is not None
        assert self.info_lancedb is not None
        assert self.info_toml is not None
        self.info_name.value = entry.name
        self.info_uri.value = entry.neo4j_uri
        self.info_user.value = entry.neo4j_user
        self.info_lancedb.value = str(entry.lancedb_path)
        self.info_toml.value = str(entry.corpus_config_path)

        cfg = self._safe_load_config(entry)
        self._populate_layer_badges(cfg)
        self._populate_ontologies(cfg)
        self._populate_extractor(cfg)
        self._populate_xrefs(cfg)
        self._populate_ingest_settings(cfg)
        self._populate_models()
        self._schedule_counts()

    def _info_empty_state(self) -> None:
        """When there's no active corpus, blank out all the fields."""
        for text in (
            self.info_name, self.info_uri, self.info_user,
            self.info_lancedb, self.info_toml,
            self.info_ontologies, self.info_extractor, self.info_xrefs,
            self.info_breakdown, self.info_chunking, self.info_ocr,
            self.info_figures, self.info_embedder, self.info_llm,
        ):
            if text is not None:
                text.value = "—"
        if self.info_counts is not None:
            self.info_counts.value = "docs: — · chunks: — · mentions: —"
        if self.info_layers is not None:
            self.info_layers.controls = [
                ft.Text("—", size=12, color=ft.Colors.GREY_600),
            ]

    def _safe_load_config(self, entry: CorpusEntry) -> CorpusConfig | None:
        try:
            return load_corpus_config(entry.corpus_config_path)
        except FileNotFoundError:
            logger.info(
                "corpus.toml not found for %r at %s",
                entry.name, entry.corpus_config_path,
            )
            return None
        except Exception as exc:
            logger.warning(
                "load_corpus_config failed for %r: %r", entry.name, exc,
            )
            return None

    def _populate_layer_badges(self, cfg: CorpusConfig | None) -> None:
        if self.info_layers is None:
            return
        badges: list[ft.Control] = []
        for field, label in _LAYER_BADGES:
            on = (
                cfg is not None
                and bool(getattr(cfg.layers, field, False))
            )
            badges.append(_badge(label, on=on))
        self.info_layers.controls = badges

    def _populate_ontologies(self, cfg: CorpusConfig | None) -> None:
        if self.info_ontologies is None:
            return
        if cfg is None:
            self.info_ontologies.value = "—"
            return
        active = [
            display
            for key, display in _ONTOLOGY_KEYS
            if bool(getattr(cfg.layers, f"ontology_{key}", False))
        ]
        self.info_ontologies.value = (
            ", ".join(active) if active else "none enabled"
        )

    def _populate_extractor(self, cfg: CorpusConfig | None) -> None:
        if self.info_extractor is None:
            return
        if cfg is None or cfg.entities is None:
            self.info_extractor.value = "not configured"
            return
        ents = cfg.entities
        extractors = list(getattr(ents, "extractors", None) or [])
        if not extractors:
            # Fall back to the legacy singular field for old configs.
            single = getattr(ents, "extractor", None)
            extractors = [single] if single else []
        parts = [", ".join(extractors) or "—"]
        mode = getattr(ents, "entity_types_mode", None)
        if mode:
            parts.append(f"mode: {mode}")
        types = ents.entity_types
        if types:
            parts.append(f"types: {', '.join(types)}")
        self.info_extractor.value = " · ".join(parts)

    def _populate_xrefs(self, cfg: CorpusConfig | None) -> None:
        if self.info_xrefs is None:
            return
        if cfg is None:
            self.info_xrefs.value = "—"
            return
        self.info_xrefs.value = cfg.layers.xrefs

    def _populate_ingest_settings(self, cfg: CorpusConfig | None) -> None:
        """Summarise the per-corpus chunk / OCR / figure settings from the
        loaded corpus.toml (all sync — no DB read)."""
        if cfg is None:
            for t in (self.info_chunking, self.info_ocr, self.info_figures):
                if t is not None:
                    t.value = "—"
            return

        def g(name: str, default: object = "?") -> object:
            return getattr(cfg, name, default)

        if self.info_chunking is not None:
            self.info_chunking.value = (
                f"{g('chunker_strategy')} · max {g('chunk_max_tokens')} tok "
                f"· merge_peers {_onoff(bool(g('merge_peers', False)))}"
            )
        if self.info_ocr is not None:
            self.info_ocr.value = (
                f"pdf {_onoff(bool(g('enable_pdf_ocr', False)))} · "
                f"image {_onoff(bool(g('enable_image_ocr', False)))}"
            )
        if self.info_figures is not None:
            self.info_figures.value = (
                f"extract {_onoff(bool(g('extract_figures', False)))} · "
                f"scale {g('images_scale')} · min {g('min_figure_bytes')} B"
            )

    def _populate_models(self) -> None:
        """Show the active embedder (provider · model · dims) + the LLM
        provider from global Settings. Read-only; failure → '—'.

        The LLM uses per-role models (synthesizer, cypher_builder, …), so
        only the provider is summarised — role models live in Settings."""
        try:
            from knowledge_agent.config import get_settings
            s = get_settings()
        except Exception as exc:
            logger.info(
                "SelectDataset: get_settings failed (%r); models —", exc
            )
            for t in (self.info_embedder, self.info_llm):
                if t is not None:
                    t.value = "—"
            return
        if self.info_embedder is not None:
            self.info_embedder.value = (
                f"{s.embedding_provider} · {s.embedding_model} · "
                f"{s.embedding_dims}-dim"
            )
        if self.info_llm is not None:
            self.info_llm.value = str(s.llm_provider)

    # ----- async counts (docs/chunks from LanceDB, mentions from Neo4j) ---

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    def _schedule_counts(self) -> None:
        """Kick off the async count fetch when a loop is running (guarded
        so build() stays test-safe with no running loop)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spawn(self._reload_counts())

    async def _reload_counts(self) -> None:
        """Fill the Counts line (docs/chunks from LanceDB, mentions from
        Neo4j) + the metadata-status breakdown. Each source degrades to
        '—' independently, so one failure doesn't blank the rest."""
        if self.info_counts is None:
            return
        docs_str = "—"
        chunks_str = "—"
        breakdown = "—"
        try:
            rows = await get_search_client().list_indexed_docs()
        except Exception as exc:
            logger.warning("SelectDataset: list_indexed_docs failed: %r", exc)
            rows = None
        if rows is not None:
            docs_str = str(len(rows))
            chunks_str = str(sum(int(r.get("n_chunks") or 0) for r in rows))
            counts = Counter(
                (r.get("metadata_status") or "unknown") for r in rows
            )
            breakdown = (
                " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
                or "—"
            )
        mentions_str = "—"
        try:
            from knowledge_agent.kg.client import get_kg_client
            mentions_str = str(await get_kg_client().count_mentions())
        except Exception as exc:
            logger.info(
                "SelectDataset: count_mentions failed (%r); showing —", exc
            )
        self.info_counts.value = (
            f"docs: {docs_str} · chunks: {chunks_str} · "
            f"mentions: {mentions_str}"
        )
        if self.info_breakdown is not None:
            self.info_breakdown.value = breakdown
        self.app.page.update()

    # ----- switch handler --------------------------------------------------

    def on_active_changed(self, e: ft.Event) -> None:
        if self.picker_radio is None or self.status is None:
            return
        new_name = self.picker_radio.value
        if new_name == self.app.gui_config.active_corpus_name:
            return
        entry = self._find_corpus(new_name)
        if entry is None:
            self.status.value = f"corpus {new_name!r} not found"
            self.app.page.update()
            return
        previous_state = self._snapshot_active_state()
        self.app.gui_config.active_corpus_name = new_name
        self.app.gui_config.neo4j_uri = entry.neo4j_uri
        self.app.gui_config.neo4j_user = entry.neo4j_user
        self.app.gui_config.lancedb_path = entry.lancedb_path
        self.app.gui_config.corpus_config_path = entry.corpus_config_path
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self._restore_active_state(previous_state)
            self.picker_radio.value = previous_state["active_corpus_name"]
            self.status.value = f"could not save: {exc}"
            self.app.page.update()
            return
        apply_connection_to_env(self.app.gui_config)
        password = get_corpus_password(new_name) or ""
        if password:
            os.environ["NEO4J_PASSWORD"] = password
        else:
            os.environ.pop("NEO4J_PASSWORD", None)
            self.status.value = (
                f"active corpus set to {new_name!r} but no Neo4j "
                f"password is stored for it — set one via Create New "
                f"first."
            )
        try:
            reset_after_key_change()
        except Exception as exc:
            logger.warning("reset_after_key_change failed: %r", exc)
        if password:
            self.status.value = f"active corpus: {new_name}"
        self._populate_info_card()
        self.app.chat_panel.append_system(f"switched to corpus {new_name!r}")
        self.app.page.update()

    def _find_corpus(self, name: str) -> CorpusEntry | None:
        for c in self.app.gui_config.corpora:
            if c.name == name:
                return c
        return None

    def _snapshot_active_state(self) -> dict[str, object]:
        return {
            "active_corpus_name": self.app.gui_config.active_corpus_name,
            "neo4j_uri": self.app.gui_config.neo4j_uri,
            "neo4j_user": self.app.gui_config.neo4j_user,
            "lancedb_path": self.app.gui_config.lancedb_path,
            "corpus_config_path": self.app.gui_config.corpus_config_path,
        }

    def _restore_active_state(self, snap: dict[str, object]) -> None:
        self.app.gui_config.active_corpus_name = snap["active_corpus_name"]  # type: ignore[assignment]
        self.app.gui_config.neo4j_uri = snap["neo4j_uri"]  # type: ignore[assignment]
        self.app.gui_config.neo4j_user = snap["neo4j_user"]  # type: ignore[assignment]
        self.app.gui_config.lancedb_path = snap["lancedb_path"]  # type: ignore[assignment]
        self.app.gui_config.corpus_config_path = snap["corpus_config_path"]  # type: ignore[assignment]

    # ----- Refresh --------------------------------------------------------

    def on_refresh_clicked(self, e: ft.Event) -> None:
        self._sync_picker()
        self._populate_info_card()
        if self.status is not None:
            self.status.value = "refreshed"
        self.app.page.update()

    def refresh_after_ingest(self) -> None:
        """Reload the corpus card (counts) + Documents list after an
        ingest / bulk-op in the Ingest sub-tab. Wired cross-tab by
        LibraryView so the user needn't hit Refresh manually."""
        self._populate_info_card()
        self.documents._schedule_reload(force=True)
        self.app.page.update()

    # ----- Rename ---------------------------------------------------------

    def on_rename_clicked(self, e: ft.Event) -> None:
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return
        new_name_field = ft.TextField(
            label="New name",
            value=active_name,
            border=ft.InputBorder.OUTLINE,
            border_color=FRAME_BORDER_COLOR,
            bgcolor=PANEL_BG,
        )
        error_text = ft.Text("", size=11, color=ft.Colors.RED_300)

        def _do_rename(_ev):
            new_name = (new_name_field.value or "").strip()
            if not new_name:
                error_text.value = "name can't be empty"
                self.app.page.update()
                return
            if new_name == active_name:
                self.app.page.pop_dialog()
                return
            if any(c.name == new_name for c in self.app.gui_config.corpora):
                error_text.value = f"name {new_name!r} already exists"
                self.app.page.update()
                return
            self._rename_corpus(active_name, new_name)
            self.app.page.pop_dialog()

        def _do_cancel(_ev):
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Rename corpus {active_name!r}"),
            content=ft.Column(
                controls=[
                    ft.Text(
                        "Rename only changes the label shown in the "
                        "picker. It does NOT rename the corpus folder, "
                        "the corpus.toml file, the LanceDB folder, or "
                        "the Neo4j DBMS — those keep their original "
                        "names on disk.",
                        size=12,
                    ),
                    new_name_field,
                    error_text,
                ],
                tight=True,
                spacing=8,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_do_cancel),
                ft.TextButton("Rename", on_click=_do_rename),
            ],
        )
        self.app.page.show_dialog(dialog)

    def _rename_corpus(self, old_name: str, new_name: str) -> None:
        if self.status is None:
            return
        old_password = get_corpus_password(old_name) or ""
        if old_password:
            try:
                set_corpus_password(new_name, old_password)
            except ConfigError as exc:
                self.status.value = f"could not migrate password: {exc}"
                self.app.page.update()
                return
        updated: list[CorpusEntry] = []
        target: CorpusEntry | None = None
        for c in self.app.gui_config.corpora:
            if c.name == old_name:
                target = c.model_copy(update={"name": new_name})
                updated.append(target)
            else:
                updated.append(c)
        self.app.gui_config.corpora = updated
        if self.app.gui_config.active_corpus_name == old_name:
            self.app.gui_config.active_corpus_name = new_name
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            if old_password:
                try:
                    set_corpus_password(new_name, "")
                except ConfigError:
                    pass
            self.status.value = f"could not save rename: {exc}"
            self.app.page.update()
            return
        if old_password:
            try:
                set_corpus_password(old_name, "")
            except ConfigError as exc:
                logger.warning(
                    "could not delete old keyring entry for %r: %r",
                    old_name, exc,
                )
        self._sync_picker()
        self._populate_info_card()
        self.status.value = f"renamed {old_name!r} → {new_name!r}"
        self.app.chat_panel.append_system(
            f"renamed corpus {old_name!r} → {new_name!r}"
        )
        self.app.page.update()

    # ----- Remove ---------------------------------------------------------

    def on_remove_clicked(self, e: ft.Event) -> None:
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return

        def _do_remove(_ev):
            self._remove_corpus(active_name)
            self.app.page.pop_dialog()

        def _do_cancel(_ev):
            self.app.page.pop_dialog()

        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Remove corpus {active_name!r}?"),
            content=ft.Text(
                "This unregisters the corpus from the GUI. It does "
                "NOT delete the Neo4j DBMS, the LanceDB folder, or "
                "the corpus.toml file — you manage those yourself.",
                size=12,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=_do_cancel),
                ft.TextButton("Remove", on_click=_do_remove),
            ],
        )
        self.app.page.show_dialog(dialog)

    def _remove_corpus(self, name: str) -> None:
        if self.status is None:
            return
        updated = [
            c for c in self.app.gui_config.corpora if c.name != name
        ]
        self.app.gui_config.corpora = updated
        if self.app.gui_config.active_corpus_name == name:
            new_active = updated[0] if updated else None
            self.app.gui_config.active_corpus_name = (
                new_active.name if new_active is not None else None
            )
            if new_active is not None:
                self.app.gui_config.neo4j_uri = new_active.neo4j_uri
                self.app.gui_config.neo4j_user = new_active.neo4j_user
                self.app.gui_config.lancedb_path = new_active.lancedb_path
                self.app.gui_config.corpus_config_path = (
                    new_active.corpus_config_path
                )
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save remove: {exc}"
            self.app.page.update()
            return
        try:
            set_corpus_password(name, "")
        except ConfigError as exc:
            logger.warning(
                "could not delete keyring entry for %r: %r", name, exc,
            )
        new_active_name = self.app.gui_config.active_corpus_name
        if new_active_name is not None:
            apply_connection_to_env(self.app.gui_config)
            password = get_corpus_password(new_active_name) or ""
            if password:
                os.environ["NEO4J_PASSWORD"] = password
            else:
                os.environ.pop("NEO4J_PASSWORD", None)
            try:
                reset_after_key_change()
            except Exception as exc:
                logger.warning("reset_after_key_change failed: %r", exc)
        self._sync_picker()
        self._populate_info_card()
        self.status.value = f"removed {name!r} from the corpus list"
        self.app.chat_panel.append_system(
            f"unregistered corpus {name!r} (external files kept)"
        )
        self.app.page.update()


# ----- module-level helpers ---------------------------------------------

def _kv_row(label: str, value_control: ft.Control) -> ft.Row:
    """A `label: value` row where the label is fixed-width so keys align."""
    return ft.Row(
        controls=[
            ft.Text(label, size=12, color=ft.Colors.GREY_500, width=110),
            value_control,
        ],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )


def _badge(label: str, *, on: bool) -> ft.Control:
    """Small pill showing a layer flag on/off state."""
    return ft.Container(
        padding=ft.Padding.symmetric(vertical=2, horizontal=6),
        border_radius=10,
        bgcolor=(
            ft.Colors.GREEN_900 if on else ft.Colors.GREY_800
        ),
        content=ft.Text(
            label,
            size=11,
            color=(
                ft.Colors.GREEN_200 if on else ft.Colors.GREY_500
            ),
        ),
    )


def _onoff(v: bool) -> str:
    """Compact on/off label for a boolean corpus setting."""
    return "on" if v else "off"
