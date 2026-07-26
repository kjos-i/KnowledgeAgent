"""Library → Select sub-tab — pick a corpus, view its status + documents.

Internal 2-column layout (fixed proportions, no dragger):

  Left column  (LEFT_COLUMN_WIDTH): read-only info card for the active
                          corpus (name, URIs, paths, counts, status,
                          layer pills, ontologies + extractor + xrefs,
                          chunk/OCR/figure settings, embedder + LLM,
                          and a "changed since last ingest" diff).
  Right column (expand):  Documents browser (filter + Refresh + card
                          list), the DocumentsView.

Corpus selection + management (Rename / Relocate / Remove / Refresh)
moved to the global top-bar `Corpus ▾` dropdown + `⚙ Manage` dialog;
those handlers still live here (the Manage dialog calls across to them).

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
  5. `save_config` + `apply_connection_to_env` + `reset_after_settings_change`.
  6. Repopulate the info card.

Rename: updates `CorpusEntry.name`; migrates keyring entry from
`neo4j-{old}` → `neo4j-{new}`; refreshes info card if the renamed
corpus is active.

Relocate: repoints a MOVED corpus — rewrites `CorpusEntry.lancedb_path`
+ `corpus_config_path` to a folder you pick (must contain `corpus.toml`).
No re-ingest; the data moved with the folder.

Remove: unregisters the `CorpusEntry` + deletes its keyring entry.
Does NOT delete the actual Neo4j DBMS / LanceDB folder / corpus.toml
— those are external artefacts the user manages themselves.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft
from pydantic import ValidationError

from knowledge_agent.config import reset_after_settings_change
from knowledge_agent.corpus_config import CorpusConfig, load_corpus_config
from knowledge_agent.gui._styles import (
    FRAME_BORDER_COLOR,
    LEFT_COLUMN_WIDTH,
    PANEL_BG,
    PANEL_RADIUS,
    panel_box,
    panel_title,
)
from knowledge_agent.gui._widgets.info_text import info
from knowledge_agent.gui.config_store import (
    ConfigError,
    apply_active_corpus_embedding_to_env,
    apply_connection_to_env,
    get_corpus_password,
    save_config,
    set_corpus_password,
)
from knowledge_agent.gui.library.config_diff import config_diff
from knowledge_agent.gui.library.documents_view import DocumentsView
from knowledge_agent.gui.library.session_state import load_session

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.config_store import CorpusEntry


logger = logging.getLogger(__name__)


# Idle placeholder for the left-column status line — always visible so it
# reads as an intentional line, not a blank gap under the info card.
_STATUS_IDLE = "No recent corpus changes."


# Sentinel `field` for the L7 pill: ontology linking is a bundle of 18
# per-ontology flags (no single LayerFlags bool), so its pill counts as
# "configured" when ANY `ontology_*` flag is on. The detail — which
# vocabularies — lives on the separate Ontologies line below.
_ONTOLOGY_ANY = "__ontology_any__"

# Layer pills — one per pipeline layer, in pipeline order: L1–L4
# (OpenAlex) · L5 chunks · L6 entities · L7 ontology · L8 triples ·
# L9 cross-doc · L10 cross-doc xrefs. Each entry is (field, label,
# tooltip). A pill is GREEN only when its layer is configured AND the
# corpus has ingested data (docs > 0); grey otherwise (not configured,
# OR configured-but-empty). `field` is the LayerFlags attribute that
# gates "configured", or `_ONTOLOGY_ANY` for the L7 bundle.
_LAYER_BADGES: tuple[tuple[str, str, str], ...] = (
    (
        "openalex_papers",
        "L1–L4",
        "L1–L4 — OpenAlex bundle: citation / author / venue / topic graph",
    ),
    ("chunks", "L5", "L5 — per-chunk :Chunk nodes for retrieval"),
    ("entities", "L6", "L6 — named-entity extraction per chunk"),
    (
        _ONTOLOGY_ANY,
        "L7",
        "L7 — ontology linking (which vocabularies: see Ontologies below)",
    ),
    ("triples", "L8", "L8 — (subject, predicate, object) edges per chunk"),
    ("cross_doc", "L9", "L9 — shared-entity links across docs"),
    (
        "cross_doc_xrefs",
        "L10",
        "L10 — shared-canonical-concept links across docs",
    ),
)


# Same ontology keys as CorpusConfigEditor's `_ONTOLOGY_DISPLAY` —
# kept local to avoid a cross-import between sibling files.
_ONTOLOGY_KEYS: tuple[tuple[str, str], ...] = (
    ("mesh", "MeSH"),
    ("go", "GO"),
    ("hpo", "HPO"),
    ("uberon", "UBERON"),
    ("mondo", "MONDO"),
    ("chebi", "ChEBI"),
    ("eco", "ECO"),
    ("so", "SO"),
    ("pr", "PR"),
    ("cl", "CL"),
    ("po", "PO"),
    ("foodon", "FOODON"),
    ("envo", "ENVO"),
    ("ncbitaxon", "NCBITaxon"),
    ("obi", "OBI"),
    ("efo", "EFO"),
    ("dron", "DRON"),
    ("fibo", "FIBO"),
)


class SelectDatasetTab:
    """Corpus picker (left) + read-only info card + docs table (right)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # Corpus SELECTION + MANAGEMENT moved to the global top-bar
        # `Corpus ▾` dropdown + `⚙ Manage` dialog (Phase 2). This tab is now
        # a read-only "browse the active corpus" view: info card + Documents.
        # The `on_{rename,relocate,remove,refresh}_clicked` handlers below
        # stay — the Manage dialog reaches across to them.
        self.status: ft.Text | None = None
        # Sticky left-column title ("Selected dataset: <name>").
        self.selected_title: ft.Text | None = None

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
        # "Changed since last ingest" section — the unsaved config draft
        # (from the Ingest tab) diffed against the ingested baseline. The
        # whole section hides when there are no pending changes.
        self.info_pending: ft.Text | None = None
        self.info_pending_section: ft.Container | None = None
        # Strong refs to in-flight count-fetch tasks (they self-discard).
        self._bg_tasks: set[asyncio.Task] = set()
        # Config of the active corpus, stashed by `_populate_info_card`
        # so the async count fetch can re-colour the layer pills (green
        # needs data, which is only known after the count returns).
        self._active_cfg: CorpusConfig | None = None

        self.documents = DocumentsView(app)
        self._create_controls()

    # ----- control construction --------------------------------------------

    def _create_controls(self) -> None:
        self.status = ft.Text(_STATUS_IDLE, size=12, color=ft.Colors.GREY_400)
        # Sticky left-column title — set in `_populate_info_card` /
        # `_info_empty_state` to the active corpus name.
        self.selected_title = panel_title("Selected dataset: —")

        # ---- info card controls ----
        self.info_name = ft.Text(
            "",
            size=15,
            weight=ft.FontWeight.BOLD,
        )
        self.info_uri = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_user = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_lancedb = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_toml = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_counts = ft.Text(
            "docs: — · chunks: — · mentions: —",
            size=12,
            color=ft.Colors.GREY_400,
        )
        self.info_layers = ft.Row(spacing=6, wrap=True, controls=[])
        self.info_ontologies = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.info_extractor = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_300,
        )
        self.info_xrefs = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_breakdown = ft.Text("", size=12, color=ft.Colors.GREY_400)
        self.info_chunking = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_ocr = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_figures = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_embedder = ft.Text("", size=12, color=ft.Colors.GREY_300)
        self.info_llm = ft.Text("", size=12, color=ft.Colors.GREY_300)
        # "Changed since last ingest" — hidden unless the corpus has an
        # unsaved config draft (edited on the Ingest tab, not yet ingested).
        self.info_pending = ft.Text("", size=12, color=ft.Colors.AMBER_300, selectable=True)
        self.info_pending_section = ft.Container(
            visible=False,
            content=ft.Column(
                spacing=4,
                controls=[
                    ft.Divider(height=1),
                    ft.Text(
                        "Changed since last ingest (unsaved)",
                        size=12,
                        weight=ft.FontWeight.BOLD,
                        color=ft.Colors.AMBER_300,
                    ),
                    self.info_pending,
                ],
            ),
        )

        self.info_container = ft.Container(
            padding=12,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            bgcolor=PANEL_BG,
            border_radius=PANEL_RADIUS,
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
                                size=12,
                                color=ft.Colors.GREY_500,
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
                    self.info_pending_section,
                ],
            ),
        )

    # ----- public API -------------------------------------------------------

    def build(self) -> ft.Control:
        # Refresh the info card every render — cheap, avoids stale state
        # after sibling-tab mutations (corpus switch / ingest).
        self._populate_info_card()

        # Left column: the read-only info card + a status line. Corpus
        # selection + management moved to the global top-bar `Corpus ▾`
        # dropdown + `⚙ Manage` dialog.
        left_pane = ft.Container(
            width=LEFT_COLUMN_WIDTH,
            content=ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                controls=[
                    self.info_container,
                    self.status,
                ],
            ),
        )

        # Right column: documents area (scrollable).
        right_pane = panel_box(
            ft.Column(
                expand=True,
                scroll=ft.ScrollMode.AUTO,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                spacing=10,
                controls=[self.documents.build()],
            )
        )

        body = ft.Row(
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=12,
            controls=[left_pane, right_pane],
        )

        # Two column titles in a fixed header above the panes (the Ingest
        # idiom): left = the selected corpus, right = "Documents (N)" — the
        # count rides in from DocumentsView, which no longer draws its own
        # header. Left container is pinned to the card column's width so the
        # titles sit squarely over their panes.
        header = ft.Row(
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=12,
            controls=[
                ft.Container(
                    width=LEFT_COLUMN_WIDTH,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=ft.Row(
                        controls=[
                            self.selected_title,
                            info(self.app, "select.selected_dataset"),
                        ],
                        spacing=4,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
                ft.Container(
                    expand=1,
                    padding=ft.Padding.symmetric(horizontal=12, vertical=10),
                    content=ft.Row(
                        controls=[
                            panel_title("Documents"),
                            self.documents.coverage_text,
                            info(self.app, "select.documents"),
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ),
            ],
        )
        return ft.Column(
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            spacing=8,
            controls=[header, body],
        )

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
        if self.selected_title is not None:
            self.selected_title.value = f'Selected dataset: "{entry.name}"'
        self.info_uri.value = entry.neo4j_uri
        self.info_user.value = entry.neo4j_user
        self.info_lancedb.value = str(entry.lancedb_path)
        self.info_toml.value = str(entry.corpus_config_path)

        cfg = self._safe_load_config(entry)
        self._active_cfg = cfg
        # Pills start grey; `_reload_counts` re-colours them once the
        # doc count is known (green needs data, not just config).
        self._populate_layer_badges(cfg, has_data=False)
        self._populate_ontologies(cfg)
        self._populate_extractor(cfg)
        self._populate_xrefs(cfg)
        self._populate_ingest_settings(cfg)
        self._populate_models()
        self._populate_pending_changes()
        self._schedule_counts()

    def refresh_pending_changes(self) -> None:
        """Live-refresh just the "changed since last ingest" section.

        Wired to the Ingest tab's config editor (`on_draft_changed`), so
        editing config on the right updates this card immediately without a
        full info-card re-fetch (which would also re-run the count query).
        """
        self._populate_pending_changes()
        self.app.page.update()

    def _populate_pending_changes(self) -> None:
        """Fill the "changed since last ingest" section from the corpus's
        unsaved config draft, diffed against the ingested baseline.

        The card keeps showing the *ingested* config (its whole reason for
        existing); this section adds what's been changed on the Ingest tab
        but not yet re-ingested. Hidden when there's no draft, or the draft
        matches the baseline.
        """
        section = self.info_pending_section
        text = self.info_pending
        if section is None or text is None:
            return
        baseline = self._active_cfg
        draft = self._load_draft_config()
        diffs = config_diff(baseline, draft) if baseline is not None and draft is not None else []
        if not diffs:
            section.visible = False
            text.value = ""
            return
        header = (
            f"{len(diffs)} setting{'s' if len(diffs) != 1 else ''} changed "
            "(edit on the Ingest tab; re-ingest to apply):"
        )
        lines = [header]
        lines.extend(f"  • {name}: {old} → {new}" for name, old, new in diffs)
        text.value = "\n".join(lines)
        section.visible = True

    def _load_draft_config(self) -> CorpusConfig | None:
        """The active corpus's persisted unsaved draft, if present and valid.

        A draft that no longer validates (schema drift) is treated as
        absent — the card falls back to showing no pending changes."""
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return None
        entry = self._find_corpus(active_name)
        if entry is None:
            return None
        session = load_session(entry.corpus_config_path)
        if session.draft_config is None:
            return None
        try:
            return CorpusConfig.model_validate(session.draft_config)
        except ValidationError:
            return None

    def _info_empty_state(self) -> None:
        """When there's no active corpus, blank out all the fields."""
        self._active_cfg = None
        if self.selected_title is not None:
            self.selected_title.value = "Selected dataset: none"
        for text in (
            self.info_name,
            self.info_uri,
            self.info_user,
            self.info_lancedb,
            self.info_toml,
            self.info_ontologies,
            self.info_extractor,
            self.info_xrefs,
            self.info_breakdown,
            self.info_chunking,
            self.info_ocr,
            self.info_figures,
            self.info_embedder,
            self.info_llm,
        ):
            if text is not None:
                text.value = "—"
        if self.info_counts is not None:
            self.info_counts.value = "docs: — · chunks: — · mentions: —"
        if self.info_layers is not None:
            self.info_layers.controls = [
                ft.Text("—", size=12, color=ft.Colors.GREY_600),
            ]
        if self.info_pending_section is not None:
            self.info_pending_section.visible = False

    def _safe_load_config(self, entry: CorpusEntry) -> CorpusConfig | None:
        try:
            return load_corpus_config(entry.corpus_config_path)
        except FileNotFoundError:
            logger.info(
                "corpus.toml not found for %r at %s",
                entry.name,
                entry.corpus_config_path,
            )
            return None
        except Exception as exc:
            logger.warning(
                "load_corpus_config failed for %r: %r",
                entry.name,
                exc,
            )
            return None

    def _populate_layer_badges(self, cfg: CorpusConfig | None, *, has_data: bool) -> None:
        """Render one pill per pipeline layer. GREEN only when the layer
        is configured AND the corpus has ingested data (`has_data`); grey
        otherwise. `has_data` is known only after the async doc count, so
        the first (sync) call passes False — an un-ingested corpus starts
        all-grey — and `_reload_counts` re-populates with the real value."""
        if self.info_layers is None:
            return
        badges: list[ft.Control] = []
        for field, label, tooltip in _LAYER_BADGES:
            configured = cfg is not None and self._layer_configured(cfg, field)
            badges.append(_badge(label, on=configured and has_data, tooltip=tooltip))
        self.info_layers.controls = badges

    @staticmethod
    def _layer_configured(cfg: CorpusConfig, field: str) -> bool:
        """True when `field` is enabled in the corpus config. `field` is a
        `LayerFlags` attribute, or `_ONTOLOGY_ANY` — the L7 bundle, on when
        any single `ontology_*` flag is set."""
        if field == _ONTOLOGY_ANY:
            return any(
                bool(getattr(cfg.layers, f"ontology_{key}", False)) for key, _ in _ONTOLOGY_KEYS
            )
        return bool(getattr(cfg.layers, field, False))

    def _populate_ontologies(self, cfg: CorpusConfig | None) -> None:
        if self.info_ontologies is None:
            return
        self.info_ontologies.value = _ontologies_str(cfg) if cfg is not None else "—"

    def _populate_extractor(self, cfg: CorpusConfig | None) -> None:
        if self.info_extractor is None:
            return
        self.info_extractor.value = _extractor_str(cfg) if cfg is not None else "not configured"

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
        if self.info_chunking is not None:
            self.info_chunking.value = _chunking_str(cfg)
        if self.info_ocr is not None:
            self.info_ocr.value = _ocr_str(cfg)
        if self.info_figures is not None:
            self.info_figures.value = _figures_str(cfg)

    def _populate_models(self) -> None:
        """Show the active embedder (provider · model · dims) + the LLM
        provider from global Settings. Read-only; failure → '—'.

        The LLM uses per-role models (synthesizer, cypher_builder, …), so
        only the provider is summarised — role models live in Settings."""
        try:
            from knowledge_agent.config import get_settings

            s = get_settings()
        except Exception as exc:
            logger.info("SelectDataset: get_settings failed (%r); models —", exc)
            for t in (self.info_embedder, self.info_llm):
                if t is not None:
                    t.value = "—"
            return
        if self.info_embedder is not None:
            self.info_embedder.value = (
                f"{s.embedding_provider} · {s.embedding_model} · {s.embedding_dims}-dim"
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
        '—' independently, so one failure doesn't blank the rest.

        Also re-colours the layer pills: green needs real data, so a pill
        lights up only once we've confirmed the corpus has ≥ 1 indexed
        doc. When the LanceDB read fails, docs is unknown → all grey (we
        never claim green we can't confirm)."""
        if self.info_counts is None:
            return
        docs_str = "—"
        chunks_str = "—"
        breakdown = "—"
        n_docs = 0
        # Local import: keeps lancedb out of GUI startup (warmed by the graph
        # pre-warm). See app.py `_prewarm_graph`.
        from knowledge_agent.search.client import get_search_client

        try:
            rows = await get_search_client().list_indexed_docs()
        except Exception as exc:
            logger.warning("SelectDataset: list_indexed_docs failed: %r", exc)
            rows = None
        if rows is not None:
            n_docs = len(rows)
            docs_str = str(n_docs)
            chunks_str = str(sum(int(r.get("n_chunks") or 0) for r in rows))
            counts = Counter((r.get("metadata_status") or "unknown") for r in rows)
            breakdown = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "—"
        mentions_str = "—"
        try:
            from knowledge_agent.kg.client import get_kg_client

            mentions_str = str(await get_kg_client().count_mentions())
        except Exception as exc:
            logger.info("SelectDataset: count_mentions failed (%r); showing —", exc)
        self.info_counts.value = (
            f"docs: {docs_str} · chunks: {chunks_str} · mentions: {mentions_str}"
        )
        if self.info_breakdown is not None:
            self.info_breakdown.value = breakdown
        # Now that we know whether the corpus holds data, re-colour the
        # pills: a configured layer only shows green when n_docs > 0.
        self._populate_layer_badges(self._active_cfg, has_data=n_docs > 0)
        self.app.page.update()

    def refresh_after_switch(self) -> None:
        """Refresh the info card + Documents after an app-wide corpus switch
        (called by `GuiApp.refresh_after_corpus_change`).

        No `page.update()` here — the app broadcast issues one at the end."""
        self._populate_info_card()
        self.documents._schedule_reload(force=True)

    def _find_corpus(self, name: str) -> CorpusEntry | None:
        for c in self.app.gui_config.corpora:
            if c.name == name:
                return c
        return None

    # ----- Refresh (reached from the Manage dialog) -----------------------

    def on_refresh_clicked(self, e: ft.Event) -> None:
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
        error_text = ft.Text("", size=12, color=ft.Colors.RED_300)

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
                with contextlib.suppress(ConfigError):
                    set_corpus_password(new_name, "")
            self.status.value = f"could not save rename: {exc}"
            self.app.page.update()
            return
        if old_password:
            try:
                set_corpus_password(old_name, "")
            except ConfigError as exc:
                logger.warning(
                    "could not delete old keyring entry for %r: %r",
                    old_name,
                    exc,
                )
        self.app.refresh_after_corpus_change()  # dropdown + card + Ingest
        self.status.value = f"renamed {old_name!r} → {new_name!r}"
        self.app.chat_panel.append_system(f"renamed corpus {old_name!r} → {new_name!r}")
        self.app.page.update()

    # ----- Relocate -------------------------------------------------------

    async def on_relocate_clicked(self, e: ft.Event) -> None:
        """Repoint the active corpus at a folder it was MOVED to.

        Opens the OS folder picker; the chosen folder becomes the corpus
        home (its `corpus.toml` + `lancedb/` are read from there). No
        re-ingest — the data moved with the folder, so only the two stored
        paths change."""
        active_name = self.app.gui_config.active_corpus_name
        if active_name is None:
            return
        entry = self._find_corpus(active_name)
        if entry is None:
            return
        current = entry.corpus_config_path.parent
        initial = str(current) if current.is_dir() else None
        try:
            chosen = await self.app.file_picker.get_directory_path(
                dialog_title=f"Pick the new folder for corpus {active_name!r}",
                initial_directory=initial,
            )
        except Exception as exc:
            logger.warning("folder picker failed: %r", exc)
            if self.status is not None:
                self.status.value = f"folder picker error: {exc}"
                self.app.page.update()
            return
        if not chosen:
            return
        self._relocate_corpus(active_name, Path(chosen))

    def _relocate_corpus(self, name: str, new_folder: Path) -> None:
        """Rewrite the corpus's `lancedb_path` + `corpus_config_path` to
        `new_folder` (its home after a move). Requires a `corpus.toml` there
        — the corpus identity marker; `lancedb/` may be absent (never
        ingested). Re-bridges to env when the relocated corpus is active."""
        if self.status is None:
            return
        toml_path = new_folder / "corpus.toml"
        if not toml_path.is_file():
            self.status.value = (
                f"no corpus.toml in {new_folder} — pick the folder that IS the corpus"
            )
            self.app.page.update()
            return
        lancedb_path = new_folder / "lancedb"
        updated: list[CorpusEntry] = []
        for c in self.app.gui_config.corpora:
            if c.name == name:
                updated.append(
                    c.model_copy(
                        update={
                            "lancedb_path": lancedb_path,
                            "corpus_config_path": toml_path,
                        }
                    )
                )
            else:
                updated.append(c)
        self.app.gui_config.corpora = updated
        is_active = self.app.gui_config.active_corpus_name == name
        if is_active:
            self.app.gui_config.lancedb_path = lancedb_path
            self.app.gui_config.corpus_config_path = toml_path
        try:
            save_config(self.app.gui_config)
        except ConfigError as exc:
            self.status.value = f"could not save relocation: {exc}"
            self.app.page.update()
            return
        if is_active:
            apply_connection_to_env(self.app.gui_config)
            try:
                reset_after_settings_change()
            except Exception as exc:
                logger.warning("reset_after_settings_change failed: %r", exc)
        self.app.refresh_after_corpus_change()  # dropdown + card + Ingest
        self.status.value = f"relocated {name!r} → {new_folder}"
        self.app.chat_panel.append_system(f"relocated corpus {name!r} → {new_folder}")
        self.app.page.update()

    # ----- Remove ---------------------------------------------------------

    def open_remove_confirm(self, name: str, *, missing: bool = False) -> None:
        """Confirm, then unregister ANY registered corpus by name (not only
        the active one).

        Shared by the active-corpus Remove button and the Manage dialog's
        corpus list, so a stale corpus whose folder is gone can be pruned
        without activating it first. Pass `missing=True` to word the prompt
        for a corpus whose files are already deleted. Removal never touches
        the corpus's files (see `_remove_corpus`).
        """
        if not name:
            return

        def _do_remove(_ev):
            self._remove_corpus(name)
            self.app.page.pop_dialog()

        def _do_cancel(_ev):
            self.app.page.pop_dialog()

        if missing:
            body = (
                "This corpus's files are already gone (its corpus.toml is "
                "missing), so this just clears the stale entry from the list. "
                "Nothing on disk is affected."
            )
        else:
            body = (
                "This unregisters the corpus from the GUI. It does NOT delete "
                "the Neo4j DBMS, the LanceDB folder, or the corpus.toml file; "
                "you manage those yourself."
            )
        dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Remove corpus {name!r}?"),
            content=ft.Text(body, size=12),
            actions=[
                ft.TextButton("Cancel", on_click=_do_cancel),
                ft.TextButton("Remove", on_click=_do_remove),
            ],
        )
        self.app.page.show_dialog(dialog)

    def _remove_corpus(self, name: str) -> None:
        if self.status is None:
            return
        updated = [c for c in self.app.gui_config.corpora if c.name != name]
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
                self.app.gui_config.corpus_config_path = new_active.corpus_config_path
            else:
                # Removed the last corpus — clear the stale path mirrors so
                # the no-corpus state is consistent: LANCEDB_PATH lands on
                # the sentinel below instead of pointing at the removed
                # corpus's folder.
                self.app.gui_config.lancedb_path = None
                self.app.gui_config.corpus_config_path = None
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
                "could not delete keyring entry for %r: %r",
                name,
                exc,
            )
        new_active_name = self.app.gui_config.active_corpus_name
        # Always re-bridge: when a corpus remains it points env at that
        # corpus; when none remain (mirror now None) it lands LANCEDB_PATH
        # on the sentinel. Either way, drop caches so the next backend read
        # rebuilds against the new target.
        apply_connection_to_env(self.app.gui_config)
        # Removing the active corpus promotes a DIFFERENT corpus (or none);
        # bridge its embedder so ingest/query match the newly-active corpus.
        # No-op when no corpus remains (corpus_config_path is None).
        apply_active_corpus_embedding_to_env(self.app.gui_config)
        if new_active_name is not None:
            password = get_corpus_password(new_active_name) or ""
            if password:
                os.environ["NEO4J_PASSWORD"] = password
            else:
                os.environ.pop("NEO4J_PASSWORD", None)
        else:
            os.environ.pop("NEO4J_PASSWORD", None)
        try:
            reset_after_settings_change()
        except Exception as exc:
            logger.warning("reset_after_settings_change failed: %r", exc)
        self.app.refresh_after_corpus_change()  # dropdown + card + Ingest
        self.status.value = f"removed {name!r} from the corpus list"
        self.app.chat_panel.append_system(f"unregistered corpus {name!r} (external files kept)")
        self.app.page.update()


# ----- module-level helpers ---------------------------------------------


def _kv_row(label: str, value_control: ft.Control) -> ft.Row:
    """A `label: value` row where the label is fixed-width so keys align.

    The value sits in an `expand=True` container so long values (LanceDB /
    corpus.toml paths, ontology lists) wrap onto multiple lines instead of
    being clipped on the right. Top-aligned so the label stays on the
    value's first line when it wraps.
    """
    return ft.Row(
        controls=[
            ft.Text(label, size=12, color=ft.Colors.GREY_500, width=110),
            ft.Container(content=value_control, expand=True),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def _badge(label: str, *, on: bool, tooltip: str | None = None) -> ft.Control:
    """Small pill showing a layer's state: green = configured + populated,
    grey otherwise. `tooltip` (hover) spells out which layer the pill is."""
    return ft.Container(
        padding=ft.Padding.symmetric(vertical=2, horizontal=6),
        border_radius=10,
        bgcolor=(ft.Colors.GREEN_900 if on else ft.Colors.GREY_800),
        tooltip=tooltip,
        content=ft.Text(
            label,
            size=12,
            color=(ft.Colors.GREEN_200 if on else ft.Colors.GREY_500),
        ),
    )


def _onoff(v: bool) -> str:
    """Compact on/off label for a boolean corpus setting."""
    return "on" if v else "off"


# ---- corpus summary formatters (single source) -------------------------------
#
# Shared by this tab's info card (`_populate_*`) AND the global Manage
# dialog's card (`corpus_card.build_corpus_card`), so the two can't drift.
# Pure cfg -> display string; callers guard `cfg is None` themselves.


def _chunking_str(cfg: CorpusConfig) -> str:
    return (
        f"{cfg.chunker_strategy} · max {cfg.chunk_max_tokens} tok "
        f"· merge_peers {_onoff(bool(cfg.merge_peers))}"
    )


def _ocr_str(cfg: CorpusConfig) -> str:
    return f"pdf {_onoff(bool(cfg.enable_pdf_ocr))} · image {_onoff(bool(cfg.enable_image_ocr))}"


def _figures_str(cfg: CorpusConfig) -> str:
    return (
        f"extract {_onoff(bool(cfg.extract_figures))} · "
        f"scale {cfg.images_scale} · min {cfg.min_figure_bytes} B"
    )


def _ontologies_str(cfg: CorpusConfig) -> str:
    active = [
        display
        for key, display in _ONTOLOGY_KEYS
        if bool(getattr(cfg.layers, f"ontology_{key}", False))
    ]
    return ", ".join(active) if active else "none enabled"


def _extractor_str(cfg: CorpusConfig) -> str:
    """`extractor(s) [· mode · types]` summary, or 'not configured' when the
    entities layer has no [entities] section."""
    if cfg.entities is None:
        return "not configured"
    ents = cfg.entities
    extractors = list(getattr(ents, "extractors", None) or [])
    if not extractors:
        single = getattr(ents, "extractor", None)  # legacy singular field
        extractors = [single] if single else []
    parts = [", ".join(extractors) or "—"]
    mode = getattr(ents, "entity_types_mode", None)
    if mode:
        parts.append(f"mode: {mode}")
    if ents.entity_types:
        parts.append(f"types: {', '.join(ents.entity_types)}")
    return " · ".join(parts)
