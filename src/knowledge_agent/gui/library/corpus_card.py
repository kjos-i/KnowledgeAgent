"""Read-only corpus summary card for the global Manage-corpora dialog.

Shows the active corpus's key facts — connection + paths, enabled layers,
extractor, chunking / OCR / figures, and (notes #34) the **embedder + LLM**
— so the user can see what a corpus was built with from anywhere, not just
Library → Select. Built fresh (its own controls) each time the dialog opens;
purely read-only (the management actions are separate buttons in the dialog).

Reuses the Library card's display helpers (`_kv_row`, `_onoff`) so the two
stay visually consistent. The field ASSEMBLY is intentionally a small,
self-contained duplicate of `SelectDatasetTab._populate_*` for now — kept
read-only and synchronous (no async counts / live badges) so it can't
regress the live Library card. Phase 2b converges both onto one component.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.library.select_dataset import (
    _LAYER_BADGES,
    _ONTOLOGY_ANY,
    _ONTOLOGY_KEYS,
    _chunking_str,
    _extractor_str,
    _figures_str,
    _kv_row,
    _ocr_str,
    _ontologies_str,
)
from knowledge_agent.kg.corpus_config import load_corpus_config

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.config_store import CorpusEntry
    from knowledge_agent.kg.corpus_config import CorpusConfig

logger = logging.getLogger(__name__)


def build_corpus_card(app: GuiApp) -> ft.Control:
    """Return a read-only summary card for the app's active corpus.

    No active corpus → a short hint. corpus.toml unreadable → connection /
    paths still show, with an amber note in place of the config summary.
    """
    entry = _active_entry(app)
    if entry is None:
        return ft.Text(
            "No active corpus — create or select one in Library.",
            size=12,
            italic=True,
            color=ft.Colors.GREY_400,
        )

    rows: list[ft.Control] = [
        ft.Text(entry.name, size=15, weight=ft.FontWeight.BOLD),
        ft.Divider(height=1),
        _kv_row("Neo4j URI", _value(entry.neo4j_uri)),
        _kv_row("Neo4j user", _value(entry.neo4j_user)),
        _kv_row("LanceDB path", _value(str(entry.lancedb_path))),
        _kv_row("corpus.toml", _value(str(entry.corpus_config_path))),
        ft.Divider(height=1),
    ]

    cfg = _safe_load_config(entry)
    if cfg is None:
        rows.append(
            ft.Text(
                "corpus.toml could not be read — showing connection only.",
                size=12,
                italic=True,
                color=ft.Colors.AMBER_300,
            )
        )
    else:
        rows += [
            _kv_row("Layers", _value(_layers_str(cfg))),
            _kv_row("Ontologies", _value(_ontologies_str(cfg))),
            _kv_row("Extractor", _value(_extractor_str(cfg))),
            _kv_row("Xrefs", _value(cfg.layers.xrefs)),
            ft.Divider(height=1),
            _kv_row("Chunking", _value(_chunking_str(cfg))),
            _kv_row("OCR", _value(_ocr_str(cfg))),
            _kv_row("Figures", _value(_figures_str(cfg))),
        ]

    embedder, llm = _models_str(app)
    rows += [
        ft.Divider(height=1),
        _kv_row("Embedder", _value(embedder)),
        _kv_row("LLM", _value(llm)),
    ]

    return ft.Column(spacing=6, tight=True, controls=rows)


# ---- data access (fail-soft) -------------------------------------------------


def _active_entry(app: GuiApp) -> CorpusEntry | None:
    active = app.gui_config.active_corpus_name
    if not active:
        return None
    return next((c for c in app.gui_config.corpora if c.name == active), None)


def _safe_load_config(entry: CorpusEntry) -> CorpusConfig | None:
    try:
        return load_corpus_config(entry.corpus_config_path)
    except Exception as exc:  # read-only card must never raise
        logger.info("build_corpus_card: load_corpus_config failed for %r: %r", entry.name, exc)
        return None


# ---- formatting (mirrors SelectDatasetTab._populate_*) -----------------------


def _value(text: str) -> ft.Text:
    return ft.Text(text, size=12, color=ft.Colors.GREY_300, selectable=True)


def _layers_str(cfg: CorpusConfig) -> str:
    enabled: list[str] = []
    for field, label, _tip in _LAYER_BADGES:
        if field == _ONTOLOGY_ANY:
            on = any(bool(getattr(cfg.layers, f"ontology_{k}", False)) for k, _ in _ONTOLOGY_KEYS)
        else:
            on = bool(getattr(cfg.layers, field, False))
        if on:
            enabled.append(label)
    return ", ".join(enabled) if enabled else "none"


def _models_str(app: GuiApp) -> tuple[str, str]:
    """(embedder, llm) summary from global Settings. Fail-soft → ('—', '—')."""
    try:
        from knowledge_agent.config import get_settings

        s = get_settings()
    except Exception as exc:  # read-only card must never raise
        logger.info("build_corpus_card: get_settings failed: %r", exc)
        return "—", "—"
    embedder = f"{s.embedding_provider} · {s.embedding_model} · {s.embedding_dims}-dim"
    return embedder, str(s.llm_provider)
