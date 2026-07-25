"""Build-path smoke tests for `CorpusConfigEditor`.

The section handlers are covered elsewhere; these pin the `build()` render
itself — that the panel assembles (info-icon headers + all 8 layer
sections) without a real Flet runtime, and that every section header
registers an `(i)` with the global teaching-mode toggle. `build()` had no
coverage before the info-icon retrofit.
"""

from __future__ import annotations

from types import SimpleNamespace

import flet as ft

from knowledge_agent.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.gui.library.corpus_config_editor import CorpusConfigEditor


def _cfg() -> CorpusConfig:
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractors=["llm"]),
    )


def _ready_editor(fake_app):
    """Editor with a corpus already 'loaded' so build() renders the
    sections rather than the empty-state / status-only fallbacks."""
    fake_app.gui_config.active_corpus_name = "c1"
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"  # == active → build() skips the disk reload
    ed._corpus_config = _cfg()
    ed._baseline_config = _cfg()
    return ed


def test_build_renders_a_column(fake_app):
    ed = _ready_editor(fake_app)
    assert isinstance(ed.build(), ft.Column)


def test_build_registers_an_info_icon_per_section(fake_app):
    """8 layer sections + the Ingest-infrastructure block + the
    cross-ontology-xrefs sub-header each register an `(i)` with the app's
    global show/hide toggle."""
    ed = _ready_editor(fake_app)
    ed.build()
    assert fake_app.register_info_icon.call_count >= 10


def test_section_title_pairs_label_with_info_icon(fake_app):
    ed = _ready_editor(fake_app)
    row = ed._section_title("My section", "help text")
    assert isinstance(row, ft.Row)
    assert isinstance(row.controls[0], ft.Text)
    assert row.controls[0].value == "My section"
    # info_icon now returns a Row of tier icons; the standard (i) is inside it.
    assert isinstance(row.controls[1], ft.Row)
    assert isinstance(row.controls[1].controls[0], ft.IconButton)


def test_refresh_availability_greys_section_fields_when_layer_off(fake_app):
    """A section's sub-controls are disabled when its own layer toggle is
    off (so it's clear those settings won't apply), and re-enabled when on.
    The `entity_types_field` is excluded — `_refresh_extractor_groups` owns
    its disabled state."""
    from knowledge_agent.corpus_config import EntityConfig, LayerFlags

    ed = CorpusConfigEditor(fake_app)  # __init__ builds the controls

    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=False, entities=False),  # triples/cross_doc default off
        entities=EntityConfig(extractors=["llm"]),
    )
    ed._refresh_availability()
    assert ed.chunk_max_tokens_field.disabled is True
    assert ed.optimize_indexes_checkbox.disabled is True
    assert ed.triples_extractor_model_field.disabled is True
    assert ed.cross_doc_threshold_field.disabled is True

    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, triples=True, cross_doc=True),
        entities=EntityConfig(extractors=["llm"]),
    )
    ed._refresh_availability()
    assert ed.chunk_max_tokens_field.disabled is False
    assert ed.triples_extractor_model_field.disabled is False
    assert ed.cross_doc_threshold_field.disabled is False


def test_freeze_locks_embedder_and_graph_layers_only(fake_app):
    """When the corpus is frozen, the embedder + the five L6–L10 section
    wrappers are disabled; Labels / Resolve / the rest of Chunks-and-embeddings
    (L5) stay editable (selective freeze)."""
    ed = _ready_editor(fake_app)
    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractors=["llm"]),
        frozen=True,
    )
    ed._baseline_config = ed._corpus_config.model_copy(deep=True)
    ed.build()  # populates _frozen_wraps + applies the lock at the end

    assert set(ed._frozen_wraps) == {
        "entities",
        "ontology",
        "triples",
        "cross_doc",
        "cross_doc_xrefs",
    }
    for wrap in ed._frozen_wraps.values():
        assert wrap.disabled is True
    assert ed.embedding_model_field.disabled is True
    # Per-batch controls are NOT locked by freeze.
    assert ed.chunk_max_tokens_field.disabled is False
    assert ed.enable_pdf_ocr_checkbox.disabled is False
    assert ed.openalex_checkbox.disabled is False


def test_unfreeze_releases_the_lock(fake_app):
    """Unfrozen: the wrappers + embedder are not force-disabled."""
    ed = _ready_editor(fake_app)
    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractors=["llm"]),
        frozen=False,
    )
    ed._baseline_config = ed._corpus_config.model_copy(deep=True)
    ed.build()

    for wrap in ed._frozen_wraps.values():
        assert wrap.disabled is False
    assert ed.embedding_model_field.disabled is False


def test_validate_pending_config_returns_config_without_writing(fake_app, monkeypatch):
    """validate_pending_config validates the draft and returns it, but must NOT
    write corpus.toml. The write is deferred to commit_config (dialog confirm)."""
    ed = _ready_editor(fake_app)
    wrote: list = []
    monkeypatch.setattr(
        "knowledge_agent.gui.library.corpus_config_editor._write_corpus_toml",
        lambda *a, **k: wrote.append(a),
    )

    config, error = ed.validate_pending_config()

    assert error is None
    assert config is not None
    assert wrote == []  # validation never touches disk


def test_commit_config_writes_and_adopts_baseline(fake_app, monkeypatch):
    """commit_config writes the validated config to disk and adopts it as the
    new in-memory baseline."""
    ed = _ready_editor(fake_app)
    wrote: list = []
    monkeypatch.setattr(
        "knowledge_agent.gui.library.corpus_config_editor._write_corpus_toml",
        lambda _path, cfg: wrote.append(cfg),
    )
    monkeypatch.setattr(
        ed, "_find_active_entry", lambda _name: SimpleNamespace(corpus_config_path="corpus.toml")
    )
    # Silence the UI-refresh / draft side effects (isolate the write path).
    for m in (
        "_clear_draft",
        "_refresh_subtitles",
        "_refresh_availability",
        "_refresh_dirty_indicator",
        "_notify_draft_changed",
    ):
        monkeypatch.setattr(ed, m, lambda *a, **k: None)

    config, _ = ed.validate_pending_config()

    assert ed.commit_config(config) is None
    assert wrote == [config]  # written to disk
    assert ed._corpus_config is config  # adopted as current
