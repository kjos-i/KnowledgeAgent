"""Build-path smoke tests for `CorpusConfigEditor`.

The section handlers are covered elsewhere; these pin the `build()` render
itself — that the panel assembles (info-icon headers + all 8 layer
sections) without a real Flet runtime, and that every section header
registers an `(i)` with the global teaching-mode toggle. `build()` had no
coverage before the info-icon retrofit.
"""

from __future__ import annotations

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
