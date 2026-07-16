"""Handler tests for the CorpusConfigEditor's per-corpus Embedding section.

One composite provider:model picker (options span installed embedders); the
provider comes with the chosen model and is staged to the in-memory
CorpusConfig alongside the DERIVED `embedding_dims`. A choice that changes the
provider to one whose vector dim would strand existing chunks hard-confirms
first (the dim guard the deleted Settings tab used to own). (The global Voyage
request-rate cap moved to the LLMs tab — see test_llm_tab.)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import flet as ft

from knowledge_agent.gui.library.corpus_config_editor import CorpusConfigEditor
from knowledge_agent.kg.corpus_config import CorpusConfig

_EDITOR = "knowledge_agent.gui.library.corpus_config_editor"


def _editor(fake_app, cfg: CorpusConfig | None = None) -> CorpusConfigEditor:
    cfg = cfg or CorpusConfig()  # voyage / voyage-multimodal-3 / 1024
    fake_app.gui_config.active_corpus_name = "c1"
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"
    ed._corpus_config = cfg
    ed._baseline_config = cfg
    return ed


def _openai_cfg() -> CorpusConfig:
    return CorpusConfig(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dims=1536,
    )


def test_populate_sets_composite_value_and_spans_installed(fake_app):
    """The picker value is the corpus's provider:model composite, and its
    options come from `embedding_model_options` (installed embedders only)."""
    ed = _editor(fake_app, _openai_cfg())
    opts = [
        ft.DropdownOption(
            key="openai:text-embedding-3-small", text="openai — text-embedding-3-small"
        ),
        ft.DropdownOption(
            key="openai:text-embedding-3-large", text="openai — text-embedding-3-large"
        ),
    ]
    with patch(f"{_EDITOR}.embedding_model_options", return_value=opts):
        ed._populate_controls()
    assert ed.embedding_model_field.value == "openai:text-embedding-3-small"
    assert "openai:text-embedding-3-large" in [o.key for o in ed.embedding_model_field.options]


def test_model_change_same_provider_stages_model_and_dims(fake_app):
    """Picking a different model of the SAME provider just stages the model +
    re-derives dims — no destructive-switch confirm."""
    ed = _editor(fake_app, _openai_cfg())
    ed.embedding_model_field.value = "openai:text-embedding-3-large"
    ed._on_embedding_model_selected(MagicMock())
    assert ed._corpus_config.embedding_provider == "openai"  # unchanged
    assert ed._corpus_config.embedding_model == "text-embedding-3-large"
    assert ed._corpus_config.embedding_dims == 1536  # openai dim, re-derived
    fake_app.page.show_dialog.assert_not_called()


def test_provider_change_no_mismatch_stages_provider_model_dims(fake_app):
    """A composite pick on a different provider (no dim mismatch) stages the
    provider, model, and derived dims together."""
    ed = _editor(fake_app)  # voyage default
    ed.embedding_model_field.value = "openai:text-embedding-3-small"
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=MagicMock(dim_mismatch=False)):
        ed._on_embedding_model_selected(MagicMock())
    assert ed._corpus_config.embedding_provider == "openai"
    assert ed._corpus_config.embedding_model == "text-embedding-3-small"
    assert ed._corpus_config.embedding_dims == 1536


def test_provider_change_to_hf_derives_model_dim(fake_app):
    ed = _editor(fake_app)
    ed.embedding_model_field.value = "huggingface:BAAI/bge-m3"
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=MagicMock(dim_mismatch=False)):
        ed._on_embedding_model_selected(MagicMock())
    assert ed._corpus_config.embedding_provider == "huggingface"
    assert ed._corpus_config.embedding_model == "BAAI/bge-m3"
    assert ed._corpus_config.embedding_dims == 1024  # bge-m3's own dim (per-model)


def test_provider_change_dim_mismatch_confirms_before_applying(fake_app):
    """A destructive switch (existing chunks at a different dim) shows a confirm
    dialog and does NOT mutate the corpus until the user confirms."""
    ed = _editor(fake_app)  # voyage
    ed.embedding_model_field.value = "openai:text-embedding-3-small"
    plan = MagicMock(
        dim_mismatch=True, summary="DESTRUCTIVE switch: voyage (1024) -> openai (1536)"
    )
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=plan):
        ed._on_embedding_model_selected(MagicMock())
    fake_app.page.show_dialog.assert_called_once()
    assert ed._corpus_config.embedding_provider == "voyage"  # unchanged pending confirm
    assert ed._corpus_config.embedding_model == "voyage-multimodal-3"  # unchanged
