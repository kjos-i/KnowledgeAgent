"""Handler tests for the CorpusConfigEditor's per-corpus Embedding section.

Provider + model are staged to the in-memory CorpusConfig (written to
corpus.toml at Ingest); `embedding_dims` is DERIVED from them. The Voyage
rate is GLOBAL — written straight to GuiConfig, not staged. A provider
switch that would strand existing chunks at a mismatched vector dim
hard-confirms first (the dim guard the deleted Settings tab used to own).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def test_populate_loads_provider_and_model(fake_app):
    ed = _editor(fake_app, _openai_cfg())
    ed._populate_controls()
    assert ed.embedding_provider_dropdown.value == "openai"
    assert ed.embedding_model_field.value == "text-embedding-3-small"
    # Model dropdown menu tracks the active provider.
    assert "text-embedding-3-large" in [o.key for o in ed.embedding_model_field.options]


def test_model_blur_stages_model_and_derives_dims(fake_app):
    ed = _editor(fake_app, _openai_cfg())
    ed.embedding_model_field.value = "text-embedding-3-large"
    ed._on_embedding_model_blur(MagicMock())
    assert ed._corpus_config.embedding_model == "text-embedding-3-large"
    assert ed._corpus_config.embedding_dims == 1536  # openai dim, re-derived


def test_provider_change_no_data_stages_provider_model_dims(fake_app):
    ed = _editor(fake_app)  # voyage default
    ed.embedding_provider_dropdown.value = "openai"
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=MagicMock(dim_mismatch=False)):
        ed._on_embedding_provider_changed(MagicMock())
    assert ed._corpus_config.embedding_provider == "openai"
    assert ed._corpus_config.embedding_model == "text-embedding-3-small"  # openai default
    assert ed._corpus_config.embedding_dims == 1536
    assert "text-embedding-3-small" in [o.key for o in ed.embedding_model_field.options]


def test_provider_change_to_hf_derives_model_and_dims(fake_app):
    ed = _editor(fake_app)
    ed.embedding_provider_dropdown.value = "huggingface"
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=MagicMock(dim_mismatch=False)):
        ed._on_embedding_provider_changed(MagicMock())
    assert ed._corpus_config.embedding_provider == "huggingface"
    assert ed._corpus_config.embedding_model == "BAAI/bge-m3"
    assert ed._corpus_config.embedding_dims == 1024  # bge-m3's own dim, not a provider default


def test_provider_change_dim_mismatch_confirms_before_applying(fake_app):
    """A destructive switch (existing chunks at a different dim) shows a
    confirm dialog and does NOT mutate the corpus until the user confirms."""
    ed = _editor(fake_app)  # voyage
    ed.embedding_provider_dropdown.value = "openai"
    plan = MagicMock(
        dim_mismatch=True, summary="DESTRUCTIVE switch: voyage (1024) -> openai (1536)"
    )
    with patch(f"{_EDITOR}.switch_embedder_plan", return_value=plan):
        ed._on_embedding_provider_changed(MagicMock())
    fake_app.page.show_dialog.assert_called_once()
    assert ed._corpus_config.embedding_provider == "voyage"  # unchanged pending confirm


def test_voyage_rate_row_visible_only_for_voyage(fake_app):
    ed_voyage = _editor(fake_app)
    ed_voyage._refresh_voyage_rate_visibility()
    assert ed_voyage.voyage_rate_row.visible is True

    ed_openai = _editor(fake_app, _openai_cfg())
    ed_openai._refresh_voyage_rate_visibility()
    assert ed_openai.voyage_rate_row.visible is False


def test_voyage_rate_blur_writes_global_gui_config(fake_app):
    """The Voyage rate is global — it writes GuiConfig + bridges JUST the
    rate (never the per-corpus provider/model)."""
    ed = _editor(fake_app)
    ed.voyage_rate_field.value = "5"
    with (
        patch(f"{_EDITOR}.save_config") as save,
        patch(f"{_EDITOR}.apply_voyage_rate_to_env") as rate_bridge,
        patch(f"{_EDITOR}.reset_after_key_change"),
    ):
        ed._on_voyage_rate_blur(MagicMock())
    assert fake_app.gui_config.voyage_requests_per_second == 5.0
    save.assert_called_once()
    rate_bridge.assert_called_once()
