"""Tests for the Library → Select corpus card (`SelectDatasetTab`).

Exercise the pure sync populators (extractor / ingest-settings / models /
empty-state) directly — no Flet render, no DB, no event loop. The async
count fetch (`_reload_counts`) is guarded off without a running loop, so
these never touch LanceDB/Neo4j.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from knowledge_agent.gui.config_store import CorpusEntry
from knowledge_agent.gui.library.select_dataset import (
    SelectDatasetTab,
    _onoff,
)

_MOD = "knowledge_agent.gui.library.select_dataset"


def _tab(fake_app) -> SelectDatasetTab:
    return SelectDatasetTab(fake_app)


# ---- _onoff ----


def test_onoff():
    assert _onoff(True) == "on"
    assert _onoff(False) == "off"


# ---- extractor (multi-extractor summary) ----


def test_populate_extractor_multi(fake_app):
    tab = _tab(fake_app)
    cfg = SimpleNamespace(
        entities=SimpleNamespace(
            extractors=["llm", "gliner"],
            entity_types_mode="add",
            entity_types=["Gene", "Disease"],
        )
    )
    tab._populate_extractor(cfg)
    v = tab.info_extractor.value
    assert "llm, gliner" in v
    assert "mode: add" in v
    assert "Gene, Disease" in v


def test_populate_extractor_legacy_singular(fake_app):
    tab = _tab(fake_app)
    # Old config: no `extractors` list, only the singular `extractor`.
    ents = SimpleNamespace(entity_types=[], entity_types_mode="replace")
    ents.extractor = "llm"
    tab._populate_extractor(SimpleNamespace(entities=ents))
    assert "llm" in tab.info_extractor.value


def test_populate_extractor_not_configured(fake_app):
    tab = _tab(fake_app)
    tab._populate_extractor(SimpleNamespace(entities=None))
    assert tab.info_extractor.value == "not configured"


# ---- ingest settings ----


def test_populate_ingest_settings(fake_app):
    tab = _tab(fake_app)
    cfg = SimpleNamespace(
        chunker_strategy="hybrid",
        chunk_max_tokens=512,
        merge_peers=True,
        enable_pdf_ocr=False,
        enable_image_ocr=True,
        extract_figures=True,
        images_scale=2.0,
        min_figure_bytes=2048,
    )
    tab._populate_ingest_settings(cfg)
    assert "hybrid" in tab.info_chunking.value
    assert "512" in tab.info_chunking.value
    assert "merge_peers on" in tab.info_chunking.value
    assert "pdf off" in tab.info_ocr.value
    assert "image on" in tab.info_ocr.value
    assert "extract on" in tab.info_figures.value
    assert "2048" in tab.info_figures.value


def test_populate_ingest_settings_none(fake_app):
    tab = _tab(fake_app)
    tab._populate_ingest_settings(None)
    assert tab.info_chunking.value == "—"
    assert tab.info_ocr.value == "—"
    assert tab.info_figures.value == "—"


# ---- models (global settings) ----


def test_populate_models(fake_app, monkeypatch):
    fake_settings = SimpleNamespace(
        embedding_provider="huggingface",
        embedding_model="bge-small",
        embedding_dims=384,
        llm_provider="anthropic",
    )
    monkeypatch.setattr("knowledge_agent.config.get_settings", lambda: fake_settings)
    tab = _tab(fake_app)
    tab._populate_models()
    assert "huggingface" in tab.info_embedder.value
    assert "bge-small" in tab.info_embedder.value
    assert "384-dim" in tab.info_embedder.value
    assert tab.info_llm.value == "anthropic"


def test_populate_models_failure_shows_dash(fake_app, monkeypatch):
    def boom():
        raise RuntimeError("no settings")

    monkeypatch.setattr("knowledge_agent.config.get_settings", boom)
    tab = _tab(fake_app)
    tab._populate_models()
    assert tab.info_embedder.value == "—"
    assert tab.info_llm.value == "—"


# ---- empty state + schedule guard ----


def test_info_empty_state_blanks_new_controls(fake_app):
    tab = _tab(fake_app)
    tab._info_empty_state()
    for ctrl in (
        tab.info_breakdown,
        tab.info_chunking,
        tab.info_ocr,
        tab.info_figures,
        tab.info_embedder,
        tab.info_llm,
    ):
        assert ctrl.value == "—"
    assert tab.info_counts.value == "docs: — · chunks: — · mentions: —"


def test_schedule_counts_noop_without_loop(fake_app):
    """No running loop (unit test) → schedule must not spawn or raise."""
    tab = _tab(fake_app)
    tab._schedule_counts()
    assert tab._bg_tasks == set()


def test_refresh_after_ingest_is_callable(fake_app):
    """Cross-tab hook the Ingest sub-tab calls after a successful op."""
    tab = _tab(fake_app)
    assert callable(tab.refresh_after_ingest)


# ---- Relocate (repoint a moved corpus) ----


def _corpus(name: str, folder) -> CorpusEntry:
    return CorpusEntry(
        name=name,
        neo4j_uri="neo4j://h:7687",
        lancedb_path=folder / "lancedb",
        corpus_config_path=folder / "corpus.toml",
    )


def test_relocate_corpus_rewrites_paths(fake_app, tmp_path):
    new = tmp_path / "moved"
    new.mkdir()
    (new / "corpus.toml").write_text("x", encoding="utf-8")
    tab = _tab(fake_app)
    fake_app.gui_config.corpora = [_corpus("c1", tmp_path / "old")]
    fake_app.gui_config.active_corpus_name = "c1"

    with (
        patch(f"{_MOD}.save_config"),
        patch(f"{_MOD}.apply_connection_to_env"),
        patch(f"{_MOD}.reset_after_key_change"),
        patch.object(tab, "_sync_picker"),
        patch.object(tab, "_populate_info_card"),
    ):
        tab._relocate_corpus("c1", new)

    entry = fake_app.gui_config.corpora[0]
    assert entry.corpus_config_path == new / "corpus.toml"
    assert entry.lancedb_path == new / "lancedb"
    # active corpus → top-level mirror updated too
    assert fake_app.gui_config.corpus_config_path == new / "corpus.toml"
    assert "relocated" in tab.status.value


def test_relocate_corpus_rejects_folder_without_corpus_toml(fake_app, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()  # no corpus.toml here
    tab = _tab(fake_app)
    fake_app.gui_config.corpora = [_corpus("c1", tmp_path / "old")]
    fake_app.gui_config.active_corpus_name = "c1"

    with patch(f"{_MOD}.save_config") as save:
        tab._relocate_corpus("c1", empty)

    save.assert_not_called()
    # entry untouched
    assert fake_app.gui_config.corpora[0].corpus_config_path == tmp_path / "old" / "corpus.toml"
    assert "corpus.toml" in tab.status.value
