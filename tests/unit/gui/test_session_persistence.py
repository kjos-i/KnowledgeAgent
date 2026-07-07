"""Wiring tests for per-corpus "resume where you left off" persistence.

Exercises the three views that touch the session sidecar, driving their
state handlers directly (fake app + page from `conftest.py`, no Flet render):

  * `CorpusConfigEditor` — persists the unsaved draft on mutate, restores it
    on load, clears it on discard / save.
  * `IngestTab` — restores the folder / file pickers per corpus (once), and
    persists a picked path.
  * `SelectDatasetTab` — shows the "changed since last ingest" section from
    the persisted draft, hidden when there's nothing pending.
"""

from __future__ import annotations

from types import SimpleNamespace

from knowledge_agent.gui.library.corpus_config_editor import CorpusConfigEditor
from knowledge_agent.gui.library.ingest import IngestTab
from knowledge_agent.gui.library.select_dataset import SelectDatasetTab
from knowledge_agent.gui.library.session_state import (
    load_session,
    update_draft,
    update_last_file,
    update_last_folder,
)
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)


def _cfg(**overrides) -> CorpusConfig:
    base = {
        "allowed_types": ["Paper"],
        "layers": LayerFlags(chunks=True, entities=True),
        "entities": EntityConfig(extractors=["llm"]),
    }
    base.update(overrides)
    return CorpusConfig(**base)


def _entry(name, toml_path):
    # Stand-in for a CorpusEntry — the editor / select only read `.name`
    # and `.corpus_config_path`.
    return SimpleNamespace(name=name, corpus_config_path=toml_path)


# ---- CorpusConfigEditor: draft persist / restore / clear --------------------


def test_editor_persists_draft_to_sidecar_when_dirty(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"
    ed._baseline_config = _cfg(chunk_max_tokens=512)
    ed._corpus_config = ed._baseline_config.model_copy(update={"chunk_max_tokens": 999})

    ed._persist_draft()

    draft = load_session(toml).draft_config
    assert draft is not None
    assert draft["chunk_max_tokens"] == 999


def test_editor_persist_clears_draft_when_clean(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    update_draft(toml, {"stale": 1})  # a leftover draft on disk
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"
    ed._baseline_config = _cfg()
    ed._corpus_config = ed._baseline_config.model_copy(deep=True)  # == baseline → clean

    ed._persist_draft()

    assert load_session(toml).draft_config is None


def test_editor_persist_fires_on_draft_changed(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"
    ed._baseline_config = _cfg()
    ed._corpus_config = _cfg()
    calls: list[int] = []
    ed.on_draft_changed = lambda: calls.append(1)

    ed._persist_draft()

    assert calls == [1]


def test_editor_restore_draft_overrides_corpus_config(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    update_draft(toml, _cfg(chunk_max_tokens=777).model_dump(mode="json"))
    ed = CorpusConfigEditor(fake_app)
    ed._baseline_config = _cfg(chunk_max_tokens=512)
    ed._corpus_config = _cfg(chunk_max_tokens=512)

    ed._restore_draft(toml)

    assert ed._corpus_config.chunk_max_tokens == 777


def test_editor_restore_drops_incompatible_draft(fake_app, tmp_path):
    """A draft that no longer validates (schema drift) is discarded and its
    sidecar entry cleared — the editor falls back to the baseline."""
    toml = tmp_path / "corpus.toml"
    update_draft(toml, {"unknown_field": 1})  # violates extra="forbid"
    ed = CorpusConfigEditor(fake_app)
    ed._baseline_config = _cfg(chunk_max_tokens=512)
    ed._corpus_config = _cfg(chunk_max_tokens=512)

    ed._restore_draft(toml)

    assert ed._corpus_config.chunk_max_tokens == 512  # unchanged (baseline)
    assert load_session(toml).draft_config is None  # bad draft cleared


def test_editor_clear_draft_removes_it(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    update_draft(toml, {"chunk_max_tokens": 800})
    ed = CorpusConfigEditor(fake_app)
    ed._loaded_for_corpus = "c1"

    ed._clear_draft()

    assert load_session(toml).draft_config is None


# ---- IngestTab: folder / file picker persistence ----------------------------


def test_ingest_restore_session_paths_prefills_pickers(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    update_last_folder(toml, "/docs/in")
    update_last_file(toml, "/docs/x.pdf")
    fake_app.gui_config.corpus_config_path = toml
    tab = IngestTab(fake_app)

    tab._restore_session_paths("c1")

    assert tab.folder_field.value == "/docs/in"
    assert tab.file_field.value == "/docs/x.pdf"


def test_ingest_restore_is_once_per_corpus(fake_app, tmp_path):
    """A re-render for the SAME corpus must not clobber a path the user is
    mid-way through editing."""
    toml = tmp_path / "corpus.toml"
    update_last_folder(toml, "/docs/in")
    fake_app.gui_config.corpus_config_path = toml
    tab = IngestTab(fake_app)

    tab._restore_session_paths("c1")
    tab.folder_field.value = "/docs/edited"
    tab._restore_session_paths("c1")  # same corpus → no reload

    assert tab.folder_field.value == "/docs/edited"


def test_ingest_persist_folder_writes_sidecar(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpus_config_path = toml
    tab = IngestTab(fake_app)

    tab._persist_folder("/docs/pick")

    assert load_session(toml).last_folder == "/docs/pick"


def test_ingest_persist_file_writes_sidecar(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.corpus_config_path = toml
    tab = IngestTab(fake_app)

    tab._persist_file("/docs/one.pdf")

    assert load_session(toml).last_file == "/docs/one.pdf"


# ---- SelectDatasetTab: "changed since last ingest" section ------------------


def test_select_pending_section_shown_when_draft_differs(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    update_draft(toml, _cfg(chunk_max_tokens=900).model_dump(mode="json"))
    fake_app.gui_config.active_corpus_name = "c1"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    tab = SelectDatasetTab(fake_app)
    tab._active_cfg = _cfg(chunk_max_tokens=512)

    tab._populate_pending_changes()

    assert tab.info_pending_section.visible is True
    assert "chunk_max_tokens" in tab.info_pending.value


def test_select_pending_section_hidden_without_draft(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    fake_app.gui_config.active_corpus_name = "c1"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    tab = SelectDatasetTab(fake_app)
    tab._active_cfg = _cfg()

    tab._populate_pending_changes()

    assert tab.info_pending_section.visible is False


def test_select_pending_section_hidden_when_draft_equals_baseline(fake_app, tmp_path):
    toml = tmp_path / "corpus.toml"
    cfg = _cfg(chunk_max_tokens=512)
    update_draft(toml, cfg.model_dump(mode="json"))
    fake_app.gui_config.active_corpus_name = "c1"
    fake_app.gui_config.corpora = [_entry("c1", toml)]
    tab = SelectDatasetTab(fake_app)
    tab._active_cfg = cfg

    tab._populate_pending_changes()

    assert tab.info_pending_section.visible is False
