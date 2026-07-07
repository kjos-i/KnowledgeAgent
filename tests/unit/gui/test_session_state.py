"""Tests for the per-corpus session sidecar (`gui.library.session_state`).

Pure file I/O against `tmp_path` — no Flet, no real corpus. Verifies the
draft / folder / file round-trip, the fail-soft reads (missing / corrupt),
and that an emptied session removes the sidecar rather than leaving a husk.
"""

from __future__ import annotations

import json

from knowledge_agent.gui.library.session_state import (
    CorpusSession,
    load_session,
    save_session,
    sidecar_path,
    update_draft,
    update_last_file,
    update_last_folder,
)


def _toml(tmp_path):
    # The sidecar lives beside corpus.toml; the toml itself need not exist.
    return tmp_path / "corpus.toml"


def test_sidecar_path_is_beside_corpus_toml(tmp_path):
    assert sidecar_path(_toml(tmp_path)) == tmp_path / ".ka_session.json"


def test_load_missing_sidecar_returns_empty(tmp_path):
    session = load_session(_toml(tmp_path))
    assert session == CorpusSession()
    assert session.is_empty()


def test_save_then_load_round_trips(tmp_path):
    toml = _toml(tmp_path)
    save_session(
        toml,
        CorpusSession(draft_config={"a": 1}, last_folder="/docs", last_file="/docs/x.pdf"),
    )
    got = load_session(toml)
    assert got.draft_config == {"a": 1}
    assert got.last_folder == "/docs"
    assert got.last_file == "/docs/x.pdf"


def test_save_empty_session_removes_sidecar(tmp_path):
    toml = _toml(tmp_path)
    save_session(toml, CorpusSession(last_folder="/docs"))
    assert sidecar_path(toml).exists()
    # Now empty → the file is removed, not left as an empty husk.
    save_session(toml, CorpusSession())
    assert not sidecar_path(toml).exists()


def test_load_corrupt_json_is_failsoft(tmp_path):
    toml = _toml(tmp_path)
    sidecar_path(toml).write_text("{not valid json", encoding="utf-8")
    assert load_session(toml) == CorpusSession()


def test_load_non_dict_json_is_failsoft(tmp_path):
    toml = _toml(tmp_path)
    sidecar_path(toml).write_text("[1, 2, 3]", encoding="utf-8")
    assert load_session(toml) == CorpusSession()


def test_load_ignores_non_dict_draft(tmp_path):
    """A malformed draft value must not corrupt the whole read — the paths
    still load, the draft just falls back to None."""
    toml = _toml(tmp_path)
    sidecar_path(toml).write_text(
        json.dumps({"draft_config": "oops", "last_folder": "/d"}),
        encoding="utf-8",
    )
    got = load_session(toml)
    assert got.draft_config is None
    assert got.last_folder == "/d"


def test_update_draft_sets_and_clears_preserving_paths(tmp_path):
    toml = _toml(tmp_path)
    update_last_folder(toml, "/docs")
    update_draft(toml, {"chunk_max_tokens": 800})
    got = load_session(toml)
    assert got.draft_config == {"chunk_max_tokens": 800}
    assert got.last_folder == "/docs"  # preserved across the draft write
    # Clearing the draft leaves the folder intact.
    update_draft(toml, None)
    got = load_session(toml)
    assert got.draft_config is None
    assert got.last_folder == "/docs"


def test_update_draft_empty_dict_is_cleared(tmp_path):
    """An empty draft dict is treated as 'no draft' (no pending changes)."""
    toml = _toml(tmp_path)
    update_draft(toml, {})
    assert load_session(toml).draft_config is None


def test_update_last_folder_blank_clears(tmp_path):
    toml = _toml(tmp_path)
    update_last_folder(toml, "/docs")
    update_last_folder(toml, "   ")  # whitespace normalises to None
    assert load_session(toml).last_folder is None


def test_update_last_file_sets_and_preserves_draft(tmp_path):
    toml = _toml(tmp_path)
    update_draft(toml, {"a": 1})
    update_last_file(toml, "/docs/x.pdf")
    got = load_session(toml)
    assert got.last_file == "/docs/x.pdf"
    assert got.draft_config == {"a": 1}  # preserved


def test_clearing_the_last_field_removes_sidecar(tmp_path):
    toml = _toml(tmp_path)
    update_last_folder(toml, "/docs")
    update_last_folder(toml, None)
    assert not sidecar_path(toml).exists()
