"""Tests for the Evaluation GUI shared helpers (`gui/evaluation/_common.py`).

The writer (Run tab) and the reader tabs must resolve the SAME per-corpus
ledger; these lock the single derivation so they can't drift apart.
"""

from __future__ import annotations

from pathlib import Path

from knowledge_agent.gui.evaluation._common import (
    active_corpus_config_path,
    active_eval_ledger,
    active_output_dir,
)


def test_active_corpus_config_path_set(fake_app):
    fake_app.gui_config.corpus_config_path = "/data/c/corpus.toml"
    assert active_corpus_config_path(fake_app) == Path("/data/c/corpus.toml")


def test_active_corpus_config_path_none(fake_app):
    fake_app.gui_config.corpus_config_path = None
    assert active_corpus_config_path(fake_app) is None


def test_active_eval_ledger_uses_corpus_folder(fake_app, tmp_path):
    """Ledger lands in <corpus folder>/eval_output — beside its lancedb."""
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"
    ledger = active_eval_ledger(fake_app)
    assert ledger.db_path == tmp_path / "eval_output" / "eval_ledger.db"


def test_active_eval_ledger_falls_back_to_cwd(fake_app, tmp_path, monkeypatch):
    """No active corpus → CWD/eval_output (chdir keeps the write in tmp)."""
    fake_app.gui_config.corpus_config_path = None
    monkeypatch.chdir(tmp_path)
    ledger = active_eval_ledger(fake_app)
    assert ledger.db_path == tmp_path / "eval_output" / "eval_ledger.db"


def test_active_output_dir_uses_corpus_folder(fake_app, tmp_path):
    """Output dir = <corpus folder>/eval_output, and resolving it for display
    is READ-ONLY — unlike an EvalLedger, it must NOT create the folder."""
    fake_app.gui_config.corpus_config_path = tmp_path / "corpus.toml"
    out = active_output_dir(fake_app)
    assert out == tmp_path / "eval_output"
    assert not out.exists()  # no disk side effect from a read-only display


def test_active_output_dir_falls_back_to_cwd(fake_app, tmp_path, monkeypatch):
    fake_app.gui_config.corpus_config_path = None
    monkeypatch.chdir(tmp_path)
    assert active_output_dir(fake_app) == tmp_path / "eval_output"
