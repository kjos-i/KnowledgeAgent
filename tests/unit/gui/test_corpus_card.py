"""Tests for the read-only corpus card (`gui.library.corpus_card`) shown in
the global Manage-corpora dialog. Construction + fail-soft only — no Flet
render, no real backend."""

from __future__ import annotations

from types import SimpleNamespace

import flet as ft

from knowledge_agent.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.gui.library.corpus_card import build_corpus_card


def _entry(tmp_path):
    return SimpleNamespace(
        name="c1",
        neo4j_uri="neo4j://h:7687",
        neo4j_user="neo4j",
        lancedb_path=tmp_path / "lancedb",
        corpus_config_path=tmp_path / "corpus.toml",
    )


def test_no_active_corpus_returns_hint(fake_app):
    fake_app.gui_config.active_corpus_name = None
    fake_app.gui_config.corpora = []
    assert isinstance(build_corpus_card(fake_app), ft.Text)


def test_active_corpus_returns_populated_column(fake_app, tmp_path, monkeypatch):
    fake_app.gui_config.active_corpus_name = "c1"
    fake_app.gui_config.corpora = [_entry(tmp_path)]
    cfg = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractors=["llm"]),
    )
    monkeypatch.setattr(
        "knowledge_agent.gui.library.corpus_card.load_corpus_config",
        lambda _p: cfg,
    )
    card = build_corpus_card(fake_app)
    assert isinstance(card, ft.Column)
    assert card.controls  # non-empty (name + rows)


def test_unreadable_config_still_builds_connection_only(fake_app, tmp_path, monkeypatch):
    """corpus.toml unreadable → the card still builds (connection/paths +
    an amber note) rather than raising."""
    fake_app.gui_config.active_corpus_name = "c1"
    fake_app.gui_config.corpora = [_entry(tmp_path)]

    def _boom(_p):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(
        "knowledge_agent.gui.library.corpus_card.load_corpus_config",
        _boom,
    )
    assert isinstance(build_corpus_card(fake_app), ft.Column)
