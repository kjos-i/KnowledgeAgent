"""Tests for `GuiApp._invoke_state_for_input_mode` — the input-mode ->
graph invoke-state mapping (Conversational / Direct query / Direct Cypher).

The helper is pure: given the mode + user text (+ the router's distilled
query in conversational mode) it returns the invoke-state overrides. This
is where the input-mode radio actually changes retrieval behaviour.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from knowledge_agent.gui.app import GuiApp

_CORPUS = object()  # opaque corpus_config sentinel; the helper just forwards it


def _app() -> GuiApp:
    return GuiApp(page=MagicMock())


def test_conversational_uses_router_distilled_query():
    app = _app()
    st = app._invoke_state_for_input_mode("conversational", "raw", _CORPUS, "distilled q")
    assert st["query"] == "distilled q"
    assert st["skip_query_builder"] is False
    assert st["retrieval_mode"] == app.gui_config.retrieval_mode
    assert st["corpus_config"] is _CORPUS
    assert "user_cypher" not in st


def test_conversational_falls_back_to_raw_text_when_no_search_query():
    app = _app()
    st = app._invoke_state_for_input_mode("conversational", "raw text", _CORPUS, None)
    assert st["query"] == "raw text"


def test_direct_query_skips_builder_and_forces_vector():
    app = _app()
    st = app._invoke_state_for_input_mode("direct_query", "find CRISPR", _CORPUS, None)
    assert st["query"] == "find CRISPR"
    assert st["skip_query_builder"] is True
    assert st["retrieval_mode"] == "lancedb_only"
    assert "user_cypher" not in st


def test_direct_cypher_passes_user_cypher_and_forces_kg():
    app = _app()
    cypher = "MATCH (n) RETURN n LIMIT 5"
    st = app._invoke_state_for_input_mode("direct_cypher", cypher, _CORPUS, None)
    assert st["query"] == cypher
    assert st["user_cypher"] == cypher
    assert st["retrieval_mode"] == "neo4j_only"
