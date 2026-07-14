"""Tests for `GuiApp._invoke_state_for_input_mode` — the input-mode ->
graph invoke-state mapping (Conversational / Refined / Direct query /
Direct Cypher).

The helper is pure: given the mode + user text (+ the router's distilled
query in conversational mode) it returns the invoke-state overrides. This
is where the input-mode radio actually changes retrieval behaviour. The
query-mode -> knob mapping is shared with the eval Dataset form via
`retrieval_form.query_mode_to_knobs`, so the store is the user's own
retrieval_mode for every mode EXCEPT direct_cypher (pinned to neo4j).
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
    assert st["user_cypher"] is None  # no Cypher in conversational


def test_conversational_falls_back_to_raw_text_when_no_search_query():
    app = _app()
    st = app._invoke_state_for_input_mode("conversational", "raw text", _CORPUS, None)
    assert st["query"] == "raw text"


def test_direct_query_skips_builder_and_keeps_user_store():
    # Direct query skips the query-builder but leaves the store INDEPENDENT —
    # it no longer force-selects lancedb; the user's retrieval_mode stands.
    app = _app()
    st = app._invoke_state_for_input_mode("direct_query", "find CRISPR", _CORPUS, None)
    assert st["query"] == "find CRISPR"
    assert st["skip_query_builder"] is True
    assert st["retrieval_mode"] == app.gui_config.retrieval_mode
    assert st["user_cypher"] is None


def test_refined_runs_builder_and_keeps_user_store():
    # Refined = the 4th chat mode: the query-builder runs on the raw text (no
    # router distillation), store = the user's retrieval_mode. Same knobs as
    # conversational, but the query is the raw text rather than the router's.
    app = _app()
    st = app._invoke_state_for_input_mode("refined", "find CRISPR", _CORPUS, None)
    assert st["query"] == "find CRISPR"
    assert st["skip_query_builder"] is False
    assert st["retrieval_mode"] == app.gui_config.retrieval_mode
    assert st["user_cypher"] is None


def test_direct_cypher_passes_user_cypher_and_forces_kg():
    app = _app()
    cypher = "MATCH (n) RETURN n LIMIT 5"
    st = app._invoke_state_for_input_mode("direct_cypher", cypher, _CORPUS, None)
    assert st["query"] == cypher
    assert st["user_cypher"] == cypher
    assert st["retrieval_mode"] == "neo4j_only"


# ---- _retrieval_snapshot (feeds the eval "capture from search / chat" flow) ----


def test_retrieval_snapshot_mirrors_gui_config():
    """The snapshot pins EVERY retrieval knob from GuiConfig (what the search ran
    under), so a captured eval case reproduces the search instead of falling back
    to form defaults."""
    from knowledge_agent.evaluation.models import RetrievalSettings

    app = _app()
    app.gui_config.retrieval_mode = "neo4j_only"
    app.gui_config.lancedb_search_mode = "vector"
    app.gui_config.top_k = 11
    app.gui_config.num_candidates = 88
    app.gui_config.rrf_rank_constant = 42
    app.gui_config.use_mmr = True
    app.gui_config.mmr_lambda = 0.7
    app.gui_config.kg_max_rows = 33
    app.gui_config.direct_retrieve = True
    app.gui_config.input_mode = "refined"  # query-builder runs → skip=False
    snap = app._retrieval_snapshot()
    assert isinstance(snap, RetrievalSettings)
    assert snap.retrieval_mode == "neo4j_only"
    assert snap.lancedb_search_mode == "vector"
    assert snap.top_k == 11
    assert snap.num_candidates == 88
    assert snap.rrf_rank_constant == 42
    assert snap.use_mmr is True
    assert snap.mmr_lambda == 0.7
    assert snap.kg_max_rows == 33
    assert snap.direct_retrieval is True
    assert snap.skip_query_builder is False


def test_retrieval_snapshot_direct_query_records_skip_builder():
    app = _app()
    app.gui_config.input_mode = "direct_query"
    assert app._retrieval_snapshot().skip_query_builder is True


def test_retrieval_snapshot_direct_cypher_pins_neo4j():
    app = _app()
    app.gui_config.input_mode = "direct_cypher"
    app.gui_config.retrieval_mode = "lancedb_only"  # would be overridden by the mode
    assert app._retrieval_snapshot().retrieval_mode == "neo4j_only"
