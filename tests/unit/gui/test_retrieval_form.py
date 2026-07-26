"""Tests for the shared retrieval-form gray-out logic (`retrieval_form.py`).

Layer-pure: this exercises the GUI's own copy of the mode → leg rule. The
backend's copy (`evaluation.models.required_knobs`) is tested separately in
test_models; the two are kept consistent by each being correct against the
retrieval semantics, not by a cross-layer import.
"""

from __future__ import annotations

import flet as ft

from knowledge_agent.gui._widgets.retrieval_form import (
    RetrievalControls,
    apply_gray_out,
    build_input_mode_radios,
    knobs_to_query_mode,
    mmr_help_text,
    query_mode_to_knobs,
    store_forced_by_mode,
)


def _controls() -> RetrievalControls:
    return RetrievalControls(
        lancedb_mode_box=ft.Container(),
        lancedb_radios=[ft.Radio(value=m) for m in ("hybrid", "fts", "vector")],
        lancedb_mode_control=ft.RadioGroup(content=ft.Row()),
        num_candidates_field=ft.TextField(),
        rrf_constant_field=ft.TextField(),
        use_mmr_checkbox=ft.Checkbox(value=True),
        mmr_lambda_control=ft.Slider(),
        kg_max_rows_field=ft.TextField(),
    )


def test_neo4j_only_disables_and_fades_lancedb_block():
    c = _controls()
    apply_gray_out(c, retrieval_mode="neo4j_only", lancedb_search_mode="fts", use_mmr=True)
    assert all(r.disabled for r in c.lancedb_radios)
    assert c.lancedb_mode_box.opacity == 0.4  # radios can't recolor → fade
    assert c.num_candidates_field.disabled is True
    assert c.rrf_constant_field.disabled is True
    assert c.use_mmr_checkbox.disabled is True
    assert c.mmr_lambda_control.disabled is True
    assert c.kg_max_rows_field.disabled is False  # the Neo4j leg runs


def test_lancedb_only_hybrid_enables_lance_disables_kg():
    c = _controls()
    apply_gray_out(c, retrieval_mode="lancedb_only", lancedb_search_mode="hybrid", use_mmr=True)
    assert not any(r.disabled for r in c.lancedb_radios)
    assert c.lancedb_mode_box.opacity == 1.0
    assert c.num_candidates_field.disabled is False
    assert c.rrf_constant_field.disabled is False
    assert c.mmr_lambda_control.disabled is False  # hybrid + MMR on
    assert c.kg_max_rows_field.disabled is True  # no Neo4j leg


def test_fts_disables_pool_rrf_and_forces_mmr_off():
    c = _controls()
    apply_gray_out(c, retrieval_mode="lancedb_only", lancedb_search_mode="fts", use_mmr=True)
    assert c.num_candidates_field.disabled is True  # fts fetches no candidate pool
    assert c.rrf_constant_field.disabled is True  # rrf only fuses in hybrid
    assert c.use_mmr_checkbox.disabled is True
    assert c.use_mmr_checkbox.value is False  # can't honour MMR on FTS
    assert c.mmr_lambda_control.disabled is True


def test_mmr_off_disables_only_the_lambda():
    c = _controls()
    apply_gray_out(c, retrieval_mode="lancedb_only", lancedb_search_mode="hybrid", use_mmr=False)
    assert c.use_mmr_checkbox.disabled is False  # can flip on
    assert c.mmr_lambda_control.disabled is True  # off → its parameter is irrelevant


def test_mmr_help_text_states():
    assert "LanceDB leg" in mmr_help_text(lance_on=False, is_fts=False, use_mmr=True)
    assert "FTS" in mmr_help_text(lance_on=True, is_fts=True, use_mmr=True)
    assert "MMR is off" in mmr_help_text(lance_on=True, is_fts=False, use_mmr=False)
    assert "relevance" in mmr_help_text(lance_on=True, is_fts=False, use_mmr=True)


# ---- shared input-mode radio + wiring (SSOT for Settings + eval Dataset) ----


def test_input_mode_radios_are_the_three_shared_modes():
    # The shared builder owns ONLY the modes both forms use; 'conversational'
    # is chat-only and prepended by the Settings tab itself, never here.
    radios = build_input_mode_radios()
    assert [r.value for r in radios] == ["refined", "direct_query", "direct_cypher"]
    assert all(r.value != "conversational" for r in radios)


def test_query_mode_to_knobs_maps_each_mode():
    assert query_mode_to_knobs("refined") == {"skip_query_builder": False, "user_cypher": None}
    assert query_mode_to_knobs("direct_query") == {"skip_query_builder": True, "user_cypher": None}
    assert query_mode_to_knobs("direct_cypher", cypher_text="MATCH (n) RETURN n") == {
        "skip_query_builder": False,
        "user_cypher": "MATCH (n) RETURN n",
    }
    # conversational carries refined-like knobs (only its query source differs).
    assert query_mode_to_knobs("conversational") == {
        "skip_query_builder": False,
        "user_cypher": None,
    }


def test_query_mode_to_knobs_blank_cypher_is_none():
    assert query_mode_to_knobs("direct_cypher", cypher_text="   ")["user_cypher"] is None


def test_knobs_to_query_mode_is_the_inverse():
    assert knobs_to_query_mode(skip_query_builder=False, user_cypher=None) == "refined"
    assert knobs_to_query_mode(skip_query_builder=True, user_cypher=None) == "direct_query"
    # Cypher wins over the skip flag.
    assert (
        knobs_to_query_mode(skip_query_builder=True, user_cypher="MATCH (n) RETURN n")
        == "direct_cypher"
    )


def test_store_forced_by_mode_only_pins_cypher_to_neo4j():
    assert store_forced_by_mode("direct_cypher") == "neo4j_only"
    assert store_forced_by_mode("refined") is None
    assert store_forced_by_mode("direct_query") is None
    assert store_forced_by_mode("conversational") is None
