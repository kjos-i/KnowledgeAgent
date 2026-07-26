"""Tests for the single-source retrieval mode-to-leg rule (`retrieval_rules`).

This is the SSOT both the GUI gray-out (`retrieval_form.apply_gray_out`) and the
eval harness (`evaluation.models.required_knobs`) read, so the FTS/num_candidates
behaviour is pinned here once.
"""

from __future__ import annotations

from knowledge_agent.retrieval_rules import (
    LANCE_MODES,
    NEO4J_MODES,
    active_knobs,
    mmr_applicable,
    runs_lance_leg,
    runs_neo4j_leg,
)


def test_leg_membership():
    assert runs_lance_leg("lancedb_only") and not runs_neo4j_leg("lancedb_only")
    assert runs_neo4j_leg("neo4j_only") and not runs_lance_leg("neo4j_only")
    # auto runs both legs (conservative: its knobs are pinned either way it routes).
    assert runs_lance_leg("auto") and runs_neo4j_leg("auto")
    assert "auto" in LANCE_MODES and "auto" in NEO4J_MODES


def test_hybrid_uses_pool_rrf_and_mmr():
    assert active_knobs("lancedb_only", "hybrid", use_mmr=True) == {
        "num_candidates",
        "rrf_rank_constant",
        "mmr_lambda",
    }


def test_fts_uses_nothing_lance_side():
    # THE FIX: fts fetches no candidate pool (limits straight to top_k), isn't
    # hybrid, and has no vectors for MMR, so it uses none of the lance knobs.
    assert active_knobs("lancedb_only", "fts", use_mmr=True) == set()


def test_vector_uses_pool_and_mmr_but_not_rrf():
    assert active_knobs("lancedb_only", "vector", use_mmr=True) == {
        "num_candidates",
        "mmr_lambda",
    }
    assert active_knobs("lancedb_only", "vector", use_mmr=False) == {"num_candidates"}


def test_neo4j_only_uses_kg_max_rows_only():
    assert active_knobs("neo4j_only", "hybrid", use_mmr=True) == {"kg_max_rows"}


def test_both_legs_union_all_knobs():
    assert active_knobs("parallel_fused", "hybrid", use_mmr=True) == {
        "num_candidates",
        "rrf_rank_constant",
        "mmr_lambda",
        "kg_max_rows",
    }


def test_mmr_applicable():
    assert mmr_applicable("lancedb_only", "hybrid")
    assert mmr_applicable("lancedb_only", "vector")
    assert not mmr_applicable("lancedb_only", "fts")  # no vectors
    assert not mmr_applicable("neo4j_only", "hybrid")  # no lance leg
