"""Tests for the deterministic metric compute functions.

Pure functions over primitives — no LLM, no KA objects, no event loop.
Covers ranking primitives, the source/chunk family functions (incl. the
no-gold → None contract), keyword checks, and citation grounding.
"""

from __future__ import annotations

from knowledge_agent.evaluation import metrics as M

# ---- helpers ----


def test_normalize_text_lowercases_strips_accents_collapses_ws():
    assert M.normalize_text("  Café   RÉSUMÉ\tX ") == "cafe resume x"


def test_safe_mean_skips_none_and_handles_empty():
    assert M.safe_mean([1.0, None, 3.0]) == 2.0
    assert M.safe_mean([None, None]) is None
    assert M.safe_mean([]) is None


# ---- ranking primitives ----


def test_hit_mrr_precision_ndcg_primitives():
    rel = [False, True, False, True]
    assert M.hit_at_k(rel) == 1.0
    assert M.hit_at_k([False, False]) == 0.0
    assert M.mrr(rel) == 0.5  # first relevant at rank 2
    assert M.mrr([False, False]) == 0.0
    assert M.precision_at_k(rel) == 0.5  # 2 of 4
    # NDCG: relevant at ranks 2 and 4; ideal has 2 relevant at ranks 1,2.
    assert 0.0 < M.ndcg_at_k(rel, total_relevant=2) < 1.0
    assert M.ndcg_at_k([True, True], total_relevant=2) == 1.0


# ---- source-level family ----


def test_source_metrics_hit_and_ranking():
    got = M.compute_source_metrics(["d3", "d1", "d9"], expected_sources=["d1", "d2"])
    assert got["hit_at_k"] == 1.0
    assert got["mrr"] == 0.5  # d1 at rank 2
    assert got["precision_at_k"] == 1 / 3
    assert got["recall_at_k"] == 0.5  # found d1 of {d1,d2}


def test_source_recall_dedups_documents():
    # d1 retrieved twice → counts once toward finding the gold doc.
    got = M.compute_source_metrics(["d1", "d1", "d1"], expected_sources=["d1", "d2"])
    assert got["recall_at_k"] == 0.5
    assert got["precision_at_k"] == 1.0  # every slot is a gold doc


def test_source_metrics_miss_all_zero():
    got = M.compute_source_metrics(["dX", "dY"], expected_sources=["d1"])
    assert got["hit_at_k"] == 0.0
    assert got["recall_at_k"] == 0.0


def test_source_metrics_none_when_no_gold():
    got = M.compute_source_metrics(["d1"], expected_sources=[])
    assert set(got.values()) == {None}


# ---- chunk-level family ----


def test_chunk_metrics_substring_match():
    texts = ["Norway has 5.5 million people", "unrelated text"]
    got = M.compute_chunk_metrics(texts, expected_chunks=["5.5 million", "Oslo"])
    assert got["chunk_hit_at_k"] == 1.0
    assert got["chunk_mrr"] == 1.0  # first chunk matches
    assert got["chunk_precision_at_k"] == 0.5  # 1 of 2 chunks relevant
    assert got["chunk_recall_at_k"] == 0.5  # found "5.5 million" of 2 snippets


def test_chunk_metrics_none_when_no_gold():
    got = M.compute_chunk_metrics(["anything"], expected_chunks=[])
    assert set(got.values()) == {None}


# ---- keyword checks ----


def test_required_keyword_hit_rate_partial_and_vacuous():
    assert M.required_keyword_hit_rate("Oslo is the capital", ["Oslo", "5.5 million"]) == 0.5
    assert M.required_keyword_hit_rate("anything", []) == 1.0  # vacuous pass


def test_disallowed_keyword_hits_counts():
    assert M.disallowed_keyword_hits("contains foo and bar", ["foo", "baz"]) == 1
    assert M.disallowed_keyword_hits("clean answer", ["foo"]) == 0


# ---- citation grounding ----


def test_chunk_source_grounding():
    # all cited chunks were retrieved → 1.0
    assert M.chunk_source_grounding(["c1", "c2"], ["c1", "c2", "c3"]) == 1.0
    # one citation not retrieved → 0.5 (fabricated citation caught)
    assert M.chunk_source_grounding(["c1", "cX"], ["c1", "c2"]) == 0.5
    # cited nothing → nothing to ground → 1.0
    assert M.chunk_source_grounding([], ["c1"]) == 1.0


# ---- KG metrics (Phase 2) ----


def test_kg_entity_metrics():
    hits = [{"name": "ESCRT-III", "type": "complex"}, {"name": "Vps4"}]
    got = M.compute_kg_entity_metrics(hits, ["ESCRT-III", "CHMP4B"])
    assert got["kg_hit_at_k"] == 1.0
    assert got["kg_entity_recall"] == 0.5  # found ESCRT-III, not CHMP4B
    # no gold entities → None
    assert set(M.compute_kg_entity_metrics(hits, []).values()) == {None}


def test_kg_source_grounding():
    assert M.kg_source_grounding([0, 1], 3) == 1.0
    assert M.kg_source_grounding([0, 5], 3) == 0.5  # index 5 out of range → ungrounded
    assert M.kg_source_grounding([], 3) == 1.0


def test_cypher_validity():
    assert M.cypher_validity(True, None) == 1.0
    assert M.cypher_validity(False, None) == 0.0  # write cypher (read-only rail failed)
    assert M.cypher_validity(True, "Neo4jError") == 0.0  # executed with error


def test_mode_routing_correct():
    assert M.mode_routing_correct("neo4j_only", "neo4j_only") == 1.0
    assert M.mode_routing_correct("lancedb_only", "neo4j_only") == 0.0
