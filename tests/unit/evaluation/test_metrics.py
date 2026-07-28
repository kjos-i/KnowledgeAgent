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


def test_normalize_text_strips_punctuation_keeping_dot_percent_dash():
    # Commas / slashes / semicolons / underscores / quotes → space; ``.`` ``%``
    # ``-`` are kept (mirrors the reference's punctuation stripping).
    assert M.normalize_text("a, b/c; d_e") == "a b c d e"
    assert M.normalize_text("5.5% co-factor") == "5.5% co-factor"


def test_normalize_text_keeps_non_ascii_letters_and_digits():
    # Greek / CJK letters are kept (only accents are stripped), so terms that
    # differ only by a non-ASCII character stay distinct — UNLIKE the reference,
    # which drops every non-ASCII char. Punctuation around them is still stripped.
    assert M.normalize_text("TGF-β/SMAD") == "tgf-β smad"
    assert M.normalize_text("北京 policy") == "北京 policy"
    assert M.normalize_text("α-catenin") != M.normalize_text("β-catenin")


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
    # NDCG: relevant at ranks 2 and 4; the ideal is those same labels sorted to
    # the top (ranks 1,2), so 0 < score < 1.
    assert 0.0 < M.ndcg_at_k(rel) < 1.0
    assert M.ndcg_at_k([True, True]) == 1.0


def test_ndcg_never_exceeds_one_with_duplicate_gold_docs():
    """A gold item retrieved in several slots must NOT push NDCG above 1.0 —
    IDCG is the ideal ordering of the same relevance labels (matches the
    reference harness; this was the >1 bug that showed 2.09)."""
    assert M.ndcg_at_k([True, True, True]) == 1.0
    assert M.ndcg_at_k([False, True, True]) <= 1.0
    # via the family fn: one gold doc returned 3× collapses to a single relevant
    # document → 1.0.
    got = M.compute_source_metrics(["d1", "d1", "d1"], expected_sources=["d1"])
    assert got["ndcg_at_k"] == 1.0


# ---- source-level family ----


def test_source_metrics_hit_and_ranking():
    got = M.compute_source_metrics(["d3", "d1", "d9"], expected_sources=["d1", "d2"])
    assert got["hit_at_k"] == 1.0
    assert got["mrr"] == 0.5  # d1 at rank 2
    assert got["precision_at_k"] == 1 / 3
    assert got["recall_at_k"] == 0.5  # found d1 of {d1,d2}


def test_source_metrics_collapse_to_documents():
    # Source metrics collapse retrieved chunks to distinct documents first, so
    # d1 retrieved 3× is a single relevant slot: recall found 1 of {d1, d2} = 0.5,
    # and precision = the one distinct retrieved doc is relevant = 1.0.
    got = M.compute_source_metrics(["d1", "d1", "d1"], expected_sources=["d1", "d2"])
    assert got["recall_at_k"] == 0.5
    assert got["precision_at_k"] == 1.0


def test_source_precision_dedups_documents():
    # Two chunks from relevant d1 + one from noise dX collapse to docs {d1, dX}:
    # precision = 1 relevant of 2 distinct docs = 0.5 (per-chunk would give 2/3).
    got = M.compute_source_metrics(["d1", "d1", "dX"], expected_sources=["d1"])
    assert got["precision_at_k"] == 0.5
    assert got["hit_at_k"] == 1.0
    assert got["recall_at_k"] == 1.0  # found the one gold doc


def test_source_mrr_ranks_first_distinct_document():
    # Same noise doc twice then the relevant doc collapses to [dX, d1], so the
    # first relevant DOCUMENT is at rank 2 → 0.5 (per-chunk would give 1/3).
    got = M.compute_source_metrics(["dX", "dX", "d1"], expected_sources=["d1"])
    assert got["mrr"] == 0.5


def test_source_ndcg_scored_on_distinct_documents():
    # [dX, dX, d1] collapses to [dX, d1]; NDCG is scored on that 2-doc list, not
    # the raw 3-chunk list — verify it equals the primitive on the deduped rel.
    got = M.compute_source_metrics(["dX", "dX", "d1"], expected_sources=["d1"])
    assert got["ndcg_at_k"] == M.ndcg_at_k([False, True])
    assert got["ndcg_at_k"] != M.ndcg_at_k([False, False, True])


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


def test_required_keyword_hit_rate_ignores_keyword_that_normalizes_to_empty():
    """A punctuation-only required keyword passes `.strip()` but normalizes to
    "" — a substring of every answer. It must be dropped, not counted as
    always-present. Regression: it inflated the hit rate (1/2 below)."""
    assert M.required_keyword_hit_rate("the capital city", ["Oslo", "!!!"]) == 0.0
    # When every keyword normalizes away, the list is empty -> vacuous 1.0.
    assert M.required_keyword_hit_rate("anything", ["***"]) == 1.0


def test_disallowed_keyword_hits_ignores_keyword_that_normalizes_to_empty():
    """A punctuation-only disallowed keyword normalizes to "", which matches
    every answer. It must be dropped, not counted — else a clean answer is
    falsely flagged (a spurious REVIEW). Regression: returned 1 / 2 here."""
    assert M.disallowed_keyword_hits("a perfectly clean answer", ["***"]) == 0
    assert M.disallowed_keyword_hits("contains foo", ["foo", "!!!"]) == 1


def test_chunk_metrics_ignore_expected_chunk_that_normalizes_to_empty():
    """A punctuation-only expected chunk normalizes to "" — a substring of every
    text — which would falsely mark every retrieved chunk relevant. It must be
    dropped. Regression: chunk_hit_at_k/precision inflated to 1.0/0.5."""
    texts = ["completely unrelated text", "also unrelated"]
    got = M.compute_chunk_metrics(texts, expected_chunks=["Oslo", "!!!"])
    assert got["chunk_hit_at_k"] == 0.0  # only "Oslo" is real, and it's absent
    assert got["chunk_recall_at_k"] == 0.0


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
