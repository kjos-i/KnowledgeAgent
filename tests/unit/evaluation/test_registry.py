"""Consistency invariants for the flat metric registry.

The registry is the single source of truth; these pin that its derived
lists stay unique + well-formed. Because the registry is FLAT (no
composites), there is no composite-drift check to write — that whole
class of assertion is designed out.
"""

from __future__ import annotations

from knowledge_agent.evaluation import registry as R


def test_keys_unique():
    keys = [m.key for m in R.METRICS]
    assert len(keys) == len(set(keys))


def test_sql_columns_unique_and_flat():
    cols = [col for col, _ in R.case_sql_columns()]
    assert len(cols) == len(set(cols))
    # Every metric with a sql_column contributes exactly one column (flat).
    assert len(cols) == sum(1 for m in R.METRICS if m.sql_column)


def test_summary_avg_keys_unique():
    keys = [col for col, _ in R.run_sql_columns()]
    assert len(keys) == len(set(keys))


def test_run_n_columns_pair_one_to_one_with_averages():
    n_cols = [col for col, _ in R.run_n_columns()]
    assert len(n_cols) == len(set(n_cols))  # unique
    assert len(n_cols) == len(R.run_sql_columns())  # one n per run-level average
    for col, sqltype in R.run_n_columns():
        assert col.startswith("n_") and sqltype == "INTEGER"
    # n_<source_key> pairs with the same source key as the average.
    assert n_cols == [f"n_{src}" for _, src in R.summary_avg_pairs()]


def test_pass_rate_has_no_per_case_column():
    pass_rate = next(m for m in R.METRICS if m.key == "pass_rate")
    assert pass_rate.sql_column is None
    assert pass_rate.summary_avg_key is None


def test_metric_directions_mark_cost_and_error_metrics_lower_is_better():
    """higher_is_better is False for cost/error metrics (latency, tokens,
    hallucination, disallowed keyword hits) and True for score metrics —
    mirrored onto each summary-avg key. Drives the dashboard's delta pills."""
    directions = R.metric_directions()
    for key in (
        "latency_seconds",
        "avg_latency_seconds",
        "judge_total_tokens",
        "avg_agent_total_tokens",
        "hallucination",
        "avg_hallucination",
        "disallowed_keyword_hits",
        "avg_disallowed_keyword_hits",
    ):
        assert directions[key] is False, key
    for key in ("pass_rate", "hit_at_k", "avg_hit_at_k", "avg_precision_at_k"):
        assert directions[key] is True, key


def test_group_labels_cover_every_metric_group():
    """Every group a metric declares has a display label, so the GUI never
    falls back to a raw group key for a section header."""
    used_groups = {m.group for m in R.METRICS}
    assert used_groups <= set(R.GROUP_LABELS)


def test_toggle_groups_resolve():
    assert R.keys_in_toggle_group("source") == [
        "hit_at_k",
        "mrr",
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
    ]
    assert R.keys_in_toggle_group("chunk") == [
        "chunk_hit_at_k",
        "chunk_mrr",
        "chunk_precision_at_k",
        "chunk_recall_at_k",
        "chunk_ndcg_at_k",
    ]
    assert R.keys_in_toggle_group("kg") == [
        "cypher_validity",
        "cypher_nonempty",
        "kg_hit_at_k",
        "kg_entity_recall",
        "kg_source_grounding",
        "mode_routing_correctness",
    ]


def test_always_on_excludes_pass_rate_and_toggled():
    always = set(R.always_on_keys())
    assert "pass_rate" not in always  # no per-case column
    assert "hit_at_k" not in always  # toggle_group="source"
    assert {
        "required_keyword_hit_rate",
        "disallowed_keyword_hits",
        "chunk_source_grounding",
    } <= always


def test_labels_fmts_decimals_cover_every_key():
    labels, fmts, decimals = R.metric_labels(), R.metric_fmts(), R.metric_decimals()
    for m in R.METRICS:
        assert m.key in labels and m.key in fmts and m.key in decimals


def test_csv_fieldnames_structure():
    fields = R.csv_fieldnames()
    assert fields[:3] == ["id", "category", "status"]
    assert fields[-1] == "error_count"
