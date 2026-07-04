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


def test_pass_rate_has_no_per_case_column():
    pass_rate = next(m for m in R.METRICS if m.key == "pass_rate")
    assert pass_rate.sql_column is None
    assert pass_rate.summary_avg_key is None


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
