"""Single source of truth for every metric in the harness.

The ledger schema, CSV columns, report summaries, and the Flet dashboard
all derive their names / labels / formats / SQL columns from the `METRICS`
list here — so adding or renaming a metric is a one-place edit.

Deliberately FLAT (no composite/nested metrics): the reference exploded
`backend_distribution` + `keyword_checks` into sub-columns via a
`composite=True` mechanism that needed a drift assertion to police it.
Here every metric — including what were sub-fields — is a first-class
`MetricDef` with its own `sql_column`. No composites, no explosion, no
composite drift check.

Adding a metric:
  1. Append a `MetricDef` here.
  2. Deterministic → write its compute fn in `metrics.py` and add the key
     to the per-case result dict in `engine.py`. Judge → add its DeepEval
     metric in `judge.py`.
Everything else (ledger schema, CSV, summaries) updates automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricGroup = Literal[
    "summary", "retrieval", "chunk", "keyword", "kg", "orchestration", "llm", "tokens", "latency"
]
ToggleGroup = Literal["source", "chunk", "kg", "judge"]

DEFAULT_FMT = ".2f"
DEFAULT_DECIMALS = 3

# ── display constants — single source for the dashboard; the GUI reads these
#    instead of hardcoding labels/thresholds. ───────────────────────────────
# Section header per metric group, in display order (dashboard renders sections
# top-to-bottom in this order). Mirrors the Streamlit reference order —
# Judge → Source → Chunk — then KA-specific groups. Groups whose metrics all
# sit in the OVERVIEW_ROW (summary, latency) render nothing and are skipped.
GROUP_LABELS: dict[str, str] = {
    "summary": "Summary",
    "llm": "Average Judge Scores",
    "retrieval": "Average Source Retrieval Quality",
    "chunk": "Average Chunk Retrieval Quality",
    "kg": "Knowledge Graph",
    "orchestration": "Orchestration",
    "keyword": "Keywords",
    "tokens": "Tokens",
    "latency": "Latency",
}
# The top "Average Performance" overview row: a curated, ordered set of headline
# metric keys pulled ACROSS groups (the one thing group membership can't express).
# The GUI renders these as the first card row and skips them in their own group
# section so they aren't shown twice. Left-to-right = list order.
OVERVIEW_ROW: tuple[str, ...] = (
    "pass_rate",
    "avg_judge_run_score",
    "avg_latency_seconds",
    "avg_agent_total_tokens",
    "avg_judge_total_tokens",
)
# Section header for the overview row.
OVERVIEW_LABEL = "Average Performance"
# Color-band thresholds for a 0–1 score: >= GOOD → green, >= BORDERLINE →
# orange, below → red. A global display convention (not per-metric).
SCORE_GOOD_THRESHOLD = 0.8
SCORE_BORDERLINE_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class MetricDef:
    """Definition of a single evaluation metric."""

    key: str
    """Internal dict key in the per-case result + JSON report."""
    label: str
    """Human-readable display label."""
    group: MetricGroup
    """Display/family classification."""
    sql_column: str | None = None
    """Column in `eval_cases`. None only for run-level structural metrics
    (e.g. pass_rate) that live in the runs preamble, not per-case."""
    sql_type: str = "REAL"
    fmt: str = DEFAULT_FMT
    decimals: int = DEFAULT_DECIMALS
    summary_avg_key: str | None = None
    """Column in `eval_runs` for this metric's run-level average. None when
    no run-level average is stored."""
    summary_avg_fmt: str | None = None
    """Format for the run-level mean when it differs from `fmt` (e.g. an
    integer per-case metric whose mean renders as a float)."""
    summary_avg_label: str | None = None
    """Label for the run-level average when it must differ from `label`."""
    toggle_group: ToggleGroup | None = None
    """Which user toggle controls whether this metric is computed. None =
    always-on regardless of `EvalConfig.enabled_groups`."""
    higher_is_better: bool = True
    """Whether a HIGHER value is an improvement. False for cost/error metrics
    where lower is better (latency, tokens, hallucination, disallowed-keyword
    hits) — drives the green/red direction of the dashboard's delta pills and
    any pass/fail coloring. Single source of truth; the GUI never hardcodes it."""


# ---------------------------------------------------------------------------
# The registry. Phase 1 = deterministic retrieval + chunk (incl. citation
# grounding) + keyword + always-on tokens/latency. KG (P2) + judge (P3)
# metrics get appended in their phases.
# ---------------------------------------------------------------------------

METRICS: list[MetricDef] = [
    # ── run-level structural (one value per run; lives in runs preamble) ──
    MetricDef(key="pass_rate", label="Pass Rate", group="summary", fmt=".0%"),
    # ── deterministic retrieval — source-level (toggle "source") ──────────
    MetricDef(
        key="hit_at_k",
        label="Hit@k",
        group="retrieval",
        sql_column="hit_at_k",
        summary_avg_key="avg_hit_at_k",
        toggle_group="source",
    ),
    MetricDef(
        key="mrr",
        label="MRR",
        group="retrieval",
        sql_column="mrr",
        summary_avg_key="avg_mrr",
        toggle_group="source",
    ),
    MetricDef(
        key="precision_at_k",
        label="Precision@k",
        group="retrieval",
        sql_column="precision_at_k",
        summary_avg_key="avg_precision_at_k",
        toggle_group="source",
    ),
    MetricDef(
        key="recall_at_k",
        label="Recall@k",
        group="retrieval",
        sql_column="recall_at_k",
        summary_avg_key="avg_recall_at_k",
        toggle_group="source",
    ),
    MetricDef(
        key="ndcg_at_k",
        label="NDCG@k",
        group="retrieval",
        sql_column="ndcg_at_k",
        summary_avg_key="avg_ndcg_at_k",
        toggle_group="source",
    ),
    # ── deterministic retrieval — chunk-level (toggle "chunk") ────────────
    MetricDef(
        key="chunk_hit_at_k",
        label="Chunk Hit@k",
        group="chunk",
        sql_column="chunk_hit_at_k",
        summary_avg_key="avg_chunk_hit_at_k",
        toggle_group="chunk",
    ),
    MetricDef(
        key="chunk_mrr",
        label="Chunk MRR",
        group="chunk",
        sql_column="chunk_mrr",
        summary_avg_key="avg_chunk_mrr",
        toggle_group="chunk",
    ),
    MetricDef(
        key="chunk_precision_at_k",
        label="Chunk Precision@k",
        group="chunk",
        sql_column="chunk_precision_at_k",
        summary_avg_key="avg_chunk_precision_at_k",
        toggle_group="chunk",
    ),
    MetricDef(
        key="chunk_recall_at_k",
        label="Chunk Recall@k",
        group="chunk",
        sql_column="chunk_recall_at_k",
        summary_avg_key="avg_chunk_recall_at_k",
        toggle_group="chunk",
    ),
    MetricDef(
        key="chunk_ndcg_at_k",
        label="Chunk NDCG@k",
        group="chunk",
        sql_column="chunk_ndcg_at_k",
        summary_avg_key="avg_chunk_ndcg_at_k",
        toggle_group="chunk",
    ),
    # ── citation grounding — a structural check that the answer's chunk
    #    citations map to actually-retrieved chunks. Grouped with the chunk
    #    metrics on the dashboard (it operates on chunk citations); always-on,
    #    unlike the toggle_group="chunk" retrieval metrics. ─────────────────
    MetricDef(
        key="chunk_source_grounding",
        label="Chunk Citation Grounding",
        group="chunk",
        sql_column="chunk_source_grounding",
        summary_avg_key="avg_chunk_source_grounding",
    ),
    # ── keyword checks — FLAT (were a composite in the reference) ──────────
    MetricDef(
        key="required_keyword_hit_rate",
        label="Required Keyword Hit Rate",
        group="keyword",
        sql_column="required_keyword_hit_rate",
        summary_avg_key="avg_required_keyword_hit_rate",
    ),
    MetricDef(
        key="disallowed_keyword_hits",
        label="Disallowed Keyword Hits",
        group="keyword",
        higher_is_better=False,
        sql_column="disallowed_keyword_hits",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_disallowed_keyword_hits",
        summary_avg_fmt=".2f",
    ),
    # ── KG group (Phase 2) — deterministic, toggle "kg" ───────────────────
    MetricDef(
        key="cypher_validity",
        label="Cypher Validity",
        group="kg",
        sql_column="cypher_validity",
        summary_avg_key="avg_cypher_validity",
        toggle_group="kg",
    ),
    MetricDef(
        key="cypher_nonempty",
        label="Cypher Non-empty",
        group="kg",
        sql_column="cypher_nonempty",
        summary_avg_key="avg_cypher_nonempty",
        toggle_group="kg",
    ),
    MetricDef(
        key="kg_hit_at_k",
        label="KG Entity Hit",
        group="kg",
        sql_column="kg_hit_at_k",
        summary_avg_key="avg_kg_hit_at_k",
        toggle_group="kg",
    ),
    MetricDef(
        key="kg_entity_recall",
        label="KG Entity Recall",
        group="kg",
        sql_column="kg_entity_recall",
        summary_avg_key="avg_kg_entity_recall",
        toggle_group="kg",
    ),
    MetricDef(
        key="kg_source_grounding",
        label="KG Citation Grounding",
        group="kg",
        sql_column="kg_source_grounding",
        summary_avg_key="avg_kg_source_grounding",
        toggle_group="kg",
    ),
    MetricDef(
        key="mode_routing_correctness",
        label="Mode Routing Correct",
        group="orchestration",
        sql_column="mode_routing_correctness",
        summary_avg_key="avg_mode_routing_correctness",
    ),
    # ── LLM-judge (DeepEval, Phase 3) — toggle "judge", stored at 4dp ─────
    MetricDef(
        key="answer_relevancy",
        label="Answer Relevancy",
        group="llm",
        sql_column="answer_relevancy",
        summary_avg_key="avg_answer_relevancy",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="faithfulness",
        label="Faithfulness",
        group="llm",
        sql_column="faithfulness",
        summary_avg_key="avg_faithfulness",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="contextual_precision",
        label="Ctx Precision",
        group="llm",
        sql_column="contextual_precision",
        summary_avg_key="avg_contextual_precision",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="contextual_recall",
        label="Ctx Recall",
        group="llm",
        sql_column="contextual_recall",
        summary_avg_key="avg_contextual_recall",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="contextual_relevancy",
        label="Ctx Relevancy",
        group="llm",
        sql_column="contextual_relevancy",
        summary_avg_key="avg_contextual_relevancy",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="hallucination",
        label="Hallucination",
        group="llm",
        higher_is_better=False,
        sql_column="hallucination",
        summary_avg_key="avg_hallucination",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="correctness_g_eval",
        label="Correctness (GEval)",
        group="llm",
        sql_column="correctness_g_eval",
        summary_avg_key="avg_correctness_g_eval",
        toggle_group="judge",
        decimals=4,
    ),
    MetricDef(
        key="avg_judge_score",
        label="Avg Judge Score",
        group="summary",
        sql_column="avg_judge_score",
        summary_avg_key="avg_judge_run_score",
        summary_avg_label="Avg Judge Score (Run)",
        toggle_group="judge",
    ),
    # ── judge token consumption — toggle "judge" ──────────────────────────
    MetricDef(
        key="judge_input_tokens",
        label="Judge Input Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="judge_input_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_judge_input_tokens",
        summary_avg_fmt=".0f",
        toggle_group="judge",
    ),
    MetricDef(
        key="judge_output_tokens",
        label="Judge Output Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="judge_output_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_judge_output_tokens",
        summary_avg_fmt=".0f",
        toggle_group="judge",
    ),
    MetricDef(
        key="judge_total_tokens",
        label="Judge Total Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="judge_total_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_judge_total_tokens",
        summary_avg_fmt=".0f",
        toggle_group="judge",
    ),
    # ── agent token consumption — always-on (int per-case, float mean) ────
    MetricDef(
        key="agent_input_tokens",
        label="Agent Input Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="agent_input_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_agent_input_tokens",
        summary_avg_fmt=".0f",
    ),
    MetricDef(
        key="agent_output_tokens",
        label="Agent Output Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="agent_output_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_agent_output_tokens",
        summary_avg_fmt=".0f",
    ),
    MetricDef(
        key="agent_total_tokens",
        label="Agent Total Tokens",
        group="tokens",
        higher_is_better=False,
        sql_column="agent_total_tokens",
        sql_type="INTEGER",
        fmt="d",
        decimals=0,
        summary_avg_key="avg_agent_total_tokens",
        summary_avg_fmt=".0f",
    ),
    # ── latency — always-on. Total wall-clock only: KA returns retrieved
    #    chunks IN the final state (no separate retriever call), so the
    #    retrieval-vs-LLM split isn't measurable without node instrumentation
    #    (deferred). One honest column beats two always-NULL ones.
    MetricDef(
        key="latency_seconds",
        label="Latency",
        group="latency",
        higher_is_better=False,
        sql_column="latency_seconds",
        summary_avg_key="avg_latency_seconds",
        fmt=".2f",
        decimals=2,
    ),
]


# ---------------------------------------------------------------------------
# Derived accessors — every consumer imports from here.
# ---------------------------------------------------------------------------


def keys_in_group(group: str) -> list[str]:
    """Ordered metric keys in a display group."""
    return [m.key for m in METRICS if m.group == group]


def keys_in_toggle_group(toggle_group: str) -> list[str]:
    """Ordered metric keys controlled by a toggle group."""
    return [m.key for m in METRICS if m.toggle_group == toggle_group]


def always_on_keys() -> list[str]:
    """Metric keys with no toggle_group (computed every run), excluding the
    run-level structural metrics that have no per-case sql_column."""
    return [m.key for m in METRICS if m.toggle_group is None and m.sql_column is not None]


def case_sql_columns() -> list[tuple[str, str]]:
    """Ordered (column, sql_type) pairs for the metric columns in
    `eval_cases`. Flat — one column per metric, no composite explosion."""
    return [(m.sql_column, m.sql_type) for m in METRICS if m.sql_column]


def run_sql_columns() -> list[tuple[str, str]]:
    """Ordered (summary_avg_key, "REAL") pairs for the average columns in
    `eval_runs`."""
    return [(m.summary_avg_key, "REAL") for m in METRICS if m.summary_avg_key]


def run_n_columns() -> list[tuple[str, str]]:
    """Ordered (n_<key>, "INTEGER") pairs for the per-metric case-count columns
    in `eval_runs`. `n_<key>` records how many cases had a non-None value for
    the metric — the denominator behind `avg_<key>`. Because a no-gold case is
    dropped from the mean (scored None, not 0/1), that denominator varies per
    metric; storing it keeps a mean over 1 case distinguishable from one over
    all of them. Paired with `summary_avg_pairs()` by the same source key."""
    return [(f"n_{m.key}", "INTEGER") for m in METRICS if m.summary_avg_key]


def summary_avg_pairs() -> list[tuple[str, str]]:
    """(summary_avg_key, source_key) pairs for building run-level summaries,
    e.g. ("avg_hit_at_k", "hit_at_k")."""
    return [(m.summary_avg_key, m.key) for m in METRICS if m.summary_avg_key]


def metric_labels() -> dict[str, str]:
    """key/column → display label, covering per-case keys + summary_avg keys."""
    labels: dict[str, str] = {m.key: m.label for m in METRICS}
    for m in METRICS:
        if m.sql_column and m.sql_column != m.key:
            labels[m.sql_column] = m.label
        if m.summary_avg_key:
            labels[m.summary_avg_key] = m.summary_avg_label or m.label
    return labels


def metric_fmts() -> dict[str, str]:
    """key → format spec, mirrored onto each summary_avg key."""
    fmts: dict[str, str] = {m.key: m.fmt for m in METRICS}
    for m in METRICS:
        if m.summary_avg_key:
            fmts[m.summary_avg_key] = m.summary_avg_fmt or m.fmt
    return fmts


def metric_directions() -> dict[str, bool]:
    """key → higher_is_better, mirrored onto each summary_avg key.

    Lets any consumer (e.g. the dashboard's delta pills) look up a metric's
    direction by its per-case OR run-level-average key without hardcoding
    which metrics are cost/error metrics where lower is better."""
    directions: dict[str, bool] = {m.key: m.higher_is_better for m in METRICS}
    for m in METRICS:
        if m.summary_avg_key:
            directions[m.summary_avg_key] = m.higher_is_better
    return directions


def metric_decimals() -> dict[str, int]:
    """key → round() precision, mirrored onto each summary_avg key."""
    decimals: dict[str, int] = {m.key: m.decimals for m in METRICS}
    for m in METRICS:
        if m.summary_avg_key:
            decimals[m.summary_avg_key] = m.decimals
    return decimals


def csv_fieldnames() -> list[str]:
    """Ordered per-case CSV columns: structural preamble → metric columns →
    trailing diagnostics."""
    preamble = ["id", "category", "status"]
    metric_cols = [col for col, _ in case_sql_columns()]
    trailing = ["error_count"]
    return preamble + metric_cols + trailing
