"""Orchestration — run each case through the adapter + deterministic
metrics, with bounded async concurrency.

`evaluate_case` is the per-case pipeline: invoke the graph (via the
adapter), compute the enabled deterministic metric families, assemble a
flat per-case result dict (keys = registry columns), and decide a PASS /
REVIEW verdict. `evaluate_cases` fans that out under a concurrency
semaphore. Case loading + corpus loading + persistence live in the
runner — the engine takes cases + a corpus_config and returns per-case
dicts, which keeps it unit-testable with a stubbed adapter.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation import metrics as M
from knowledge_agent.evaluation.adapter import run_case
from knowledge_agent.evaluation.registry import (
    keys_in_toggle_group,
    metric_decimals,
    metric_directions,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.corpus_config import CorpusConfig
    from knowledge_agent.evaluation.adapter import CaseRun
    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.evaluation.models import EvalCase

_DECIMALS = metric_decimals()


def _round(key: str, value: float | int | None) -> float | int | None:
    """Round a metric to its registry-declared precision (None passes through)."""
    if value is None:
        return None
    return round(value, _DECIMALS.get(key, 3))


_KG_KEYS = (
    "cypher_validity",
    "cypher_nonempty",
    "kg_hit_at_k",
    "kg_entity_recall",
    "kg_source_grounding",
)


def _kg_metrics(case: EvalCase, run: CaseRun) -> dict[str, Any]:
    """The 5 deterministic KG metrics. Each is None when not applicable:
    the cypher metrics + KG-source grounding need a Cypher to have run; the
    entity metrics need gold `expected_entities`."""
    values: dict[str, Any] = dict.fromkeys(_KG_KEYS, None)
    values.update(M.compute_kg_entity_metrics(run.kg_hits, case.expected_entities))

    if run.cypher_query and run.cypher_query.strip():
        values["cypher_validity"] = M.cypher_validity(run.cypher_read_only, run.kg_retrieval_error)
        values["cypher_nonempty"] = 1.0 if run.kg_hits else 0.0
        values["kg_source_grounding"] = M.kg_source_grounding(
            run.cited_kg_indices, len(run.kg_hits)
        )

    return values


def _compute_metrics(case: EvalCase, run: CaseRun, cfg: EvalConfig) -> dict[str, Any]:
    """Assemble the flat metric dict. Disabled groups → None columns (stored
    as NULL); families with no gold → None from the compute fns."""
    values: dict[str, Any] = {}

    if "source" in cfg.enabled_groups:
        values.update(M.compute_source_metrics(run.retrieved_doc_ids, case.expected_sources))
    else:
        values.update(dict.fromkeys(keys_in_toggle_group("source"), None))

    if "chunk" in cfg.enabled_groups:
        values.update(M.compute_chunk_metrics(run.retrieved_texts, case.expected_chunks))
    else:
        values.update(dict.fromkeys(keys_in_toggle_group("chunk"), None))

    if "kg" in cfg.enabled_groups:
        values.update(_kg_metrics(case, run))
    else:
        values.update(dict.fromkeys(keys_in_toggle_group("kg"), None))

    # Always-on families.
    values["required_keyword_hit_rate"] = M.required_keyword_hit_rate(
        run.answer, case.required_keywords
    )
    values["disallowed_keyword_hits"] = M.disallowed_keyword_hits(
        run.answer, case.disallowed_keywords
    )
    values["chunk_source_grounding"] = M.chunk_source_grounding(
        run.cited_chunk_ids, run.retrieved_chunk_ids
    )
    # Orchestration: routing correctness self-gates to None unless the case is
    # auto-mode with an expected_mode (not tied to any store's toggle group).
    values["mode_routing_correctness"] = (
        M.mode_routing_correct(run.routed_mode, case.expected_mode)
        if case.expected_mode and case.retrieval.retrieval_mode == "auto"
        else None
    )
    values["agent_input_tokens"] = run.input_tokens
    values["agent_output_tokens"] = run.output_tokens
    values["agent_total_tokens"] = run.total_tokens
    values["latency_seconds"] = run.latency_seconds

    return {k: _round(k, v) for k, v in values.items()}


def _status(case: EvalCase, values: dict[str, Any], run: CaseRun, cfg: EvalConfig) -> str:
    """PASS / REVIEW verdict. Any error forces REVIEW. Retrieval gate applies
    only when the source group ran AND the case has gold sources.

    A `direct_retrieval` case returns chunks with no synthesized answer, so
    the answer-based gates (keywords, judge) don't apply — it's scored on
    retrieval alone."""
    if run.error:
        return "REVIEW"
    retrieval_ok = True
    if "source" in cfg.enabled_groups and case.expected_sources:
        retrieval_ok = values.get("hit_at_k") == 1.0

    if case.retrieval.direct_retrieval:
        return "PASS" if retrieval_ok else "REVIEW"

    keywords_ok = (values.get("required_keyword_hit_rate") or 0.0) >= cfg.required_keyword_threshold
    keywords_ok = keywords_ok and (values.get("disallowed_keyword_hits") or 0) == 0
    judge_ok = True
    if "judge" in cfg.enabled_groups:
        # Gate on the two core generation-quality judges when present.
        present = [values.get(k) for k in ("faithfulness", "answer_relevancy")]
        present = [s for s in present if s is not None]
        judge_ok = bool(present) and all(s >= cfg.judge_threshold for s in present)
    return "PASS" if retrieval_ok and keywords_ok and judge_ok else "REVIEW"


async def _judge_metrics(case: EvalCase, run: CaseRun, cfg: EvalConfig) -> dict[str, Any]:
    """Run the DeepEval judge panel (async) → the 7 judge scores + their
    mean + judge token usage, rounded to registry precision."""
    from knowledge_agent.evaluation import judge as J

    data = J.build_judge_input(run, case)
    models = J.resolve_judge_models(cfg.judge_models)
    scores, in_tok, out_tok = await J.run_judge_panel(data, models, cfg.judge_threshold)

    values: dict[str, Any] = {k: _round(k, scores.get(k)) for k in J.JUDGE_METRIC_KEYS}
    # Align every judge metric to "higher = better" before averaging: a
    # lower-is-better metric (hallucination) is inverted via its registry
    # direction, else a high (worse) hallucination score would inflate the mean
    # and make the answer read as BETTER. Per-metric columns keep the raw score;
    # only the blended average is direction-normalised.
    directions = metric_directions()
    present = [
        (s if directions.get(k, True) else 1.0 - s)
        for k in J.JUDGE_METRIC_KEYS
        if (s := scores.get(k)) is not None
    ]
    values["avg_judge_score"] = _round("avg_judge_score", M.safe_mean(present)) if present else None
    values["judge_input_tokens"] = in_tok
    values["judge_output_tokens"] = out_tok
    values["judge_total_tokens"] = (
        in_tok + out_tok if in_tok is not None and out_tok is not None else None
    )
    return values


async def evaluate_case(
    case: EvalCase, corpus_config: CorpusConfig | None, cfg: EvalConfig
) -> dict[str, Any]:
    """Run + score one case → a flat result dict (keys = registry columns +
    structural fields)."""
    run = await run_case(case, corpus_config)
    values = _compute_metrics(case, run, cfg)
    # direct_retrieval returns chunks with no synthesized answer, so the LLM
    # judge has nothing to score — skip it (and don't spend judge tokens).
    if "judge" in cfg.enabled_groups and not case.retrieval.direct_retrieval:
        values.update(await _judge_metrics(case, run, cfg))
    else:
        values.update(dict.fromkeys(keys_in_toggle_group("judge"), None))
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "origin": case.origin,
        # The per-case retrieval settings this case ran under (mode / search
        # mode / top_k / tuning knobs), snapshotted so Run Summary → Case
        # Details can show exactly how each case was retrieved. Display only.
        "case_settings": case.retrieval.model_dump(mode="json"),
        # Chat provenance (origin="chat"): as plain dicts so the ledger can JSON
        # it. Display only — never scored.
        "source_conversation": [t.model_dump() for t in case.source_conversation],
        "expected_output": "\n".join(case.expected_answer_points),
        "answer": run.answer,
        "status": _status(case, values, run, cfg),
        "errors": [run.error] if run.error else [],
        **values,
    }


async def evaluate_cases(
    cases: list[EvalCase],
    corpus_config: CorpusConfig | None,
    cfg: EvalConfig,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every case under a concurrency semaphore, preserving order.

    `on_progress(done, total)` — when given — fires once per case as it
    COMPLETES (completion order, not submission order), so a GUI progress bar
    or the CLI can report advancement. Same emit-in-backend / render-in-caller
    split as the ledger readers; the harness never imports a UI. `done` is
    captured before the callback so concurrent completions report distinct
    counts.
    """
    sem = asyncio.Semaphore(max(1, cfg.concurrency))
    total = len(cases)
    done = 0

    async def _one(case: EvalCase) -> dict[str, Any]:
        nonlocal done
        async with sem:
            result = await evaluate_case(case, corpus_config, cfg)
        done += 1
        if on_progress is not None:
            on_progress(done, total)
        return result

    return await asyncio.gather(*[_one(c) for c in cases])
