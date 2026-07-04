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
from knowledge_agent.evaluation.registry import keys_in_toggle_group, metric_decimals

if TYPE_CHECKING:
    from knowledge_agent.evaluation.adapter import CaseRun
    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.evaluation.models import EvalCase
    from knowledge_agent.kg.corpus_config import CorpusConfig

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
    "mode_routing_correctness",
)


def _kg_metrics(case: EvalCase, run: CaseRun) -> dict[str, Any]:
    """The 6 deterministic KG metrics. Each is None when not applicable:
    the cypher metrics + KG-source grounding need a Cypher to have run; the
    entity metrics need gold `expected_entities`; mode-routing needs an
    auto-mode case with an `expected_mode`."""
    values: dict[str, Any] = dict.fromkeys(_KG_KEYS, None)
    values.update(M.compute_kg_entity_metrics(run.kg_hits, case.expected_entities))

    if run.cypher_query and run.cypher_query.strip():
        values["cypher_validity"] = M.cypher_validity(run.cypher_read_only, run.kg_retrieval_error)
        values["cypher_nonempty"] = 1.0 if run.kg_hits else 0.0
        values["kg_source_grounding"] = M.kg_source_grounding(
            run.cited_kg_indices, len(run.kg_hits)
        )

    if case.expected_mode and case.retrieval.retrieval_mode == "auto":
        values["mode_routing_correctness"] = M.mode_routing_correct(
            run.routed_mode, case.expected_mode
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
    values["agent_input_tokens"] = run.input_tokens
    values["agent_output_tokens"] = run.output_tokens
    values["agent_total_tokens"] = run.total_tokens
    values["latency_seconds"] = run.latency_seconds

    return {k: _round(k, v) for k, v in values.items()}


def _status(case: EvalCase, values: dict[str, Any], run: CaseRun, cfg: EvalConfig) -> str:
    """PASS / REVIEW verdict. Any error forces REVIEW. Retrieval gate applies
    only when the source group ran AND the case has gold sources."""
    if run.error:
        return "REVIEW"
    retrieval_ok = True
    if "source" in cfg.enabled_groups and case.expected_sources:
        retrieval_ok = values.get("hit_at_k") == 1.0
    keywords_ok = (values.get("required_keyword_hit_rate") or 0.0) >= cfg.required_keyword_threshold
    keywords_ok = keywords_ok and (values.get("disallowed_keyword_hits") or 0) == 0
    return "PASS" if retrieval_ok and keywords_ok else "REVIEW"


async def evaluate_case(
    case: EvalCase, corpus_config: CorpusConfig | None, cfg: EvalConfig
) -> dict[str, Any]:
    """Run + score one case → a flat result dict (keys = registry columns +
    structural fields)."""
    run = await run_case(case, corpus_config)
    values = _compute_metrics(case, run, cfg)
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "answer": run.answer,
        "status": _status(case, values, run, cfg),
        "errors": [run.error] if run.error else [],
        **values,
    }


async def evaluate_cases(
    cases: list[EvalCase], corpus_config: CorpusConfig | None, cfg: EvalConfig
) -> list[dict[str, Any]]:
    """Evaluate every case under a concurrency semaphore, preserving order."""
    sem = asyncio.Semaphore(max(1, cfg.concurrency))

    async def _one(case: EvalCase) -> dict[str, Any]:
        async with sem:
            return await evaluate_case(case, corpus_config, cfg)

    return await asyncio.gather(*[_one(c) for c in cases])
