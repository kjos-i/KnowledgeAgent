"""LLM-judge track (Phase 3) — DeepEval metrics over a user-set panel.

The judge is a PANEL: a list of model IDs (`EvalConfig.judge_models`),
one per judge, run on the user's configured LLM provider — the same
choose-your-provider pattern as the search LLM, no hardcoded default.
Each model scores the case independently; scores are aggregated (mean
per metric) across the panel, which dilutes single-model bias. The GUI
Evaluation tab just fills that list later.

Ported from the HybridSearchAgent reference's proven DeepEval usage, with
three KA adaptations: (1) provider-agnostic — the judge LLM is built via
`llm_factory.get_llm` (active provider) instead of a hardcoded ChatOpenAI;
(2) a panel of models rather than one; (3) importable WITHOUT `deepeval`
installed (the deterministic harness must not depend on it) — `deepeval`
is imported lazily and the base-class subclass degrades to `object`.

`deepeval` ships as the optional `eval` extra: `pip install
knowledge-agent[eval]`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation.metrics import safe_mean

if TYPE_CHECKING:
    from knowledge_agent.evaluation.adapter import CaseRun
    from knowledge_agent.evaluation.models import EvalCase

# The 7 DeepEval metric keys — must match the registry's "llm" group.
JUDGE_METRIC_KEYS: tuple[str, ...] = (
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
    "hallucination",
    "correctness_g_eval",
)

# Subclass the DeepEval base when available; fall back to `object` so this
# module imports even without deepeval (the judge track just won't run).
try:
    from deepeval.models.base_model import DeepEvalBaseLLM as _BaseLLM
except ImportError:  # pragma: no cover - exercised only without the extra
    _BaseLLM = object


def resolve_judge_models(judge_models: tuple[str, ...] | list[str]) -> list[str]:
    """The panel to run: the user's list, or a single default from the
    active provider when unset. The default reuses the mode-classifier
    model (a cheap tier in the configured provider, typically distinct from
    the synthesizer) — never a hardcoded cross-provider model."""
    if judge_models:
        return list(judge_models)
    from knowledge_agent.config import get_settings

    return [get_settings().mode_classifier_model]


def build_judge_input(run: CaseRun, case: EvalCase) -> dict[str, Any]:
    """Assemble the DeepEval test-case fields from the normalized run + gold.

    `retrieval_context` includes BOTH the retrieved chunk texts and the KG
    rows, so faithfulness/correctness judge the answer against the full
    evidence (this is how the KG leg reaches the judge track — no separate
    KG judge metric needed)."""
    retrieval_context = list(run.retrieved_texts) + [str(h) for h in run.kg_hits]
    return {
        "input": case.question,
        "actual_output": run.answer or "",
        "expected_output": "\n".join(case.expected_answer_points),
        "context": list(case.expected_answer_points),
        "retrieval_context": retrieval_context,
    }


class _JudgeLLM(_BaseLLM):
    """DeepEval-compatible judge LLM wrapping a KA `get_llm` client, with
    per-instance token accounting. Provider-agnostic: the model runs on the
    active `llm_provider`."""

    def __init__(self, model: str) -> None:
        from knowledge_agent.llm_factory import get_llm

        self._model_name = model
        self._client = get_llm(model, 0.0)  # temp 0 for judge determinism
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = asyncio.Lock()

    def load_model(self) -> Any:
        return self._client

    def get_model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = structured.invoke(prompt)
            self._record(result.get("raw"))
            return result["parsed"]
        response = self._client.invoke(prompt)
        self._record(response)
        return response.content

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        if schema is not None:
            structured = self._client.with_structured_output(schema, include_raw=True)
            result = await structured.ainvoke(prompt)
            async with self._lock:
                self._record(result.get("raw"))
            return result["parsed"]
        response = await self._client.ainvoke(prompt)
        async with self._lock:
            self._record(response)
        return response.content

    def _record(self, message: Any) -> None:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            self.input_tokens += int(usage.get("input_tokens", 0) or 0)
            self.output_tokens += int(usage.get("output_tokens", 0) or 0)


def _build_metrics(judge_llm: _JudgeLLM, threshold: float) -> dict[str, Any]:
    """Construct the 7 DeepEval metrics bound to one judge LLM (lazy import)."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
    )
    from deepeval.test_case import LLMTestCaseParams

    return {
        "answer_relevancy": AnswerRelevancyMetric(threshold=threshold, model=judge_llm),
        "faithfulness": FaithfulnessMetric(threshold=threshold, model=judge_llm),
        "contextual_precision": ContextualPrecisionMetric(threshold=threshold, model=judge_llm),
        "contextual_recall": ContextualRecallMetric(threshold=threshold, model=judge_llm),
        "contextual_relevancy": ContextualRelevancyMetric(threshold=threshold, model=judge_llm),
        "hallucination": HallucinationMetric(threshold=threshold, model=judge_llm),
        "correctness_g_eval": GEval(
            name="Grounded Correctness",
            criteria=(
                "Determine whether the actual output correctly answers the input, "
                "covers the important facts from the expected output, and avoids "
                "unsupported claims."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold,
            model=judge_llm,
        ),
    }


def _build_test_case(data: dict[str, Any]) -> Any:
    from deepeval.test_case import LLMTestCase

    return LLMTestCase(
        input=data["input"],
        actual_output=data["actual_output"],
        expected_output=data["expected_output"],
        context=data["context"],
        retrieval_context=data["retrieval_context"],
    )


async def _score_one_model(
    model: str, data: dict[str, Any], threshold: float
) -> tuple[dict[str, float | None], int, int]:
    """Run the 7 metrics for one judge model concurrently; return its scores
    + token usage. A metric that raises scores None (one judge failure must
    not sink the panel)."""
    judge_llm = _JudgeLLM(model)
    metrics = _build_metrics(judge_llm, threshold)
    test_case = _build_test_case(data)

    async def _one(name: str, metric: Any) -> tuple[str, float | None]:
        try:
            await metric.a_measure(test_case)
            score = getattr(metric, "score", None)
            return name, float(score) if score is not None else None
        except Exception:
            return name, None

    pairs = await asyncio.gather(*(_one(n, m) for n, m in metrics.items()))
    return dict(pairs), judge_llm.input_tokens, judge_llm.output_tokens


async def run_judge_panel(
    data: dict[str, Any], judge_models: list[str], threshold: float
) -> tuple[dict[str, float | None], int | None, int | None]:
    """Run the whole panel; aggregate each metric as the mean across models
    (None-safe), and sum judge tokens. Returns (scores, in_tokens, out_tokens)."""
    per_model = await asyncio.gather(*(_score_one_model(m, data, threshold) for m in judge_models))
    scores: dict[str, float | None] = {}
    for key in JUDGE_METRIC_KEYS:
        scores[key] = safe_mean(model_scores.get(key) for model_scores, _, _ in per_model)
    in_tok = sum(t[1] for t in per_model)
    out_tok = sum(t[2] for t in per_model)
    return scores, (in_tok or None), (out_tok or None)
