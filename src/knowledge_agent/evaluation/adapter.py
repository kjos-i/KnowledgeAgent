"""KA graph adapter — the one place the harness couples to the agent.

Isolates everything KA-specific so the metric layer stays provider- and
framework-agnostic. It invokes the compiled graph, attaches a usage
callback for provider-normalized token counts, and reads the TYPED final
state into a flat `CaseRun` that the engine + metric functions consume.

Why this is clean by construction: every KA graph node runs through
`with_structured_output`, so the state carries typed objects
(`AgentAnswer` / `RetrievedChunk` / `KGHit`) — never raw provider JSON.
The agent's provider (anthropic / openai / google / ollama) is invisible
here. Tokens come from LangChain's usage callback (input/output/total,
provider-normalized); Ollama may report none → None, which `safe_mean`
drops from run-level means downstream.

`retrieved_chunks` come back IN the final state, so — unlike the
reference harness — there's no separate retriever call, and thus no
retrieval-vs-LLM latency split (only total wall-clock). `lancedb_search_mode`
is a Settings-level knob, not a graph-state field, so it is NOT injected
per case here (a P1 limitation; the EvalCase field is reserved).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import UsageMetadataCallbackHandler

if TYPE_CHECKING:
    from knowledge_agent.evaluation.models import EvalCase
    from knowledge_agent.kg.corpus_config import CorpusConfig


@dataclass(slots=True)
class CaseRun:
    """Normalized result of running one case through the graph — the single
    struct both metric tracks read. KG fields are populated now but only
    consumed by the Phase-2 KG metrics."""

    question: str
    answer: str
    # ---- LanceDB retrieval side ----
    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    # ---- KG side (Phase 2) ----
    kg_hits: list[dict[str, Any]] = field(default_factory=list)
    cited_kg_indices: list[int] = field(default_factory=list)
    cypher_query: str | None = None
    routed_mode: str | None = None
    search_query: str | None = None
    # ---- usage / timing ----
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float = 0.0
    # ---- error (str repr, None on success) ----
    error: str | None = None


def _sum_tokens(cb: UsageMetadataCallbackHandler) -> tuple[int | None, int | None, int | None]:
    """Sum token usage across every model the callback saw. Returns
    (None, None, None) when nothing was reported (e.g. Ollama)."""
    usage = getattr(cb, "usage_metadata", None) or {}
    if not usage:
        return None, None, None
    inp = sum(int(u.get("input_tokens", 0) or 0) for u in usage.values())
    out = sum(int(u.get("output_tokens", 0) or 0) for u in usage.values())
    tot = sum(int(u.get("total_tokens", 0) or 0) for u in usage.values())
    return inp, out, tot


def build_case_run(
    question: str,
    final_state: dict[str, Any],
    cb: UsageMetadataCallbackHandler,
    latency_seconds: float,
    error: str | None,
) -> CaseRun:
    """Read a graph `final_state` (typed) into a flat `CaseRun`.

    Pure w.r.t. the graph — takes the already-run state so it's unit-testable
    without invoking anything.
    """
    answer_obj = final_state.get("final_answer")
    answer = getattr(answer_obj, "answer", "") or ""
    chunk_sources = getattr(answer_obj, "chunk_sources", []) or []
    kg_sources = getattr(answer_obj, "kg_sources", []) or []

    chunks = final_state.get("retrieved_chunks", []) or []
    kg_hits = final_state.get("kg_hits", []) or []
    inp, out, tot = _sum_tokens(cb)

    return CaseRun(
        question=question,
        answer=answer,
        retrieved_doc_ids=[c.doc_id for c in chunks],
        retrieved_chunk_ids=[c.chunk_id for c in chunks],
        retrieved_texts=[c.text for c in chunks],
        cited_chunk_ids=[cs.chunk_id for cs in chunk_sources],
        kg_hits=[h.data for h in kg_hits],
        cited_kg_indices=[ks.hit_index for ks in kg_sources],
        cypher_query=final_state.get("cypher_query"),
        routed_mode=final_state.get("routed_mode"),
        search_query=final_state.get("search_query"),
        input_tokens=inp,
        output_tokens=out,
        total_tokens=tot,
        latency_seconds=latency_seconds,
        error=error,
    )


async def run_case(
    case: EvalCase,
    corpus_config: CorpusConfig | None,
    *,
    graph: Any = None,
) -> CaseRun:
    """Invoke the KA graph for one case and normalize the typed result.

    Injects the state-supported per-case overrides (query, retrieval_mode,
    top_k, corpus_config) + a usage callback. A graph error is captured as
    a string on the returned `CaseRun` (empty answer / empty retrieval) so
    one bad case can't sink the whole run. `graph` is injectable for tests;
    None lazy-imports the real compiled graph.
    """
    if graph is None:
        from knowledge_agent.graph import graph as compiled_graph

        graph = compiled_graph

    cb = UsageMetadataCallbackHandler()
    invoke_state: dict[str, Any] = {
        "query": case.question,
        "retrieval_mode": case.retrieval.retrieval_mode,
        "top_k": case.retrieval.top_k,
        "corpus_config": corpus_config,
    }

    started = time.perf_counter()
    error: str | None = None
    final_state: dict[str, Any] = {}
    try:
        final_state = await graph.ainvoke(invoke_state, config={"callbacks": [cb]})
    except Exception as exc:
        # One bad case must not sink the whole run — capture + carry on.
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started

    return build_case_run(case.question, final_state, cb, latency, error)
