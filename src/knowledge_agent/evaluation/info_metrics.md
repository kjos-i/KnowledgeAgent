# Evaluation Metrics Reference

This document describes every metric the **KnowledgeAgent (KA)** evaluation
harness produces. Metrics come in two kinds: **LLM-judged** (scored by a
judge-model panel via DeepEval) and **deterministic** (computed directly
from retrieval results and the agent's answer).

The harness has **four toggleable metric groups** (`EvalConfig.enabled_groups`):

- **`source`** — source-level retrieval (hit@k, mrr, precision@k, recall@k,
  ndcg@k). Relevance is decided by **doc_id** (a document's SHA-256
  content hash) against `expected_sources`.
- **`chunk`** — chunk-level retrieval (the same five, snippet-substring
  relevance against `expected_chunks`) plus chunk-source grounding.
- **`kg`** — knowledge-graph metrics (Cypher validity / non-empty, KG
  hit@k, entity recall, KG-source grounding, mode-routing correctness).
- **`judge`** — DeepEval LLM-judged metrics. **Off by default** — it calls
  judge LLMs and costs money.

Always-on regardless of toggles: required/disallowed keyword checks, agent
tokens, latency, `pass_rate`, and `avg_judge_score`. Metrics from a disabled
group are stored NULL in the ledger and render as **"Not evaluated"**.

**The judge is a panel.** Each judge scores the *same* rubric (the 7 metrics
below); the only difference between judges is the **model**. Scores are
averaged per metric across the panel to dilute any single model's bias. Use
*different* models for a meaningful panel; an empty panel = one default
judge from the active provider. Judges run at temperature 0.

For the PASS/REVIEW gate logic jump to [Verdict Logic](#verdict-logic).

---

## LLM-Judged Metrics (DeepEval)

Each judge LLM scores the test case on these seven metrics. Range 0–1,
higher is better (except **hallucination**, where lower is better). Toggle
group: `judge`.

- **Answer Relevancy** — is the answer on-topic for the question? Catches an
  answer that drifts from what was asked (does not check truth). *One of the
  two judge gates.*
- **Faithfulness** — is every claim in the answer supported by the retrieved
  context? The primary hallucination safeguard. *The second judge gate.*
- **Contextual Precision** — are the relevant chunks ranked above irrelevant
  ones? Surfaces ranking problems in the fusion pipeline.
- **Contextual Recall** — does the retrieved context contain everything
  needed to produce the expected answer? Catches retrieval coverage gaps.
- **Contextual Relevancy** — what fraction of the retrieved context is
  relevant to the question? Quantifies noise in the context window.
- **Hallucination** — does the answer *contradict* the context? **Lower is
  better** (0.0 = no contradictions). Complements faithfulness (which checks
  *support*, not contradiction).
- **Grounded Correctness (GEval)** — a holistic check: does the answer
  correctly address the question, cover the expected facts, and avoid
  unsupported claims? A single custom-criteria GEval call over input /
  actual output / expected output.

---

## Deterministic — Source Retrieval (by doc_id)

Computed from the retrieved chunks' **doc_ids** against `expected_sources`.
Toggle group: `source`. Cases with no `expected_sources` score 1.0 (nothing
to check), so they don't poison run-level averages.

- **Hit@k** — 1.0 if at least one expected doc_id appears in the retrieved
  set, else 0.0. The most basic retrieval health check. *Half of the
  retrieval gate.*
- **MRR** — reciprocal rank of the first relevant result (`1/rank`). How
  early the first expected document appears.
- **Precision@k** — fraction of retrieved chunks whose doc_id is expected.
  How much of the retrieval budget is spent on relevant documents.
- **Recall@k** — fraction of expected doc_ids that were retrieved. Retrieval
  completeness at the document level.
- **NDCG@k** — ranking quality vs. the ideal ranking, with logarithmic
  position decay. Sensitive to relevant documents scattered down the list.

---

## Deterministic — Chunk Retrieval (by snippet)

The same five metrics with a finer relevance notion: a retrieved chunk is
relevant when any snippet from `expected_chunks` appears (normalized
substring) in the chunk text. Toggle group: `chunk`. Prefixed `chunk_`.
Catches pipelines that fetch the right *document* but the wrong *passage*.

- **Chunk Hit@k / Chunk MRR / Chunk Precision@k / Chunk Recall@k /
  Chunk NDCG@k** — chunk-level analogues of the source metrics above.
- **Chunk-Source Grounding** — of the chunk citations the answer makes
  (`chunk_sources`), what fraction point at a chunk that was actually
  retrieved? A structural citation-accuracy check (the judge's *faithfulness*
  is the semantic counterpart).

---

## Knowledge Graph (KA-specific)

Deterministic metrics over the Neo4j leg — the Cypher the agent ran and the
rows it returned. Toggle group: `kg`. Each is **N/A (None)** when it doesn't
apply: the Cypher metrics need a Cypher to have run; the entity metrics need
gold `expected_entities`; mode-routing needs an `auto`-mode case with an
`expected_mode`.

- **Cypher Validity** — did the generated (or user-supplied) Cypher parse,
  pass the read-only safety rails, and execute without error? (1.0 / 0.0)
- **Cypher Non-empty** — did the Cypher return at least one row?
- **KG Hit@k** — is at least one gold entity (`expected_entities`) present in
  the returned KG rows?
- **KG Entity Recall** — fraction of the gold entities found in the KG rows.
- **KG-Source Grounding** — of the KG citations the answer makes
  (`kg_sources` by row index), what fraction point at a real returned row?
  The graph twin of chunk-source grounding.
- **Mode-Routing Correctness** — for an `auto`-mode case, did the
  mode-classifier route to the `expected_mode` leg? (1.0 / 0.0)

---

## Keyword Checks (always-on)

Applied to the answer text; normalized (lowercased, accents stripped) before
matching.

- **Required Keyword Hit Rate** — fraction of `required_keywords` present in
  the answer. 1.0 when none are defined. *Half of the keyword gate.*
- **Disallowed Keyword Hits** — count of `disallowed_keywords` present.
  **Lower is better**; any non-zero value fails the keyword gate.

---

## Tokens & Latency (always-on)

- **Agent Input / Output / Total Tokens** — provider-normalized token usage
  for the agent run (via LangChain's usage callback). Null when the provider
  reports none (e.g. Ollama).
- **Judge Input / Output / Total Tokens** — token usage for the judge panel
  (evaluation-only cost; separate from the agent's production cost).
- **Latency** — total wall-clock seconds for the agent to answer one case.
  (KA times only total latency — there is no separate retrieval/LLM split.)

---

## Summary

- **Pass Rate** — fraction of a run's cases with a PASS verdict. The headline
  "is this run good?" number; track it over time to catch regressions.
- **Avg Judge Score** — per case, the mean of that case's judge scores; at
  run level, the mean across cases. Null when the judge group is off.

---

## Pathways

A KA case pins its own retrieval **pathway** (in `RetrievalSettings`):
`retrieval_mode` (which store/leg), `lancedb_search_mode` (hybrid / fts /
vector), `skip_query_builder`, `direct_retrieval`, and optional raw
`user_cypher`. Two have scoring consequences:

- **`direct_retrieval`** — the agent returns the retrieved chunks as sources
  with **no synthesized answer** (by design — no LLM call). Such a case is
  scored on **retrieval only**: the judge and answer-keyword gates don't
  apply, and an empty answer is expected, not a failure.
- **`user_cypher`** — raw read-only Cypher runs verbatim (bypassing the
  Cypher builder); the KG metrics score the rows it returns.

---

<a id="verdict-logic"></a>
## Verdict Logic (PASS / REVIEW)

Each case's `status` is PASS only when **all applicable gates** pass; any
failing gate or runtime error yields REVIEW. The `errors` field records every
failure that fired.

| Gate | Condition |
|------|-----------|
| **retrieval** | `hit_at_k == 1.0` — applied only when the `source` group ran **and** the case has gold `expected_sources` |
| **keywords** | `required_keyword_hit_rate >= required_keyword_threshold` **and** `disallowed_keyword_hits == 0` |
| **judge** | `faithfulness >= judge_threshold` **and** `answer_relevancy >= judge_threshold` — applied only when the `judge` group ran |

**Pathway carve-out:** a `direct_retrieval` case is gated on the **retrieval**
condition alone (there's no answer, so the keyword + judge gates are skipped
and the judge panel isn't run).

### Thresholds

Two configurable thresholds (`EvalConfig`, persisted per run in
`eval_runs.gate_thresholds`):

| Threshold | Applies to |
|-----------|-----------|
| `required_keyword_threshold` | required keyword hit rate (keyword gate) |
| `judge_threshold` | faithfulness + answer_relevancy (judge gate), and each DeepEval metric's own pass line |

The binary conditions (`hit_at_k == 1.0`, `disallowed_keyword_hits == 0`) are
intentionally not configurable. Keep thresholds stable across runs so trend
comparisons stay apples-to-apples.
