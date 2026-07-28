# Evaluation Metrics Reference

This document describes every metric the **KnowledgeAgent (KA)** evaluation harness produces. Metrics come in two categories: **LLM-judged metrics** (scored by a judge-model panel via DeepEval) and **deterministic metrics** (computed directly from retrieval results and the agent's answer).

The harness has four toggleable metric groups configured via `EvalConfig.enabled_groups` in `config.py`:

- **`judge`** — DeepEval LLM-judged metrics (faithfulness, answer_relevancy, contextual_*, hallucination, correctness_g_eval). **Off by default** — it calls judge LLMs and costs money.
- **`source`** — source-level retrieval metrics (hit_at_k, mrr, precision_at_k, recall_at_k, ndcg_at_k). Relevance is decided by **doc_id** (a document's SHA-256 content hash) against `expected_sources`.
- **`chunk`** — chunk-level retrieval metrics (snippet-substring match against `expected_chunks`), plus chunk-source grounding.
- **`kg`** — knowledge-graph metrics (Cypher validity / non-empty, KG hit@k, entity recall, KG-source grounding). On by default — its metrics are None (skipped) for cases whose leg didn't run, so it costs nothing on non-KG corpora.

Always-on regardless of toggles: `required_keyword_hit_rate`, `disallowed_keyword_hits`, chunk-source grounding, mode-routing correctness, agent tokens, latency, `pass_rate`, and `avg_judge_score`. Disabled metrics are stored as NULL in the ledger and CSV, and the dashboard renders them as "Not evaluated". Verdict gates skip sub-conditions whose metric group is disabled.

**The judge is a panel.** Each judge scores the *same* rubric (the 7 metrics below); the only difference between judges is the **model**. Scores are averaged per metric across the panel to dilute any single model's bias. Use *different* models for a meaningful panel; an empty panel = one default judge resolved from the active provider. Judges run at temperature 0.

<a id="no-gold"></a>
**No-gold cases and `n`.** A deterministic metric returns `None` (not 0.0 or 1.0) for a case that carries no gold for it — e.g. a case with no `expected_sources` isn't scored on source retrieval. `None` cases are *dropped* from the run-level average rather than scored, so a vacuous case never inflates or drags a mean. Because the denominator therefore varies per metric, each run records a per-metric **`n`** (how many cases actually fed each average), stored as `n_<metric>` and shown on the dashboard as `n = X of Y` — so a mean over 1 case is distinguishable from one over all of them.

For information about the verdict logic (which metrics actually flip a case from PASS to REVIEW) jump to [PASS/REVIEW Gates Verdict Logic](#verdict-logic) at the bottom of this document. For the authoritative sources each metric's definition was checked against, jump to [Sources and references](#sources) at the very bottom.

### Metrics at a Glance

**LLM-Judged (DeepEval)**

| Metric | Description |
|--------|-------------|
| [Answer Relevancy](#answer-relevancy) | Is the answer on-topic for the question? |
| [Faithfulness](#faithfulness) | Are all claims supported by the retrieved context? |
| [Contextual Precision](#contextual-precision) | Are relevant chunks ranked above irrelevant ones? |
| [Contextual Recall](#contextual-recall) | Does the context cover all needed information? |
| [Contextual Relevancy](#contextual-relevancy) | What fraction of retrieved chunks are relevant? |
| [Hallucination](#hallucination) | Does the answer contradict the context? |
| [Grounded Correctness (GEval)](#grounded-correctness-geval) | Is the answer correct compared to the expected output? |

**Deterministic — Source-level Retrieval** (relevance by doc_id)

| Metric | Description |
|--------|-------------|
| [Hit@k](#hitk) | Did at least one expected source appear in the retrieved set? |
| [MRR](#mean-reciprocal-rank-mrr) | How early does the first relevant result appear? |
| [Precision@k](#precisionk) | What fraction of retrieved results are relevant? |
| [Recall@k](#recallk) | What fraction of expected sources were retrieved? |
| [NDCG@k](#ndcgk-normalized-discounted-cumulative-gain) | How good is the overall ranking quality? |

**Deterministic — Chunk-level Retrieval** (relevance by snippet substring match)

| Metric | Description |
|--------|-------------|
| [Chunk Hit@k](#chunk-hitk) | Did at least one expected snippet appear in any retrieved chunk? |
| [Chunk MRR](#chunk-mrr) | How early does the first snippet-matching chunk appear? |
| [Chunk Precision@k](#chunk-precisionk) | What fraction of retrieved chunks contain any expected snippet? |
| [Chunk Recall@k](#chunk-recallk) | What fraction of expected snippets were found in a retrieved chunk? |
| [Chunk NDCG@k](#chunk-ndcgk) | How good is the chunk ranking under snippet-level relevance? |
| [Chunk Citation Grounding](#chunk-citation-grounding) | Do the answer's chunk citations point at actually-retrieved chunks? |

**Knowledge Graph** (KA-specific)

| Metric | Description |
|--------|-------------|
| [Cypher Validity](#cypher-validity) | Did the Cypher pass the read-only rail and execute without error? |
| [Cypher Non-empty](#cypher-non-empty) | Did the Cypher return at least one row? |
| [KG Entity Hit](#kg-entity-hit) | Is at least one expected entity present in the returned KG rows? |
| [KG Entity Recall](#kg-entity-recall) | What fraction of expected entities appear in the KG rows? |
| [KG Citation Grounding](#kg-citation-grounding) | Do the answer's KG citations point at real returned rows? |

**Orchestration** (KA-specific)

| Metric | Description |
|--------|-------------|
| [Mode Routing Correct](#mode-routing-correct) | Did an auto-mode case route to the expected leg? |

**Deterministic — Keyword Checks**

| Metric | Description |
|--------|-------------|
| [Required Keyword Hit Rate](#required-keyword-hit-rate) | Does the answer contain the required key terms? |
| [Disallowed Keyword Hits](#disallowed-keyword-hits) | Does the answer avoid disallowed terms? |

**Tokens and Latency**

| Metric | Description |
|--------|-------------|
| [Agent Tokens](#agent-tokens) | Provider-normalized token usage for the agent run. |
| [Judge Tokens](#judge-tokens) | Token usage for the judge panel (evaluation-only cost). |
| [Latency](#latency) | Total wall-clock time for the agent to answer a case. |

**Summary**

| Metric | Description |
|--------|-------------|
| [Pass Rate](#pass-rate) | Fraction of cases with a PASS verdict (run-level). |
| [Avg Judge Score](#avg-judge-score) | Mean of a case's LLM-judged scores, then meaned across cases at run level. |

---

## LLM-Judged Metrics (DeepEval)

These metrics use a **judge panel** (the models in `EvalConfig.judge_models`, one judge each) to score every test case. Each judge runs on the active LLM provider at temperature 0 and is wrapped in a token-tracking client so judge-panel token usage accumulates per case. Every judge scores the same 7 metrics, so each case ends up with **one averaged score per metric** (seven in all). For each metric *separately*, the panel's per-judge scores are combined as the **mean across the judges** (None-safe: a judge that errors is skipped), which dilutes any single model's bias. Each metric returns a 0–1 score. These seven per-case scores are later blended into a single [Avg Judge Score](#avg-judge-score) as a summary; the per-metric columns keep the raw mean.

The DeepEval test case is built in `judge.build_judge_input`: `input` = the case question, `actual_output` = the agent's answer, `expected_output` = the case's `expected_answer_points`, `context` = those same points, and `retrieval_context` = the retrieved chunk texts **plus the KG rows** (so the KG leg's evidence reaches the judge — no separate KG judge metric needed). Sparse cases fall back to sentinels so the contextual/hallucination metrics stay well-defined.

---

<a id="answer-relevancy"></a>
### Answer Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.answer_relevancy`, `eval_runs.avg_answer_relevancy` |
| Computed by | DeepEval `AnswerRelevancyMetric` (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (configurable in `config.py`); also one half of the judge gate for the case verdict |

**What it evaluates:** Whether the agent's answer is on-topic for the user's question, i.e. that the response actually addresses what was asked rather than providing tangential or off-topic information. It does not check whether the information is true.

**Why it is included:** A RAG system can retrieve perfect context but still produce an answer that drifts from the question. This metric catches cases where the LLM ignores the input or over-generalises. It is one of the two judge gates that flips a case to REVIEW.

**How it is calculated:** The judge LLM splits the actual output into individual statements and rates each for relevance to the input query. Score = relevant statements / total statements, then averaged across the panel.

---

<a id="faithfulness"></a>
### Faithfulness

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.faithfulness`, `eval_runs.avg_faithfulness` |
| Computed by | DeepEval `FaithfulnessMetric` (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (configurable in `config.py`); also one half of the judge gate for the case verdict |

**What it evaluates:** Whether each factual claim in the answer contradicts the retrieved context or not. A faithful answer makes no statements that conflict with what was retrieved.

**Why it is included:** A safeguard against answers that contradict the retrieved evidence in RAG. If the agent states something that conflicts with the retrieved documents, the answer is unreliable even if it sounds correct. It is the second of the two judge gates that flips a case to REVIEW.

**How it is calculated:** The judge LLM extracts individual claims from the answer and checks each against the retrieval context. Score = non-contradicting claims / total claims, then averaged across the panel. A claim counts as faithful as long as it doesn't contradict the retrieved context, so a claim the context simply doesn't mention still passes. That makes it weaker than strict entailment: it catches statements that conflict with the evidence, not fabrications the context happens to be silent on. (This "unmentioned still passes" behavior follows DeepEval's default `penalize_ambiguous_claims=False`, which the harness does not override; setting it True would additionally penalize claims the retrieved context can't verify.)

---

<a id="contextual-precision"></a>
### Contextual Precision

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_precision`, `eval_runs.avg_contextual_precision` |
| Computed by | DeepEval `ContextualPrecisionMetric` (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the relevant chunks in the retrieval context are ranked higher than irrelevant ones. It measures ranking quality, not just presence.

**Why it is included:** Retrieval order matters: if relevant documents are buried below noise, the LLM may miss or deprioritise them. This metric surfaces ranking problems that source-level Hit@k and Recall@k can't see.

**How it is calculated:** The judge LLM labels each chunk in the retrieval context as relevant or irrelevant, judging relevance from the question and the expected answer. DeepEval then computes a **weighted cumulative precision** over that ranked list, using the same idea as Average Precision in information retrieval: at each position where a *relevant* chunk sits, it measures the precision up to that point (of the chunks seen so far, how many are relevant), and averages those values over all the relevant chunks. A relevant chunk that appears after many irrelevant ones has a low precision-up-to-that-point, so the score is dragged down when relevant chunks are buried deep, and approaches 1.0 when they're all near the top. It scores ranking *order*, not just whether relevant chunks are present.

---

<a id="contextual-recall"></a>
### Contextual Recall

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_recall`, `eval_runs.avg_contextual_recall` |
| Computed by | DeepEval `ContextualRecallMetric` (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the retrieval context contains all the information needed to produce the expected output. It checks completeness of the retrieved evidence.

**Why it is included:** A retrieval system might return chunks that are individually relevant but collectively miss key facts. Catches gaps in retrieval coverage that show up as the LLM "not knowing" something it should have been told.

**How it is calculated:** The judge LLM extracts each individual statement from the expected output and checks whether it can be attributed to any chunk in the retrieved context. Score = attributable statements / total statements in the expected output. ("Attributed" means the statement is backed up by something in a retrieved chunk, so a high score means the retrieved context covers everything the ideal answer needs, and a low score means the retriever left gaps.)

---

<a id="contextual-relevancy"></a>
### Contextual Relevancy

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.contextual_relevancy`, `eval_runs.avg_contextual_relevancy` |
| Computed by | DeepEval `ContextualRelevancyMetric` (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether each chunk in the retrieval context is relevant to the input question. Unlike contextual precision (which focuses on ranking), this measures the overall noise level in the context window.

**Why it is included:** Retrieving too many irrelevant chunks dilutes the LLM's attention and can produce worse answers and higher latency. Quantifies how much noise the retrieval pipeline is introducing independent of where in the ranking the noise appears.

**How it is calculated:** The judge LLM splits the retrieval context into individual statements and rates each for relevance to the input. Score = relevant statements / total statements in the context. (So a high score means most of what was retrieved is on-topic with little noise, and a low score means the context is padded with irrelevant material.)

---

<a id="hallucination"></a>
### Hallucination

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | **lower is better** (target: 0.0; this is the only judge metric where high is bad) |
| Toggle group | judge |
| Stored as | `eval_cases.hallucination`, `eval_runs.avg_hallucination` |
| Computed by | DeepEval `HallucinationMetric` (built in `judge._build_metrics`) |
| Pass condition | `score <= judge_threshold` (DeepEval inverts the comparison for this metric; not a verdict gate) |

**What it evaluates:** Whether the agent's answer contradicts its reference context. Both Faithfulness and Hallucination are contradiction checks; they differ in two ways. First, *what* each compares the answer against: Faithfulness uses the retrieved context (the chunks), while Hallucination uses the case's gold expected-answer points (what the harness passes to DeepEval as the `context` field). Second, the *direction*: Faithfulness counts the claims that don't contradict (higher is better), while Hallucination counts the contexts that are contradicted (so it is inverted: lower is better).

**Why it is included:** A hallucinated answer is worse than an incomplete one because it actively misleads the user. It complements Faithfulness: Hallucination checks contradictions against the gold answer points, Faithfulness against the retrieved chunks.

**How it is calculated:** The judge LLM emits one verdict per context item (here each of the case's gold expected-answer points) indicating whether the answer contradicts it. Score = contradicting verdicts / total verdicts, so 0.0 means no contradictions detected and 1.0 means every context item was contradicted. DeepEval's `is_successful()` for this metric returns `score <= threshold`, the opposite of every other judge metric, so on the dashboard, treat the hallucination column as "smaller = better".

---

<a id="grounded-correctness-geval"></a>
### Grounded Correctness (GEval)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | judge |
| Stored as | `eval_cases.correctness_g_eval`, `eval_runs.avg_correctness_g_eval` |
| Computed by | DeepEval `GEval` with custom criteria (built in `judge._build_metrics`) |
| Pass condition | `score >= judge_threshold` (judge metric line, not a verdict gate) |

**What it evaluates:** Whether the agent's answer correctly addresses the user's request, covers the important facts from the expected output, and avoids unsupported claims. A holistic correctness check.

**Why it is included:** The other judge metrics evaluate individual dimensions (relevancy, faithfulness, context quality), but none directly ask "is this answer correct?". GEval fills that gap by comparing the actual answer against the expected answer points.

**How it is calculated:** A single GEval call with criteria *"Determine whether the actual output correctly answers the user's request, covers the important facts from the expected output, and avoids unsupported claims."*. From that criteria, DeepEval first generates a set of chain-of-thought **evaluation steps**, then applies those steps to `INPUT`, `ACTUAL_OUTPUT`, and `EXPECTED_OUTPUT` to produce **one holistic score**. That score is normalized via a weighted summation of the LLM's output-token probabilities (the G-Eval technique for reducing scoring bias), yielding the final 0–1 value. Each judge model in the panel runs this independently, generating its **own** chain-of-thought steps and score; those per-judge scores are then averaged, as for every judge metric.

<!--section-rule-->

## Deterministic Retrieval Metrics

These metrics are computed directly from the retrieval results without an LLM judge, so they isolate retrieval quality from generation behavior. Retrieval quality is measured on two axes:

- **Source-level** — relevance is decided by **doc_id** match against `expected_sources`. A `doc_id` is a document's SHA-256 content hash — the identity KA assigns at ingest and carries on every chunk — so relevance is content-addressed: unique per document and stable across renames (unlike a filename).
- **Chunk-level** — relevance is decided by substring match against `expected_chunks` (a list of short representative text snippets on the case). A retrieved chunk is relevant when any expected snippet appears inside the chunk's text after normalization. Snippets are preferred over chunk IDs because chunk IDs change whenever the chunking strategy is tuned, while short representative text remains stable across re-chunking.

Both axes produce the same five metrics (hit@k, MRR, precision@k, recall@k, NDCG@k). Chunk-level variants are prefixed with `chunk_`. Throughout, **k** is the retrieval depth: the number of chunks the case's pathway actually retrieved (at most its `top_k`). The **source** axis is document-level: those retrieved chunks are first collapsed to the distinct documents they came from (first-occurrence order, so a document's rank is the position of its first retrieved chunk), and all five metrics are scored over that document list. So several chunks from one document count as a single slot everywhere at source level (precision@k here means distinct relevant documents over distinct retrieved documents). The **chunk** axis instead keeps every retrieved chunk as its own slot, which is where per-passage granularity lives. Cases with no `expected_sources` or no `expected_chunks` score **None** for their axis (nothing to check) and are dropped from run-level averages — see [No-gold cases and `n`](#no-gold) above.

**Text normalization** (`metrics.normalize_text`): lowercase, strip accents, drop punctuation (keeping only `.` `%` `-` among symbols), and collapse whitespace — so keyword and snippet matching ignore case/accent/punctuation/spacing noise. Non-ASCII letters and digits (Greek, CJK, accented bases) are **kept**, so `β` stays distinct from `α` and multilingual corpora aren't degraded.

---

<a id="hitk"></a>
### Hit@k

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.hit_at_k`, `eval_runs.avg_hit_at_k` |
| Computed by | `compute_source_metrics` in `metrics.py` |
| Pass condition | `value == 1.0` (binary by design; the retrieval gate) |

**What it evaluates:** A binary signal of whether at least one expected source appears anywhere in the retrieved results. Relevance is decided at the source (doc_id) level, not the chunk level.

**Why it is included:** The most basic retrieval health check. Did the system find *anything* right? If no expected source appears at all, no amount of good ranking or generation can save the answer. Complements [recall@k](#recallk), which reports *how much* of the expected set was found.

**How it is calculated:** `1.0 if (expected_sources ∩ retrieved_doc_ids) else 0.0`. Both sides are doc_id sets, so duplicates collapse (five chunks from one document count as one hit, not five). `None` when the case defines no expected sources.

**Worked example:** expected sources = `{d1, d2, d3}` (doc_ids).

| Retrieved chunks' doc_ids | Unique retrieved docs | Intersection | Score |
|---------------------------|-----------------------|--------------|-------|
| `d1`, `d2`, `d9` | `{d1, d2, d9}` | `{d1, d2}` | 1.0 |
| `d1`, `d1`, `d2` | `{d1, d2}` | `{d1, d2}` | 1.0 |
| `d1` ×10 | `{d1}` | `{d1}` | 1.0 |
| `d7`, `d8`, `d9` | `{d7, d8, d9}` | `{}` | 0.0 |
| `d1`, `d2`, `d3` | `{d1, d2, d3}` | `{d1, d2, d3}` | 1.0 |

---

<a id="mean-reciprocal-rank-mrr"></a>
### Mean Reciprocal Rank (MRR)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.mrr`, `eval_runs.avg_mrr` |
| Computed by | `compute_source_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** How early the first relevant document appears in the ranked list, after the retrieved chunks are collapsed to distinct documents. 1.0 if the first document is relevant, 0.5 if the second is, 0.33 if the third is, and so on. Relevance is decided at the source (doc_id) level.

**Why it is included:** In RAG, the top-ranked result has the most influence on the LLM's generation. MRR tells you whether the retrieval pipeline is placing the most important document first or burying it.

**How it is calculated:** `1 / rank`, where `rank` is the 1-indexed position of the first **distinct retrieved document** that is an expected source (the retrieved chunks are collapsed to distinct documents in first-occurrence order first). `0.0` if no expected source is found, `None` if the case defines no expected sources.

Strictly, a single case's value is the **Reciprocal Rank**; the run-level average (`avg_mrr`) is the **Mean** Reciprocal Rank. The per-case field is named `mrr` for convenience.

---

<a id="precisionk"></a>
### Precision@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.precision_at_k`, `eval_runs.avg_precision_at_k` |
| Computed by | `compute_source_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of the **distinct retrieved documents** that are expected sources, i.e. how document-precise the retrieval is. Relevance is decided at the source (doc_id) level.

**Why it is included:** A low precision means the retrieval pipeline is surfacing many irrelevant *documents* alongside the relevant ones. Reported alongside [recall@k](#recallk) for the standard precision/recall trade-off picture at the document level. (For how much of the *context window* is spent on answer-bearing passages rather than whole documents, see [chunk_precision_at_k](#chunk-precisionk).)

**How it is calculated:** `(distinct retrieved documents that are expected sources) / (total distinct retrieved documents)`. The retrieved chunks are collapsed to distinct documents first (first-occurrence order), so several chunks from one document count once. `None` if the case defines no expected sources; `0.0` if nothing was retrieved.

---

<a id="recallk"></a>
### Recall@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.recall_at_k`, `eval_runs.avg_recall_at_k` |
| Computed by | `compute_source_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of expected source documents that appear in the retrieved results. Retrieval completeness, decided at the source (doc_id) level.

**Why it is included:** Where [hit@k](#hitk) tells you *whether* anything relevant was found (binary health check), recall@k tells you *how much* of the expected set was covered. Together with precision@k it reveals the trade-off between retrieving broadly and retrieving precisely.

**How it is calculated:** `|expected_sources ∩ retrieved_doc_ids| / |expected_sources|` — deduped, so a document retrieved several times counts once toward finding the gold doc. Note this measures whether each expected *document* was retrieved, not whether specific expected passages were (for that, see [chunk_recall_at_k](#chunk-recallk)). `None` if the case defines no expected sources.

---

<a id="ndcgk-normalized-discounted-cumulative-gain"></a>
### NDCG@k (Normalized Discounted Cumulative Gain)

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | source |
| Stored as | `eval_cases.ndcg_at_k`, `eval_runs.avg_ndcg_at_k` |
| Computed by | `compute_source_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Ranking quality compared to the ideal ranking of the same relevance labels. Penalises relevant results appearing lower in the list, on a logarithmic decay.

**Why it is included:** MRR only looks at the first relevant result. NDCG evaluates the entire ranked list, so it's sensitive to cases where multiple relevant documents exist but are scattered across positions. The standard metric for evaluating ranked retrieval in information-retrieval research.

**How it is calculated:** First the retrieved chunks are collapsed to distinct documents (first-occurrence order), and each document gets a binary relevance label (1 if its doc_id ∈ `expected_sources`, else 0). NDCG is then built up in three steps:

1. **Cumulative Gain (CG)** simply adds up the relevance labels of the retrieved documents. With binary labels that is just the count of relevant documents found. It captures *how many* relevant docs you retrieved, but says nothing about *where* in the ranking they landed.

2. **Discounted Cumulative Gain (DCG)** fixes that by dividing each document's relevance by a discount that grows with its position, so a relevant document near the top counts for more than one buried lower down. `DCG = Σ rel_i / log2(rank_i + 2)`, where `rank_i` is the **0-indexed** position (the first slot's discount is `log2(0 + 2)` = 1, the second `log2(3)` ≈ 1.58, and so on; a 1-indexed position would instead read `log2(rank + 1)`).

3. **Normalizing by the ideal (IDCG)** turns DCG into a 0-to-1 score. The ideal DCG is the DCG you would get if those same labels were ranked perfectly (every relevant document first), i.e. the best score achievable for this case. `NDCG = DCG / IDCG`. A real ranking can never beat the ideal, so the score always lands in [0, 1]: 1.0 means the relevant documents were ranked as high as possible; lower means they were scattered further down.

`None` if the case defines no expected sources.

<!--section-rule-->

## Chunk-level Retrieval Metrics

These five metrics mirror the source-level ones but apply a finer notion of relevance: a retrieved chunk is relevant when any snippet from the case's `expected_chunks` appears as a substring of the chunk's text. Both the snippets and the chunk text are normalized (lowercased, accents stripped, punctuation dropped) before substring comparison, making the check tolerant to formatting differences.

Why a separate axis? A chunk from the right *document* can still miss the *passage* that actually answers the question. Chunk-level scores catch retrieval pipelines that find the right documents but rank the wrong chunk within them. Cases with an empty `expected_chunks` list score `None` for every chunk metric. All are computed by `compute_chunk_metrics` in `metrics.py` and toggle with the `chunk` group.

---

<a id="chunk-hitk"></a>
### Chunk Hit@k

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_hit_at_k`, `eval_runs.avg_chunk_hit_at_k` |
| Computed by | `compute_chunk_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** A binary signal of whether *any* retrieved chunk contains *any* expected snippet.

**Why it is included:** The chunk-level analogue of source-level Hit@k. Confirms the retriever surfaced at least one chunk containing answer-bearing text, not just a chunk from the right document. Catches the case where the right document was retrieved but the wrong passage was selected within it.

**How it is calculated:** `1.0 if any expected snippet is a substring of any retrieved chunk's normalized text, else 0.0`. `None` if `expected_chunks` is empty.

---

<a id="chunk-mrr"></a>
### Chunk MRR

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_mrr`, `eval_runs.avg_chunk_mrr` |
| Computed by | `compute_chunk_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The reciprocal rank of the first retrieved chunk that contains any expected snippet. How early in the ranked list the right *passage* (not just the right document) appears.

**Why it is included:** Source-level MRR can show a "perfect" 1.0 when the top chunk is from the right document but doesn't contain the answer text. Chunk MRR is the stricter version that demands the answer-bearing passage actually surface near the top.

**How it is calculated:** Iterate retrieved chunks in rank order; return `1 / rank` (1-indexed) for the first chunk whose normalized text contains any normalized expected snippet. `0.0` if no chunk matches, `None` if `expected_chunks` is empty.

As with [MRR](#mean-reciprocal-rank-mrr), a single case's value is strictly the **Reciprocal Rank**; the run-level `avg_chunk_mrr` is the **Mean** across cases.

---

<a id="chunk-precisionk"></a>
### Chunk Precision@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_precision_at_k`, `eval_runs.avg_chunk_precision_at_k` |
| Computed by | `compute_chunk_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of retrieved chunks that contain at least one expected snippet. How much of the context window is spent on actually useful passages, not just chunks from useful documents.

**Why it is included:** A retriever can hit perfect source-level precision while flooding the context with non-answer-bearing chunks from the right documents. This metric exposes that failure mode.

**How it is calculated:** `(chunks containing any snippet) / (total retrieved chunks)`. `None` if `expected_chunks` is empty; `0.0` if nothing was retrieved.

---

<a id="chunk-recallk"></a>
### Chunk Recall@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_recall_at_k`, `eval_runs.avg_chunk_recall_at_k` |
| Computed by | `compute_chunk_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of expected snippets that appear in at least one retrieved chunk. Coverage of the known-good passages.

**Why it is included:** Unlike source-level recall, this tracks distinct *passages* rather than distinct documents, so two expected passages from the same document each contribute one unit of recall. This catches the case where the retriever consistently grabs only one expected passage per document and misses the others.

**How it is calculated:** `|{snippet : snippet ∈ some retrieved chunk}| / |expected_chunks|`. `None` if `expected_chunks` is empty; `0.0` if nothing was retrieved.

---

<a id="chunk-ndcgk"></a>
### Chunk NDCG@k

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | chunk |
| Stored as | `eval_cases.chunk_ndcg_at_k`, `eval_runs.avg_chunk_ndcg_at_k` |
| Computed by | `compute_chunk_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Ranking quality when each retrieved chunk is labelled 1 if it contains any expected snippet, else 0. The chunk-level analogue of source-level NDCG.

**Why it is included:** Combines the snippet-aware relevance signal with NDCG's logarithmic position decay. Penalises pipelines that surface answer-bearing chunks but rank them below chunks that just happen to share a document.

**How it is calculated:** Build a binary relevance list by checking each retrieved chunk for any snippet match, then apply the same CG → DCG → IDCG → ratio computation described under [NDCG@k](#ndcgk-normalized-discounted-cumulative-gain) (here scored over retrieved chunks rather than distinct documents). `None` if `expected_chunks` is empty; `0.0` if nothing was retrieved.

---

<a id="chunk-citation-grounding"></a>
### Chunk Citation Grounding

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on |
| Stored as | `eval_cases.chunk_source_grounding`, `eval_runs.avg_chunk_source_grounding` |
| Computed by | `chunk_source_grounding` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of the answer's chunk citations that point at an actually-retrieved chunk — a structural anti-hallucination check on the citations themselves (the judge's *faithfulness* is the semantic counterpart).

**Why it is included:** An answer can cite a chunk_id it never actually retrieved (a fabricated citation). This catches that structurally, without an LLM. Always-on because it needs no gold — only the answer's citations and the retrieved set.

**How it is calculated:** `(cited chunk_ids that were retrieved) / (total cited chunk_ids)`. Returns 1.0 when the answer cited nothing (nothing to ground).

<!--section-rule-->

## Knowledge Graph Metrics (KA-specific)

Deterministic metrics over the Neo4j leg: the Cypher the agent ran and the rows it returned. Toggle group: `kg`. Each is **None (N/A)** when it doesn't apply: the Cypher metrics + KG-source grounding need a Cypher to have run; the entity metrics need gold `expected_entities`. Assembled in `engine._kg_metrics`.

---

<a id="cypher-validity"></a>
### Cypher Validity

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | kg |
| Stored as | `eval_cases.cypher_validity`, `eval_runs.avg_cypher_validity` |
| Computed by | `cypher_validity` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Whether the generated (or user-supplied) Cypher parsed, passed the read-only safety rails, and executed without a retrieval error.

**Why it is included:** A malformed or write-attempting Cypher is a functional failure of the KG leg that no content metric would catch. `None` when no Cypher ran.

**How it is calculated:** The score is `1.0` only when the query is both **read-only** and **error-free**, and `0.0` otherwise. Read-only means it passed our safety check that rejects any write (like `CREATE` / `DELETE` / `SET` / `MERGE`), so only read queries are allowed; error-free means it actually ran against the graph without throwing an error. A query that tries to write, or that fails to run, scores `0.0`, while a read-only query that runs cleanly scores `1.0`. (`None` when the case ran no Cypher.)

---

<a id="cypher-non-empty"></a>
### Cypher Non-empty

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | kg |
| Stored as | `eval_cases.cypher_nonempty`, `eval_runs.avg_cypher_nonempty` |
| Computed by | `engine._kg_metrics` (1.0 if the KG leg returned rows) |
| Pass condition | (not gated) |

**What it evaluates:** Whether the Cypher returned at least one row.

**Why it is included:** A valid Cypher that returns nothing usually means the query was too narrow or mistargeted, a distinct failure mode from an invalid query. `None` when no Cypher ran.

**How it is calculated:** `1.0` if the returned KG rows are non-empty, else `0.0`.

---

<a id="kg-entity-hit"></a>
### KG Entity Hit

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | kg |
| Stored as | `eval_cases.kg_hit_at_k`, `eval_runs.avg_kg_hit_at_k` |
| Computed by | `compute_kg_entity_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** Whether at least one gold entity (`expected_entities`) is present in the returned KG rows.

**Why it is included:** The KG analogue of source Hit@k: did the graph leg surface any of the entities the case expects? `None` when the case has no `expected_entities`.

**How it is calculated:** `1.0` if any normalized expected entity appears in the normalized text of any returned KG row, else `0.0`.

---

<a id="kg-entity-recall"></a>
### KG Entity Recall

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | kg |
| Stored as | `eval_cases.kg_entity_recall`, `eval_runs.avg_kg_entity_recall` |
| Computed by | `compute_kg_entity_metrics` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of gold entities that appear in the KG rows.

**Why it is included:** Where KG Entity Hit is binary, entity recall reports *how much* of the expected entity set the graph leg surfaced. `None` when the case has no `expected_entities`.

**How it is calculated:** `(expected entities found in any KG row) / (total expected entities)`, after normalization.

---

<a id="kg-citation-grounding"></a>
### KG Citation Grounding

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | kg |
| Stored as | `eval_cases.kg_source_grounding`, `eval_runs.avg_kg_source_grounding` |
| Computed by | `kg_source_grounding` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** The fraction of the answer's KG citations (by returned-row index) that point at a real returned row, the graph twin of [Chunk Citation Grounding](#chunk-citation-grounding).

**Why it is included:** Catches an answer that cites a KG row index that doesn't exist in the returned set (a fabricated graph citation). `None` when no Cypher ran.

**How it is calculated:** `(cited row indices that are in range) / (total cited row indices)`. Returns 1.0 when the answer cited no KG rows.

<!--section-rule-->

## Orchestration (KA-specific)

Metrics over the agent's orchestration decisions (which store or leg to use), not over the retrieved content. Always-on, and each self-gates to `None` when it doesn't apply. This is where future decision-quality metrics (query-rewrite quality, fusion effectiveness, tool selection) will live.

<a id="mode-routing-correct"></a>
### Mode Routing Correct

| Field | Value |
|-------|-------|
| Range | 0.0 or 1.0 |
| Direction | higher is better |
| Toggle group | always-on |
| Stored as | `eval_cases.mode_routing_correctness`, `eval_runs.avg_mode_routing_correctness` |
| Computed by | `mode_routing_correct` in `metrics.py` |
| Pass condition | (not gated) |

**What it evaluates:** For an `auto`-mode case, whether the mode-classifier routed to the leg the case expects (`expected_mode`).

**Why it is included:** When retrieval is left to `auto`, the classifier's routing decision is itself a thing to score, and a case can fail purely because it was routed to the wrong store. `None` unless the case is `auto`-mode with an `expected_mode`.

**How it is calculated:** `1.0` if the routed mode equals `expected_mode`, else `0.0`.

<!--section-rule-->

## Deterministic — Keyword Checks

Applied to the answer text; both the keywords and the answer are run through `normalize_text` (lowercased, accents stripped, punctuation dropped) before substring matching. Always-on regardless of toggles; computed by `required_keyword_hit_rate` / `disallowed_keyword_hits` in `metrics.py`.

---

<a id="required-keyword-hit-rate"></a>
### Required Keyword Hit Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always-on |
| Stored as | `eval_cases.required_keyword_hit_rate`, `eval_runs.avg_required_keyword_hit_rate` |
| Computed by | `required_keyword_hit_rate` in `metrics.py` |
| Pass condition | `value >= required_keyword_threshold` (configurable in `config.py`); one half of the keyword gate |

**What it evaluates:** The fraction of the case's `required_keywords` that appear in the agent's answer.

**Why it is included:** Some questions demand specific terms in the answer (e.g. a regulation number, a product name). This enforces that the answer contains the expected key terms — it catches the case where every judge metric passes but the answer paraphrases away from a required term.

**How it is calculated:** Score = (required keywords found in the answer) / (total required keywords), after normalization. Returns 1.0 when no required keywords are defined (vacuous pass).

---

<a id="disallowed-keyword-hits"></a>
### Disallowed Keyword Hits

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 |
| Direction | lower is better (target: 0) |
| Toggle group | always-on |
| Stored as | `eval_cases.disallowed_keyword_hits`, `eval_runs.avg_disallowed_keyword_hits` |
| Computed by | `disallowed_keyword_hits` in `metrics.py` |
| Pass condition | `value == 0` (binary by design; one half of the keyword gate) |

**What it evaluates:** The count of disallowed keywords that appear in the agent's answer.

**Why it is included:** Some answers should avoid certain terms (e.g. a refusal phrase, a deprecated product name, a competitor mention). Any non-zero count is a failure signal; binary by design so even one hit fails the gate.

**How it is calculated:** The count of distinct disallowed keywords that appear as substrings in the normalized answer. Returns 0 when no disallowed keywords are defined.

<!--section-rule-->

## Tokens and Latency

Always-on cost/performance signals captured per case. They have no pass thresholds — they surface regressions alongside quality metrics. All are lower-is-better.

---

<a id="agent-tokens"></a>
### Agent Tokens (Input / Output / Total)

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (or None when unreported) |
| Direction | lower is better |
| Toggle group | always-on |
| Stored as | `eval_cases.agent_input_tokens` / `agent_output_tokens` / `agent_total_tokens` (+ `eval_runs.avg_*`) |
| Computed by | LangChain usage callback in `adapter.run_case` |
| Pass condition | (not gated) |

**What it evaluates:** Provider-normalized token usage for the agent's run — the production cost of answering the case.

**Why it is included:** Token usage is a direct cost and a proxy for prompt/retrieval bloat. Tracking it per case catches cost regressions after prompt or model changes.

**How it is calculated:** Summed across every model the agent invoked, via LangChain's standardized usage metadata. `None` when the provider reports no usage (e.g. Ollama), so "couldn't measure" stays distinct from "actually zero".

---

<a id="judge-tokens"></a>
### Judge Tokens (Input / Output / Total)

| Field | Value |
|-------|-------|
| Range | integer ≥ 0 (or None) |
| Direction | lower is better |
| Toggle group | judge |
| Stored as | `eval_cases.judge_input_tokens` / `judge_output_tokens` / `judge_total_tokens` (+ `eval_runs.avg_*`) |
| Computed by | token-tracking judge client in `judge.py` |
| Pass condition | (not gated) |

**What it evaluates:** Token usage for the whole judge panel on a case — the *evaluation* cost, kept separate from the agent's production cost.

**Why it is included:** The judge panel can be the dominant cost of a run. Isolating it lets you see what evaluation itself costs. `None` when the judge group is off.

**How it is calculated:** Summed across every judge model in the panel via the token-tracking wrapper each metric is bound to.

---

<a id="latency"></a>
### Latency

| Field | Value |
|-------|-------|
| Range | float seconds ≥ 0 |
| Direction | lower is better |
| Toggle group | always-on |
| Stored as | `eval_cases.latency_seconds`, `eval_runs.avg_latency_seconds` |
| Computed by | `time.perf_counter()` around the graph invocation in `adapter.run_case` |
| Pass condition | (not gated) |

**What it evaluates:** Total wall-clock time, in seconds, for the agent to produce an answer for a single case — the full graph invocation from input to final answer.

**Why it is included:** End-to-end latency is what a user experiences. Tracking it per case spots slow queries and catches run-to-run regressions.

**How it is calculated:** `perf_counter()` around `graph.ainvoke`. KA measures **total latency only** — the retrieved chunks come back in the final state (there is no separate retriever call), so there is no retrieval-vs-LLM split to report.

<!--section-rule-->

## Summary Metrics

Run-level aggregates surfaced at the top of each report and in the `eval_runs` row, so you can compare runs over time without drilling into per-case details.

---

<a id="pass-rate"></a>
### Pass Rate

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary) |
| Stored as | `eval_runs.pass_rate` |
| Computed by | `build_summary` in `report.py` |
| Pass condition | (not a gate; this is the result of all gates) |

**What it evaluates:** The fraction of cases in a run whose final `status` is `"PASS"` (every applicable gate succeeded). Cases that fail any gate end with status `"REVIEW"`.

**Why it is included:** The headline number for "is this run good?". The trend over time tells you whether agent quality is regressing or improving across commits and configuration changes.

**How it is calculated:** `pass_count / case_count`. Unlike the drop-`None` metrics, pass_rate always uses the full case count. See [Verdict Logic](#verdict-logic) for how each case's status is determined.

---

<a id="avg-judge-score"></a>
### Avg Judge Score

| Field | Value |
|-------|-------|
| Range | 0.0 – 1.0 |
| Direction | higher is better |
| Toggle group | always (run-level summary; only meaningful when the judge group is enabled) |
| Stored as | `eval_cases.avg_judge_score` (per case), `eval_runs.avg_judge_run_score` (run level) |
| Computed by | `engine.evaluate_case` (per case) and `build_summary` in `report.py` (run level) |
| Pass condition | (not gated) |

**What it evaluates:** Per case: the mean of that case's DeepEval judge scores (on the 0–1 scale the judges use). Run level (`avg_judge_run_score`): the mean of those per-case values across the run.

**Why it is included:** A single judge-quality number per case and per run, useful as a sanity check alongside pass rate. A high pass rate with a low judge score might mean gates are too lenient; a low pass rate with a high judge score might mean retrieval/keyword gates are breaking even though content quality is fine. `None` when the judge group is disabled.

**How it is calculated:** Per case: the None-safe mean of the case's 7 judge scores, after direction-normalizing so higher is always better. Hallucination is lower-is-better, so it is inverted (`1 - score`) before the mean; otherwise a high (bad) hallucination score would inflate the overall. Only this blended average inverts, the per-metric columns keep the raw scores. Run level: the None-safe mean of those per-case values (NULL cases excluded).

<!--section-rule-->

<a id="verdict-logic"></a>
## PASS/REVIEW Gates Verdict Logic

Each case's `status` is `"PASS"` only when **all applicable gates** pass; any failing gate or runtime error yields `"REVIEW"`. Computed by `engine._status`. The `errors` field records every failure that fired.

| Gate | Condition |
|------|-----------|
| **retrieval** | `hit_at_k == 1.0` — applied only when the `source` group ran **and** the case has gold `expected_sources` |
| **keywords** | `required_keyword_hit_rate >= required_keyword_threshold` **and** `disallowed_keyword_hits == 0` |
| **judge** | `faithfulness >= judge_threshold` **and** `answer_relevancy >= judge_threshold` — applied only when the `judge` group ran (and both scores are present) |

**Pathway carve-out:** a `direct_retrieval` case is gated on the **retrieval** condition alone (there's no answer, so the keyword + judge gates are skipped and the judge panel isn't run).

### Gate thresholds

Two configurable thresholds gate the verdict, set **per run in the Run Evaluation tab** (the "Gate thresholds" fields). Their definitions and defaults live in `config.py` (`EvalConfig`), so this document doesn't hardcode the numbers. Each run's chosen values are persisted in `eval_runs.gate_thresholds`, and the dashboard flags drift between runs.

| Threshold | Applies to |
|-----------|-----------|
| `judge_threshold` | faithfulness + answer_relevancy (judge gate), and each DeepEval metric's own pass line |
| `required_keyword_threshold` | required_keyword_hit_rate (keyword gate) |

The binary-by-design conditions (`hit_at_k == 1.0`, `disallowed_keyword_hits == 0`) are intentionally not configurable. Keep thresholds stable across runs so trend comparisons stay apples-to-apples.

### Gate behaviour when metric groups are disabled

- **retrieval** — if `source` is disabled (or the case has no `expected_sources`), the gate is skipped and doesn't block a PASS.
- **keywords** — always evaluated; keyword checks are always-on regardless of toggles.
- **judge** — if `judge` is disabled, no DeepEval scores exist, so the gate is skipped and doesn't block a PASS.

<!--section-rule-->

<a id="sources"></a>
## Sources and references

Where each metric's definition was checked against an authoritative source.

**LLM-judged metrics** are implemented by the [DeepEval](https://deepeval.com) library, so DeepEval's own documentation is the source of truth: [Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy), [Faithfulness](https://deepeval.com/docs/metrics-faithfulness), [Contextual Precision](https://deepeval.com/docs/metrics-contextual-precision), [Contextual Recall](https://deepeval.com/docs/metrics-contextual-recall), [Contextual Relevancy](https://deepeval.com/docs/metrics-contextual-relevancy), [Hallucination](https://deepeval.com/docs/metrics-hallucination), [Grounded Correctness (GEval)](https://deepeval.com/docs/metrics-llm-evals).

**Deterministic retrieval metrics** (Hit@k, MRR, Precision@k, Recall@k, NDCG@k, and their chunk-level variants) are standard information-retrieval measures. Definitions checked against:

- EvidentlyAI: [Hit rate](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems), [Mean Reciprocal Rank](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr), [NDCG](https://www.evidentlyai.com/ranking-metrics/ndcg-metric), [Precision and Recall at K](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k)
- Meilisearch: [A practical guide to search relevance metrics](https://www.meilisearch.com/blog/search-relevance-metrics)

**Knowledge-graph and grounding metrics** rest on established concepts. KG Entity Hit and KG Entity Recall are the hit-rate and recall concepts (EvidentlyAI / Meilisearch, above). Chunk and KG Citation Grounding are the *structural* side of RAG groundedness / citation ([EvidentlyAI RAG guide](https://www.evidentlyai.com/llm-guide/rag-evaluation), [DeepEval RAG triad](https://deepeval.com/guides/guides-rag-triad)); the *semantic* side is Faithfulness / Hallucination above. Cypher Validity is a Cypher *valid-query* check: Neo4j's blog defines a "KG Valid Query" metric via **static** syntax / schema / property validation ([Neo4j — Verifying Cypher queries with CyVer](https://neo4j.com/blog/developer/verify-neo4j-cypher-queries-with-cyver/)); our version is the lighter **runtime** flavor (the query ran without error and passed the read-only rail), not static schema validation.

**Harness-defined (no external reference):** Cypher Non-empty (did the query return rows), Mode Routing Correct, the keyword checks, tokens, latency, and the run-level summaries are defined by this harness itself. Their behavior is the code in `evaluation/metrics.py`, `evaluation/judge.py`, and `evaluation/engine.py`.
