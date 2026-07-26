# Evaluation

The Evaluation harness measures how well retrieval and answering work on the
selected corpus, using **test cases** you author and **metrics** it scores. Metric
*definitions* live in the **Metrics Guide** sub-tab; this page is about *using* the
harness.

Jump to: [Test cases](#cases) · [Running](#running) · [Suites](#suites) ·
[Test-case matrix](#matrix) · [Dashboards](#dashboards) · [Good cases](#good-cases)

<a id="cases"></a>
## Create test cases

A test case is a question plus what a good result looks like, saved into a
**dataset** (a `.json` in the corpus folder). Build cases on the **Create test
cases** sub-tab: write a question, pin the retrieval knobs it should run under, and
add the expected passage(s) / answer. You can also **capture** a case straight from
a real search or chat, so it pins exactly what that query used.

<a id="running"></a>
## Run evaluation

On the **Run evaluation** sub-tab, pick a dataset and run it. Each case is
retrieved, answered, and scored against its metrics, and the results are written to
a report in the corpus's `eval_output/`. They then appear in the dashboards.

<a id="suites"></a>
## Suites

A **suite** is several datasets that share the same facts but differ in their
**knobs** (the same questions run under different retrieval settings) so you can
compare settings head-to-head on identical content.

### Example: a mode-comparison suite

A natural suite locks the *same* questions to a different retrieval mode per member
dataset, so **Compare Datasets** can overlay them:

| Dataset | Every case forced to |
|---|---|
| forced_vector | `lancedb_only` (vector) |
| forced_hybrid | `lancedb_only` (hybrid) |
| forced_graph | `neo4j_only` |
| forced_fused | `parallel_fused` |
| auto | `auto` (+ an `expected_mode` per case) |

Overlaying their metrics shows which facts live in text (chunk hit@k) vs cleanly in
the graph (KG hit@k), whether `parallel_fused` earns its extra latency + tokens with
higher grounded correctness / recall, and, via the `auto` member's mode-routing
correctness, whether the classifier routes each question to the mode that wins. Run
each member with only the relevant metric groups on (e.g. the graph dataset → `kg` +
`judge`, not `chunk` / `source`).

<a id="matrix"></a>
## Test-case matrix

What you want to test → which fields to fill, how to set the knobs, and which
metrics to watch:

| Test objective | Fill these fields | Knob strategy | Watch these metrics |
|---|---|---|---|
| Semantic text retrieval | `expected_chunks`, `required_keywords` | defaults (`lancedb_only` + hybrid) | chunk hit@k, chunk recall@k, required-keyword hit rate |
| Graph structure / relationships | `expected_entities` | `neo4j_only` or `parallel_fused` | KG hit@k, KG entity recall, Cypher non-empty |
| Answer truth / hallucination | `expected_answer_points` | fixed knobs, judge group **on** | faithfulness, hallucination, grounded correctness |
| Fusion / ranking knobs | `expected_chunks`, `expected_sources` | vary `rrf_rank_constant`; vector vs hybrid | MRR, NDCG@k, contextual precision |
| Result diversity (MMR) | `expected_chunks` | `use_mmr` on; `mmr_lambda` 0.1 vs 1.0 | contextual relevancy, chunk precision@k |
| Graph payload limits | `expected_entities` | vary `kg_max_rows` (5 vs 200) | latency, agent tokens, contextual relevancy |
| Isolate retrieval (no LLM) | `expected_chunks`, `expected_sources` | `direct_retrieval: true` | hit@k, chunk hit@k (judge + keyword gates skipped) |
| Raw graph execution | `user_cypher`, `expected_entities` | `neo4j_only` + read-only Cypher | Cypher validity, KG-source grounding |
| Routing / classifier | `expected_mode` | `retrieval_mode: auto` | mode-routing correctness |

<a id="dashboards"></a>
## Dashboards

- **Run Summary:** per-run cards, cases, and details.
- **Run Charts:** metric scores by case, metric balance, score distributions,
  correlations, latency, and token usage.
- **Compare Datasets:** one suite-run's members side by side.
- **Trends:** a metric tracked over time for a dataset.

<a id="good-cases"></a>
## Writing good test cases

- **One fact per case:** a focused question with a clear right answer scores more
  meaningfully than a broad one.
- **Make it self-contained:** the expected answer should be supported by the
  passages you pin, not by outside knowledge.
- **Pin the retrieval knobs:** a case saves the settings it runs under, so a run
  is reproducible regardless of your current global settings.
- **Capture from real use:** turning an actual search or chat into a case keeps it
  realistic and records what that query actually used.
