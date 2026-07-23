"""Smoke test for the evaluation harness end-to-end (deterministic, judge OFF).

Ingests the first test PDF into the TEST instance (`.env.test`), builds a tiny
1-case gold dataset whose `expected_sources` is that doc_id, then runs
`evaluation.runner.run()` and prints the report + per-case metrics. This is the
harness's first REAL end-to-end pass: the real agent graph per case, real
LanceDB retrieval, real deterministic metrics + SQLite ledger + JSON/CSV
report. The LLM judge is OFF (the default), so the only LLM cost is the
agent's own nodes (~one lancedb_only query).

Hits real services (same as smoke_agent): docling, OpenAlex, Voyage, LanceDB,
Neo4j, Anthropic. Requires `.env.test` (NEO4J_* + VOYAGE_API_KEY +
ANTHROPIC_API_KEY).

Lifecycle: clear any leftover chunks for this doc_id -> ingest -> run eval ->
print report -> clean up chunks. Report files land in a temp folder (printed)
for inspection.

Run from the project root:
    python scripts/smoke_eval.py

Automated counterpart: the harness is unit-tested throughout
tests/unit/evaluation/ (config / registry / metrics / engine / adapter / judge
/ ledger / report / runner, all mocked). This smoke is the live complement —
it is the first time the harness runs against a real corpus.
"""

import asyncio
import tempfile
from pathlib import Path

# Switch to the test instance BEFORE any import that triggers get_settings()
# (mirrors smoke_agent) — a wrong-instance-running state fails at auth rather
# than touching real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.corpus_config import CorpusConfig, LayerFlags  # noqa: E402
from knowledge_agent.evaluation.config import EvalConfig  # noqa: E402
from knowledge_agent.evaluation.models import (  # noqa: E402
    EvalCase,
    EvalDataset,
    RetrievalSettings,
    save_dataset,
)
from knowledge_agent.evaluation.runner import run as run_eval  # noqa: E402
from knowledge_agent.ingestion.ids import compute_doc_id  # noqa: E402
from knowledge_agent.ingestion.pipeline import ingest_document  # noqa: E402
from knowledge_agent.search.client import get_search_client  # noqa: E402

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


async def main() -> None:
    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return
    target = pdfs[0]
    doc_id = compute_doc_id(target)
    print(f"Source PDF: {target.name}")
    print(f"doc_id    : {doc_id[:12]}...\n")

    search_client = get_search_client()
    print("Clearing any leftover chunks for this doc_id...")
    await search_client.delete_chunks_by_doc_id(doc_id)

    print("Ingesting the PDF (LanceDB + Neo4j)...")
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    result = await ingest_document(target, config, "Document", "Paper")
    print(f"  n_chunks={result.n_chunks}, lancedb_ok={result.lancedb_ok}\n")

    # ---- tiny 1-case gold dataset (expected_sources = the ingested doc) ----
    out = Path(tempfile.mkdtemp(prefix="ka_smoke_eval_"))
    dataset_path = out / "smoke_gold.json"
    dataset = EvalDataset(
        name="smoke-eval",
        status="draft",
        cases=[
            EvalCase(
                id="smoke-01",
                question="What role does nutrition play in chronic disease?",
                expected_sources=[doc_id],
                required_keywords=["nutrition"],
                retrieval=RetrievalSettings(retrieval_mode="lancedb_only", top_k=5),
                category="smoke",
                origin="manual",
            )
        ],
    )
    save_dataset(dataset, dataset_path)

    # ---- run the harness (judge OFF by default) ----
    print("Running the eval harness (deterministic metrics, judge OFF)...")
    print("(one lancedb_only query: query_builder + synthesizer LLM calls)\n")
    cfg = EvalConfig(dataset_path=dataset_path, output_dir=out)
    run_result = await run_eval(cfg)

    summary = run_result.report["summary"]
    print("=== Eval report ===")
    print(f"cases : {summary['case_count']}")
    print(f"pass  : {summary['pass_count']}  ({summary['pass_rate']:.0%})")
    for key, value in summary.items():
        if key.startswith("avg_") and value is not None:
            print(f"  {key:26s} {value:.3f}")
    print(f"\nreport json : {run_result.json_path}")
    print(f"summary csv : {run_result.csv_path}")
    print(f"ledger      : {cfg.ledger_path} (run_id={run_result.run_id})")

    # ---- self-check ----
    scored = summary["case_count"] == 1 and any(
        k.startswith("avg_") and v is not None for k, v in summary.items()
    )
    print(f"\n[{'OK' if scored else 'FAIL'}] one case scored with deterministic metrics")

    print("\nCleaning up ingested chunks...")
    await search_client.delete_chunks_by_doc_id(doc_id)
    if not scored:
        raise SystemExit("SMOKE FAILED - see report above")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
