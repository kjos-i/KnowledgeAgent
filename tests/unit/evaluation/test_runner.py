"""End-to-end runner test — the whole pipeline with a stubbed graph.

The adapter (`run_case`) is stubbed so no graph / LLM / DB is touched;
provenance capture is patched to skip git/settings. Uses the real
bootstrap dataset (so it also validates that file parses) and a tmp
output dir. Verifies load → evaluate → report → JSON/CSV → ledger.
"""

from __future__ import annotations

import asyncio
import sqlite3

from knowledge_agent.evaluation import engine as E
from knowledge_agent.evaluation import report as RP
from knowledge_agent.evaluation import runner as RN
from knowledge_agent.evaluation.adapter import CaseRun
from knowledge_agent.evaluation.config import load_eval_config


def test_end_to_end_with_stubbed_graph(tmp_path, monkeypatch):
    async def fake_run_case(case, corpus_config):
        return CaseRun(
            question=case.question,
            answer="ESCRT-III remodels the membrane and drives scission.",
            retrieved_texts=[
                "ESCRT-III filaments remodel the membrane",
                "scission occurs at the membrane neck",
            ],
            retrieved_doc_ids=["d1", "d2"],
            retrieved_chunk_ids=["c1", "c2"],
            input_tokens=100,
            output_tokens=40,
            total_tokens=140,
            latency_seconds=0.5,
        )

    monkeypatch.setattr(E, "run_case", fake_run_case)
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )

    # dataset defaults to the shipped escrt_bootstrap.json (9 cases, one per
    # pathway).
    cfg = load_eval_config(output_dir=tmp_path / "out")
    report = asyncio.run(RN.run(cfg))

    # ---- report shape ----
    assert report["summary"]["case_count"] == 9
    assert report["summary"]["avg_chunk_hit_at_k"] == 1.0  # every answer chunk matched
    assert report["summary"]["avg_agent_total_tokens"] == 140

    # ---- files written ----
    out = tmp_path / "out"
    assert len(list(out.glob("eval_report_*.json"))) == 1
    assert len(list(out.glob("eval_summary_*.csv"))) == 1

    # ---- ledger persisted ----
    ledger = out / "eval_ledger.db"
    assert ledger.exists()
    with sqlite3.connect(ledger) as conn:
        n_runs = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
        n_cases = conn.execute("SELECT COUNT(*) FROM eval_cases").fetchone()[0]
    assert n_runs == 1
    assert n_cases == 9
