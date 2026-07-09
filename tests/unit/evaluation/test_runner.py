"""End-to-end runner test — the whole pipeline with a stubbed graph.

The adapter (`run_case`) is stubbed so no graph / LLM / DB is touched;
provenance capture is patched to skip git/settings. Uses the real
bootstrap dataset (so it also validates that file parses) and a tmp
output dir. Verifies load → evaluate → report → JSON/CSV → ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3

import pytest

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

    # Build a tiny valid dataset (the harness has no baked-in default now).
    dataset = tmp_path / "gold.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "question": "How does ESCRT-III remodel the membrane?",
                    "expected_chunks": ["ESCRT-III", "membrane"],
                    "retrieval": {"num_candidates": 100, "rrf_rank_constant": 60},
                },
                {
                    "id": "c2",
                    "question": "How do ESCRTs drive scission?",
                    "expected_chunks": ["scission"],
                    "retrieval": {"num_candidates": 100, "rrf_rank_constant": 60},
                },
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_eval_config(dataset_path=dataset, output_dir=tmp_path / "out")
    result = asyncio.run(RN.run(cfg))

    # ---- run() is presentation-free: returns a RunResult, prints nothing ----
    assert isinstance(result, RN.RunResult)
    assert result.run_id == 1
    report = result.report

    # ---- report shape ----
    assert report["summary"]["case_count"] == 2
    # dataset hash computed from the cases + carried on the report
    assert isinstance(report["dataset_hash"], str) and len(report["dataset_hash"]) == 64
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
        stored_hash = conn.execute("SELECT dataset_hash FROM eval_runs").fetchone()[0]
    assert n_runs == 1
    assert n_cases == 2
    assert stored_hash == report["dataset_hash"]  # persisted to the ledger


def test_run_refuses_when_no_dataset_selected(tmp_path):
    """No dataset set (the harness has no baked-in default) → run aborts with a
    clear error before anything runs."""
    cfg = load_eval_config(output_dir=tmp_path / "out")  # dataset_path stays None
    with pytest.raises(ValueError, match="[Nn]o dataset"):
        asyncio.run(RN.run(cfg))


def test_run_refuses_dataset_with_missing_required_knob(tmp_path, monkeypatch):
    """run() validates up-front: a case leaving a required retrieval knob blank
    aborts with a clear, case-named error BEFORE any case runs — the guard for
    datasets imported / hand-edited outside the GUI form."""

    async def must_not_run(case, corpus_config):
        raise AssertionError("no case should run when the dataset is invalid")

    monkeypatch.setattr(E, "run_case", must_not_run)

    bad = tmp_path / "bad.json"
    # lancedb_only + hybrid (defaults) but num_candidates / rrf_rank_constant blank.
    bad.write_text(
        json.dumps(
            [{"id": "leaky", "question": "q?", "retrieval": {"retrieval_mode": "lancedb_only"}}]
        ),
        encoding="utf-8",
    )
    cfg = load_eval_config(dataset_path=bad, output_dir=tmp_path / "out")

    with pytest.raises(ValueError) as exc:
        asyncio.run(RN.run(cfg))
    msg = str(exc.value)
    assert "leaky" in msg
    assert "num_candidates" in msg and "rrf_rank_constant" in msg


def test_overrides_from_args_maps_fields(tmp_path):
    args = argparse.Namespace(
        dataset=tmp_path / "d.json",
        corpus=tmp_path / "corpus.toml",
        groups="source, chunk ,kg",  # whitespace tolerated
        max_cases=3,
        output_dir=tmp_path / "out",
    )
    ov = RN._overrides_from_args(args)
    assert ov["dataset_path"] == tmp_path / "d.json"
    assert ov["corpus_config_path"] == tmp_path / "corpus.toml"
    assert ov["enabled_groups"] == frozenset({"source", "chunk", "kg"})
    assert ov["max_cases"] == 3
    assert ov["output_dir"] == tmp_path / "out"


def test_run_from_args_prints_summary_and_returns_zero(tmp_path, monkeypatch, capsys):
    """The CLI wrapper builds config, runs, prints a summary, returns 0.
    run() is stubbed so no graph/engine/ledger is touched — this exercises
    the presentation layer that both `ka eval` and `python -m` reuse."""
    fake = RN.RunResult(
        report={
            "summary": {"case_count": 2, "pass_count": 2, "pass_rate": 1.0},
            "run_timestamp": "2026-07-05T00:00:00+00:00",
            "enabled_groups": ["source"],
        },
        json_path=tmp_path / "r.json",
        csv_path=tmp_path / "r.csv",
        run_id=7,
    )

    async def fake_run(cfg, on_progress=None, *, trace=False, langsmith_project=None):
        return fake

    monkeypatch.setattr(RN, "run", fake_run)
    args = argparse.Namespace(
        dataset=None,
        corpus=None,
        groups="source",
        max_cases=None,
        output_dir=tmp_path / "out",
        history=False,
        show=None,
        export=None,
        trace=False,
        project=None,
    )
    code = asyncio.run(RN.run_from_args(args))
    assert code == 0
    out = capsys.readouterr().out
    assert "run 7" in out
    assert "pass_rate: 100%" in out


def _seed_run(cfg) -> int:
    """Save one minimal run into cfg's ledger; return its run_id."""
    from knowledge_agent.evaluation.ledger import EvalLedger

    report = {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": "d.json",
        "git_commit": None,
        "prompts_snapshot": {},
        "enabled_groups": ["source"],
        "gate_thresholds": {},
        "summary": {"case_count": 2, "pass_count": 2, "pass_rate": 1.0},
        "results": [
            {"id": "c1", "category": "Factual", "question": "q1", "status": "PASS", "errors": []},
            {"id": "c2", "category": "", "question": "q2", "status": "REVIEW", "errors": []},
        ],
    }
    return EvalLedger(cfg.ledger_path).save_run(report)


def test_show_history_empty(tmp_path, capsys):
    cfg = load_eval_config(output_dir=tmp_path / "out")
    assert RN._show_history(cfg, None) == 0
    assert "No eval runs" in capsys.readouterr().out


def test_show_history_lists_and_exports_csv(tmp_path, capsys):
    cfg = load_eval_config(output_dir=tmp_path / "out")
    run_id = _seed_run(cfg)
    export = tmp_path / "hist.csv"
    assert RN._show_history(cfg, export) == 0
    out = capsys.readouterr().out
    assert str(run_id) in out and "pass_rate" in out
    assert export.exists()
    assert "run_id" in export.read_text(encoding="utf-8")  # CSV header row


def test_show_run_found_and_missing(tmp_path, capsys):
    cfg = load_eval_config(output_dir=tmp_path / "out")
    run_id = _seed_run(cfg)
    assert RN._show_run(cfg, run_id, None) == 0
    out = capsys.readouterr().out
    assert "c1" in out and "PASS" in out
    assert RN._show_run(cfg, 9999, None) == 1  # no such run


def test_run_from_args_history_mode_does_not_run(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "out"
    _seed_run(load_eval_config(output_dir=out_dir))

    async def boom(cfg, on_progress=None, *, trace=False, langsmith_project=None):
        raise AssertionError("run() must not be called in --history mode")

    monkeypatch.setattr(RN, "run", boom)
    args = argparse.Namespace(
        dataset=None,
        corpus=None,
        groups=None,
        max_cases=None,
        output_dir=out_dir,
        history=True,
        show=None,
        export=None,
        trace=False,
        project=None,
    )
    assert asyncio.run(RN.run_from_args(args)) == 0
    assert "pass_rate" in capsys.readouterr().out  # printed the history table


def test_add_eval_args_registers_trace_and_project():
    """The eval CLI surface exposes --trace (opt-in) + --project."""
    p = argparse.ArgumentParser()
    RN.add_eval_args(p)
    on = p.parse_args(["--trace", "--project", "my-proj"])
    assert on.trace is True
    assert on.project == "my-proj"
    # Off by default: no flag → no tracing.
    off = p.parse_args([])
    assert off.trace is False
    assert off.project is None


def test_run_from_args_trace_without_key_errors(tmp_path, monkeypatch, capsys):
    """--trace with no LangSmith key in env fails fast (exit 1) and does not
    run — so the user isn't told a run traced when it silently couldn't."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    async def must_not_run(cfg, on_progress=None, *, trace=False, langsmith_project=None):
        raise AssertionError("run() must not be called without a trace key")

    monkeypatch.setattr(RN, "run", must_not_run)
    args = argparse.Namespace(
        dataset=None,
        corpus=None,
        groups="source",
        max_cases=None,
        output_dir=tmp_path / "out",
        history=False,
        show=None,
        export=None,
        trace=True,
        project=None,
    )
    assert asyncio.run(RN.run_from_args(args)) == 1
    assert "LangSmith API key" in capsys.readouterr().err


def test_run_from_args_trace_passes_project_through(tmp_path, monkeypatch):
    """With a key present, --trace + --project reach run() as kwargs."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    seen: dict = {}

    async def fake_run(cfg, on_progress=None, *, trace=False, langsmith_project=None):
        seen["trace"] = trace
        seen["project"] = langsmith_project
        return RN.RunResult(
            report={
                "summary": {"case_count": 1, "pass_count": 1, "pass_rate": 1.0},
                "run_timestamp": "t",
                "enabled_groups": ["source"],
            },
            json_path=tmp_path / "r.json",
            csv_path=tmp_path / "r.csv",
            run_id=1,
        )

    monkeypatch.setattr(RN, "run", fake_run)
    args = argparse.Namespace(
        dataset=None,
        corpus=None,
        groups="source",
        max_cases=None,
        output_dir=tmp_path / "out",
        history=False,
        show=None,
        export=None,
        trace=True,
        project="proj-x",
    )
    assert asyncio.run(RN.run_from_args(args)) == 0
    assert seen == {"trace": True, "project": "proj-x"}


def test_run_wraps_evaluate_in_langsmith_when_trace(tmp_path, monkeypatch):
    """run(trace=True) evaluates INSIDE tracing_v2_enabled, scoped to the run
    (never the global env flag), with the chosen project name."""
    import contextlib

    entered: dict = {}

    async def fake_run_case(case, corpus_config):
        return CaseRun(
            question=case.question,
            answer="a",
            retrieved_texts=["t"],
            retrieved_doc_ids=["d"],
            retrieved_chunk_ids=["c"],
        )

    @contextlib.contextmanager
    def fake_tracing(project_name=None):
        entered["project"] = project_name
        yield None

    monkeypatch.setattr(E, "run_case", fake_run_case)
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )
    import langchain_core.tracers.context as lcc

    monkeypatch.setattr(lcc, "tracing_v2_enabled", fake_tracing)

    dataset = tmp_path / "gold.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "question": "q?",
                    "retrieval": {"num_candidates": 100, "rrf_rank_constant": 60},
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_eval_config(dataset_path=dataset, output_dir=tmp_path / "out", max_cases=1)
    result = asyncio.run(RN.run(cfg, trace=True, langsmith_project="custom-proj"))
    assert result.run_id == 1
    assert entered["project"] == "custom-proj"
