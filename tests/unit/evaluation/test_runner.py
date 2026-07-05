"""End-to-end runner test — the whole pipeline with a stubbed graph.

The adapter (`run_case`) is stubbed so no graph / LLM / DB is touched;
provenance capture is patched to skip git/settings. Uses the real
bootstrap dataset (so it also validates that file parses) and a tmp
output dir. Verifies load → evaluate → report → JSON/CSV → ledger.
"""

from __future__ import annotations

import argparse
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
    result = asyncio.run(RN.run(cfg))

    # ---- run() is presentation-free: returns a RunResult, prints nothing ----
    assert isinstance(result, RN.RunResult)
    assert result.run_id == 1
    report = result.report

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

    async def fake_run(cfg):
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

    async def boom(cfg):
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
    )
    assert asyncio.run(RN.run_from_args(args)) == 0
    assert "pass_rate" in capsys.readouterr().out  # printed the history table
