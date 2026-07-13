"""Tests for the airtight SQLite ledger.

Everything runs against a tmp_path DB — no real instance touched. Covers
schema generation, registry-column coverage, FK enforcement (the PRAGMA
the reference omitted), and a save/read round-trip.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from knowledge_agent.evaluation import registry as R
from knowledge_agent.evaluation.ledger import EvalLedger


def _report() -> dict:
    return {
        "run_timestamp": "2026-07-05T10:00:00",
        "dataset_path": "escrt_bootstrap.json",
        "dataset_hash": "deadbeef" * 8,
        "git_commit": "abc123",
        "prompts_snapshot": {"synthesizer": "PROMPT"},
        "enabled_groups": ["source", "chunk"],
        "gate_thresholds": {"required_keyword_threshold": 0.5},
        "summary": {"case_count": 2, "pass_count": 1, "pass_rate": 0.5, "avg_hit_at_k": 0.5},
        "results": [
            {
                "id": "c1",
                "category": "Factual",
                "question": "q1?",
                "status": "PASS",
                "answer": "a1",
                "errors": [],
                "hit_at_k": 1.0,
                "mrr": 1.0,
            },
            {
                "id": "c2",
                "category": "",
                "question": "q2?",
                "status": "REVIEW",
                "answer": "a2",
                "errors": ["boom"],
                "hit_at_k": 0.0,
            },
        ],
    }


def test_creates_tables_and_indexes(tmp_path):
    EvalLedger(tmp_path / "l.db")
    with sqlite3.connect(tmp_path / "l.db") as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"eval_runs", "eval_cases"} <= tables
    assert {"idx_cases_run_id", "idx_runs_ts"} <= idx


def test_case_columns_match_registry(tmp_path):
    EvalLedger(tmp_path / "l.db")
    with sqlite3.connect(tmp_path / "l.db") as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(eval_cases)")}
    for col, _ in R.case_sql_columns():
        assert col in cols


def test_run_columns_include_per_metric_n(tmp_path):
    EvalLedger(tmp_path / "l.db")
    with sqlite3.connect(tmp_path / "l.db") as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(eval_runs)")}
    for col, _ in R.run_n_columns():
        assert col in cols


def test_foreign_keys_are_enforced(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    with pytest.raises(sqlite3.IntegrityError), led._connect() as conn:
        conn.execute(
            "INSERT INTO eval_cases (run_id, run_timestamp, case_id) VALUES (999, 't', 'c')"
        )
        conn.commit()


def test_save_run_roundtrip(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    run_id = led.save_run(_report())
    assert run_id == 1
    with sqlite3.connect(tmp_path / "l.db") as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["pass_count"] == 1
        assert run["dataset_hash"] == "deadbeef" * 8
        assert run["git_commit"] == "abc123"
        assert json.loads(run["prompts_snapshot"])["synthesizer"] == "PROMPT"
        assert run["avg_hit_at_k"] == 0.5
        assert json.loads(run["enabled_groups"]) == ["chunk", "source"]

        cases = conn.execute("SELECT * FROM eval_cases ORDER BY case_id").fetchall()
        assert len(cases) == 2
        assert cases[0]["case_id"] == "c1" and cases[0]["hit_at_k"] == 1.0
        assert cases[0]["run_id"] == run_id  # FK wired
        assert json.loads(cases[1]["errors"]) == ["boom"]
        assert cases[1]["hit_at_k"] == 0.0


def test_save_run_persists_filter_columns(tmp_path):
    """C1 filter columns (added 2026-07-13): the run-level model / dataset /
    corpus / judge fields + the per-case origin round-trip into their own
    columns (so the dashboard can query them without parsing JSON)."""
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report.update(
        {
            "dataset_name": "escrt_bootstrap",
            "corpus_name": "corpus_beta",
            "llm_provider": "anthropic",
            "synthesizer_model": "claude-sonnet-5",
            "judge_models": ["claude-haiku-4-5", "claude-opus-4-8"],
        }
    )
    report["results"][0]["origin"] = "search"
    report["results"][1]["origin"] = "llm"
    led.save_run(report)
    with sqlite3.connect(tmp_path / "l.db") as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["dataset_name"] == "escrt_bootstrap"
        assert run["corpus_name"] == "corpus_beta"
        assert run["llm_provider"] == "anthropic"
        assert run["synthesizer_model"] == "claude-sonnet-5"
        assert json.loads(run["judge_models"]) == ["claude-haiku-4-5", "claude-opus-4-8"]
        cases = conn.execute("SELECT * FROM eval_cases ORDER BY case_id").fetchall()
        assert cases[0]["origin"] == "search"
        assert cases[1]["origin"] == "llm"


def test_save_run_filter_columns_default_when_absent(tmp_path):
    """A bare / legacy report (no filter keys) stores NULL / empty-JSON, not
    an error — save_run reads the new keys defensively."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())  # _report() carries none of the new keys
    with sqlite3.connect(tmp_path / "l.db") as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["dataset_name"] is None
        assert run["synthesizer_model"] is None
        assert json.loads(run["judge_models"]) == []
        case = conn.execute("SELECT * FROM eval_cases LIMIT 1").fetchone()
        assert case["origin"] is None  # _report()'s cases have no origin


def test_save_run_persists_per_metric_n(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["summary"]["n_hit_at_k"] = 2
    report["summary"]["n_chunk_hit_at_k"] = 1
    led.save_run(report)
    with sqlite3.connect(tmp_path / "l.db") as conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["n_hit_at_k"] == 2
        assert run["n_chunk_hit_at_k"] == 1


def test_list_runs_and_get_run(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    id1 = led.save_run(_report())
    id2 = led.save_run(_report())

    runs = led.list_runs()
    assert [r["run_id"] for r in runs] == [id2, id1]  # newest first
    assert runs[0]["pass_count"] == 1
    assert led.list_runs(limit=1) == [runs[0]]  # limit caps the count

    assert led.get_run(id1)["run_id"] == id1
    assert led.get_run(9999) is None  # miss → None, not an error


def test_get_run_cases(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    run_id = led.save_run(_report())
    cases = led.get_run_cases(run_id)
    assert [c["case_id"] for c in cases] == ["c1", "c2"]  # insertion order
    assert cases[0]["status"] == "PASS" and cases[0]["hit_at_k"] == 1.0
    assert led.get_run_cases(9999) == []  # unknown run → empty
