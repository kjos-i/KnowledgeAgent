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
