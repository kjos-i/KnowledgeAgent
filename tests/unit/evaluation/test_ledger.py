"""Tests for the airtight SQLite ledger.

Everything runs against a tmp_path DB — no real instance touched. Covers
schema generation, registry-column coverage, FK enforcement (the PRAGMA
the reference omitted), and a save/read round-trip.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

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
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert {"eval_runs", "eval_cases"} <= tables
    assert {"idx_cases_run_id", "idx_runs_ts"} <= idx


def test_case_columns_match_registry(tmp_path):
    EvalLedger(tmp_path / "l.db")
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(eval_cases)")}
    for col, _ in R.case_sql_columns():
        assert col in cols


def test_run_columns_include_per_metric_n(tmp_path):
    EvalLedger(tmp_path / "l.db")
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
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
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
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
            "recipe_hash": "cafe" * 16,
        }
    )
    report["results"][0]["origin"] = "search"
    report["results"][1]["origin"] = "llm"
    led.save_run(report)
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["dataset_name"] == "escrt_bootstrap"
        assert run["corpus_name"] == "corpus_beta"
        assert run["llm_provider"] == "anthropic"
        assert run["synthesizer_model"] == "claude-sonnet-5"
        assert json.loads(run["judge_models"]) == ["claude-haiku-4-5", "claude-opus-4-8"]
        assert run["recipe_hash"] == "cafe" * 16
        cases = conn.execute("SELECT * FROM eval_cases ORDER BY case_id").fetchall()
        assert cases[0]["origin"] == "search"
        assert cases[1]["origin"] == "llm"


def test_save_run_persists_source_conversation(tmp_path):
    """A chat-sourced case's conversation is snapshotted into eval_cases as JSON
    (display only — so Run Summary → Case Details can show it) and round-trips
    via get_run_cases. A case with no conversation stores [] (not NULL / error)."""
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["results"][0]["origin"] = "chat"
    report["results"][0]["source_conversation"] = [
        {"role": "user", "content": "why did the valve fail?"},
        {"role": "assistant", "content": "Let me search the corpus."},
    ]
    # results[1] carries no source_conversation key → stored as [].
    led.save_run(report)
    cases = led.get_run_cases(led.list_runs()[0]["run_id"])
    conv = json.loads(cases[0]["source_conversation"])
    assert [t["role"] for t in conv] == ["user", "assistant"]
    assert conv[0]["content"] == "why did the valve fail?"
    assert json.loads(cases[1]["source_conversation"]) == []  # absent → empty list


def test_save_run_persists_case_settings(tmp_path):
    """A case's retrieval settings are snapshotted into eval_cases as JSON
    (display only — so Run Summary → Case Details can show them) and round-trip
    via get_run_cases. A case with no settings stores NULL (not "{}" / error)."""
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["results"][0]["case_settings"] = {
        "retrieval_mode": "lancedb_only",
        "top_k": 8,
        "num_candidates": None,  # unpinned → global
    }
    # results[1] carries no case_settings key → stored as NULL.
    led.save_run(report)
    cases = led.get_run_cases(led.list_runs()[0]["run_id"])
    settings = json.loads(cases[0]["case_settings"])
    assert settings["retrieval_mode"] == "lancedb_only"
    assert settings["top_k"] == 8
    assert settings["num_candidates"] is None
    assert cases[1]["case_settings"] is None  # absent → NULL


def test_save_run_persists_facts_hash_and_suite_run_id(tmp_path):
    """The suite identity (facts_hash) + the shared launch id (suite_run_id)
    round-trip onto the run row."""
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["facts_hash"] = "f" * 64
    report["knob_hash"] = "k" * 64
    report["suite_run_id"] = "2026-07-15T10:00:00+00:00"
    led.save_run(report)
    run = led.list_runs()[0]
    assert run["facts_hash"] == "f" * 64
    assert run["knob_hash"] == "k" * 64
    assert run["suite_run_id"] == "2026-07-15T10:00:00+00:00"


def test_get_suite_run_groups_members(tmp_path):
    """get_suite_run returns exactly the runs sharing a suite_run_id (the members
    of one suite execution), and [] for an unknown id."""
    led = EvalLedger(tmp_path / "l.db")
    sid = "2026-07-15T10:00:00+00:00"
    for _ in range(2):  # two members of one suite-run
        r = _report()
        r["suite_run_id"] = sid
        led.save_run(r)
    led.save_run(_report())  # an unrelated single-file run (no suite_run_id)
    members = led.get_suite_run(sid)
    assert len(members) == 2
    assert {m["suite_run_id"] for m in members} == {sid}
    assert led.get_suite_run("nope") == []


def test_save_run_filter_columns_default_when_absent(tmp_path):
    """A bare / legacy report (no filter keys) stores NULL / empty-JSON, not
    an error — save_run reads the new keys defensively."""
    led = EvalLedger(tmp_path / "l.db")
    led.save_run(_report())  # _report() carries none of the new keys
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
        conn.row_factory = sqlite3.Row
        run = conn.execute("SELECT * FROM eval_runs").fetchone()
        assert run["dataset_name"] is None
        assert run["synthesizer_model"] is None
        assert json.loads(run["judge_models"]) == []
        assert run["recipe_hash"] is None
        assert run["facts_hash"] is None
        assert run["knob_hash"] is None
        assert run["suite_run_id"] is None
        case = conn.execute("SELECT * FROM eval_cases LIMIT 1").fetchone()
        assert case["origin"] is None  # _report()'s cases have no origin
        assert case["case_settings"] is None  # absent → NULL (not "{}")


def test_save_run_persists_per_metric_n(tmp_path):
    led = EvalLedger(tmp_path / "l.db")
    report = _report()
    report["summary"]["n_hit_at_k"] = 2
    report["summary"]["n_chunk_hit_at_k"] = 1
    led.save_run(report)
    with closing(sqlite3.connect(tmp_path / "l.db")) as conn, conn:
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
