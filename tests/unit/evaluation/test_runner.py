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
                    "retrieval": {
                        "retrieval_mode": "lancedb_only",
                        "num_candidates": 100,
                        "rrf_rank_constant": 60,
                    },
                },
                {
                    "id": "c2",
                    "question": "How do ESCRTs drive scission?",
                    "expected_chunks": ["scission"],
                    "retrieval": {
                        "retrieval_mode": "lancedb_only",
                        "num_candidates": 100,
                        "rrf_rank_constant": 60,
                    },
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


def test_run_builds_report_off_the_event_loop(tmp_path, monkeypatch):
    """build_report captures provenance via a blocking `git rev-parse`
    subprocess. run() is async and the GUI drives it on the Flet event loop, so
    the report must be assembled via asyncio.to_thread, not inline — else the
    git call stalls the UI. Regression: run() called build_report() directly on
    the loop, so `build_report` never appeared among the offloaded callables."""

    async def fake_run_case(case, corpus_config):
        return CaseRun(
            question=case.question,
            answer="a",
            retrieved_texts=["t"],
            retrieved_doc_ids=["d"],
            retrieved_chunk_ids=["c"],
        )

    monkeypatch.setattr(E, "run_case", fake_run_case)
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )

    # Spy on asyncio.to_thread (call through so run() completes normally) and
    # record which callables were offloaded.
    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(fn, *args, **kwargs):
        offloaded.append(getattr(fn, "__name__", repr(fn)))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(RN.asyncio, "to_thread", spy_to_thread)

    dataset = tmp_path / "gold.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "question": "q?",
                    "retrieval": {
                        "retrieval_mode": "lancedb_only",
                        "num_candidates": 100,
                        "rrf_rank_constant": 60,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_eval_config(dataset_path=dataset, output_dir=tmp_path / "out", max_cases=1)
    result = asyncio.run(RN.run(cfg))

    assert result.run_id == 1
    assert "build_report" in offloaded  # report assembled off the event loop


def test_run_suite_shares_one_id_across_members(tmp_path, monkeypatch):
    """run_suite runs each member (same facts, swept knobs) and stamps them all
    with ONE shared suite_run_id — so members share a facts_hash but differ by
    dataset_hash, and on_run_complete fires once per member, in order."""

    async def fake_run_case(case, corpus_config):
        return CaseRun(
            question=case.question,
            answer="a",
            retrieved_texts=["x"],
            retrieved_doc_ids=["d1"],
            retrieved_chunk_ids=["c1"],
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            latency_seconds=0.1,
        )

    monkeypatch.setattr(E, "run_case", fake_run_case)
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )
    out = tmp_path / "out"

    def _member(name, cid, mode):
        path = tmp_path / name
        path.write_text(
            json.dumps(
                [
                    {
                        "id": cid,
                        "question": "q?",
                        "expected_chunks": ["x"],
                        # pin every knob either leg needs so validate_dataset passes
                        "retrieval": {
                            "retrieval_mode": mode,
                            "num_candidates": 100,
                            "rrf_rank_constant": 60,
                            "kg_max_rows": 50,
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        return load_eval_config(dataset_path=path, output_dir=out)

    cfg_v = _member("facts_vector.json", "q__vector", "lancedb_only")
    cfg_g = _member("facts_graph.json", "q__graph", "neo4j_only")

    seen: list[tuple[int, int]] = []
    suite = asyncio.run(
        RN.run_suite(
            [cfg_v, cfg_g],
            on_run_complete=lambda done, total, _r: seen.append((done, total)),
            suite="mode-sweep",
        )
    )

    assert isinstance(suite, RN.SuiteRunResult)
    assert len(suite.results) == 2
    assert seen == [(1, 2), (2, 2)]  # progress fired per member, in order

    with sqlite3.connect(out / "eval_ledger.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT suite, suite_run_id, facts_hash, knob_hash, dataset_hash "
            "FROM eval_runs ORDER BY run_id"
        ).fetchall()
    # one shared suite_run_id + suite name across both members
    assert rows[0]["suite_run_id"] == rows[1]["suite_run_id"] == suite.suite_run_id
    assert rows[0]["suite"] == rows[1]["suite"] == "mode-sweep"
    # same facts -> same facts_hash; different knobs -> different knob_hash AND
    # dataset_hash (facts_hash ⊕ knob_hash ≈ dataset_hash)
    assert rows[0]["facts_hash"] == rows[1]["facts_hash"]
    assert rows[0]["knob_hash"] != rows[1]["knob_hash"]
    assert rows[0]["dataset_hash"] != rows[1]["dataset_hash"]


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
                    "retrieval": {
                        "retrieval_mode": "lancedb_only",
                        "num_candidates": 100,
                        "rrf_rank_constant": 60,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_eval_config(dataset_path=dataset, output_dir=tmp_path / "out", max_cases=1)
    result = asyncio.run(RN.run(cfg, trace=True, langsmith_project="custom-proj"))
    assert result.run_id == 1
    assert entered["project"] == "custom-proj"


# ---------------------------------------------------------------------------
# Save-options toggles — the Run tab's CSV/JSON checkboxes gate ONLY the file
# exports; the ledger write is unconditional (run_id + row always exist).
# ---------------------------------------------------------------------------


def _stub_case(monkeypatch):
    """Stub the graph + provenance so a run touches no LLM/DB/git."""

    async def fake_run_case(case, corpus_config):
        return CaseRun(
            question=case.question,
            answer="a",
            retrieved_texts=["t"],
            retrieved_doc_ids=["d"],
            retrieved_chunk_ids=["c"],
        )

    monkeypatch.setattr(E, "run_case", fake_run_case)
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )


def _one_case_dataset(path):
    path.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "question": "q?",
                    "retrieval": {
                        "retrieval_mode": "lancedb_only",
                        "num_candidates": 100,
                        "rrf_rank_constant": 60,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(("save_json", "save_csv"), [(True, False), (False, True), (False, False)])
def test_run_save_toggles_skip_files_but_always_ledger(tmp_path, monkeypatch, save_json, save_csv):
    """A skipped export is absent on disk AND None on the result; whatever the
    toggles, the ledger row + run_id always exist so the dashboards can read it."""
    _stub_case(monkeypatch)
    cfg = load_eval_config(
        dataset_path=_one_case_dataset(tmp_path / "gold.json"),
        output_dir=tmp_path / "out",
        max_cases=1,
    )
    result = asyncio.run(RN.run(cfg, save_json=save_json, save_csv=save_csv))

    out = tmp_path / "out"
    assert len(list(out.glob("eval_report_*.json"))) == (1 if save_json else 0)
    assert (result.json_path is not None) == save_json
    assert len(list(out.glob("eval_summary_*.csv"))) == (1 if save_csv else 0)
    assert (result.csv_path is not None) == save_csv
    # Ledger is always written, whatever the toggles.
    assert result.run_id == 1
    with sqlite3.connect(out / "eval_ledger.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM eval_cases").fetchone()[0] == 1


def test_run_suite_forwards_save_toggles_to_members(tmp_path, monkeypatch):
    """run_suite passes the Save-options flags to every member: both off → no
    JSON/CSV files land, but each member still writes its ledger row."""
    _stub_case(monkeypatch)
    out = tmp_path / "out"
    members = [
        load_eval_config(dataset_path=_one_case_dataset(tmp_path / f"m{i}.json"), output_dir=out)
        for i in (1, 2)
    ]
    suite = asyncio.run(RN.run_suite(members, suite="s", save_json=False, save_csv=False))

    assert all(r.json_path is None and r.csv_path is None for r in suite.results)
    assert list(out.glob("eval_report_*.json")) == []
    assert list(out.glob("eval_summary_*.csv")) == []
    with sqlite3.connect(out / "eval_ledger.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0] == 2
