"""Tests for report building + persistence.

Provenance capture (git/subprocess + get_settings) is patched out so the
tests are hermetic; files are written to tmp_path.
"""

from __future__ import annotations

import json

from knowledge_agent.evaluation import report as RP
from knowledge_agent.evaluation.config import EvalConfig


def _results() -> list[dict]:
    return [
        {
            "id": "c1",
            "category": "F",
            "status": "PASS",
            "answer": "a",
            "errors": [],
            "hit_at_k": 1.0,
            "mrr": 1.0,
            "chunk_hit_at_k": None,
        },
        {
            "id": "c2",
            "category": "",
            "status": "REVIEW",
            "answer": "b",
            "errors": ["x"],
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "chunk_hit_at_k": 1.0,
        },
    ]


def _fake_provenance(git="abc", **_):
    return {
        "git_commit": git,
        "model_config": {"llm_provider": "anthropic"},
        "prompts": {"synthesizer_system": "P"},
    }


def test_build_summary_none_safe_means():
    s = RP.build_summary(_results())
    assert s["case_count"] == 2
    assert s["pass_count"] == 1
    assert s["pass_rate"] == 0.5
    assert s["avg_hit_at_k"] == 0.5  # mean(1.0, 0.0)
    assert s["avg_chunk_hit_at_k"] == 1.0  # None skipped → mean(1.0)


def test_build_report_structure(monkeypatch):
    monkeypatch.setattr(RP, "capture_provenance", _fake_provenance)
    rep = RP.build_report(EvalConfig(), _results(), "2026-07-05T10:00:00")
    assert rep["git_commit"] == "abc"
    assert rep["prompts_snapshot"]["model_config"]["llm_provider"] == "anthropic"
    assert rep["prompts_snapshot"]["prompts"]["synthesizer_system"] == "P"
    assert rep["summary"]["pass_rate"] == 0.5
    assert set(rep["enabled_groups"]) == {"source", "chunk", "kg"}
    assert len(rep["results"]) == 2


def test_write_report_creates_json_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )
    rep = RP.build_report(EvalConfig(), _results(), "2026-07-05T10:00:00")
    json_path, csv_path = RP.write_report(rep, tmp_path)
    assert json_path.exists() and csv_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["pass_count"] == 1

    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.replace(" ", "").startswith("id,category,status")
    assert "error_count" in csv_text
    rows = [line for line in csv_text.splitlines() if line.strip()]
    assert len(rows) == 3  # header + 2 cases


def test_filename_stamp():
    assert RP._filename_stamp("2026-07-05T10:00:00") == "20260705100000"
