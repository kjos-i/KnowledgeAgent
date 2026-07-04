"""Tests for EvalConfig defaults + env-override loading."""

from __future__ import annotations

from pathlib import Path

from knowledge_agent.evaluation.config import (
    DEFAULT_ENABLED_GROUPS,
    EvalConfig,
    load_eval_config,
)


def test_defaults():
    cfg = EvalConfig()
    assert cfg.dataset_path.name == "escrt_bootstrap.json"
    assert cfg.enabled_groups == DEFAULT_ENABLED_GROUPS
    assert cfg.max_cases is None


def test_ledger_path_derives_from_output_dir():
    cfg = EvalConfig(output_dir=Path("/tmp/ka_eval_x"))
    assert cfg.ledger_path == Path("/tmp/ka_eval_x") / "eval_ledger.db"


def test_gate_thresholds_dict():
    gates = EvalConfig().gate_thresholds()
    assert set(gates) == {
        "judge_threshold",
        "metadata_match_threshold",
        "required_keyword_threshold",
    }


def test_load_eval_config_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("KA_EVAL_MAX_CASES", "7")
    monkeypatch.setenv("KA_EVAL_CONCURRENCY", "2")
    monkeypatch.setenv("KA_EVAL_GROUPS", "source, chunk , kg")
    monkeypatch.setenv("KA_EVAL_OUTPUT_DIR", str(tmp_path / "out"))
    cfg = load_eval_config()
    assert cfg.max_cases == 7
    assert cfg.concurrency == 2
    assert cfg.enabled_groups == frozenset({"source", "chunk", "kg"})
    assert cfg.output_dir == tmp_path / "out"


def test_explicit_overrides_win_over_env(monkeypatch):
    monkeypatch.setenv("KA_EVAL_MAX_CASES", "7")
    cfg = load_eval_config(max_cases=99)
    assert cfg.max_cases == 99
