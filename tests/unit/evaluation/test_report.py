"""Tests for report building + persistence.

Provenance capture (git/subprocess + get_settings) is patched out so the
tests are hermetic; files are written to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def test_build_summary_records_per_metric_n():
    # n_<key> = how many cases fed the mean. Both cases had hit_at_k; only c2
    # had chunk_hit_at_k (c1 was None → dropped), so its mean rests on n=1.
    s = RP.build_summary(_results())
    assert s["n_hit_at_k"] == 2
    assert s["n_mrr"] == 2
    assert s["n_chunk_hit_at_k"] == 1


def test_build_report_structure(monkeypatch):
    monkeypatch.setattr(RP, "capture_provenance", _fake_provenance)
    rep = RP.build_report(EvalConfig(), _results(), "2026-07-05T10:00:00")
    assert rep["git_commit"] == "abc"
    assert rep["prompts_snapshot"]["model_config"]["llm_provider"] == "anthropic"
    assert rep["prompts_snapshot"]["prompts"]["synthesizer_system"] == "P"
    assert rep["summary"]["pass_rate"] == 0.5
    assert set(rep["enabled_groups"]) == {"source", "chunk", "kg"}
    assert len(rep["results"]) == 2
    assert rep["dataset_hash"] is None  # not passed → None
    rep2 = RP.build_report(EvalConfig(), _results(), "t", dataset_hash="h123")
    assert rep2["dataset_hash"] == "h123"


def test_build_report_lifts_filter_keys(monkeypatch):
    """C1: model / dataset / corpus / judge fields are surfaced at the TOP
    level (not only inside prompts_snapshot) so the ledger can column them."""
    monkeypatch.setattr(
        RP,
        "capture_provenance",
        lambda: {
            "git_commit": "abc",
            "model_config": {"llm_provider": "anthropic", "synthesizer_model": "claude-sonnet-5"},
            "prompts": {},
        },
    )
    cfg = EvalConfig(
        dataset_path=Path("/data/escrt_bootstrap.json"),
        corpus_config_path=Path("/corpora/corpus_beta/corpus.toml"),
        enabled_groups=frozenset({"source", "judge"}),
        judge_models=("claude-haiku-4-5",),
    )
    rep = RP.build_report(cfg, _results(), "t")
    assert rep["llm_provider"] == "anthropic"
    assert rep["synthesizer_model"] == "claude-sonnet-5"
    assert rep["dataset_name"] == "escrt_bootstrap"  # stem of the dataset file
    assert rep["corpus_name"] == "corpus_beta"  # the corpus FOLDER name
    assert rep["judge_models"] == ["claude-haiku-4-5"]  # the panel that ran


def test_build_report_judge_models_empty_when_group_off(monkeypatch):
    """Judge group off → judge_models empty (no judge ran) even if a panel
    was configured — so a filter on judge model reflects reality."""
    monkeypatch.setattr(
        RP,
        "capture_provenance",
        lambda: {"git_commit": None, "model_config": {}, "prompts": {}},
    )
    cfg = EvalConfig(
        dataset_path=Path("d.json"),
        judge_models=("m",),
        enabled_groups=frozenset({"source"}),
    )
    rep = RP.build_report(cfg, _results(), "t")
    assert rep["judge_models"] == []


def test_build_report_stamps_recipe_provenance(monkeypatch):
    """C2: recipe_hash comes from the dataset's saved recipe."""
    from knowledge_agent.evaluation.models import EvalRecipe, compute_recipe_hash

    monkeypatch.setattr(
        RP,
        "capture_provenance",
        lambda: {"git_commit": None, "model_config": {}, "prompts": {}},
    )
    recipe = EvalRecipe(judge_threshold=0.9)
    rep = RP.build_report(EvalConfig(dataset_path=Path("d.json")), _results(), "t", recipe=recipe)
    assert rep["recipe_hash"] == compute_recipe_hash(recipe)


def test_build_report_recipe_provenance_none_without_recipe(monkeypatch):
    """No recipe on the dataset → recipe_hash is None (not an error)."""
    monkeypatch.setattr(
        RP,
        "capture_provenance",
        lambda: {"git_commit": None, "model_config": {}, "prompts": {}},
    )
    rep = RP.build_report(EvalConfig(dataset_path=Path("d.json")), _results(), "t")
    assert rep["recipe_hash"] is None


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


def test_write_report_toggles_skip_files(tmp_path, monkeypatch):
    """save_json / save_csv gate each file independently: a skipped file is not
    written and its returned path is None. With both off nothing is written (the
    output dir isn't even created)."""
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )
    rep = RP.build_report(EvalConfig(), _results(), "2026-07-05T10:00:00")

    # JSON only
    j, c = RP.write_report(rep, tmp_path / "a", save_csv=False)
    assert j is not None and j.exists()
    assert c is None
    assert list((tmp_path / "a").glob("eval_summary_*.csv")) == []

    # CSV only
    j, c = RP.write_report(rep, tmp_path / "b", save_json=False)
    assert c is not None and c.exists()
    assert j is None
    assert list((tmp_path / "b").glob("eval_report_*.json")) == []

    # neither -> nothing written, dir not even created
    j, c = RP.write_report(rep, tmp_path / "c", save_json=False, save_csv=False)
    assert j is None and c is None
    assert not (tmp_path / "c").exists()


def test_filename_stamp():
    assert RP._filename_stamp("2026-07-05T10:00:00") == "20260705100000"


def test_write_report_names_by_dataset_avoids_suite_collision(tmp_path, monkeypatch):
    """Two suite members sharing a run timestamp write DISTINCT files (named by
    dataset), and a same-name same-timestamp re-run gets a uniqueness suffix — so
    no per-member report is silently overwritten."""
    monkeypatch.setattr(
        RP, "capture_provenance", lambda: {"git_commit": None, "model_config": {}, "prompts": {}}
    )

    def _rep(name):
        r = RP.build_report(EvalConfig(), _results(), "2026-07-05T10:00:00")
        r["dataset_name"] = name
        return r

    # same timestamp, different datasets -> two distinct, dataset-named files
    j1, c1 = RP.write_report(_rep("facts_vector"), tmp_path)
    j2, c2 = RP.write_report(_rep("facts_graph"), tmp_path)
    assert j1 != j2 and c1 != c2
    assert "facts_vector" in j1.name and "facts_graph" in j2.name
    # same timestamp AND same dataset -> uniqueness counter keeps it distinct
    j3, _ = RP.write_report(_rep("facts_vector"), tmp_path)
    assert j3 != j1 and j3.exists() and j1.exists()
    assert len(list(tmp_path.glob("eval_report_*.json"))) == 3
