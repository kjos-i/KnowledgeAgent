"""Tests for the EvalCase schema + queryset loader."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge_agent.evaluation.models import (
    EvalCase,
    EvalDataset,
    RetrievalSettings,
    append_case,
    compute_dataset_hash,
    load_cases,
    load_dataset,
    required_knobs,
    save_dataset,
    validate_case,
    validate_dataset,
)


def test_minimal_case_defaults():
    c = EvalCase(id="x", question="what?")
    assert c.expected_sources == [] and c.expected_chunks == []
    assert c.retrieval.retrieval_mode == "lancedb_only"
    assert c.retrieval.lancedb_search_mode == "hybrid"
    assert c.retrieval.top_k == 5
    assert c.expected_mode is None


def test_pathway_knob_defaults():
    c = EvalCase(id="x", question="what?")
    assert c.retrieval.skip_query_builder is False
    assert c.retrieval.direct_retrieval is False
    assert c.user_cypher is None


def test_pathway_knobs_load(tmp_path):
    data = [
        {
            "id": "raw",
            "question": "escrt filament",
            "retrieval": {"retrieval_mode": "lancedb_only", "skip_query_builder": True},
        },
        {
            "id": "direct",
            "question": "escrt",
            "retrieval": {"retrieval_mode": "lancedb_only", "direct_retrieval": True},
        },
        {
            "id": "cyph",
            "question": "list escrt",
            "user_cypher": "MATCH (n) RETURN n LIMIT 5",
            "retrieval": {"retrieval_mode": "neo4j_only"},
        },
    ]
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    cases = load_cases(p)
    assert cases[0].retrieval.skip_query_builder is True
    assert cases[1].retrieval.direct_retrieval is True
    assert cases[2].user_cypher == "MATCH (n) RETURN n LIMIT 5"


def test_case_rejects_blank_id():
    with pytest.raises(ValidationError):
        EvalCase(id="", question="q?")


def test_retrieval_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        EvalCase(id="x", question="q?", retrieval={"retrieval_mode": "nope"})


def test_load_cases_roundtrip(tmp_path):
    data = [
        {"id": "a", "question": "q1?", "expected_sources": ["d1"]},
        {
            "id": "b",
            "question": "q2?",
            "retrieval": {"retrieval_mode": "neo4j_only", "top_k": 3},
            "expected_entities": ["ESCRT-III"],
            "expected_mode": "neo4j_only",
        },
    ]
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    cases = load_cases(p)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[1].retrieval.retrieval_mode == "neo4j_only"
    assert cases[1].expected_entities == ["ESCRT-III"]


def test_load_dataset_accepts_both_shapes(tmp_path):
    # New object shape: header + cases.
    obj = tmp_path / "obj.json"
    obj.write_text(
        json.dumps({"status": "final", "name": "gold", "cases": [{"id": "a", "question": "q?"}]}),
        encoding="utf-8",
    )
    ds = load_dataset(obj)
    assert ds.status == "final" and ds.name == "gold"
    assert [c.id for c in ds.cases] == ["a"]

    # Legacy bare array → default header, same cases; load_cases stays
    # backward-compatible for the engine/runner.
    arr = tmp_path / "arr.json"
    arr.write_text(json.dumps([{"id": "b", "question": "q?"}]), encoding="utf-8")
    ds2 = load_dataset(arr)
    assert ds2.status == "draft"  # default header
    assert [c.id for c in load_cases(arr)] == ["b"]


def test_load_rejects_json_scalar(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array or object"):
        load_cases(p)


def test_case_origin_default_and_values():
    assert EvalCase(id="x", question="q?").origin == "manual"
    assert EvalCase(id="x", question="q?", origin="llm").origin == "llm"
    with pytest.raises(ValidationError):
        EvalCase(id="x", question="q?", origin="bogus")


def test_save_and_append_roundtrip(tmp_path):
    p = tmp_path / "gold.json"
    # Append to a non-existent file → creates a default dataset with one case.
    ds = append_case(p, EvalCase(id="a", question="q1?", origin="search"))
    assert [c.id for c in ds.cases] == ["a"]
    # Persisted in OBJECT form (dict with "cases"), reloadable.
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert isinstance(on_disk, dict) and "cases" in on_disk
    assert load_dataset(p).cases[0].origin == "search"
    # Append a second, then persist an edited header.
    append_case(p, EvalCase(id="b", question="q2?"))
    ds2 = load_dataset(p)
    ds2.status = "final"
    save_dataset(ds2, p)
    assert load_dataset(p).status == "final"
    assert [c.id for c in load_dataset(p).cases] == ["a", "b"]


def test_dataset_hash_is_content_addressed():
    a = EvalCase(id="a", question="q1?", required_keywords=["x"])
    b = EvalCase(id="b", question="q2?")
    h = compute_dataset_hash([a, b])
    assert len(h) == 64
    # order-independent (same set of cases → same hash)
    assert compute_dataset_hash([b, a]) == h
    # header-independent: hashing takes cases, so status/name never affect it
    assert compute_dataset_hash(EvalDataset(status="final", cases=[a, b]).cases) == h
    # content-sensitive: editing a case changes the hash
    a2 = EvalCase(id="a", question="q1?", required_keywords=["y"])
    assert compute_dataset_hash([a2, b]) != h


# ---- per-case tuning knobs: defaults + conditional-required validation ----


def test_tuning_knob_defaults_are_blank():
    """The nullable tuning knobs default to None ('not pinned'); use_mmr is a
    plain bool default False."""
    rs = EvalCase(id="x", question="q?").retrieval
    assert rs.num_candidates is None
    assert rs.rrf_rank_constant is None
    assert rs.mmr_lambda is None
    assert rs.kg_max_rows is None
    assert rs.use_mmr is False


def test_required_knobs_lancedb_hybrid_default():
    # Default case (lancedb_only + hybrid, no MMR): pool + RRF constant.
    assert required_knobs(RetrievalSettings()) == {"num_candidates", "rrf_rank_constant"}


def test_required_knobs_fts_and_vector_drop_rrf():
    assert required_knobs(RetrievalSettings(lancedb_search_mode="fts")) == {"num_candidates"}
    assert required_knobs(RetrievalSettings(lancedb_search_mode="vector")) == {"num_candidates"}


def test_required_knobs_mmr_adds_lambda():
    assert required_knobs(RetrievalSettings(use_mmr=True)) == {
        "num_candidates",
        "rrf_rank_constant",
        "mmr_lambda",
    }


def test_required_knobs_fts_never_requires_mmr_lambda():
    # MMR needs vectors → never runs under FTS, so mmr_lambda isn't required
    # even with use_mmr set. (rrf also drops — FTS isn't hybrid.)
    assert required_knobs(RetrievalSettings(lancedb_search_mode="fts", use_mmr=True)) == {
        "num_candidates"
    }


def test_required_knobs_neo4j_only_is_kg_max_rows_only():
    assert required_knobs(RetrievalSettings(retrieval_mode="neo4j_only")) == {"kg_max_rows"}


@pytest.mark.parametrize("mode", ["auto", "parallel_fused", "lancedb_then_neo4j"])
def test_required_knobs_both_legs_modes(mode):
    # Modes that run BOTH legs (auto is conservative) require both legs' knobs.
    assert required_knobs(RetrievalSettings(retrieval_mode=mode)) == {
        "num_candidates",
        "rrf_rank_constant",
        "kg_max_rows",
    }


def test_validate_case_flags_blank_required_knobs():
    problems = validate_case(EvalCase(id="x", question="q?"))
    assert any("num_candidates" in p for p in problems)
    assert any("rrf_rank_constant" in p for p in problems)


def test_validate_case_ok_when_required_pinned():
    case = EvalCase(
        id="x",
        question="q?",
        retrieval={"num_candidates": 50, "rrf_rank_constant": 60},
    )
    assert validate_case(case) == []


def test_validate_case_ignores_inert_knobs():
    # neo4j_only: LanceDB knobs are inert (never read), so leaving them blank
    # is fine — only kg_max_rows is required.
    case = EvalCase(
        id="x",
        question="q?",
        retrieval={"retrieval_mode": "neo4j_only", "kg_max_rows": 25},
    )
    assert validate_case(case) == []


def test_validate_case_flags_pool_smaller_than_top_k():
    case = EvalCase(
        id="x",
        question="q?",
        retrieval={"top_k": 10, "num_candidates": 5, "rrf_rank_constant": 60},
    )
    problems = validate_case(case)
    assert any("num_candidates" in p and "top_k" in p for p in problems)


def test_validate_dataset_maps_only_invalid_ids():
    good = EvalCase(
        id="good", question="q?", retrieval={"num_candidates": 50, "rrf_rank_constant": 60}
    )
    bad = EvalCase(id="bad", question="q?")  # blank required knobs
    result = validate_dataset([good, bad])
    assert "good" not in result and "bad" in result
