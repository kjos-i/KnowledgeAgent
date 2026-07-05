"""Tests for the EvalCase schema + queryset loader."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge_agent.evaluation.models import (
    EvalCase,
    EvalDataset,
    append_case,
    compute_dataset_hash,
    load_cases,
    load_dataset,
    save_dataset,
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
