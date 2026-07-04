"""Tests for the EvalCase schema + queryset loader."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge_agent.evaluation.models import EvalCase, load_cases


def test_minimal_case_defaults():
    c = EvalCase(id="x", question="what?")
    assert c.expected_sources == [] and c.expected_chunks == []
    assert c.retrieval.retrieval_mode == "lancedb_only"
    assert c.retrieval.lancedb_search_mode == "hybrid"
    assert c.retrieval.top_k == 5
    assert c.expected_mode is None


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


def test_load_cases_rejects_non_list(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"id": "a"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON array"):
        load_cases(p)
