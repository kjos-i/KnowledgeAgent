"""Tests for the Evaluation Dataset sub-tab (browse + edit, slice 3).

Loads a tiny gold dataset from tmp_path and exercises: list + select → form
populate, New/Save (creates a file), Edit (persists), Delete (persists), and
invalid-input handling. Writes go through the real `save_dataset`, so the
round-trips reload from disk to verify.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from knowledge_agent.evaluation.models import load_dataset
from knowledge_agent.gui.evaluation.dataset_tab import DatasetTab


def _dataset_file(tmp_path) -> object:
    cases = [
        {
            "id": "c1",
            "question": "What drives membrane scission?",
            "required_keywords": ["ESCRT-III", "scission"],
            "retrieval": {"retrieval_mode": "neo4j_only", "direct_retrieval": True},
            "user_cypher": "MATCH (e:Entity) RETURN e.key LIMIT 5",
            "origin": "llm",
        },
        {"id": "c2", "question": "second question"},
    ]
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(cases), encoding="utf-8")
    return p


def _tab(fake_app) -> DatasetTab:
    tab = DatasetTab(fake_app, coordinator=MagicMock())
    tab.build()
    return tab


def test_dataset_tab_builds(fake_app):
    assert DatasetTab(fake_app, coordinator=MagicMock()).build() is not None


def test_load_and_select_populates_form(fake_app, tmp_path):
    tab = _tab(fake_app)
    tab._load(_dataset_file(tmp_path))
    assert len(tab._cases) == 2
    tab._select(0)
    assert tab.f["id"].value == "c1"
    assert tab.f["retrieval_mode"].value == "neo4j_only"
    assert tab.f["direct_retrieval"].value is True
    assert tab.f["user_cypher"].value == "MATCH (e:Entity) RETURN e.key LIMIT 5"
    assert "ESCRT-III" in tab.f["required_keywords"].value  # one-per-line text
    assert tab.f["origin"].value == "llm"


def test_load_bad_dataset_surfaces_error(fake_app, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("42", encoding="utf-8")  # JSON scalar → neither array nor object
    tab = _tab(fake_app)
    tab._load(bad)
    assert "could not load" in tab.status.value
    assert tab._cases == []


def test_new_case_saves_to_file(fake_app, tmp_path):
    p = tmp_path / "new.json"
    tab = _tab(fake_app)
    tab.dataset_dropdown.value = str(p)
    tab._on_new(MagicMock())
    tab.f["id"].value = "brandnew"
    tab.f["question"].value = "a new question?"
    tab.f["required_keywords"].value = "alpha\nbeta"
    tab.name_field.value = "My set"
    tab.status_dropdown.value = "final"
    tab._on_save_case(MagicMock())

    assert p.exists()
    ds = load_dataset(p)
    assert ds.name == "My set" and ds.status == "final"  # header rode along
    assert [c.id for c in ds.cases] == ["brandnew"]
    assert ds.cases[0].required_keywords == ["alpha", "beta"]  # lines → list


def test_edit_selected_case_persists(fake_app, tmp_path):
    p = _dataset_file(tmp_path)
    tab = _tab(fake_app)
    tab._load(p)
    tab._select(0)
    tab.f["question"].value = "edited question?"
    tab._on_save_case(MagicMock())

    reloaded = load_dataset(p)
    assert reloaded.cases[0].question == "edited question?"
    assert len(reloaded.cases) == 2  # edited in place, not appended


def test_delete_selected_case_persists(fake_app, tmp_path):
    p = _dataset_file(tmp_path)
    tab = _tab(fake_app)
    tab._load(p)
    tab._select(0)
    tab._on_delete(MagicMock())

    assert [c.id for c in load_dataset(p).cases] == ["c2"]


def test_save_invalid_case_surfaces_error(fake_app, tmp_path):
    p = tmp_path / "x.json"
    tab = _tab(fake_app)
    tab.dataset_dropdown.value = str(p)
    tab._on_new(MagicMock())
    tab.f["id"].value = ""  # id is required → ValidationError
    tab.f["question"].value = "q?"
    tab._on_save_case(MagicMock())

    assert "invalid case" in tab.status.value
    assert not p.exists()  # nothing written on invalid input


def test_capture_from_search_prefills_new_case(fake_app):
    fake_app.last_query = "why did the valve fail?"
    fake_app.last_answer = SimpleNamespace(
        chunk_sources=[
            SimpleNamespace(doc_id="doc_a"),
            SimpleNamespace(doc_id="doc_b"),
            SimpleNamespace(doc_id="doc_a"),  # dup → collapsed
        ]
    )
    tab = _tab(fake_app)
    tab._on_capture_from_search(MagicMock())
    assert tab._selected is None  # new-case mode
    assert tab.f["question"].value == "why did the valve fail?"
    assert tab.f["origin"].value == "search"
    assert tab.f["expected_sources"].value == "doc_a\ndoc_b"  # deduped, order-preserved


def test_capture_from_search_no_result(fake_app):
    fake_app.last_answer = None
    fake_app.last_query = None
    tab = _tab(fake_app)
    tab._on_capture_from_search(MagicMock())
    assert "No search result" in tab.status.value
