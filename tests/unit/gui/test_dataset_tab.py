"""Tests for the Evaluation Dataset sub-tab (read-only browser, slice 1).

Loads a tiny gold dataset from tmp_path and exercises list render + select →
full-field detail. No backend writes (slice 1 is read-only).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.evaluation.dataset_tab import DatasetTab


def _dataset(tmp_path) -> object:
    cases = [
        {
            "id": "c1",
            "question": "What drives membrane scission?",
            "expected_sources": ["doc-abc"],
            "required_keywords": ["ESCRT-III", "scission"],
            "retrieval": {"retrieval_mode": "neo4j_only", "direct_retrieval": True},
            "user_cypher": "MATCH (e:Entity) RETURN e.key LIMIT 5",
            "category": "factual",
        },
        {"id": "c2", "question": "second question"},
    ]
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def _all_text(controls) -> str:
    """Collect every Text value from a nested control tree."""
    out: list[str] = []

    def walk(c: object) -> None:
        if isinstance(c, ft.Text) and isinstance(c.value, str):
            out.append(c.value)
        for attr in ("controls", "content"):
            child = getattr(c, attr, None)
            if isinstance(child, list):
                for x in child:
                    walk(x)
            elif child is not None:
                walk(child)

    for c in controls:
        walk(c)
    return "\n".join(out)


def test_dataset_tab_builds(fake_app):
    assert DatasetTab(fake_app, coordinator=MagicMock()).build() is not None


def test_load_and_select_renders_all_fields(fake_app, tmp_path):
    tab = DatasetTab(fake_app, coordinator=MagicMock())
    tab.build()
    tab._load(_dataset(tmp_path))

    assert len(tab._cases) == 2
    assert tab.case_list.controls  # list rendered

    tab._select(0)
    assert tab._selected == 0
    rendered = _all_text(tab.detail.controls)
    # every group + the selected case's values are visible (the form doubles
    # as documentation of what a case contains).
    assert "neo4j_only" in rendered  # retrieval settings
    assert "MATCH (e:Entity)" in rendered  # KG gold / user_cypher
    assert "ESCRT-III" in rendered  # keyword checks


def test_load_bad_dataset_surfaces_error(fake_app, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("42", encoding="utf-8")  # JSON scalar → neither array nor object
    tab = DatasetTab(fake_app, coordinator=MagicMock())
    tab.build()
    tab._load(bad)
    assert "could not load" in tab.status.value
    assert tab._cases == []
