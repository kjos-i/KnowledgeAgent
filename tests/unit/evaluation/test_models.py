"""Tests for the EvalCase schema + queryset loader."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge_agent.evaluation.models import (
    EvalCase,
    EvalDataset,
    EvalRecipe,
    RetrievalSettings,
    append_case,
    compute_dataset_hash,
    compute_facts_hash,
    compute_recipe_hash,
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


def test_legacy_in_progress_status_coerces_to_draft(tmp_path):
    """The retired 'in progress' status (dropped 2026-07-13; draft/final only)
    maps forward to 'draft' so an older dataset still loads instead of failing
    the Literal."""
    p = tmp_path / "legacy.json"
    p.write_text(
        json.dumps({"status": "in progress", "cases": [{"id": "a", "question": "q?"}]}),
        encoding="utf-8",
    )
    assert load_dataset(p).status == "draft"


def test_frozen_roundtrips_and_requires_final(tmp_path):
    """`frozen` persists in the header for a final dataset; the invariant
    validator clears it for any non-final status."""
    p = tmp_path / "f.json"
    save_dataset(
        EvalDataset(status="final", frozen=True, cases=[EvalCase(id="a", question="q?")]), p
    )
    assert load_dataset(p).frozen is True
    assert EvalDataset(status="draft", frozen=True).frozen is False  # invariant coerces
    assert EvalDataset().frozen is False  # default is unfrozen


def test_load_rejects_json_scalar(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array or object"):
        load_cases(p)


def test_case_origin_default_and_values():
    assert EvalCase(id="x", question="q?").origin == "manual"
    assert EvalCase(id="x", question="q?", origin="llm").origin == "llm"
    assert EvalCase(id="x", question="q?", origin="chat").origin == "chat"
    with pytest.raises(ValidationError):
        EvalCase(id="x", question="q?", origin="bogus")


def test_case_chat_provenance_roundtrips(tmp_path):
    """A chat-sourced case's provenance — origin='chat' + the conversation turns
    + the router model — persists in the dataset JSON and reloads. Provenance
    only; never read by scoring."""
    p = tmp_path / "d.json"
    case = EvalCase(
        id="chatcase",
        question="why did the valve fail?",
        origin="chat",
        source_conversation=[
            {"role": "user", "content": "valve?"},
            {"role": "assistant", "content": "searching the corpus…"},
        ],
        chat_router_model="claude-haiku-4-5",
    )
    save_dataset(EvalDataset(cases=[case]), p)
    loaded = load_dataset(p).cases[0]
    assert loaded.origin == "chat"
    assert [t.role for t in loaded.source_conversation] == ["user", "assistant"]
    assert loaded.source_conversation[0].content == "valve?"
    assert loaded.chat_router_model == "claude-haiku-4-5"


def test_case_chat_provenance_defaults_empty():
    c = EvalCase(id="x", question="q?")
    assert c.source_conversation == []
    assert c.chat_router_model is None


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


def test_facts_hash_is_the_suite_identity():
    """facts_hash = the SUITE identity: identical for cases with the same gold
    but different retrieval knobs AND different (strategy-suffixed) ids, so a
    knob-sweep's member files hash to one suite. It moves only when the gold
    itself changes."""
    base = EvalCase(
        id="q1__vector",
        question="q1?",
        expected_sources=["d1"],
        retrieval={"retrieval_mode": "lancedb_only", "top_k": 5},
    )
    # same facts, swept knob + strategy-suffixed id -> SAME facts_hash
    sibling = EvalCase(
        id="q1__graph",
        question="q1?",
        expected_sources=["d1"],
        retrieval={"retrieval_mode": "neo4j_only", "top_k": 20},
    )
    h = compute_facts_hash([base])
    assert len(h) == 64
    assert compute_facts_hash([sibling]) == h
    # ...while dataset_hash (facts + knobs + id) DOES separate the two files
    assert compute_dataset_hash([base]) != compute_dataset_hash([sibling])
    # editing the gold moves facts_hash
    edited = EvalCase(id="q1__vector", question="q1?", expected_sources=["d2"])
    assert compute_facts_hash([edited]) != h


def test_facts_hash_order_independent():
    a = EvalCase(id="a", question="q1?")
    b = EvalCase(id="b", question="q2?")
    assert compute_facts_hash([a, b]) == compute_facts_hash([b, a])


# ---- recipe (C2) ----


def test_recipe_defaults_mirror_evalconfig():
    r = EvalRecipe()
    assert sorted(r.enabled_groups) == ["chunk", "kg", "source"]
    assert r.judge_threshold == 0.5
    assert r.metadata_match_threshold == 0.8
    assert r.required_keyword_threshold == 0.5
    assert r.judge_models == []


def test_recipe_roundtrips_on_the_dataset_header(tmp_path):
    p = tmp_path / "gold.json"
    ds = EvalDataset(
        recipe=EvalRecipe(enabled_groups=["source"], judge_threshold=0.9),
        cases=[EvalCase(id="a", question="q?")],
    )
    save_dataset(ds, p)
    loaded = load_dataset(p)
    assert loaded.recipe is not None
    assert loaded.recipe.enabled_groups == ["source"]
    assert loaded.recipe.judge_threshold == 0.9


def test_recipe_is_excluded_from_the_dataset_hash():
    """The recipe lives on the header, so flipping it must NOT move the
    dataset's content fingerprint (run comparability vs the gold)."""
    cases = [EvalCase(id="a", question="q?")]
    bare = compute_dataset_hash(EvalDataset(cases=cases).cases)
    with_recipe = compute_dataset_hash(
        EvalDataset(recipe=EvalRecipe(judge_threshold=0.1), cases=cases).cases
    )
    assert bare == with_recipe


def test_recipe_hash_none_when_absent_and_content_addressed():
    assert compute_recipe_hash(None) is None
    h = compute_recipe_hash(EvalRecipe(judge_threshold=0.5))
    assert len(h) == 64
    # a value change moves it; the same content re-hashes identically
    assert compute_recipe_hash(EvalRecipe(judge_threshold=0.9)) != h
    assert compute_recipe_hash(EvalRecipe(judge_threshold=0.5)) == h


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
