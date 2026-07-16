"""Tests for the suite clone-generator (`suite_gen.generate_suite`)."""

from __future__ import annotations

import pytest

from knowledge_agent.evaluation.models import (
    EvalCase,
    EvalDataset,
    RetrievalSettings,
    compute_facts_hash,
    validate_dataset,
)
from knowledge_agent.evaluation.suite_gen import (
    PRESET_GROUPS,
    STRATEGIES,
    STRATEGY_LABELS,
    generate_suite,
    strategy_members,
)


def _master() -> EvalDataset:
    return EvalDataset(
        name="escrt",
        cases=[
            EvalCase(id="c1", question="q1?", expected_sources=["d1"]),
            EvalCase(id="c2", question="q2?", expected_sources=["d2"]),
        ],
    )


def _all_members() -> list[tuple[str, RetrievalSettings]]:
    return strategy_members(list(STRATEGIES))


def test_generate_suite_one_file_per_member():
    """One file per member; each keeps the master's facts, takes the member's
    knobs, __suffixes case ids, and is tagged into the suite."""
    files = generate_suite(_master(), "escrt-sweep", _all_members())
    assert [stem for stem, _ in files] == [f"escrt-sweep__{s}" for s in STRATEGIES]
    _stem, ds = files[0]  # "vector"
    assert ds.suites == ["escrt-sweep"]
    assert [c.id for c in ds.cases] == ["c1__vector", "c2__vector"]
    assert ds.cases[0].retrieval.retrieval_mode == "lancedb_only"
    assert ds.cases[0].retrieval.lancedb_search_mode == "vector"


def test_all_members_share_facts_and_are_runnable():
    """All members share ONE facts_hash (same gold) — the suite invariant — and
    every generated file is runnable (its leg's required knobs are pinned)."""
    files = generate_suite(_master(), "sweep", _all_members())
    hashes = {compute_facts_hash(ds.cases) for _, ds in files}
    assert len(hashes) == 1  # same facts across the whole suite
    assert hashes == {compute_facts_hash(_master().cases)}
    for _, ds in files:
        assert validate_dataset(ds.cases) == {}  # every preset is runnable as-is


def test_generate_suite_accepts_a_custom_form_member():
    """A custom knob-set (e.g. captured from the retrieval form) is a first-class
    member — not limited to the premade presets."""
    custom = RetrievalSettings(
        retrieval_mode="lancedb_only", lancedb_search_mode="vector", num_candidates=42, top_k=7
    )
    files = generate_suite(_master(), "sweep", [("my custom", custom)])
    assert [stem for stem, _ in files] == ["sweep__my-custom"]
    _stem, ds = files[0]
    assert [c.id for c in ds.cases] == ["c1__my-custom", "c2__my-custom"]
    assert ds.cases[0].retrieval.num_candidates == 42
    assert ds.cases[0].retrieval.top_k == 7


def test_strategy_members_rejects_unknown_and_empty_suite_raises():
    with pytest.raises(ValueError, match="unknown"):
        strategy_members(["bogus"])
    with pytest.raises(ValueError, match="at least one"):
        generate_suite(_master(), "sweep", [])


def test_preset_groups_and_labels_cover_every_strategy():
    """Every strategy sits in exactly one group and has a label (so the picker
    can render them all, grouped)."""
    grouped = [name for names in PRESET_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(STRATEGIES)  # a partition — no dupes/gaps
    assert len(grouped) == len(set(grouped))
    assert set(STRATEGY_LABELS) == set(STRATEGIES)


def test_generate_suite_does_not_mutate_master():
    master = _master()
    generate_suite(master, "sweep", _all_members())
    assert [c.id for c in master.cases] == ["c1", "c2"]  # untouched
    assert master.suites == []
