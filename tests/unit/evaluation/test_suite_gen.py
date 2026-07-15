"""Tests for the suite clone-generator (`suite_gen.generate_suite`)."""

from __future__ import annotations

import pytest

from knowledge_agent.evaluation.models import (
    EvalCase,
    EvalDataset,
    compute_facts_hash,
    validate_dataset,
)
from knowledge_agent.evaluation.suite_gen import STRATEGIES, generate_suite


def _master() -> EvalDataset:
    return EvalDataset(
        name="escrt",
        cases=[
            EvalCase(id="c1", question="q1?", expected_sources=["d1"]),
            EvalCase(id="c2", question="q2?", expected_sources=["d2"]),
        ],
    )


def test_generate_suite_clones_all_strategies():
    """One file per strategy; each keeps the master's facts, takes the strategy's
    knobs, __suffixes case ids, and is tagged into the suite."""
    files = generate_suite(_master(), "escrt-sweep")
    assert [stem for stem, _ in files] == [f"escrt-sweep__{s}" for s in STRATEGIES]
    _stem, ds = files[0]  # "vector"
    assert ds.suites == ["escrt-sweep"]
    assert [c.id for c in ds.cases] == ["c1__vector", "c2__vector"]
    assert ds.cases[0].retrieval.retrieval_mode == "lancedb_only"
    assert ds.cases[0].retrieval.lancedb_search_mode == "vector"


def test_generate_suite_members_share_facts_and_are_runnable():
    """All members share ONE facts_hash (same gold) — the suite invariant — and
    every generated file is runnable (its leg's required knobs are pinned)."""
    files = generate_suite(_master(), "sweep")
    hashes = {compute_facts_hash(ds.cases) for _, ds in files}
    assert len(hashes) == 1  # same facts across the whole suite
    # ...and it matches the master's facts (id + knobs excluded from facts_hash)
    assert hashes == {compute_facts_hash(_master().cases)}
    for _, ds in files:
        assert validate_dataset(ds.cases) == {}  # runnable as-is


def test_generate_suite_subset_and_unknown():
    files = generate_suite(_master(), "sweep", strategies=["graph", "auto"])
    assert [stem for stem, _ in files] == ["sweep__graph", "sweep__auto"]
    with pytest.raises(ValueError, match="unknown"):
        generate_suite(_master(), "sweep", strategies=["bogus"])


def test_generate_suite_does_not_mutate_master():
    master = _master()
    generate_suite(master, "sweep")
    assert [c.id for c in master.cases] == ["c1", "c2"]  # untouched
    assert master.suites == []
