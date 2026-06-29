"""Tests for `state.py` helpers (`effective_mode`).

The `AgentState` TypedDict itself has no behaviour worth testing - it's a
dict-shape contract. The `effective_mode` helper does have logic, and is
load-bearing: every retriever node and the graph router reads modes
through it, so the precedence rules need explicit coverage.
"""

from dataclasses import dataclass

from knowledge_agent.state import effective_mode


@dataclass
class FakeSettings:
    """Stub for `Settings` - only the field `effective_mode` reads."""

    default_retrieval_mode: str


def test_effective_mode_prefers_routed_mode_over_retrieval_mode():
    state = {
        "routed_mode": "neo4j_only",
        "retrieval_mode": "lancedb_only",
    }
    assert effective_mode(
        state, FakeSettings(default_retrieval_mode="parallel_fused")
    ) == "neo4j_only"


def test_effective_mode_prefers_retrieval_mode_over_default():
    state = {"retrieval_mode": "neo4j_only"}
    assert effective_mode(
        state, FakeSettings(default_retrieval_mode="lancedb_only")
    ) == "neo4j_only"


def test_effective_mode_falls_back_to_settings_default():
    state: dict = {}
    assert effective_mode(
        state, FakeSettings(default_retrieval_mode="parallel_fused")
    ) == "parallel_fused"


def test_effective_mode_none_routed_mode_does_not_shadow_retrieval_mode():
    """routed_mode set explicitly to None must fall through to retrieval_mode."""
    state = {
        "routed_mode": None,
        "retrieval_mode": "neo4j_only",
    }
    assert effective_mode(
        state, FakeSettings(default_retrieval_mode="lancedb_only")
    ) == "neo4j_only"
