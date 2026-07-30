"""Tests for kg.ontologies.xrefs - the L7 cross-ontology xref primitives.

Covers the four single-purpose per-ontology primitives:
  - materialize_xref_edges_for_ontology  (create :<X>_XREF edges)
  - remove_xref_edges_for_ontology       (delete :<X>_XREF edges)
  - strip_materialized_xrefs_for_ontology (prune resolved dangling entries)
  - remove_dangling_xrefs_for_ontology   (remove the dangling_xrefs property)
plus the two diagnostics:
  - count_dangling_xrefs (diagnostic)
  - count_xref_edges (per-ontology + all-ontologies)

Tests work with a stub driver / session that records every Cypher
call + parameters. The actual query strings carry the ontology
sub-label and the derived xref edge type, so assertions check both
shape and the right ontology routing. Each primitive opens ONE session
and runs ONE query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.kg.ontologies import xrefs


@dataclass
class _StubResult:
    rows: list[dict[str, Any]] = field(default_factory=list)

    async def single(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


@dataclass
class _StubSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results: list[_StubResult] = field(default_factory=list)

    async def run(self, query: str, **params: Any):
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        idx = len(self.calls) - 1
        if idx < len(self.canned_results):
            return self.canned_results[idx]
        return _StubResult()

    async def __aenter__(self) -> _StubSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class _StubDriver:
    sessions: list[_StubSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results_per_session: list[list[_StubResult]] = field(default_factory=list)

    def session(self) -> _StubSession:
        idx = len(self.sessions)
        canned = (
            self.canned_results_per_session[idx]
            if idx < len(self.canned_results_per_session)
            else []
        )
        sess = _StubSession(
            raise_on_run=self.raise_on_run,
            canned_results=canned,
        )
        self.sessions.append(sess)
        return sess


@dataclass
class _StubClient:
    driver: _StubDriver = field(default_factory=_StubDriver)


def _one_row(n: int) -> _StubClient:
    """A client whose first session's first query returns {"n": n}."""
    client = _StubClient()
    client.driver.canned_results_per_session = [[_StubResult(rows=[{"n": n}])]]
    return client


# ---- materialize_xref_edges_for_ontology ----


async def test_materialize_returns_count_and_uses_derived_xref_type():
    """Resolves dangling xrefs into edges: one MERGE query using the
    ontology's derived xref type; returns the count."""
    client = _one_row(9)
    n = await xrefs.materialize_xref_edges_for_ontology(client, "MeSHTerm")
    assert n == 9
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert ":OntologyTerm" in cypher
    assert "MERGE (s)-[r:MESH_XREF]" in cypher
    # Edges only — it must NOT strip the property.
    assert "REMOVE s.dangling_xrefs" not in cypher
    assert "WHERE NOT x IN resolved" not in cypher


async def test_materialize_rejects_unknown_term_label():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.materialize_xref_edges_for_ontology(client, "NotATerm")
    assert client.driver.sessions == []


async def test_materialize_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.materialize_xref_edges_for_ontology(client, "MeSHTerm")


# ---- remove_xref_edges_for_ontology ----


async def test_remove_edges_deletes_only_outbound_typed_edges():
    """One DELETE query on the derived xref type; returns edges deleted.
    Leaves the dangling_xrefs property alone."""
    client = _one_row(12)
    n = await xrefs.remove_xref_edges_for_ontology(client, "ChEBITerm")
    assert n == 12
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":ChEBITerm" in cypher
    assert ":CHEBI_XREF" in cypher
    assert "DELETE r" in cypher
    assert "REMOVE s.dangling_xrefs" not in cypher


async def test_remove_edges_rejects_unknown_term_label():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.remove_xref_edges_for_ontology(client, "NotATerm")
    assert client.driver.sessions == []


async def test_remove_edges_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.remove_xref_edges_for_ontology(client, "MeSHTerm")


# ---- strip_materialized_xrefs_for_ontology ----


async def test_strip_materialized_prunes_resolved_entries_only():
    """One property-rewrite query keyed off existing edges of the derived
    type; returns source nodes tidied. Touches no edges."""
    client = _one_row(7)
    n = await xrefs.strip_materialized_xrefs_for_ontology(client, "GOTerm")
    assert n == 7
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":GOTerm" in cypher
    assert ":GO_XREF" in cypher
    assert "WHERE NOT x IN resolved" in cypher
    assert "SET s.dangling_xrefs" in cypher
    # Property tidy only — no edge deletion / creation.
    assert "DELETE r" not in cypher
    assert "MERGE (s)-[r" not in cypher


async def test_strip_materialized_rejects_unknown_term_label():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.strip_materialized_xrefs_for_ontology(client, "NotATerm")
    assert client.driver.sessions == []


async def test_strip_materialized_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.strip_materialized_xrefs_for_ontology(client, "MeSHTerm")


# ---- remove_dangling_xrefs_for_ontology ----


async def test_remove_dangling_removes_whole_property():
    """One REMOVE query wiping the entire dangling_xrefs property; returns
    source nodes cleared. Touches no edges."""
    client = _one_row(4)
    n = await xrefs.remove_dangling_xrefs_for_ontology(client, "MeSHTerm")
    assert n == 4
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert "REMOVE s.dangling_xrefs" in cypher
    # Full wipe, not the selective strip.
    assert "WHERE NOT x IN resolved" not in cypher
    assert "DELETE r" not in cypher


async def test_remove_dangling_rejects_unknown_term_label():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.remove_dangling_xrefs_for_ontology(client, "NotATerm")
    assert client.driver.sessions == []


async def test_remove_dangling_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.remove_dangling_xrefs_for_ontology(client, "MeSHTerm")


# ---- count_dangling_xrefs ----


async def test_count_dangling_xrefs_returns_int_for_known_ontology():
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 42}])],
    ]
    n = await xrefs.count_dangling_xrefs(client, "MeSHTerm")
    assert n == 42
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert "size(s.dangling_xrefs) > 0" in cypher


async def test_count_dangling_xrefs_unknown_ontology_raises():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.count_dangling_xrefs(client, "NotATerm")


async def test_count_dangling_xrefs_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.count_dangling_xrefs(client, "MeSHTerm")


# ---- count_xref_edges ----


async def test_count_xref_edges_per_ontology_uses_derived_type():
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 17}])],
    ]
    n = await xrefs.count_xref_edges(client, "GOTerm")
    assert n == 17
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":GO_XREF" in cypher


async def test_count_xref_edges_global_uses_pipe_union_of_all_18():
    """`term_label=None` unions all 18 xref edge types via pipe syntax."""
    from knowledge_agent.kg.schema import ONTOLOGY_XREF_RELS

    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 1234}])],
    ]
    n = await xrefs.count_xref_edges(client, None)
    assert n == 1234
    cypher, _ = client.driver.sessions[0].calls[0]
    # Every one of the 18 edge types appears in the pipe union.
    for rel in ONTOLOGY_XREF_RELS:
        assert rel in cypher


async def test_count_xref_edges_unknown_ontology_raises():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await xrefs.count_xref_edges(client, "NotATerm")


async def test_count_xref_edges_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.count_xref_edges(client, "MeSHTerm")


async def test_count_xref_edges_global_propagates_cypher_exception():
    client = _StubClient(driver=_StubDriver(raise_on_run=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        await xrefs.count_xref_edges(client, None)
