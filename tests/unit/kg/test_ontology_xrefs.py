"""Tests for kg.ontology_xrefs - the L7 cross-ontology xref primitives.

Covers the four public primitives:
  - backfill_resolved_xrefs (per-ontology resolve + strip passes)
  - clear_xref_edges_for_ontology (edges + dangling_xrefs property)
  - count_dangling_xrefs (diagnostic)
  - count_xref_edges (per-ontology + all-ontologies)

Tests work with a stub driver / session that records every Cypher
call + parameters. The actual query strings carry the ontology
sub-label and the derived xref edge type, so assertions check both
shape and the right ontology routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from knowledge_agent.kg import ontology_xrefs


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

    async def __aenter__(self) -> "_StubSession":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


@dataclass
class _StubDriver:
    sessions: list[_StubSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    canned_results_per_session: list[list[_StubResult]] = field(
        default_factory=list
    )

    def session(self) -> _StubSession:
        idx = len(self.sessions)
        canned = (
            self.canned_results_per_session[idx]
            if idx < len(self.canned_results_per_session)
            else []
        )
        sess = _StubSession(
            raise_on_run=self.raise_on_run, canned_results=canned,
        )
        self.sessions.append(sess)
        return sess


@dataclass
class _StubClient:
    driver: _StubDriver = field(default_factory=_StubDriver)


# ---- backfill_resolved_xrefs ----


async def test_backfill_returns_per_ontology_dict_for_all_18():
    """Every shipped sub-label gets an entry in the returned dict."""
    from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS

    client = _StubClient()
    # Per-call canned: 2 queries per ontology (resolve + strip), 18 ontologies = 36 results.
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 0}]) for _ in range(36)],
    ]
    result = await ontology_xrefs.backfill_resolved_xrefs(client)
    assert result is not None
    assert set(result.keys()) == set(ONTOLOGY_SUB_LABELS)
    for label, counts in result.items():
        assert "n_edges_attempted" in counts
        assert "n_sources_cleaned" in counts


async def test_backfill_resolve_pass_queries_use_derived_xref_type():
    """The resolve query for MeSHTerm uses :MESH_XREF, for GOTerm uses
    :GO_XREF, etc. Verifies the per-ontology routing via the helper."""
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 0}]) for _ in range(36)],
    ]
    await ontology_xrefs.backfill_resolved_xrefs(client)

    # 36 calls (2 per ontology, 18 ontologies).
    calls = client.driver.sessions[0].calls
    # Resolve query for MeSHTerm comes first (matches ONTOLOGY_SUB_LABELS
    # iteration order: MESH is first).
    resolve_cypher, _ = calls[0]
    assert ":MeSHTerm" in resolve_cypher
    assert ":MESH_XREF" in resolve_cypher
    assert "MERGE (s)-[r:MESH_XREF]" in resolve_cypher


async def test_backfill_strip_pass_queries_use_derived_xref_type():
    """The strip query for the second ontology (GO) uses :GO_XREF."""
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 0}]) for _ in range(36)],
    ]
    await ontology_xrefs.backfill_resolved_xrefs(client)

    calls = client.driver.sessions[0].calls
    # GO is the 2nd ontology; its 2 queries are at indices 2 and 3.
    go_strip_cypher, _ = calls[3]
    assert ":GOTerm" in go_strip_cypher
    assert ":GO_XREF" in go_strip_cypher
    assert "WHERE NOT x IN resolved" in go_strip_cypher


async def test_backfill_aggregates_counts_per_ontology():
    """`n_edges_attempted` and `n_sources_cleaned` reflect the returned
    counts from each pass per ontology."""
    client = _StubClient()
    # 18 ontologies * (resolve_count, strip_count). Give MeSH = (5, 3), GO = (7, 4), rest 0.
    canned = []
    # MeSH (first)
    canned.append(_StubResult(rows=[{"n": 5}]))
    canned.append(_StubResult(rows=[{"n": 3}]))
    # GO (second)
    canned.append(_StubResult(rows=[{"n": 7}]))
    canned.append(_StubResult(rows=[{"n": 4}]))
    # Remaining 16 with zeros.
    for _ in range(16 * 2):
        canned.append(_StubResult(rows=[{"n": 0}]))
    client.driver.canned_results_per_session = [canned]

    result = await ontology_xrefs.backfill_resolved_xrefs(client)
    assert result["MeSHTerm"] == {
        "n_edges_attempted": 5, "n_sources_cleaned": 3,
    }
    assert result["GOTerm"] == {
        "n_edges_attempted": 7, "n_sources_cleaned": 4,
    }


async def test_backfill_per_ontology_failure_records_zeros_keeps_going():
    """Cypher errors on a per-ontology pass mark that ontology as 0/0
    but don't break the rest of the loop (caller still gets results
    for all 18 ontologies — failed ones at zero, successful ones at
    their real counts).

    Setup: a session whose every `await run()` raises. Both passes for each
    ontology fail, so every ontology gets {0, 0} but the loop still
    completes."""
    from knowledge_agent.kg.schema import ONTOLOGY_SUB_LABELS

    client = _StubClient(
        driver=_StubDriver(raise_on_run=RuntimeError("constraint violation"))
    )
    result = await ontology_xrefs.backfill_resolved_xrefs(client)
    # NOT None — the outer session opened cleanly; only the per-query
    # MATCHes raised, caught by the helpers' try/except.
    assert result is not None
    # All 18 ontologies present, all with zero counts.
    assert set(result.keys()) == set(ONTOLOGY_SUB_LABELS)
    for label, counts in result.items():
        assert counts == {
            "n_edges_attempted": 0, "n_sources_cleaned": 0,
        }


async def test_backfill_propagates_on_session_failure():
    """Outer session exception (e.g. driver gone) propagates; the
    orchestrator boundary catches and records the failure."""
    @dataclass
    class _BrokenDriver:
        def session(self) -> Any:
            raise RuntimeError("driver closed")

    client = _StubClient(driver=_BrokenDriver())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="driver closed"):
        await ontology_xrefs.backfill_resolved_xrefs(client)


# ---- clear_xref_edges_for_ontology ----


async def test_clear_xref_edges_deletes_edges_and_clears_dangling_property():
    """Two passes per call: DELETE the typed edges, then REMOVE the
    dangling_xrefs property. Returned int is `n_edges + n_props`."""
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [
            _StubResult(rows=[{"n": 12}]),  # edges deleted
            _StubResult(rows=[{"n": 7}]),   # props cleared
        ],
    ]
    n = await ontology_xrefs.clear_xref_edges_for_ontology(client, "MeSHTerm")
    assert n == 12 + 7

    calls = client.driver.sessions[0].calls
    edge_cypher, _ = calls[0]
    assert ":MeSHTerm" in edge_cypher
    assert ":MESH_XREF" in edge_cypher
    assert "DELETE r" in edge_cypher

    prop_cypher, _ = calls[1]
    assert ":MeSHTerm" in prop_cypher
    assert "REMOVE s.dangling_xrefs" in prop_cypher


async def test_clear_xref_edges_uses_derived_xref_type_for_each_ontology():
    """ChEBI -> CHEBI_XREF; NCBITaxon -> NCBITAXON_XREF."""
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 0}]), _StubResult(rows=[{"n": 0}])],
    ]
    await ontology_xrefs.clear_xref_edges_for_ontology(client, "ChEBITerm")
    edge_cypher, _ = client.driver.sessions[0].calls[0]
    assert ":CHEBI_XREF" in edge_cypher


async def test_clear_xref_edges_rejects_unknown_term_label():
    """Unknown sub-label raises ValueError without issuing any Cypher."""
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await ontology_xrefs.clear_xref_edges_for_ontology(client, "NotATerm")
    assert client.driver.sessions == []


async def test_clear_xref_edges_propagates_cypher_exception():
    """Cypher exception propagates; orchestrator boundary catches."""
    client = _StubClient(
        driver=_StubDriver(raise_on_run=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_xrefs.clear_xref_edges_for_ontology(client, "MeSHTerm")


# ---- count_dangling_xrefs ----


async def test_count_dangling_xrefs_returns_int_for_known_ontology():
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 42}])],
    ]
    n = await ontology_xrefs.count_dangling_xrefs(client, "MeSHTerm")
    assert n == 42
    cypher, _ = client.driver.sessions[0].calls[0]
    assert ":MeSHTerm" in cypher
    assert "size(s.dangling_xrefs) > 0" in cypher


async def test_count_dangling_xrefs_unknown_ontology_raises():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await ontology_xrefs.count_dangling_xrefs(client, "NotATerm")


async def test_count_dangling_xrefs_propagates_cypher_exception():
    client = _StubClient(
        driver=_StubDriver(raise_on_run=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_xrefs.count_dangling_xrefs(client, "MeSHTerm")


# ---- count_xref_edges ----


async def test_count_xref_edges_per_ontology_uses_derived_type():
    client = _StubClient()
    client.driver.canned_results_per_session = [
        [_StubResult(rows=[{"n": 17}])],
    ]
    n = await ontology_xrefs.count_xref_edges(client, "GOTerm")
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
    n = await ontology_xrefs.count_xref_edges(client, None)
    assert n == 1234
    cypher, _ = client.driver.sessions[0].calls[0]
    # Every one of the 18 edge types appears in the pipe union.
    for rel in ONTOLOGY_XREF_RELS:
        assert rel in cypher


async def test_count_xref_edges_unknown_ontology_raises():
    client = _StubClient()
    with pytest.raises(ValueError, match="unknown term_label"):
        await ontology_xrefs.count_xref_edges(client, "NotATerm")


async def test_count_xref_edges_propagates_cypher_exception():
    client = _StubClient(
        driver=_StubDriver(raise_on_run=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_xrefs.count_xref_edges(client, "MeSHTerm")


async def test_count_xref_edges_global_propagates_cypher_exception():
    client = _StubClient(
        driver=_StubDriver(raise_on_run=RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="boom"):
        await ontology_xrefs.count_xref_edges(client, None)
