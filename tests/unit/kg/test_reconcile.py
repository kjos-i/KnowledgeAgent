"""Tests for `kg.reconcile` — the corpus-wide layer reconciliation that
WIPES KG data no longer covered by the current `CorpusConfig`.

This module is destructive (DETACH DELETE on nodes, DELETE on edges) and
was previously exercised only transitively through its callers
(`test_pipeline`, `test_bulk_ops`). These tests cover each of the 5 helpers
directly with a mocked async driver — reusing the `RecordingDriver` pattern
from `test_cross_doc_writes.py` — so the "wipe only when the config drops
the layer" gate and the fail-hard contract are pinned down in isolation.

Verified per helper:
  - the config gate: no session is even opened when the layer is still on;
  - the wipe fires (right label / rel type) when the layer is off;
  - the wiped count is read back from the RETURN row;
  - fail-hard: a Cypher error propagates (the orchestrator boundary decides
    whether to surface it — reconcile never swallows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from knowledge_agent.kg import reconcile
from knowledge_agent.kg.schema import ENTITY_LABEL, RELATED_BY_XREF_REL, RELATED_TO_REL

# ---- Async driver harness (mirrors test_cross_doc_writes) ----


@dataclass
class _RunResult:
    single_value: dict[str, Any] | None = None

    async def single(self) -> dict[str, Any] | None:
        return self.single_value


@dataclass
class RecordingSession:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    raise_on_run: Exception | None = None
    single_returns: list[dict[str, Any] | None] = field(default_factory=list)

    async def run(self, query: str, **params: Any) -> _RunResult:
        if self.raise_on_run is not None:
            raise self.raise_on_run
        self.calls.append((query, params))
        sv = self.single_returns.pop(0) if self.single_returns else None
        return _RunResult(single_value=sv)

    async def __aenter__(self) -> RecordingSession:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None


@dataclass
class RecordingDriver:
    sessions: list[RecordingSession] = field(default_factory=list)
    raise_on_run: Exception | None = None
    single_returns: list[dict[str, Any] | None] = field(default_factory=list)

    def session(self) -> RecordingSession:
        s = RecordingSession(
            raise_on_run=self.raise_on_run,
            single_returns=list(self.single_returns),
        )
        self.sessions.append(s)
        return s


def _client(driver: RecordingDriver) -> SimpleNamespace:
    """A minimal stand-in — reconcile only touches `client.driver.session()`."""
    return SimpleNamespace(driver=driver)


def _config(
    *,
    entities: bool = True,
    triples: bool = True,
    cross_doc: bool = True,
    cross_doc_xrefs: bool = True,
    entity_types: list[str] | None = None,
    enabled_onts: tuple[str, ...] = (),
) -> SimpleNamespace:
    layers = SimpleNamespace(
        entities=entities,
        triples=triples,
        cross_doc=cross_doc,
        cross_doc_xrefs=cross_doc_xrefs,
    )
    ent = None if entity_types is None else SimpleNamespace(entity_types=list(entity_types))
    return SimpleNamespace(
        layers=layers,
        entities=ent,
        _enabled_ontology_layers=lambda: set(enabled_onts),
    )


# ---- L7: ontology reconciliation ----


async def test_reconcile_ontologies_wipes_only_disabled_and_imported(monkeypatch):
    """Disabled-and-imported → wiped; disabled-but-absent → skipped;
    still-enabled → not even probed."""
    mesh = {"is_imported_fn": AsyncMock(return_value=True), "delete_fn": AsyncMock()}
    go = {"is_imported_fn": AsyncMock(return_value=False), "delete_fn": AsyncMock()}
    hpo = {"is_imported_fn": AsyncMock(return_value=True), "delete_fn": AsyncMock()}
    monkeypatch.setattr(reconcile, "ONTOLOGY_REGISTRY", {"mesh": mesh, "go": go, "hpo": hpo})

    # hpo stays enabled; mesh + go are disabled.
    cfg = _config(enabled_onts=("hpo",))
    wiped = await reconcile.reconcile_ontologies_to_config(_client(RecordingDriver()), cfg)

    assert wiped == ["mesh"]
    mesh["delete_fn"].assert_awaited_once()
    go["delete_fn"].assert_not_awaited()  # disabled but not imported
    hpo["is_imported_fn"].assert_not_awaited()  # enabled → skipped before probe
    hpo["delete_fn"].assert_not_awaited()


# ---- L6: entity reconciliation ----


async def test_reconcile_entities_layer_off_wipes_all():
    driver = RecordingDriver(single_returns=[{"n": 7}])
    result = await reconcile.reconcile_entities_to_config(_client(driver), _config(entities=False))
    assert result == {"n_wiped": 7}
    cypher, _params = driver.sessions[0].calls[0]
    assert f":{ENTITY_LABEL}" in cypher
    assert "DETACH DELETE" in cypher
    # Guard the count fix: collect + size, never the per-node `WITH e, count(e)`
    # grouping that made n_wiped always 1. Validated against real Neo4j
    # (n_wiped == the true delete count) in the reconcile smoke.
    assert "count(" not in cypher


async def test_reconcile_entities_open_vocab_is_noop():
    """Layer on + no entity_types (open-vocab) → nothing wiped, no session."""
    driver = RecordingDriver()
    result = await reconcile.reconcile_entities_to_config(
        _client(driver), _config(entities=True, entity_types=None)
    )
    assert result == {"n_wiped": 0}
    assert driver.sessions == []  # gate returned before opening a session


async def test_reconcile_entities_narrowed_types_wipes_by_type():
    driver = RecordingDriver(single_returns=[{"n": 3}])
    result = await reconcile.reconcile_entities_to_config(
        _client(driver), _config(entities=True, entity_types=["Gene", "Protein"])
    )
    assert result == {"n_wiped": 3}
    cypher, params = driver.sessions[0].calls[0]
    assert "NOT e.entity_type IN $kept" in cypher
    assert params["kept"] == ["Gene", "Protein"]


# ---- L8: triples reconciliation ----


async def test_reconcile_triples_layer_on_is_noop():
    driver = RecordingDriver()
    assert await reconcile.reconcile_triples_to_config(_client(driver), _config(triples=True)) == {
        "n_wiped": 0
    }
    assert driver.sessions == []


async def test_reconcile_triples_layer_off_deletes_edges():
    driver = RecordingDriver(single_returns=[{"n": 4}])
    result = await reconcile.reconcile_triples_to_config(_client(driver), _config(triples=False))
    assert result == {"n_wiped": 4}
    cypher, _ = driver.sessions[0].calls[0]
    assert "DELETE x" in cypher  # FOREACH deletes each collected edge
    assert "count(" not in cypher  # no per-row count() grouping (the always-1 bug)


# ---- L9 / L10: cross-doc edge reconciliation ----


async def test_reconcile_cross_doc_layer_off_deletes_related_to():
    driver = RecordingDriver(single_returns=[{"n": 2}])
    result = await reconcile.reconcile_cross_doc_to_config(
        _client(driver), _config(cross_doc=False)
    )
    assert result == {"n_wiped": 2}
    cypher, _ = driver.sessions[0].calls[0]
    assert RELATED_TO_REL in cypher


async def test_reconcile_cross_doc_layer_on_is_noop():
    driver = RecordingDriver()
    assert await reconcile.reconcile_cross_doc_to_config(
        _client(driver), _config(cross_doc=True)
    ) == {"n_wiped": 0}
    assert driver.sessions == []


async def test_reconcile_cross_doc_xrefs_layer_off_deletes_xref_edges():
    driver = RecordingDriver(single_returns=[{"n": 1}])
    result = await reconcile.reconcile_cross_doc_xrefs_to_config(
        _client(driver), _config(cross_doc_xrefs=False)
    )
    assert result == {"n_wiped": 1}
    cypher, _ = driver.sessions[0].calls[0]
    assert RELATED_BY_XREF_REL in cypher


async def test_reconcile_cross_doc_xrefs_layer_on_is_noop():
    driver = RecordingDriver()
    assert await reconcile.reconcile_cross_doc_xrefs_to_config(
        _client(driver), _config(cross_doc_xrefs=True)
    ) == {"n_wiped": 0}
    assert driver.sessions == []


# ---- fail-hard contract ----


async def test_reconcile_propagates_cypher_error():
    """A delete failure propagates (fail-hard) — reconcile never swallows;
    the caller (ingest / bulk_op) is the boundary that decides."""
    driver = RecordingDriver(raise_on_run=RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await reconcile.reconcile_triples_to_config(_client(driver), _config(triples=False))
