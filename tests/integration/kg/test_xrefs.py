"""Integration tests for `kg/ontologies/xrefs` (L7 cross-ontology xref
edges).

Exercises the four xref edge/property primitives (materialize / remove
edges / strip materialized / remove dangling) + the count diagnostics
against a real Neo4j test instance using SYNTHETIC :OntologyTerm nodes — does NOT
trigger the heavy `import_<ontology>` paths. The actual ontology
imports are exercised by the manual smoke (`scripts/smoke_kg_l7_xrefs_l10.py`)
which downloads + parses HPO + MONDO + MeSH (heavy, slow, not
integration-tier).

Manual interactive counterpart: `scripts/smoke_kg_l7_xrefs_l10.py`.

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.kg.ontologies import xrefs
from knowledge_agent.kg.schema import (
    HPO_TERM_LABEL,
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
)

pytestmark = pytest.mark.integration


async def _seed_terms(
    client: Any,
    sub_label: str,
    terms: list[tuple[str, list[str]]],
) -> None:
    """Seed a list of (term_id, dangling_xrefs) for the given ontology
    sub-label."""
    async with client.driver.session() as session:
        for term_id, dangling in terms:
            await session.run(
                f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{sub_label} {{id: $id}}) "
                f"SET t.label = $id, t.dangling_xrefs = $dangling",
                id=term_id,
                dangling=dangling,
            )


async def _seed_target(client: Any, sub_label: str, term_id: str) -> None:
    """Seed a single ontology term that other terms' dangling_xrefs
    can resolve to."""
    async with client.driver.session() as session:
        await session.run(
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{sub_label} {{id: $id}}) SET t.label = $id",
            id=term_id,
        )


async def _resolve_and_tidy(client: Any, term_label: str) -> None:
    """Replicate the old fused backfill for one ontology: resolve its
    dangling xrefs into edges, then prune the now-resolved entries."""
    await xrefs.materialize_xref_edges_for_ontology(client, term_label)
    await xrefs.strip_materialized_xrefs_for_ontology(client, term_label)


async def test_materialize_resolves_dangling_xrefs_when_target_exists(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """A MONDO term with `dangling_xrefs = ["MESH:D003920", "HPO:0001"]`,
    where one target exists and one doesn't, resolves the existing
    target into a :MONDO_XREF edge and leaves the unresolved string
    in dangling_xrefs for a future materialize."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D003920")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D003920", "HPO:0001"]),
        ],
    )

    n_edges = await xrefs.materialize_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)
    n_tidied = await xrefs.strip_materialized_xrefs_for_ontology(kg_client, MONDO_TERM_LABEL)
    # One target present -> one edge resolved and one source tidied.
    assert n_edges >= 1
    assert n_tidied >= 1

    # Edge landed; dangling stripped of the resolved entry; HPO:0001
    # remains because no HPO term with that id was seeded.
    async with kg_client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0001'}})"
            f"-[:MONDO_XREF]->(t:{ONTOLOGY_TERM_LABEL}) RETURN t.id AS id"
        )
        edge = await result.single()
        result = await session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0001'}}) RETURN s.dangling_xrefs AS d"
        )
        dangling = (await result.single())["d"]
    assert edge is not None
    assert edge["id"] == "MESH:D003920"
    assert "MESH:D003920" not in dangling
    assert "HPO:0001" in dangling


async def test_materialize_strip_is_idempotent(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """A second materialize+strip on a fully-resolved corpus must not
    change state — MERGE is idempotent at the edge level."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D003920")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D003920"]),
        ],
    )

    await _resolve_and_tidy(kg_client, MONDO_TERM_LABEL)
    await _resolve_and_tidy(kg_client, MONDO_TERM_LABEL)

    # Edge-level idempotency is the load-bearing contract: a second
    # materialize on a fully-resolved corpus does not duplicate edges
    # (MERGE dedupes). The strip pass is a no-op the second time since
    # the resolved entries were already pruned on the first pass.
    n_edges = await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL)
    assert n_edges == 1


async def test_count_xref_edges_returns_per_ontology_count(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`count_xref_edges(client, term_label)` returns the outgoing
    edge count for that ontology sub-label."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D002")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001"]),
            ("MONDO:0002", ["MESH:D002"]),
        ],
    )
    await xrefs.materialize_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)

    n_mondo = await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL)
    n_mesh = await xrefs.count_xref_edges(kg_client, MESH_TERM_LABEL)
    assert n_mondo == 2
    assert n_mesh == 0  # MeSH terms had no dangling_xrefs themselves


async def test_count_dangling_xrefs_counts_source_nodes(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`count_dangling_xrefs(client, term_label)` returns the count of
    source nodes that still have unresolved entries in their
    `dangling_xrefs` list."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001", "MESH:UNKNOWN_1"]),
            ("MONDO:0002", ["MESH:UNKNOWN_2"]),
        ],
    )
    await _resolve_and_tidy(kg_client, MONDO_TERM_LABEL)

    # MONDO:0001 still has MESH:UNKNOWN_1 dangling; MONDO:0002 still
    # has MESH:UNKNOWN_2 dangling — both count.
    n_dangling = await xrefs.count_dangling_xrefs(kg_client, MONDO_TERM_LABEL)
    assert n_dangling == 2


async def test_remove_xref_edges_for_ontology_drops_only_target(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`remove_xref_edges_for_ontology(client, term_label)` deletes the
    outgoing :<X>_XREF edges of the target ontology only; other
    ontologies' edges survive, and the dangling_xrefs property is left
    intact."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001"]),
        ],
    )
    await _seed_terms(
        kg_client,
        HPO_TERM_LABEL,
        [
            ("HPO:0001", ["MESH:D001"]),
        ],
    )
    await xrefs.materialize_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)
    await xrefs.materialize_xref_edges_for_ontology(kg_client, HPO_TERM_LABEL)

    # Pre-clear: both MONDO and HPO have 1 outgoing edge.
    assert await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 1
    assert await xrefs.count_xref_edges(kg_client, HPO_TERM_LABEL) == 1

    n_deleted = await xrefs.remove_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)
    # Edges only: the single MONDO->MESH edge.
    assert n_deleted == 1

    # Post-clear: MONDO edges gone, HPO untouched; MONDO's property survives.
    assert await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 0
    assert await xrefs.count_xref_edges(kg_client, HPO_TERM_LABEL) == 1
    async with kg_client.driver.session() as session:
        res = await session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0001'}}) RETURN s.dangling_xrefs AS d"
        )
        row = await res.single()
    assert row["d"] is not None


async def test_remove_dangling_xrefs_for_ontology_wipes_property_keeps_edges(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`remove_dangling_xrefs_for_ontology` removes the dangling_xrefs
    property but leaves the :<X>_XREF edges in place."""
    await _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    await _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001"]),
        ],
    )
    await xrefs.materialize_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)
    assert await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 1

    n_cleared = await xrefs.remove_dangling_xrefs_for_ontology(kg_client, MONDO_TERM_LABEL)
    assert n_cleared == 1

    # Property gone, edge kept.
    assert await xrefs.count_dangling_xrefs(kg_client, MONDO_TERM_LABEL) == 0
    assert await xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 1
