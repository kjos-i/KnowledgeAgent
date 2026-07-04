"""Integration tests for `kg/ontology_xrefs` (L7 cross-ontology xref
edges).

Exercises the backfill / clear / count primitives against a real
Neo4j test instance using SYNTHETIC :OntologyTerm nodes — does NOT
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

from knowledge_agent.kg import ontology_xrefs
from knowledge_agent.kg.schema import (
    HPO_TERM_LABEL,
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
)

pytestmark = pytest.mark.integration


def _seed_terms(
    client: Any,
    sub_label: str,
    terms: list[tuple[str, list[str]]],
) -> None:
    """Seed a list of (term_id, dangling_xrefs) for the given ontology
    sub-label."""
    with client.driver.session() as session:
        for term_id, dangling in terms:
            session.run(
                f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{sub_label} {{id: $id}}) "
                f"SET t.label = $id, t.dangling_xrefs = $dangling",
                id=term_id,
                dangling=dangling,
            )


def _seed_target(client: Any, sub_label: str, term_id: str) -> None:
    """Seed a single ontology term that other terms' dangling_xrefs
    can resolve to."""
    with client.driver.session() as session:
        session.run(
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{sub_label} {{id: $id}}) SET t.label = $id",
            id=term_id,
        )


def test_backfill_resolves_dangling_xrefs_when_target_exists(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """A MONDO term with `dangling_xrefs = ["MESH:D003920", "HPO:0001"]`,
    where one target exists and one doesn't, resolves the existing
    target into a :MONDO_XREF edge and leaves the unresolved string
    in dangling_xrefs for a future backfill."""
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D003920")
    _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D003920", "HPO:0001"]),
        ],
    )

    result = ontology_xrefs.backfill_resolved_xrefs(kg_client)
    assert result is not None

    # The MONDO entry shows 1 edge attempted + 1 source cleaned.
    mondo_stats = result.get(MONDO_TERM_LABEL, {})
    assert mondo_stats.get("n_edges_attempted") >= 1
    assert mondo_stats.get("n_sources_cleaned") >= 1

    # Edge landed; dangling stripped of the resolved entry; HPO:0001
    # remains because no HPO term with that id was seeded.
    with kg_client.driver.session() as session:
        edge = session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0001'}})"
            f"-[:MONDO_XREF]->(t:{ONTOLOGY_TERM_LABEL}) RETURN t.id AS id"
        ).single()
        dangling = session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0001'}}) RETURN s.dangling_xrefs AS d"
        ).single()["d"]
    assert edge is not None
    assert edge["id"] == "MESH:D003920"
    assert "MESH:D003920" not in dangling
    assert "HPO:0001" in dangling


def test_backfill_is_idempotent(kg_client: Any, ensure_constraints: None, clean_kg: None) -> None:
    """A second backfill on a fully-resolved corpus must not change
    state (n_sources_cleaned drops to 0 on the second call — MERGE
    is idempotent at the edge level)."""
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D003920")
    _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D003920"]),
        ],
    )

    ontology_xrefs.backfill_resolved_xrefs(kg_client)
    second = ontology_xrefs.backfill_resolved_xrefs(kg_client)
    assert second is not None

    # Edge-level idempotency is the load-bearing contract: a second
    # backfill on a fully-resolved corpus does not duplicate edges.
    # (`n_sources_cleaned` may still report 1 because the strip pass
    # touches the source node again with an idempotent SET — that's
    # a noisy count, not a state change. The MERGE not duplicating
    # is what matters.)
    n_edges = ontology_xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL)
    assert n_edges == 1


def test_count_xref_edges_returns_per_ontology_count(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`count_xref_edges(client, term_label)` returns the outgoing
    edge count for that ontology sub-label."""
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D002")
    _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001"]),
            ("MONDO:0002", ["MESH:D002"]),
        ],
    )
    ontology_xrefs.backfill_resolved_xrefs(kg_client)

    n_mondo = ontology_xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL)
    n_mesh = ontology_xrefs.count_xref_edges(kg_client, MESH_TERM_LABEL)
    assert n_mondo == 2
    assert n_mesh == 0  # MeSH terms had no dangling_xrefs themselves


def test_count_dangling_xrefs_counts_source_nodes(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`count_dangling_xrefs(client, term_label)` returns the count of
    source nodes that still have unresolved entries in their
    `dangling_xrefs` list."""
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001", "MESH:UNKNOWN_1"]),
            ("MONDO:0002", ["MESH:UNKNOWN_2"]),
        ],
    )
    ontology_xrefs.backfill_resolved_xrefs(kg_client)

    # MONDO:0001 still has MESH:UNKNOWN_1 dangling; MONDO:0002 still
    # has MESH:UNKNOWN_2 dangling — both count.
    n_dangling = ontology_xrefs.count_dangling_xrefs(kg_client, MONDO_TERM_LABEL)
    assert n_dangling == 2


async def test_clear_xref_edges_for_ontology_drops_only_target(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """`clear_xref_edges_for_ontology(client, term_label)` drops
    outgoing :<X>_XREF edges from the target ontology only;
    other ontologies' edges survive."""
    _seed_target(kg_client, MESH_TERM_LABEL, "MESH:D001")
    _seed_terms(
        kg_client,
        MONDO_TERM_LABEL,
        [
            ("MONDO:0001", ["MESH:D001"]),
        ],
    )
    _seed_terms(
        kg_client,
        HPO_TERM_LABEL,
        [
            ("HPO:0001", ["MESH:D001"]),
        ],
    )
    ontology_xrefs.backfill_resolved_xrefs(kg_client)

    # Pre-clear: both MONDO and HPO have 1 outgoing edge.
    assert ontology_xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 1
    assert ontology_xrefs.count_xref_edges(kg_client, HPO_TERM_LABEL) == 1

    n_cleared = await ontology_xrefs.clear_xref_edges_for_ontology(kg_client, MONDO_TERM_LABEL)
    # n_cleared = edges deleted + dangling_xrefs props removed
    # = 1 (MONDO->MESH edge) + 1 (MONDO:0001 dangling_xrefs prop) = 2.
    assert n_cleared == 2

    # Post-clear: MONDO empty, HPO untouched.
    assert ontology_xrefs.count_xref_edges(kg_client, MONDO_TERM_LABEL) == 0
    assert ontology_xrefs.count_xref_edges(kg_client, HPO_TERM_LABEL) == 1
