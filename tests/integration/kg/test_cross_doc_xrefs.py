"""Integration tests for `kg/cross_doc_xrefs_writes` (L10 cross-document
synthesis via xref equivalence).

L10 mirrors L9 but at the canonical-concept level instead of the
raw-entity level: docs are "related by xref" when they canonicalise
some of their entities to ontology terms that are either identical
OR connected via :<X>_XREF edges.

Tests use synthetic Document + Chunk + Entity + CANONICAL_TO +
OntologyTerm setup so the L10 query has something to walk. No real
ontology imports — that's the smoke's domain.

Manual interactive counterpart: `scripts/smoke_kg_l7_xrefs_l10.py`.

Requires the test Neo4j instance from `.env.test`. Skipped by
default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import pytest

from knowledge_agent.ingestion.ids import make_chunk_id
from knowledge_agent.kg.schema import (
    HPO_TERM_LABEL,
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
)

DOC_ALPHA = "integ-doc-l10-alpha"
DOC_BETA = "integ-doc-l10-beta"

pytestmark = pytest.mark.integration


def _seed_doc_with_canonical_entities(
    client: Any,
    doc_id: str,
    canonical_targets: list[tuple[str, str]],
) -> str:
    """Create :Document + :Chunk + :Entity + :CANONICAL_TO ->
    :OntologyTerm for each (entity_key, ontology_term_id) tuple.

    The :OntologyTerm nodes are pre-seeded if missing (with the same
    sub_label as supplied).
    """
    chunk_id = make_chunk_id(doc_id, 0)
    with client.driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.in_corpus = true, d:Paper "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.section = 'Synthetic', c.page = 1, c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id, chunk_id=chunk_id,
        )
        for entity_key, term_id in canonical_targets:
            session.run(
                "MERGE (e:Entity {key: $key, entity_type: 'concept'}) "
                "WITH e "
                "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                "MERGE (c)-[:MENTIONS]->(e) "
                "WITH e "
                f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
                f"  {{id: $tid}}) "
                "MERGE (e)-[:CANONICAL_TO]->(t)",
                key=entity_key, chunk_id=chunk_id, tid=term_id,
            )
    return chunk_id


async def test_recompute_l10_writes_related_by_xref_when_canonicals_match(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Two docs canonicalising entities to the same :MeSHTerm concepts
    get a :RELATED_BY_XREF edge with shared_count + shared_concepts."""
    shared = [("metformin", "MESH:D008687"), ("diabetes", "MESH:D003920")]
    _seed_doc_with_canonical_entities(kg_client, DOC_ALPHA, shared)
    _seed_doc_with_canonical_entities(kg_client, DOC_BETA, shared)

    n = await kg_client.recompute_cross_doc_xrefs_edges(DOC_ALPHA, 2)
    assert n == 1

    with kg_client.driver.session() as session:
        row = session.run(
            "MATCH (a:Document {doc_id: $a})-[r:RELATED_BY_XREF]-"
            "(b:Document {doc_id: $b}) "
            "RETURN r.shared_count AS n, r.shared_concepts AS concepts",
            a=DOC_ALPHA, b=DOC_BETA,
        ).single()
    assert row is not None
    assert row["n"] == 2
    assert set(row["concepts"]) == {"MESH:D008687", "MESH:D003920"}


async def test_recompute_l10_writes_via_xref_edge_equivalence(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """When alpha canonicalises to MeSH:X and beta canonicalises to
    HPO:Y, AND a `:MESH_XREF` edge connects MeSH:X to HPO:Y, the
    recompute treats them as the SAME shared concept."""
    # alpha → MeSH terms; beta → HPO terms.
    _seed_doc_with_canonical_entities(
        kg_client, DOC_ALPHA,
        [("metformin", "MESH:D008687"), ("diabetes", "MESH:D003920")],
    )
    with kg_client.driver.session() as session:
        # Manually create HPO targets + xref edges from MeSH to HPO.
        session.run(
            f"MERGE (h:{ONTOLOGY_TERM_LABEL}:{HPO_TERM_LABEL} "
            f"  {{id: 'HPO:0001'}}) "
            f"MERGE (m:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
            f"  {{id: 'MESH:D008687'}}) "
            f"MERGE (m)-[:MESH_XREF]->(h)"
        )
        session.run(
            f"MERGE (h:{ONTOLOGY_TERM_LABEL}:{HPO_TERM_LABEL} "
            f"  {{id: 'HPO:0002'}}) "
            f"MERGE (m:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
            f"  {{id: 'MESH:D003920'}}) "
            f"MERGE (m)-[:MESH_XREF]->(h)"
        )
    # beta canonicalises to the HPO equivalents — but the seed function
    # above uses MeSH sub-label by default; manually set up beta's
    # entities → HPO term canonicals. Beta's entity KEYS deliberately
    # differ from alpha's (suffix `_beta`) because the schema's
    # `(key, entity_type)` composite key would otherwise collapse
    # them into one shared :Entity node, which would canonicalise to
    # BOTH MeSH and HPO and break the xref-equivalence test signal.
    beta_chunk = make_chunk_id(DOC_BETA, 0)
    with kg_client.driver.session() as session:
        session.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "ON CREATE SET d.in_corpus = true, d:Paper "
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.doc_id = $doc_id, c.chunk_index = 0, "
            "c.section = 'Synthetic', c.page = 1, c.content_type = 'text' "
            "MERGE (c)-[:PART_OF]->(d) "
            "WITH c "
            "MERGE (e1:Entity {key: 'metformin_beta', entity_type: 'concept'}) "
            "MERGE (c)-[:MENTIONS]->(e1) "
            "MERGE (h1:OntologyTerm:HPOTerm {id: 'HPO:0001'}) "
            "MERGE (e1)-[:CANONICAL_TO]->(h1) "
            "WITH c "
            "MERGE (e2:Entity {key: 'diabetes_beta', entity_type: 'concept'}) "
            "MERGE (c)-[:MENTIONS]->(e2) "
            "MERGE (h2:OntologyTerm:HPOTerm {id: 'HPO:0002'}) "
            "MERGE (e2)-[:CANONICAL_TO]->(h2)",
            doc_id=DOC_BETA, chunk_id=beta_chunk,
        )

    n = await kg_client.recompute_cross_doc_xrefs_edges(DOC_ALPHA, 2)
    assert n == 1

    with kg_client.driver.session() as session:
        row = session.run(
            "MATCH (a:Document {doc_id: $a})-[r:RELATED_BY_XREF]-"
            "(b:Document {doc_id: $b}) "
            "RETURN r.shared_count AS n",
            a=DOC_ALPHA, b=DOC_BETA,
        ).single()
    assert row is not None
    assert row["n"] == 2


async def test_recompute_l10_writes_no_edge_below_threshold(
    kg_client: Any, ensure_constraints: None, clean_kg: None
) -> None:
    """Below the threshold, no edge."""
    shared = [("metformin", "MESH:D008687")]  # only 1 shared concept
    _seed_doc_with_canonical_entities(kg_client, DOC_ALPHA, shared)
    _seed_doc_with_canonical_entities(kg_client, DOC_BETA, shared)

    n = await kg_client.recompute_cross_doc_xrefs_edges(DOC_ALPHA, 2)
    assert n == 0
