"""Smoke test for L9 cross-document synthesis (with manual-inspection
pause).

L9 walks `(focal)<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(:Entity)
<-[:MENTIONS]-(:Chunk)-[:PART_OF]->(other)` and writes one
undirected `:RELATED_TO` edge per (this, other) pair whose distinct
shared-entity count meets `threshold`. Pure Cypher pass; no LLM
involved.

This smoke seeds TWO synthetic docs sharing 3 entities so the L9
recompute has something to find, then runs
`recompute_cross_doc_edges` and verifies the resulting :RELATED_TO
edge + its `shared_keys` / `shared_count` properties.

Requires Neo4j running with NEO4J_PASSWORD set in `.env.test` per
[[test-instance-setup]] — script switches to test instance up
front. No external services; pure Cypher.

Lifecycle:
  1. Clear any leftover smoke nodes (2 synthetic docs + their chunks
     + shared entities + :RELATED_TO edges).
  2. Seed two synthetic docs sharing 3 entities (alpha + beta share
     {a, b, c}; alpha additionally has {d}; beta additionally has
     {e}).
  3. Run `client.recompute_cross_doc_edges(alpha_doc_id, threshold)`.
  4. Print the edge count + read back :RELATED_TO + its properties.
  5. Pause — you inspect in Neo4j Desktop.
  6. Press Enter to clean up, Ctrl+C to keep the nodes.

Run from the project root:
    python scripts/smoke_kg_l9_cross_doc.py
    python scripts/smoke_kg_l9_cross_doc.py --threshold 2  # default
    python scripts/smoke_kg_l9_cross_doc.py --threshold 5  # too high; expect 0 edges

Automated counterpart (for regression catching, no inspection):
  tests/integration/kg/test_cross_doc_writes.py  (threshold semantics,
                                                  wipe-and-rewrite
                                                  contract, multi-doc
                                                  neighbourhoods)
Run via `pytest -m integration tests/integration/kg/`.
"""

import argparse

# Switch to the smoke-test Neo4j instance BEFORE any other RLA import.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.ingestion.ids import make_chunk_id  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    CHUNK_LABEL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
)

DOC_ALPHA = "smoke-doc-l9-alpha"
DOC_BETA = "smoke-doc-l9-beta"

# Shared entities (3) + unique-to-alpha (1) + unique-to-beta (1).
SHARED_ENTITIES: list[tuple[str, str]] = [
    ("metformin", "drug"),
    ("ampk", "protein"),
    ("type 2 diabetes mellitus", "disease"),
]
ALPHA_UNIQUE: list[tuple[str, str]] = [("glut4", "protein")]
BETA_UNIQUE: list[tuple[str, str]] = [("hba1c", "biomarker")]


def _delete_smoke_nodes(client) -> None:
    """Drop both synthetic docs + chunks + their entity orphans +
    :RELATED_TO edges between them. Idempotent."""
    with client.driver.session() as session:
        # :RELATED_TO is undirected by convention; one query covers both.
        session.run(
            f"MATCH (d1:{DOCUMENT_LABEL})-[r:RELATED_TO]-(d2:{DOCUMENT_LABEL}) "
            f"WHERE d1.doc_id IN $ids AND d2.doc_id IN $ids "
            f"DELETE r",
            ids=[DOC_ALPHA, DOC_BETA],
        )
        # Drop :MENTIONS from these docs' chunks, then orphan-GC entities.
        session.run(
            f"MATCH (d:{DOCUMENT_LABEL})<-[:PART_OF]-"
            f"(c:{CHUNK_LABEL})-[m:MENTIONS]->(e:{ENTITY_LABEL}) "
            f"WHERE d.doc_id IN $ids DELETE m",
            ids=[DOC_ALPHA, DOC_BETA],
        )
        session.run(
            f"MATCH (e:{ENTITY_LABEL}) "
            f"WHERE NOT (e)<-[:MENTIONS]-() "
            f"DETACH DELETE e"
        )
        # Drop chunks + docs.
        session.run(
            f"MATCH (c:{CHUNK_LABEL}) "
            f"WHERE c.doc_id IN $ids "
            f"DETACH DELETE c",
            ids=[DOC_ALPHA, DOC_BETA],
        )
        session.run(
            f"MATCH (d:{DOCUMENT_LABEL}) "
            f"WHERE d.doc_id IN $ids "
            f"DETACH DELETE d",
            ids=[DOC_ALPHA, DOC_BETA],
        )


def _seed_doc(
    client, doc_id: str, entities: list[tuple[str, str]]
) -> str:
    """Write a synthetic :Document + one :Chunk + :MENTIONS edges to
    each entity. Returns chunk_id."""
    chunk_id = make_chunk_id(doc_id, 0)
    with client.driver.session() as session:
        session.run(
            f"MERGE (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"ON CREATE SET d.in_corpus = true, d:Paper "
            f"MERGE (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}}) "
            f"SET c.doc_id = $doc_id, c.chunk_index = 0, "
            f"c.section = 'Synthetic', c.page = 1, "
            f"c.content_type = 'text' "
            f"MERGE (c)-[:PART_OF]->(d)",
            doc_id=doc_id,
            chunk_id=chunk_id,
        )
        for key, etype in entities:
            session.run(
                f"MERGE (e:{ENTITY_LABEL} {{key: $key, type: $type}}) "
                f"WITH e "
                f"MATCH (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}}) "
                f"MERGE (c)-[m:MENTIONS]->(e) "
                f"ON CREATE SET m.start_char = 0, m.end_char = $len",
                key=key,
                type=etype,
                chunk_id=chunk_id,
                len=len(key),
            )
    return chunk_id


def _show_state(client) -> None:
    """Read back the :RELATED_TO edge between the two synthetic docs."""
    with client.driver.session() as session:
        rows = list(session.run(
            f"MATCH (a:{DOCUMENT_LABEL} {{doc_id: $a_id}})"
            f"-[r:RELATED_TO]-(b:{DOCUMENT_LABEL} {{doc_id: $b_id}}) "
            f"RETURN r.shared_count AS n, r.shared_keys AS keys, "
            f"r.computed_at AS ts",
            a_id=DOC_ALPHA,
            b_id=DOC_BETA,
        ))
    if not rows:
        print("  no :RELATED_TO edge found between the two docs")
        return
    for r in rows:
        print(f"  :RELATED_TO  shared_count={r['n']}")
        print(f"               shared_keys={r['keys']}")
        print(f"               computed_at={r['ts']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help=(
            "Shared-entity threshold for :RELATED_TO edge creation. "
            "Default 2. With the seeded 3 shared entities, threshold "
            "<=3 produces an edge; threshold >=4 produces none."
        ),
    )
    args = parser.parse_args()

    client = get_kg_client()

    print("Applying constraints...")
    client.ensure_constraints()

    print("Clearing any leftover smoke nodes from previous runs...")
    _delete_smoke_nodes(client)

    print(f"Seeding two synthetic docs sharing {len(SHARED_ENTITIES)} entities...")
    alpha_chunk = _seed_doc(client, DOC_ALPHA, SHARED_ENTITIES + ALPHA_UNIQUE)
    beta_chunk = _seed_doc(client, DOC_BETA, SHARED_ENTITIES + BETA_UNIQUE)
    print(f"  alpha doc: {DOC_ALPHA} (chunk {alpha_chunk[:24]}..., "
          f"{len(SHARED_ENTITIES) + len(ALPHA_UNIQUE)} entities)")
    print(f"  beta doc : {DOC_BETA} (chunk {beta_chunk[:24]}..., "
          f"{len(SHARED_ENTITIES) + len(BETA_UNIQUE)} entities)")
    print(f"  shared   : {[k for k, _ in SHARED_ENTITIES]}")

    print(
        f"Running recompute_cross_doc_edges(alpha, threshold={args.threshold})..."
    )
    n = client.recompute_cross_doc_edges(DOC_ALPHA, args.threshold)
    print(f"  :RELATED_TO edges written: {n}")
    expected_present = args.threshold <= len(SHARED_ENTITIES)
    if expected_present and n == 0:
        print(
            f"  WARNING: expected an edge (threshold={args.threshold} "
            f"<= shared={len(SHARED_ENTITIES)}) but got 0. Investigate."
        )
    elif not expected_present and (n or 0) > 0:
        print(
            f"  WARNING: expected no edge (threshold={args.threshold} "
            f"> shared={len(SHARED_ENTITIES)}) but got {n}. Investigate."
        )

    print()
    print("Post-recompute KG state:")
    _show_state(client)

    print()
    print("In Neo4j Desktop -> Query, try:")
    print(
        f"  MATCH (a:Document {{doc_id: '{DOC_ALPHA}'}})"
        f"-[r:RELATED_TO]-(b:Document {{doc_id: '{DOC_BETA}'}}) "
        f"RETURN r.shared_count, r.shared_keys"
    )
    print(
        f"  MATCH (d1:Document)-[:RELATED_TO]-(d2:Document) "
        f"WHERE d1.doc_id IN ['{DOC_ALPHA}','{DOC_BETA}'] "
        f"RETURN d1.doc_id, d2.doc_id"
    )
    print()

    try:
        input("Press Enter to delete the smoke nodes, Ctrl+C to keep them. ")
    except KeyboardInterrupt:
        print()
        print("Keeping smoke nodes. Re-run this script to clean them up later.")
        client.close()
        return

    print("Deleting smoke nodes...")
    _delete_smoke_nodes(client)
    print("Done.")
    client.close()


if __name__ == "__main__":
    main()
