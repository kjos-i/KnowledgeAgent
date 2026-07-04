"""Smoke test for L8 typed-relation triples (with manual-inspection
pause).

L8 layers entity-to-entity edges on top of L6 mentions: for each
chunk that has L6 entities, the LLM is shown the chunk text + the
extracted entity vocabulary and asked to emit `(subject, predicate,
object)` triples constrained to a fixed predicate vocabulary (the 15
edge types in `schema.TRIPLE_PREDICATE_RELS`). Out-of-vocabulary
predicates are dropped per row.

This smoke pre-seeds a synthetic :Chunk + L6 entities, runs the L8
extractor on the synthetic chunk text, and writes the resulting
triples via `client.write_triples`. The graph state is then surfaced
for manual inspection.

Requires:
  - Neo4j running with NEO4J_PASSWORD set in `.env.test` per
    [[test-instance-setup]] — script switches to test instance up
    front
  - ANTHROPIC_API_KEY in `.env.test` (Haiku call for triples
    extraction; cost ~$0.001 per smoke run)

Lifecycle:
  1. Clear any leftover smoke nodes (synthetic doc + chunks +
     entities + triples).
  2. Seed synthetic :Document + :Chunk + 4 :Entity nodes.
  3. Run the triples extractor on the synthetic chunk text.
  4. Write triples via `client.write_triples`.
  5. Print breakdown by predicate + show the in-graph state.
  6. Pause — you inspect in Neo4j Desktop.
  7. Press Enter to clean up, Ctrl+C to keep the nodes.

Run from the project root:
    python scripts/smoke_kg_l8_triples.py

Automated counterpart (for regression catching, no inspection):
  tests/integration/kg/test_triples_writes.py  (KG-write contract
                                                with synthetic
                                                ExtractedTriple
                                                objects)
The LLM extractor itself is unit-tested in
  tests/unit/ingestion/test_triples_extractor.py
Run via `pytest -m integration tests/integration/kg/`.
"""

import argparse
import asyncio

# Switch to the smoke-test Neo4j instance BEFORE any other RLA
# import. Per [[test-instance-setup]] the test instance has a
# different password — wrong-instance state fails auth rather than
# corrupting real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.ingestion import triples_extractor  # noqa: E402
from knowledge_agent.ingestion.ids import make_chunk_id  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    CHUNK_LABEL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
    TRIPLE_PREDICATE_RELS,
)

# Synthetic doc — recognisable for cleanup queries.
SYNTHETIC_DOC_ID = "smoke-doc-l8-001"
SYNTHETIC_CHUNK_INDEX = 0

# Text chosen so the LLM has clear subject-predicate-object signals
# across multiple predicates. Each sentence sets up a distinct
# predicate from the 15 in TRIPLE_PREDICATE_RELS so the smoke
# exercises the vocabulary check + multi-predicate write path.
SYNTHETIC_CHUNK_TEXT = (
    "Metformin inhibits AMPK signalling, which in turn activates the "
    "GLUT4 transporter. The drug is associated with reduced incidence "
    "of type 2 diabetes mellitus. AMPK is part of the broader cellular "
    "energy regulation pathway. Recent studies show metformin causes "
    "a measurable reduction in HbA1c levels."
)

# Pre-seeded entity vocabulary for this chunk. The triples extractor
# is shown these as the allowed subject/object identifiers; predicates
# come from the constrained vocabulary in schema.TRIPLE_PREDICATE_RELS.
# Format: (lowercased_key, entity_type) — matches how L6 stores the
# :Entity composite key.
SYNTHETIC_ENTITIES: list[tuple[str, str]] = [
    ("metformin", "drug"),
    ("ampk", "protein"),
    ("glut4", "protein"),
    ("type 2 diabetes mellitus", "disease"),
    ("hba1c", "biomarker"),
]


async def _delete_smoke_nodes(client) -> None:
    """Drop synthetic doc + chunk + entities + L8 edges. Idempotent."""
    chunk_id = make_chunk_id(SYNTHETIC_DOC_ID, SYNTHETIC_CHUNK_INDEX)
    async with client.driver.session() as session:
        # Drop L8 edges sourced by this doc. The pipe-union covers all
        # 15 predicate types.
        rel_union = "|".join(TRIPLE_PREDICATE_RELS)
        await session.run(
            f"MATCH ()-[r:{rel_union}]->() WHERE r.doc_id = $doc_id DELETE r",
            doc_id=SYNTHETIC_DOC_ID,
        )
        # Drop :MENTIONS from this chunk, then orphan-GC :Entity.
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}})"
            f"-[m:MENTIONS]->(e:{ENTITY_LABEL}) DELETE m",
            chunk_id=chunk_id,
        )
        await session.run(f"MATCH (e:{ENTITY_LABEL}) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")
        # Drop chunk + doc.
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}}) DETACH DELETE c",
            doc_id=SYNTHETIC_DOC_ID,
        )
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DETACH DELETE d",
            doc_id=SYNTHETIC_DOC_ID,
        )


async def _seed_l5_and_l6(client) -> str:
    """Write synthetic :Document + :Chunk + :Entity nodes (pre-L8
    state). Returns the chunk_id."""
    chunk_id = make_chunk_id(SYNTHETIC_DOC_ID, SYNTHETIC_CHUNK_INDEX)
    async with client.driver.session() as session:
        # Focal + chunk.
        await session.run(
            f"MERGE (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"ON CREATE SET d.in_corpus = true, d:Paper "
            f"MERGE (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}}) "
            f"SET c.doc_id = $doc_id, c.chunk_index = $idx, "
            f"c.section = 'Results', c.page = 3, c.content_type = 'text' "
            f"MERGE (c)-[:PART_OF]->(d)",
            doc_id=SYNTHETIC_DOC_ID,
            chunk_id=chunk_id,
            idx=SYNTHETIC_CHUNK_INDEX,
        )
        # Entities + :MENTIONS edges (one per entity, start_char=0 stub).
        for key, etype in SYNTHETIC_ENTITIES:
            await session.run(
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


async def _show_state(client) -> None:
    """Read back the L8 edges sourced by this doc + summarise by predicate."""
    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (s:{ENTITY_LABEL})-[r:{rel_union}]->(o:{ENTITY_LABEL}) "
            f"WHERE r.doc_id = $doc_id "
            f"RETURN type(r) AS predicate, s.key AS subject, "
            f"o.key AS object ORDER BY predicate, subject",
            doc_id=SYNTHETIC_DOC_ID,
        )
        rows = await result.data()
    print(f"  L8 edges sourced by this doc: {len(rows)}")
    by_predicate: dict[str, int] = {}
    for r in rows:
        by_predicate[r["predicate"]] = by_predicate.get(r["predicate"], 0) + 1
    if by_predicate:
        print("  breakdown by predicate:")
        for p, n in sorted(by_predicate.items(), key=lambda kv: -kv[1]):
            print(f"    {p:>20}: {n}")
    for r in rows[:15]:
        print(f"    {r['subject']!r:>30} -[:{r['predicate']}]-> {r['object']!r}")
    if len(rows) > 15:
        print(f"    ... and {len(rows) - 15} more")


async def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    client = get_kg_client()

    print("Applying constraints...")
    await client.ensure_constraints()

    print("Clearing any leftover smoke nodes from previous runs...")
    await _delete_smoke_nodes(client)

    print(f"Seeding L5 + L6 state ({SYNTHETIC_DOC_ID})...")
    chunk_id = await _seed_l5_and_l6(client)
    print(f"  doc_id  : {SYNTHETIC_DOC_ID}")
    print(f"  chunk_id: {chunk_id}")
    print(f"  entities: {len(SYNTHETIC_ENTITIES)}")

    print("Extracting triples via triples_extractor (Haiku call)...")
    triples = triples_extractor.extract(SYNTHETIC_CHUNK_TEXT, SYNTHETIC_ENTITIES)
    print(f"  extractor returned {len(triples)} triples")
    for t in triples[:10]:
        print(f"    {t.subject_key!r:>30} -[:{t.predicate}]-> {t.object_key!r}")
    if len(triples) > 10:
        print(f"    ... and {len(triples) - 10} more")

    print("Writing triples via client.write_triples ...")
    await client.write_triples(SYNTHETIC_DOC_ID, [(chunk_id, triples)])
    print("  write_triples -> ok")

    print()
    print("Post-write KG state:")
    await _show_state(client)

    print()
    print("In Neo4j Desktop -> Query, try:")
    rel_union = "|".join(TRIPLE_PREDICATE_RELS)
    print(
        f"  MATCH (s:Entity)-[r:{rel_union}]->(o:Entity) "
        f"WHERE r.doc_id = '{SYNTHETIC_DOC_ID}' RETURN s, r, o"
    )
    print(
        f"  MATCH ()-[r]->() WHERE r.doc_id = '{SYNTHETIC_DOC_ID}' "
        f"RETURN type(r) AS predicate, count(r) AS n ORDER BY n DESC"
    )
    print()

    try:
        input("Press Enter to delete the smoke nodes, Ctrl+C to keep them. ")
    except KeyboardInterrupt:
        print()
        print("Keeping smoke nodes. Re-run this script to clean them up later.")
        await client.close()
        return

    print("Deleting smoke nodes...")
    await _delete_smoke_nodes(client)
    print("Done.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
