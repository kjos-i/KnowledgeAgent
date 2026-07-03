"""Smoke test for L6 entity extraction + KG writes (with manual-
inspection pause).

Exercises ONE entity-extractor adapter at a time against a synthetic
:Chunk pre-seeded in Neo4j, then writes the extracted mentions via
`write_entities` and surfaces the result in the KG so you can inspect
the :Entity nodes + :MENTIONS edges in Neo4j Desktop.

Adapters covered (select via --adapter):
  llm            Anthropic Haiku, one call per chunk. Requires
                 ANTHROPIC_API_KEY. Default.
  gliner         General-domain GLiNER (multi-v2.1). First run
                 downloads ~1.1 GB; subsequent runs reuse HF cache.
  gliner_biomed  Biomedical GLiNER. First run downloads ~1.1 GB.
  hunflair2      Biomedical Flair PrefixedSequenceTagger. First run
                 downloads ~1.24 GB.

Why one adapter per run: each NER backend has its own model loading
cost (seconds to minutes) and label vocabulary. Running them all in
one process would conflate failures and triple the wall time.

Requires Neo4j running and NEO4J_PASSWORD set in `.env.test` per
[[test-instance-setup]]. The script switches to the test instance
BEFORE any other RLA import that might trigger `get_settings()`.

Lifecycle (matches the clear-at-start + pause + optional-cleanup-at-
end convention — see [[feedback-smoke-test-cleanup]]):
  1. Clear any leftover smoke nodes (synthetic doc + its entities)
     from prior runs.
  2. Write a synthetic :Document + :Chunk so the L6 write has a
     target.
  3. Run the chosen adapter on the chunk text.
  4. Write the mentions via `client.write_entities`.
  5. Print extraction summary + show the in-graph state.
  6. Pause — you inspect in Neo4j Desktop.
  7. Press Enter to clean up, Ctrl+C to keep the nodes.

Run from the project root:
    python scripts/smoke_kg_l6_entities.py                       # llm (default)
    python scripts/smoke_kg_l6_entities.py --adapter gliner
    python scripts/smoke_kg_l6_entities.py --adapter gliner_biomed
    python scripts/smoke_kg_l6_entities.py --adapter hunflair2

Automated counterpart (for regression catching, no inspection):
  tests/integration/kg/test_entity_writes.py   (KG-write contract
                                                with synthetic
                                                Mention objects)
The extractors themselves are unit-tested in
  tests/unit/entity_extractors/
Run via `pytest -m integration tests/integration/kg/`.
"""

import argparse
import asyncio

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# RLA import that might trigger `get_settings()`. Per
# [[test-instance-setup]] the test instance has a different password
# so a wrong-instance state fails auth rather than corrupting real
# data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.entity_extractors import get_extractor  # noqa: E402
from knowledge_agent.ingestion.ids import make_chunk_id  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    CHUNK_LABEL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
)

# Synthetic doc + chunk — recognisable so cleanup is unambiguous.
SYNTHETIC_DOC_ID = "smoke-doc-l6-001"
SYNTHETIC_CHUNK_INDEX = 0
SYNTHETIC_CHUNK_TEXT = (
    "Aspirin reduces the risk of myocardial infarction in patients "
    "with type 2 diabetes mellitus, according to a 2021 meta-analysis "
    "by Smith et al. The effect was observed across multiple ethnic "
    "groups, including BRCA1 mutation carriers. EGFR-positive lung "
    "cancers also showed reduced inflammation markers."
)

# Default entity_types per adapter. None / empty for the LLM (open
# vocabulary); the four NER adapters have their own DEFAULT_LABELS
# baked in — passing an empty tuple here makes each adapter use them.
_DEFAULT_TYPES_PER_ADAPTER: dict[str, tuple[str, ...]] = {
    "llm": ("disease", "drug", "gene", "protein"),
    "gliner": (),
    "gliner_biomed": (),
    "hunflair2": (),
}


async def _delete_smoke_nodes(client) -> None:
    """Wipe synthetic doc + its chunks + its entity orphans. Idempotent."""
    chunk_id = make_chunk_id(SYNTHETIC_DOC_ID, SYNTHETIC_CHUNK_INDEX)
    async with client.driver.session() as session:
        # Drop :MENTIONS edges from this chunk's mentions, then
        # garbage-collect :Entity nodes that have no remaining
        # :MENTIONS edges (the canonical L6 orphan-GC pattern).
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}})"
            f"-[m:MENTIONS]->(e:{ENTITY_LABEL}) DELETE m",
            chunk_id=chunk_id,
        )
        await session.run(
            f"MATCH (e:{ENTITY_LABEL}) "
            f"WHERE NOT (e)<-[:MENTIONS]-() "
            f"DETACH DELETE e"
        )
        # Drop the chunk + the doc.
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}}) "
            f"DETACH DELETE c",
            doc_id=SYNTHETIC_DOC_ID,
        )
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"DETACH DELETE d",
            doc_id=SYNTHETIC_DOC_ID,
        )


async def _write_synthetic_doc_and_chunk(client) -> str:
    """Create the focal :Document + one :Chunk, return chunk_id."""
    chunk_id = make_chunk_id(SYNTHETIC_DOC_ID, SYNTHETIC_CHUNK_INDEX)
    async with client.driver.session() as session:
        await session.run(
            f"MERGE (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"ON CREATE SET d.in_corpus = true, d:Paper "
            f"MERGE (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}}) "
            f"SET c.doc_id = $doc_id, c.chunk_index = $idx, "
            f"c.section = 'Introduction', c.page = 1, "
            f"c.content_type = 'text' "
            f"MERGE (c)-[:PART_OF]->(d)",
            doc_id=SYNTHETIC_DOC_ID,
            chunk_id=chunk_id,
            idx=SYNTHETIC_CHUNK_INDEX,
        )
    return chunk_id


async def _show_state(client) -> None:
    """Read back the :Entity nodes + :MENTIONS edges for this chunk."""
    chunk_id = make_chunk_id(SYNTHETIC_DOC_ID, SYNTHETIC_CHUNK_INDEX)
    async with client.driver.session() as session:
        result = await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{chunk_id: $chunk_id}})"
            f"-[m:MENTIONS]->(e:{ENTITY_LABEL}) "
            f"RETURN e.key AS key, e.type AS type, m.offset AS offset "
            f"ORDER BY m.offset",
            chunk_id=chunk_id,
        )
        rows = await result.data()
    print(f"  :MENTIONS edges from this chunk: {len(rows)}")
    for r in rows[:20]:
        offset_str = f"@{r['offset']}" if r["offset"] is not None else "(no offset)"
        print(f"    {offset_str:>14} {r['type']:>10}  {r['key']!r}")
    if len(rows) > 20:
        print(f"    ... and {len(rows) - 20} more")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        choices=tuple(_DEFAULT_TYPES_PER_ADAPTER),
        default="llm",
        help=(
            "Entity-extractor adapter to exercise. Default 'llm' is "
            "cheapest + needs no model download. The three NER adapters "
            "trigger a ~1+ GB Hugging Face download on first run."
        ),
    )
    args = parser.parse_args()

    client = get_kg_client()

    print("Applying constraints...")
    await client.ensure_constraints()

    print("Clearing any leftover smoke nodes from previous runs...")
    await _delete_smoke_nodes(client)

    print(f"Writing synthetic :Document + :Chunk ({SYNTHETIC_DOC_ID})...")
    chunk_id = await _write_synthetic_doc_and_chunk(client)

    print(f"Loading extractor: {args.adapter} ...")
    extractor = get_extractor(args.adapter)
    entity_types = _DEFAULT_TYPES_PER_ADAPTER[args.adapter]
    print(f"  entity_types passed: {entity_types or '(adapter default)'}")

    print(f"Extracting mentions from synthetic chunk text...")
    mentions = extractor.extract(SYNTHETIC_CHUNK_TEXT, entity_types)
    print(f"  extractor returned {len(mentions)} mentions")
    for m in mentions[:10]:
        offset_str = f"@{m.offset}" if m.offset is not None else "(no offset)"
        print(
            f"    {offset_str:>14} {m.entity_type:>10}  {m.raw_text!r}"
        )
    if len(mentions) > 10:
        print(f"    ... and {len(mentions) - 10} more")

    print("Writing mentions to KG via write_entities ...")
    await client.write_entities(SYNTHETIC_DOC_ID, [(chunk_id, mentions)])
    print("  write_entities -> ok")

    print()
    print("Post-write KG state:")
    await _show_state(client)

    print()
    print("In Neo4j Desktop -> Query, try:")
    print(
        f"  MATCH (c:Chunk {{chunk_id: '{chunk_id}'}})-[:MENTIONS]->(e:Entity) "
        f"RETURN e.key, e.type ORDER BY e.type"
    )
    print(
        f"  MATCH (d:Document {{doc_id: '{SYNTHETIC_DOC_ID}'}}) "
        f"OPTIONAL MATCH (d)<-[:PART_OF]-(c:Chunk)-[:MENTIONS]->(e:Entity) "
        f"RETURN d.doc_id, count(DISTINCT e) AS distinct_entities"
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
