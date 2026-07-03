"""Smoke test for L1-L5 KG writes (with manual-inspection pause).

Constructs a synthetic OpenAlex-shaped work dict + a synthetic chunks
list, applies constraints, then writes:
  L1: focal :Document:Paper + shadow cited :Document:Paper + :CITES edges
  L2: :Author nodes + :AUTHORED edges (with position + is_corresponding)
  L3: :Venue node + :PUBLISHED_IN edge
  L4: :Topic nodes + :ABOUT_TOPIC edges (with score)
  L5: :Chunk nodes + :PART_OF edges (with chunk_index, section, page,
      content_type - text + embedding stay in LanceDB)

Document nodes carry two labels: :Document (universal) and :Paper (subtype).
Queries can target either - :Document matches any ingested doc, :Paper
filters to scholarly papers specifically.

Requires Neo4j running and NEO4J_PASSWORD set in the shared `.env` file.

Lifecycle:
  1. Idempotent: clear any leftover smoke nodes from previous runs.
  2. Write the synthetic KG.
  3. Pause - you inspect in Neo4j Desktop.
  4. Press Enter to clean up, Ctrl+C to keep the nodes for further poking.

Run from the project root:
    python scripts/smoke_kg_l1_l5.py
    python scripts/smoke_kg_l1_l5.py --drop-constraints   # nuke schema too

Automated counterparts (for regression catching, no inspection):
  tests/integration/kg/test_openalex_writes.py   (L1-L4 — citations,
                                                  authors, venue, topics)
  tests/integration/kg/test_chunk_writes.py      (L5 — chunks + PART_OF)
Run via `pytest -m integration tests/integration/kg/`.
"""

import argparse
import asyncio
from dataclasses import dataclass

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# RLA import that might trigger `get_settings()`. The test instance's
# password mismatches the real instance, so a wrong-instance-running
# state fails at authentication rather than corrupting real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    AUTHOR_LABEL,
    CHUNK_LABEL,
    CONSTRAINT_STATEMENTS,
    DOCUMENT_LABEL,
    TOPIC_LABEL,
    VENUE_LABEL,
)

# Synthetic locally-generated doc_id for the focal document. In the real
# pipeline this is `compute_doc_id(path)` (SHA-256 of file bytes); here we
# use a recognisable string for easy cleanup queries.
SYNTHETIC_DOC_ID = "smoke-doc-001"

# Synthetic OpenAlex-shaped work covering L1-L4. IDs start with W999 /
# A999 / S999 / T999 so they can't collide with any real OpenAlex entity.
SYNTHETIC_WORK = {
    "id": "https://openalex.org/W9990000001",
    "doi": "https://doi.org/10.9990/smoke-test-1",
    "referenced_works": [
        "https://openalex.org/W9990000100",
        "https://openalex.org/W9990000101",
        "https://openalex.org/W9990000102",
    ],
    "authorships": [
        {
            "author_position": "first",
            "author": {
                "id": "https://openalex.org/A9990000001",
                "display_name": "Jane Smoketest",
            },
            "is_corresponding": True,
        },
        {
            "author_position": "middle",
            "author": {
                "id": "https://openalex.org/A9990000002",
                "display_name": "Bob Synthetic",
            },
            "is_corresponding": False,
        },
        {
            "author_position": "last",
            "author": {
                "id": "https://openalex.org/A9990000003",
                "display_name": "Carol Stub",
            },
            "is_corresponding": False,
        },
    ],
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S9990000001",
            "display_name": "Journal of Synthetic Smoke Tests",
            "type": "journal",
            "issn_l": "9999-0001",
        },
    },
    "topics": [
        {
            "id": "https://openalex.org/T9990000001",
            "display_name": "Smoke-Test Topic Alpha",
            "score": 0.95,
        },
        {
            "id": "https://openalex.org/T9990000002",
            "display_name": "Smoke-Test Topic Beta",
            "score": 0.71,
        },
    ],
}


@dataclass
class _SyntheticChunk:
    """Duck-typed stand-in for ParsedChunk, used by the L5 smoke write."""

    chunk_index: int
    section: str | None
    page: int | None
    content_type: str = "text"


# Three synthetic chunks - enough to exercise the UNWIND batching + see
# multiple :PART_OF edges in the inspection step.
SYNTHETIC_CHUNKS = [
    _SyntheticChunk(0, "Introduction", 1, "text"),
    _SyntheticChunk(1, "Methods", 2, "text"),
    _SyntheticChunk(2, None, 3, "figure"),
]


async def _delete_smoke_nodes(client) -> None:
    """Remove every smoke-test node + its relationships. Idempotent."""
    async with client.driver.session() as session:
        # :Chunk nodes for this doc_id (must go before the focal so the
        # PART_OF edges don't dangle; DETACH DELETE handles it either way).
        await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}}) "
            f"DETACH DELETE c",
            doc_id=SYNTHETIC_DOC_ID,
        )
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL}) "
            f"WHERE d.doc_id = $doc_id "
            f"OR d.openalex_id STARTS WITH 'W999' "
            f"DETACH DELETE d",
            doc_id=SYNTHETIC_DOC_ID,
        )
        await session.run(
            f"MATCH (a:{AUTHOR_LABEL}) "
            f"WHERE a.openalex_id STARTS WITH 'A999' "
            f"DETACH DELETE a"
        )
        await session.run(
            f"MATCH (v:{VENUE_LABEL}) "
            f"WHERE v.openalex_id STARTS WITH 'S999' "
            f"DETACH DELETE v"
        )
        await session.run(
            f"MATCH (t:{TOPIC_LABEL}) "
            f"WHERE t.openalex_id STARTS WITH 'T999' "
            f"DETACH DELETE t"
        )


async def _drop_constraints(client) -> None:
    """Drop every constraint declared in schema.CONSTRAINT_STATEMENTS.

    Derives the constraint name from each CREATE statement (one word after
    `CONSTRAINT`) so this stays in sync with whatever schema.py declares.
    """
    async with client.driver.session() as session:
        for stmt in CONSTRAINT_STATEMENTS:
            # Format: "CREATE CONSTRAINT <name> IF NOT EXISTS FOR ..."
            name = stmt.split("CONSTRAINT", 1)[1].strip().split()[0]
            await session.run(f"DROP CONSTRAINT {name} IF EXISTS")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop-constraints",
        action="store_true",
        help=(
            "Also drop the KG schema constraints during cleanup. Default off "
            "(constraints survive runs since they're schema, not data)."
        ),
    )
    args = parser.parse_args()

    client = get_kg_client()

    # All kg.client write/maintenance methods raise on failure under
    # the typed-errors contract; success is silent. The smoke script
    # surfaces the steps via prints alone and lets any exception
    # propagate so a failing run aborts visibly.
    print("Applying constraints...")
    await client.ensure_constraints()
    print("  ensure_constraints -> ok")

    print("Clearing any leftover smoke nodes from previous runs...")
    await _delete_smoke_nodes(client)

    print(f"L1: writing focal document ({SYNTHETIC_DOC_ID}) + 3 citations...")
    await client.write_citations(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    print("  write_citations -> ok")

    print("L2: writing 3 authors + AUTHORED edges...")
    await client.write_authorships(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    print("  write_authorships -> ok")

    print("L3: writing venue + PUBLISHED_IN edge...")
    await client.write_venue(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    print("  write_venue -> ok")

    print("L4: writing 2 topics + ABOUT_TOPIC edges...")
    await client.write_topics(SYNTHETIC_DOC_ID, SYNTHETIC_WORK)
    print("  write_topics -> ok")

    print("L5: writing 3 chunks + PART_OF edges...")
    await client.write_chunks(
        SYNTHETIC_DOC_ID, SYNTHETIC_CHUNKS, "Document", "Paper"
    )
    print("  write_chunks -> ok")

    print()
    print("Smoke data written. In Neo4j Desktop -> Query, try:")
    print(
        "  MATCH (d:Document) WHERE d.openalex_id STARTS WITH 'W999' "
        f"OR d.doc_id = '{SYNTHETIC_DOC_ID}' RETURN d"
    )
    print("  MATCH (a:Author) WHERE a.openalex_id STARTS WITH 'A999' RETURN a")
    print("  MATCH (v:Venue) WHERE v.openalex_id STARTS WITH 'S999' RETURN v")
    print("  MATCH (t:Topic) WHERE t.openalex_id STARTS WITH 'T999' RETURN t")
    print(
        f"  MATCH (c:Chunk {{doc_id: '{SYNTHETIC_DOC_ID}'}}) RETURN c"
    )
    print(
        "Expected: 4 :Document + 3 :Author + 1 :Venue + 2 :Topic + 3 :Chunk, "
        "3 :CITES + 3 :AUTHORED + 1 :PUBLISHED_IN + 2 :ABOUT_TOPIC + 3 :PART_OF."
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
    if args.drop_constraints:
        print("Dropping schema constraints...")
        await _drop_constraints(client)
    print("Done.")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
