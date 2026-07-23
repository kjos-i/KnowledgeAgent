"""Smoke test for the multi-extractor priority-ordered UNION (with
manual-inspection pause).

Ingests the first `.pdf` in `test_documents/` with the L6 entities
layer on and a chosen ORDERED list of extractors, then shows the
resulting `:Entity` nodes + the `sources` provenance on each
`:MENTIONS` edge — i.e. which extractor(s) found each span, and (for
overlaps) that the base extractor owned the type while every finder
was recorded.

Purpose: verify the union end-to-end — that N adapters run per chunk,
merge fragmentation-free (one owner per span), and that the `sources`
provenance lands in Neo4j.

Choose the extractors + order via `--extractors` (comma-separated,
priority order; index 0 = base). Default is a single `llm` run (always
available, no model download) which still exercises the union plumbing
(sources = ["llm"]). A real 2-way union needs a NER installed:

    python scripts/smoke_multi_extractor.py
    python scripts/smoke_multi_extractor.py --extractors gliner_biomed,llm
    python scripts/smoke_multi_extractor.py --extractors hunflair2,llm

Hits real services: docling (local), the chosen extractors (LLM = HTTP;
NER = local model), Voyage (HTTP embed), LanceDB (local), Neo4j (local
Bolt). Requires the test instance up (loaded via `load_test_env`) +
VOYAGE_API_KEY / ANTHROPIC_API_KEY in `.env.test`. First NER run
downloads its ~1 GB weights.

Lifecycle (matches the other smokes — see [[smoke-test-cleanup]]):
  1. Clear any leftover data for this doc_id from both stores.
  2. Ingest with the chosen extractors.
  3. Print the entity + sources breakdown.
  4. Pause — inspect in Neo4j Desktop.
  5. Press Enter to clean up, Ctrl+C to keep the data.

Run from the project root.
"""

import argparse
import asyncio
from pathlib import Path

from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.corpus_config import (  # noqa: E402
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.entity_extractors.extractor_lifecycle import (  # noqa: E402
    is_extractor_ready,
)
from knowledge_agent.ingestion.ids import compute_doc_id  # noqa: E402
from knowledge_agent.ingestion.pipeline import ingest_document  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    AUTHOR_LABEL,
    AUTHORED_REL,
    CHUNK_LABEL,
    CITES_REL,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
    MENTIONS_REL,
)
from knowledge_agent.search.client import get_search_client  # noqa: E402

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


async def _cleanup_doc(doc_id: str) -> None:
    """Remove everything this script wrote for `doc_id` from both stores."""
    await get_search_client().delete_chunks_by_doc_id(doc_id)
    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        await session.run(
            f"MATCH (a:{AUTHOR_LABEL})-[:{AUTHORED_REL}]->"
            f"(:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"WHERE NOT EXISTS {{ "
            f"  MATCH (a)-[:{AUTHORED_REL}]->(other:{DOCUMENT_LABEL}) "
            f"  WHERE other.doc_id <> $doc_id }} "
            f"DETACH DELETE a",
            doc_id=doc_id,
        )
        await session.run(
            f"MATCH (:{DOCUMENT_LABEL} {{doc_id: $doc_id}})-[:{CITES_REL}]->"
            f"(c:{DOCUMENT_LABEL}) WHERE c.in_corpus = false "
            f"AND NOT EXISTS {{ "
            f"  MATCH (other:{DOCUMENT_LABEL})-[:{CITES_REL}]->(c) "
            f"  WHERE other.doc_id <> $doc_id }} "
            f"DETACH DELETE c",
            doc_id=doc_id,
        )
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DETACH DELETE d",
            doc_id=doc_id,
        )
    await kg_client.delete_entities_by_doc_id(doc_id)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extractors",
        default="llm",
        help=(
            "Comma-separated ORDERED extractor list (index 0 = base, "
            "owns overlaps). Default 'llm'. Example: 'hunflair2,llm'."
        ),
    )
    args = parser.parse_args()
    extractor_names = [e.strip() for e in args.extractors.split(",") if e.strip()]
    if not extractor_names:
        print("No extractors given.")
        return

    # Fail early + clearly if any chosen extractor isn't installed.
    not_ready = [n for n in extractor_names if not is_extractor_ready(n)]
    if not_ready:
        print(
            f"These extractors aren't installed/downloaded: {not_ready}. "
            f"Open Library → Installs (or use the extractor_lifecycle "
            f"install/download ops) first, or pass --extractors llm."
        )
        return

    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return
    target = pdfs[0]
    doc_id = compute_doc_id(target)
    print(f"Ingesting (multi-extractor union): {target.name}")
    print(f"doc_id     : {doc_id[:12]}...")
    print(f"extractors : {extractor_names}  (index 0 = base / owns overlaps)")
    print()

    print("Clearing any leftover smoke data for this doc_id...")
    await _cleanup_doc(doc_id)

    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractors=extractor_names),
    )
    result = await ingest_document(target, config, "Document", "Paper")

    print()
    print("Result:")
    print(f"  n_chunks          : {result.n_chunks}")
    print(f"  kg_entities_ok    : {result.kg_entities_ok}")
    print(f"  n_entity_mentions : {result.n_entity_mentions}")
    print()

    # ---- Provenance breakdown: which extractor(s) found what ----
    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        result_cursor = await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}})"
            f"-[m:{MENTIONS_REL}]->(e:{ENTITY_LABEL}) "
            f"RETURN e.key AS key, e.entity_type AS type, "
            f"       m.sources AS sources "
            f"ORDER BY size(coalesce(m.sources, [])) DESC, e.key "
            f"LIMIT 25",
            doc_id=doc_id,
        )
        rows = await result_cursor.data()

        # Count how many spans each distinct source-set produced.
        combo_cursor = await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}})"
            f"-[m:{MENTIONS_REL}]->(:{ENTITY_LABEL}) "
            f"RETURN m.sources AS sources, count(*) AS n "
            f"ORDER BY n DESC",
            doc_id=doc_id,
        )
        combos = await combo_cursor.data()

    print("Mentions by source-set (which extractor(s) found the span):")
    if not combos:
        print("  (no :MENTIONS edges)")
    for row in combos:
        print(f"  sources={row['sources']!r:30} count={row['n']}")
    print()

    print("First 25 mentions with provenance:")
    for r in rows:
        print(f"  {r['type']:>12}  {r['key'][:34]:<34}  sources={r['sources']}")
    print()

    print("Inspect in Neo4j Desktop → Query:")
    print(
        f"  MATCH (c:Chunk {{doc_id: '{doc_id}'}})-[m:MENTIONS]->(e:Entity) "
        f"RETURN e.key, e.entity_type, m.sources ORDER BY size(m.sources) DESC"
    )
    print()

    try:
        input("Press Enter to delete the smoke data, Ctrl+C to keep it. ")
    except KeyboardInterrupt:
        print()
        print("Keeping smoke data. Re-run this script to clean it up later.")
        return

    print("Deleting smoke data...")
    await _cleanup_doc(doc_id)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
