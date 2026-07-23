"""Smoke test for the full ingestion pipeline (with manual-inspection pause).

Picks the first .pdf file (alphabetically) from `test_documents/` and runs
`ingest_document(path, config)` end-to-end: hash, parse, resolve metadata,
embed, write to LanceDB, write to Neo4j (L1-L5 + L6 + L7 MeSH linking).
Prints the `IngestResult` summary, the top-10 most-mentioned entities,
and the top-10 entity -> MeSH canonical links.

Uses an in-memory `CorpusConfig` (biomedical, every layer on) so the smoke
exercises the whole stack - mirrors what a real biomedical paper corpus
would have in its `corpus.toml`.

Hits real services: docling (local models), OpenAlex (HTTP), Voyage (HTTP),
Anthropic Haiku (HTTP, L6 extraction), NLM MeSH download (~600 MB, first
run only - cached under the OS cache dir),
LanceDB (local files), Neo4j (local Bolt). Requires:
  - Neo4j running with NEO4J_PASSWORD set in .env
  - Internet access for OpenAlex + Voyage + Anthropic + NLM
  - VOYAGE_API_KEY and ANTHROPIC_API_KEY set in .env

Cost: ~$0.05 per 200-chunk paper at Haiku pricing (one extraction call
per chunk). First run also pays the MeSH download (~600 MB, ~minutes),
parse (~tens of seconds), and import (~30K nodes) once - subsequent runs
reuse the cache + the imported :MeSHTerm nodes in Neo4j.

Run from the project root:
    python scripts/smoke_pipeline.py            # normal per-doc cleanup
    python scripts/smoke_pipeline.py --reset    # drop LanceDB table +
                                                # MeSH nodes first (use
                                                # after schema or ontology
                                                # release changes)

Lifecycle (mirrors `smoke_kg_l1_l5.py`):
  1. Compute this PDF's doc_id; clean any leftover data for that doc_id
     from previous runs. With `--reset`: also drop the LanceDB chunks
     table AND every :MeSHTerm node so the next run re-imports fresh.
  2. Run the pipeline. First L7 run triggers MeSH import + a global
     linking pass; subsequent runs link only this doc's new entities.
  3. Pause - you inspect in Neo4j Desktop / LanceDB. The top-10
     entities + top-10 canonical-link tables are printed for quick
     eyeballing.
  4. Press Enter to clean up, Ctrl+C to keep the data.

Automated counterpart (for regression catching, no inspection):
  tests/integration/ingestion/test_pipeline.py  (golden L1-L5 ingest;
                                                 both-store write
                                                 parity; idempotency
                                                 on doc_id; delete_doc
                                                 wipes both stores).
                                                 Higher layers (L6 /
                                                 L7 / L8 / L9 / L10)
                                                 covered per-layer in
                                                 tests/integration/kg/
                                                 to keep test_pipeline
                                                 cost bounded.
Run via `pytest -m integration tests/integration/ingestion/`.
"""

import argparse
import asyncio
from pathlib import Path

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# knowledge_agent import that might trigger `get_settings()`. The test instance's
# password mismatches the real instance, so a wrong-instance-running
# state fails at authentication rather than corrupting real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.corpus_config import (  # noqa: E402
    CorpusConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
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
    """Remove everything this script wrote for `doc_id` from both stores.

    LanceDB: drop chunk rows.
    Neo4j: drop authors solely connected to this doc, shadow papers cited
    only by this doc, then the focal Document.
    """
    # LanceDB chunks.
    search_client = get_search_client()
    await search_client.delete_chunks_by_doc_id(doc_id)

    # Neo4j cleanup (password is now required at Settings load, so a
    # client always has credentials).
    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        # Authors that authored ONLY this doc (not shared with other docs).
        await session.run(
            f"MATCH (a:{AUTHOR_LABEL})-[:{AUTHORED_REL}]->(:{DOCUMENT_LABEL} "
            f"{{doc_id: $doc_id}}) "
            f"WHERE NOT EXISTS {{ "
            f"  MATCH (a)-[:{AUTHORED_REL}]->(other:{DOCUMENT_LABEL}) "
            f"  WHERE other.doc_id <> $doc_id "
            f"}} "
            f"DETACH DELETE a",
            doc_id=doc_id,
        )
        # Shadow citations only referenced by this doc.
        await session.run(
            f"MATCH (:{DOCUMENT_LABEL} {{doc_id: $doc_id}})-[:{CITES_REL}]->"
            f"(c:{DOCUMENT_LABEL}) "
            f"WHERE c.in_corpus = false "
            f"AND NOT EXISTS {{ "
            f"  MATCH (other:{DOCUMENT_LABEL})-[:{CITES_REL}]->(c) "
            f"  WHERE other.doc_id <> $doc_id "
            f"}} "
            f"DETACH DELETE c",
            doc_id=doc_id,
        )
        # The focal document itself.
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DETACH DELETE d",
            doc_id=doc_id,
        )
    # L6: GC orphan :Entity nodes whose last incoming :MENTIONS edge
    # came from this doc's chunks. Mirrors the pipeline's
    # `delete_entities_by_doc_id` GC step so re-runs leave a clean DB.
    await kg_client.delete_entities_by_doc_id(doc_id)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1-L5 + L6 end-to-end smoke test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Drop the LanceDB chunks table entirely before ingesting. Use "
            "after a schema change so the table is recreated with current "
            "columns (e.g. when 'main_label' / 'sub_label' don't yet exist "
            "in an older table). Without this flag the smoke does per-doc "
            "cleanup only - safe for normal iteration."
        ),
    )
    args = parser.parse_args()

    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return

    target = pdfs[0]
    print(f"Ingesting: {target.name}")
    doc_id = compute_doc_id(target)
    print(f"doc_id    : {doc_id[:12]}...")
    print()

    # `--reset` drops the entire LanceDB chunks table AND wipes the L7
    # ontology data - explicit opt-in for schema-migration or ontology-
    # release scenarios. Has to run BEFORE per-doc cleanup since
    # delete_chunks_by_doc_id needs an existing (compatible) LanceDB
    # table.
    if args.reset:
        print("--reset: dropping LanceDB chunks table (schema-fresh start)...")
        await get_search_client().drop_chunks_table()
        print("--reset: dropping :MeSHTerm nodes (ontology will re-import)...")
        await get_kg_client().delete_mesh()

    # Idempotent re-run: clear any leftover data for this doc_id first.
    print("Clearing any leftover smoke data for this doc_id...")
    await _cleanup_doc(doc_id)

    # In-memory corpus config - biomedical paper corpus with every layer
    # on: OpenAlex (L1-L4) + chunks (L5) + entity extraction (L6) +
    # MeSH ontology linking (L7). The first run with ontology_mesh=true
    # will download MeSH (~600 MB, gzipped N-Triples) and import ~30K
    # :MeSHTerm nodes; subsequent runs reuse the cached file + the
    # existing :MeSHTerm nodes in Neo4j. A real corpus would carry
    # this configuration in a `corpus.toml` alongside its data.
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=True,
            ontology_mesh=True,
        ),
        entities=EntityConfig(
            extractor="llm",
            entity_types=["GENE", "DISEASE", "CHEMICAL"],
        ),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )
    print(f"Config    : allowed_types={config.allowed_types}, layers={config.layers.model_dump()}")
    print()

    result = await ingest_document(target, config, "Document", "Paper")

    print()
    print("Result:")
    print(f"  doc_id            : {result.doc_id[:12]}...")
    print(f"  n_chunks          : {result.n_chunks}")
    print(f"  metadata_status   : {result.metadata_status!r}")
    print(f"  embed_ok          : {result.embed_ok}")
    print(f"  lancedb_ok        : {result.lancedb_ok}")
    print(f"  kg_citations_ok   : {result.kg_citations_ok}")
    print(f"  kg_authorships_ok : {result.kg_authorships_ok}")
    print(f"  kg_chunks_ok      : {result.kg_chunks_ok}")
    print(f"  kg_entities_ok    : {result.kg_entities_ok}")
    print(f"  n_entity_mentions : {result.n_entity_mentions}")
    if result.kg_ontology_results:
        print("  kg_ontology_results:")
        for name, info in result.kg_ontology_results.items():
            print(
                f"    {name:8s} : imported={info['imported']}, "
                f"import_ok={info['import_ok']}, "
                f"n_links={info['n_links']}"
            )

    if result.work is not None:
        title = result.work.get("title") or result.work.get("display_name")
        year = result.work.get("publication_year")
        n_refs = len(result.work.get("referenced_works") or [])
        n_authors = len(result.work.get("authorships") or [])
        print()
        print("Resolved metadata:")
        print(f"  title             : {title!r}")
        print(f"  year              : {year}")
        print(f"  n authors         : {n_authors}")
        print(f"  n referenced_works: {n_refs}")
    else:
        print()
        print("No metadata resolved.")

    # Top-10 entities by mention count for this doc - eyeball check that
    # L6 found sensible biomedical entities (genes / diseases / chemicals)
    # rather than garbage / nothing.
    if result.kg_entities_ok and result.n_entity_mentions > 0:
        kg_client = get_kg_client()
        async with kg_client.driver.session() as session:
            result_cursor = await session.run(
                f"MATCH (e:{ENTITY_LABEL})"
                f"<-[:{MENTIONS_REL}]-(c:{CHUNK_LABEL} "
                f"{{doc_id: $doc_id}}) "
                f"RETURN e.key AS key, e.entity_type AS entity_type, "
                f"count(*) AS mentions "
                f"ORDER BY mentions DESC LIMIT 10",
                doc_id=result.doc_id,
            )
            records = await result_cursor.data()
        print()
        print("Top 10 entities by mention count:")
        print(f"  {'entity_type':<10s}  {'mentions':>8s}  key")
        for row in records:
            print(f"  {row['entity_type']:<10s}  {row['mentions']:>8d}  {row['key']!r}")

    # L7 canonical-links eyeball check: show a few examples of
    # (:Entity)-[:CANONICAL_TO]->(:MeSHTerm) for this doc so we can see
    # which entities the linking pass managed to resolve.
    if result.kg_ontology_results.get("mesh", {}).get("n_links", 0) > 0:
        kg_client = get_kg_client()
        async with kg_client.driver.session() as session:
            result_cursor = await session.run(
                f"MATCH (e:{ENTITY_LABEL})"
                f"<-[:{MENTIONS_REL}]-(:{CHUNK_LABEL} "
                f"{{doc_id: $doc_id}}) "
                f"MATCH (e)-[r:CANONICAL_TO]->(t:MeSHTerm) "
                f"RETURN DISTINCT e.key AS key, t.id AS mesh_id, "
                f"       t.label AS mesh_label, r.strategy AS strategy "
                f"ORDER BY e.key LIMIT 10",
                doc_id=result.doc_id,
            )
            records = await result_cursor.data()
        print()
        print("Top 10 entity -> MeSH canonical links:")
        print(f"  {'strategy':<8s}  {'mesh_id':<14s}  {'entity key':<24s}  mesh_label")
        for row in records:
            print(
                f"  {row['strategy']:<8s}  {row['mesh_id']:<14s}  "
                f"{row['key'][:24]:<24s}  {row['mesh_label']!r}"
            )

    print()
    print("Inspect in your tools:")
    print(f"  LanceDB: chunks where doc_id = {result.doc_id!r}")
    print(f"  Neo4j:   MATCH (d:Document {{doc_id: '{result.doc_id}'}}) RETURN d")
    print(
        f"  Neo4j:   MATCH (c:Chunk {{doc_id: '{result.doc_id}'}})"
        f"-[:MENTIONS]->(e:Entity) RETURN c.chunk_index, e.key, e.entity_type"
    )
    print(
        "  Neo4j:   MATCH (e:Entity)-[:CANONICAL_TO]->(t:MeSHTerm) "
        "RETURN e.key, t.id, t.label LIMIT 20"
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
