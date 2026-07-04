"""Smoke test for the agent end-to-end across all 6 retrieval modes.

Picks the first .pdf from `test_documents/`, ingests it (so the agent has
something to retrieve from), then runs the agent in the chosen retrieval
mode and prints the structured answer + the state fields populated by
each node so you can see which legs of the graph fired.

Hits real services:
  - docling (local models), OpenAlex (HTTP), Voyage (HTTP) - via ingest
  - LanceDB (local files) - hybrid retrieval at query time
  - Neo4j (local Bolt) - structural metadata + KG retrieval (modes 2-6)
  - Anthropic Claude (HTTP) - up to four LLM nodes:
      mode_classifier (Haiku, auto only)
      query_builder   (Haiku, lancedb side)
      cypher_builder  (Sonnet, neo4j side)
      synthesizer     (Sonnet, always)

Requires:
  - Neo4j running with NEO4J_PASSWORD set in the shared .env
  - Internet access for OpenAlex + Voyage + Anthropic
  - VOYAGE_API_KEY + ANTHROPIC_API_KEY set in .env

Run from the project root:
    python scripts/smoke_agent.py                            # mode 1 (default)
    python scripts/smoke_agent.py --mode neo4j_only
    python scripts/smoke_agent.py --mode auto
    python scripts/smoke_agent.py --mode parallel_fused \\
        --question "Comprehensive overview of nutrition and chronic disease"

Modes:
  lancedb_only        Pure semantic search over chunks.
  neo4j_only          Pure Cypher over the KG.
  lancedb_then_neo4j  Lance first, KG enriches via the Lance doc_ids.
  neo4j_then_lancedb  KG first, Lance scoped to KG-returned doc_ids.
  parallel_fused      Both legs in parallel, synthesizer fuses.
  auto                Mode classifier (Haiku) picks one of the above.

Different modes are useful for different questions - if you switch modes,
you'll often want to switch the question too (see --question).

Lifecycle (mirrors `smoke_pipeline.py`):
  1. Compute this PDF's doc_id; clean any leftover data for that doc_id.
  2. Ingest the PDF (LanceDB + Neo4j writes).
  3. Run the agent in the chosen mode.
  4. Pause - you inspect.
  5. Press Enter to clean up, Ctrl+C to keep the data.

Automated counterpart (for regression catching, no inspection):
  tests/integration/test_nodes.py  (lancedb_only mode end-to-end;
                                    AgentAnswer structured-output
                                    contract; skip_query_builder +
                                    direct_retrieval state overrides).
                                    The other 5 retrieval modes
                                    (neo4j_only, lancedb_then_neo4j,
                                    neo4j_then_lancedb, parallel_fused,
                                    auto) stay smoke-only to bound LLM
                                    cost at the integration tier.
Run via `pytest -m integration tests/integration/`.
"""

import argparse
import asyncio
from pathlib import Path

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# RLA import that might trigger `get_settings()`. The test instance's
# password mismatches the real instance, so a wrong-instance-running
# state fails at authentication rather than corrupting real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.graph import graph  # noqa: E402
from knowledge_agent.ingestion.ids import compute_doc_id  # noqa: E402
from knowledge_agent.ingestion.pipeline import ingest_document  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.corpus_config import CorpusConfig, LayerFlags  # noqa: E402
from knowledge_agent.kg.schema import (  # noqa: E402
    AUTHOR_LABEL,
    AUTHORED_REL,
    CITES_REL,
    DOCUMENT_LABEL,
)
from knowledge_agent.search.client import get_search_client  # noqa: E402

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"

# Question default. Picked so the lancedb_only default makes sense without
# any flags; pass `--question` for modes that need a different shape.
DEFAULT_QUESTION = "What role does nutrition play in chronic disease, according to the literature?"

VALID_MODES: tuple[str, ...] = (
    "lancedb_only",
    "neo4j_only",
    "lancedb_then_neo4j",
    "neo4j_then_lancedb",
    "parallel_fused",
    "auto",
)


async def _cleanup_doc(doc_id: str) -> None:
    """Remove everything this script wrote for `doc_id` from both stores."""
    search_client = get_search_client()
    await search_client.delete_chunks_by_doc_id(doc_id)

    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        # Authors that authored ONLY this doc.
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
        # Shadow citations referenced only by this doc.
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
    # L6: GC orphan :Entity nodes left behind if entity extraction ran.
    # Mirrors the pipeline's `delete_entities_by_doc_id` GC step so
    # repeated smokes don't accumulate entity nodes.
    await kg_client.delete_entities_by_doc_id(doc_id)


async def _run_agent(question: str, mode: str, config: CorpusConfig) -> dict:
    """Invoke the compiled graph and return the final state.

    `config` is seeded into the initial state so the cypher_builder node
    can scope the schema description it shows the LLM. Real callers
    (CLI / future GUI) load this from the corpus folder's `corpus.yaml`;
    here we pass the same in-memory config the ingest leg used so the
    schema the agent sees matches what's actually in the KG.
    """
    return await graph.ainvoke(
        {
            "query": question,
            "retrieval_mode": mode,
            "corpus_config": config,
        }
    )


def _print_routing_summary(final_state: dict) -> None:
    """Print which graph fields each node populated.

    Presence of `routed_mode`    -> mode_classifier fired (auto mode).
    Presence of `search_query`   -> query_builder fired.
    Presence of `cypher_query`   -> cypher_builder fired.
    Non-empty `retrieved_chunks` -> lancedb_retriever fired.
    Non-empty `kg_hits`          -> neo4j_retriever fired.
    """
    print("Routing summary:")
    routed = final_state.get("routed_mode")
    if routed is not None:
        print(f"  routed_mode (classifier picked) : {routed!r}")
    sq = final_state.get("search_query")
    if sq is not None:
        print(f"  search_query (query_builder)    : {sq!r}")
    cq = final_state.get("cypher_query")
    if cq is not None:
        print(f"  cypher_query (cypher_builder)   : {cq!r}")
    chunks = final_state.get("retrieved_chunks") or []
    print(f"  retrieved_chunks                : {len(chunks)}")
    kg_hits = final_state.get("kg_hits") or []
    print(f"  kg_hits                         : {len(kg_hits)}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agent in one of 6 retrieval modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="lancedb_only",
        help="Retrieval mode (default: %(default)s).",
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help=(
            "Question for the agent. The default suits lancedb_only; "
            "for graph-shaped modes (neo4j_only / cross-store) override "
            "with something like 'Who authored these papers?'"
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Drop the LanceDB chunks table entirely before ingesting. Use "
            "after a schema change so the table is recreated with current "
            "columns. Without this flag the smoke does per-doc cleanup only."
        ),
    )
    args = parser.parse_args()

    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return

    target = pdfs[0]
    doc_id = compute_doc_id(target)
    print(f"Source PDF: {target.name}")
    print(f"doc_id    : {doc_id[:12]}...")
    print(f"mode      : {args.mode}")
    print(f"question  : {args.question!r}")
    print()

    if args.reset:
        print("--reset: dropping LanceDB chunks table (schema-fresh start)...")
        await get_search_client().drop_chunks_table()

    print("Clearing any leftover smoke data for this doc_id...")
    await _cleanup_doc(doc_id)

    print("Ingesting the PDF (populates LanceDB + Neo4j)...")
    # In-memory corpus config - biomedical paper corpus with both
    # OpenAlex (L1-L4) and chunks (L5) enabled. A real corpus would
    # carry this in a `corpus.toml` alongside its data.
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    result = await ingest_document(target, config, "Document", "Paper")
    print(
        f"  n_chunks={result.n_chunks}, "
        f"metadata_status={result.metadata_status!r}, "
        f"lancedb_ok={result.lancedb_ok}, "
        f"kg_citations_ok={result.kg_citations_ok}, "
        f"kg_authorships_ok={result.kg_authorships_ok}"
    )
    print()

    print(f"Running the agent (mode={args.mode})...")
    print("(first run is slow - Anthropic LLM calls)")
    final_state = await _run_agent(args.question, args.mode, config)
    print()

    _print_routing_summary(final_state)
    print()

    answer = final_state.get("final_answer")
    if answer is None:
        print("(no AgentAnswer in final state)")
    else:
        print("Answer:")
        print(answer.answer or "(empty)")
        print()
        print(f"Chunk sources ({len(answer.chunk_sources)}):")
        for i, src in enumerate(answer.chunk_sources, start=1):
            quote = f" - {src.quote!r}" if src.quote else ""
            print(f"  [{i}] chunk_id={src.chunk_id} doc_id={src.doc_id[:12]}...{quote}")
        print(f"KG sources ({len(answer.kg_sources)}):")
        for i, src in enumerate(answer.kg_sources):
            quote = f" - {src.quote!r}" if src.quote else ""
            print(f"  [K{i}] hit_index={src.hit_index}{quote}")

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
