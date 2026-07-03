"""Smoke test for multimodal ingestion (with manual-inspection pause).

Picks the first `.pdf` file (alphabetically) from `test_documents/` and
runs `ingest_document(path, config)` with `extract_figures=true` +
`embed_images=true`. Then prints:

  1. IngestResult summary (chunks + embed + LanceDB + KG status).
  2. Chunk breakdown by content_type ("text" vs "figure").
  3. Contents of `<corpus folder>/figures/<doc_id>/` (the saved PNGs).
  4. First 5 figure chunk rows from LanceDB with content_type + image_ref.
  5. Same in Neo4j: :Chunk nodes with content_type='figure'.

Purpose: close the "end-to-end verification" gap for multimodal
ingestion. Everything upstream (parser, chunker, embedder, storage)
is unit-tested; this script actually runs the pipeline against a real
PDF, real Docling, real Voyage `multimodal_embed`, real Neo4j / LanceDB
and lets the developer confirm figures were saved + embedded + written.

Hits real services: docling (local models), Voyage (HTTP — multimodal
embed API), LanceDB (local files), Neo4j (local Bolt).

Requires:
  - Neo4j running with NEO4J_PASSWORD set in the test-instance `.env`
    (loaded via `load_test_env`)
  - VOYAGE_API_KEY set in the test env
  - Internet access for Voyage

Lifecycle (matches other smoke scripts):
  1. Idempotent: clean any leftover data for this PDF's doc_id from
     both stores AND wipe the figures directory (`<corpus folder>/
     figures/<doc_id>/`) so the next parse re-saves fresh.
  2. Run the pipeline with multimodal flags on.
  3. Print the breakdown so the developer can eyeball figure chunks +
     saved PNGs.
  4. Pause — you inspect the PNGs in a viewer / open the GUI to see
     the Latest view figure gallery / poke around LanceDB + Neo4j.
  5. Press Enter to clean up, Ctrl+C to keep the data for further poking.

Run from the project root:
    python scripts/smoke_multimodal.py
"""

import asyncio
from pathlib import Path
import shutil

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# import that might trigger `get_settings()`. Mirrors other smokes.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.ingestion.ids import compute_doc_id  # noqa: E402
from knowledge_agent.ingestion.pipeline import ingest_document  # noqa: E402
from knowledge_agent.kg.client import get_kg_client  # noqa: E402
from knowledge_agent.kg.corpus_config import (  # noqa: E402
    CorpusConfig,
    LayerFlags,
)
from knowledge_agent.kg.schema import (  # noqa: E402
    AUTHOR_LABEL,
    AUTHORED_REL,
    CHUNK_LABEL,
    CITES_REL,
    DOCUMENT_LABEL,
)
from knowledge_agent.config import get_settings  # noqa: E402
from knowledge_agent.search.client import get_search_client  # noqa: E402

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


def _figures_dir_for(doc_id: str) -> Path:
    """Figures live INSIDE the LanceDB dir (option 1 locked 2026-07-03)
    so test data stays self-contained in one folder."""
    return get_settings().lancedb_path / "figures" / doc_id


async def _cleanup_doc(doc_id: str) -> None:
    """Remove everything this script wrote for `doc_id` from both stores
    AND the figures directory. Same delete pattern as `smoke_pipeline.py`."""
    search_client = get_search_client()
    await search_client.delete_chunks_by_doc_id(doc_id)

    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        await session.run(
            f"MATCH (a:{AUTHOR_LABEL})-[:{AUTHORED_REL}]->"
            f"(:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) "
            f"WHERE NOT EXISTS {{ "
            f"  MATCH (a)-[:{AUTHORED_REL}]->(other:{DOCUMENT_LABEL}) "
            f"  WHERE other.doc_id <> $doc_id "
            f"}} "
            f"DETACH DELETE a",
            doc_id=doc_id,
        )
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
        await session.run(
            f"MATCH (d:{DOCUMENT_LABEL} {{doc_id: $doc_id}}) DETACH DELETE d",
            doc_id=doc_id,
        )
    await kg_client.delete_entities_by_doc_id(doc_id)

    # Figures directory: wipe the whole per-doc subtree so re-runs
    # start fresh (parse re-saves i.png from scratch).
    fig_dir = _figures_dir_for(doc_id)
    if fig_dir.exists():
        shutil.rmtree(fig_dir)


async def main() -> None:
    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {TEST_DOCS}")
        return

    target = pdfs[0]
    print(f"Ingesting (multimodal): {target.name}")
    doc_id = compute_doc_id(target)
    print(f"doc_id       : {doc_id[:12]}...")
    print(f"lancedb dir  : {get_settings().lancedb_path}")
    print(f"figures dir  : {_figures_dir_for(doc_id)}")
    print()

    print("Clearing any leftover smoke data for this doc_id...")
    await _cleanup_doc(doc_id)

    # Multimodal config: parse extracts PDF pictures, chunker emits
    # figure chunks, embedder uses Voyage multimodal_embed. L6/L7/L8/
    # L9/L10 are OFF here to keep the smoke focused on the multimodal
    # WRITE + STORE round trip; extending to L6+ is `smoke_pipeline.py`'s
    # job. `allowed_types=["Paper"]` because test_documents PDFs are
    # research papers.
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
        ),
        extract_figures=True,
        embed_images=True,
    )
    print(
        f"Config       : extract_figures={config.extract_figures}, "
        f"embed_images={config.embed_images}, "
        f"layers={config.layers.model_dump()}"
    )
    print()

    result = await ingest_document(target, config, "Document", "Paper")

    print()
    print("Result:")
    print(f"  n_chunks     : {result.n_chunks}")
    print(f"  embed_ok     : {result.embed_ok}")
    print(f"  lancedb_ok   : {result.lancedb_ok}")
    print(f"  kg_chunks_ok : {result.kg_chunks_ok}")
    if result.embed_error:
        print(f"  embed_error  : {result.embed_error}")
    print()

    # ---- Chunk breakdown by content_type (LanceDB) ----
    print("Chunk breakdown from LanceDB:")
    rows = await get_search_client().get_chunks_by_doc_id(doc_id)
    if rows is None:
        print("  (LanceDB returned None — read failed)")
    else:
        by_type: dict[str, int] = {}
        for r in rows:
            ct = r.get("content_type") or "(null)"
            by_type[ct] = by_type.get(ct, 0) + 1
        for ct, n in sorted(by_type.items()):
            print(f"  content_type={ct!r:12s} count={n}")
    print()

    # ---- Figures directory contents ----
    fig_dir = _figures_dir_for(doc_id)
    print(f"Figures on disk in {fig_dir}:")
    if not fig_dir.exists():
        print("  (directory not created — extract_figures may have found no "
              "PictureItems, or docling failed to save)")
    else:
        pngs = sorted(fig_dir.glob("*.png"))
        for p in pngs:
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:8s}  {size_kb:>7.1f} KB")
        if not pngs:
            print("  (empty — no PNGs saved)")
    print()

    # ---- First 5 figure rows from LanceDB with multimodal fields ----
    print("First 5 figure chunk rows in LanceDB:")
    if rows:
        figs = [r for r in rows if r.get("content_type") == "figure"]
        if not figs:
            print("  (no rows with content_type='figure')")
        for r in figs[:5]:
            print(
                f"  chunk_id={r['chunk_id']!r} "
                f"image_ref={r.get('image_ref')!r} "
                f"text[:80]={(r.get('text') or '')[:80]!r}"
            )
    print()

    # ---- Same check in Neo4j ----
    print("First 5 :Chunk figure nodes in Neo4j:")
    kg_client = get_kg_client()
    async with kg_client.driver.session() as session:
        records = await session.run(
            f"MATCH (c:{CHUNK_LABEL} {{doc_id: $doc_id}}) "
            f"WHERE c.content_type = 'figure' "
            f"RETURN c.chunk_id AS chunk_id, "
            f"       c.image_ref AS image_ref, "
            f"       c.content_type AS content_type, "
            f"       c.page AS page "
            f"LIMIT 5",
            doc_id=doc_id,
        )
        data = await records.data()
    if not data:
        print("  (no :Chunk nodes with content_type='figure')")
    for row in data:
        print(
            f"  chunk_id={row['chunk_id']!r} "
            f"image_ref={row['image_ref']!r} "
            f"page={row['page']!r}"
        )

    print()
    print("Inspect in your tools:")
    print(f"  Figures dir:  {fig_dir}")
    print(
        f"  Neo4j:  MATCH (c:Chunk {{doc_id: '{doc_id}'}}) "
        f"WHERE c.content_type = 'figure' "
        f"RETURN c LIMIT 20"
    )
    print(
        f"  LanceDB: chunks where doc_id = {doc_id!r} "
        f"AND content_type = 'figure'"
    )
    print(
        "  GUI:     open the Search tab, ask a question, look for the "
        "'Figures cited' section in the Latest view."
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
