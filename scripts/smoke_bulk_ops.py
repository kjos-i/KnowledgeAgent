"""Smoke test for bulk_ops + ontology_lifecycle plan/execute pattern.

Exercises a realistic GUI workflow end-to-end against real LanceDB +
Neo4j + (one) OpenAlex lookup, with a controlled `test_documents/`
folder so the diff buckets are predictable.

Workflow (mirrors what a user does in the Library tab):

  1. Build a temporary "corpus folder" by copying the first .pdf in
     `test_documents/` into a tmp dir. Use this folder for ingest +
     sync + add scenarios so we can manipulate paths between runs.
  2. `bulk_ops.ingest_folder_plan` -> show summary -> `_execute`.
     Verifies the planning hash + LanceDB exists-check + per-file
     ingest path.
  3. `bulk_ops.sync_plan` on the same folder. Expect: UNCHANGED bucket
     populated; NEW/MOVED/EDITED/ORPHAN empty.
  4. Move the file inside the folder (rename) and `bulk_ops.sync_plan`
     again. Expect: MOVED bucket populated; sync_execute patches the
     stored `source_path` and skips re-ingest.
  5. `bulk_ops.bulk_resolve_openalex_plan` -> execute. Verifies the
     metadata-stage iteration.
  6. `bulk_ops.delete_doc_plan(doc_id)` -> execute. Verifies the
     unified delete (LanceDB + KG L1-L4 + L5 + L6).
  7. `ontology_lifecycle.delete_ontology_plan("mesh")` -> execute.
     Verifies the surgical wipe path.

Lifecycle (matches the clear-at-start + pause + optional-cleanup-at-end
convention - see [[feedback-smoke-test-cleanup]]):
  - Start: ALWAYS drop the LanceDB chunks table (schema may have
    changed since the last run) AND wipe any prior KG state for the
    test doc_id. Unconditional - no `--reset` flag, no opt-in.
  - Run the workflow above.
  - Pause for inspection in Neo4j Desktop / LanceDB.
  - On Enter, clean up the doc; on Ctrl+C, leave the data for inspection.

MeSH stays imported across runs (re-importing 30K nodes per smoke is
wasteful and unnecessary - it's reference data, not test artefact).
The chunks table is dropped because its schema can drift between
roadmap-aligned column changes.

Run from the project root:
    python scripts/smoke_bulk_ops.py

Automated counterpart (for regression catching, no inspection):
  tests/integration/ingestion/test_bulk_ops.py  (ingest_folder plan +
                                                 execute; sync_plan
                                                 NEW/UNCHANGED/MOVED/
                                                 ORPHAN classification;
                                                 delete_doc plan +
                                                 execute wiping both
                                                 stores).
                                                 Resolve_openalex +
                                                 delete_ontology
                                                 surfaces stay smoke-
                                                 only (cost / fixture
                                                 complexity).
Run via `pytest -m integration tests/integration/ingestion/`.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

# Switch the process to the smoke-test Neo4j instance BEFORE any other
# knowledge_agent import that might trigger `get_settings()`. The test instance's
# password mismatches the real instance, so a wrong-instance-running
# state fails at authentication rather than corrupting real data.
from knowledge_agent.config import load_test_env

load_test_env()

from knowledge_agent.ingestion import bulk_ops, pipeline  # noqa: E402
from knowledge_agent.ingestion.ids import compute_doc_id  # noqa: E402
from knowledge_agent.kg import ontology_lifecycle  # noqa: E402
from knowledge_agent.kg.corpus_config import (  # noqa: E402
    CorpusConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
)
from knowledge_agent.search.client import get_search_client  # noqa: E402

TEST_DOCS = Path(__file__).resolve().parent.parent / "test_documents"


def _biomedical_config() -> CorpusConfig:
    """Same shape as the smoke_pipeline.py corpus - all KG layers on."""
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=True,
            ontology_mesh=True,
        ),
        entities=EntityConfig(extractor="llm"),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )


def _print_plan(label: str, plan) -> None:
    print(f"\n--- {label} ---")
    print(f"  summary: {plan.summary}")


def _print_result(label: str, result) -> None:
    print(f"  {label} result: {result}")


async def _clear_at_start(doc_id: str) -> None:
    """Wipe every persistent footprint this smoke could collide with.

    Per the smoke-test cleanup memory, ALL smokes that write persistent
    data clear unconditionally at start. We drop the chunks table
    (schema-tolerant - the next write recreates it from current
    schema), and run the unified `pipeline.delete_doc` for the test
    doc_id (LanceDB chunks + KG L1-L4 / L5 / L6 / L7 edges that hang
    off them). MeSH terms stay imported - they're reference data, not
    smoke output.
    """
    await get_search_client().drop_chunks_table()
    await pipeline.delete_doc(doc_id)
    print(f"[clear-at-start] dropped chunks table + wiped {doc_id[:12]}")


async def main() -> None:
    # Pick the first PDF in test_documents/ and copy into a tmp corpus folder
    # so we can rename it between sync runs without polluting the source.
    pdfs = sorted(TEST_DOCS.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No .pdf found in {TEST_DOCS}")
    source_pdf = pdfs[0]
    doc_id_for_first_pdf = compute_doc_id(source_pdf)

    await _clear_at_start(doc_id_for_first_pdf)

    config = _biomedical_config()

    with tempfile.TemporaryDirectory(prefix="smoke_bulk_ops_") as tmpdir:
        corpus_folder = Path(tmpdir) / "corpus"
        corpus_folder.mkdir()
        first_path = corpus_folder / source_pdf.name
        shutil.copy(source_pdf, first_path)

        doc_id = compute_doc_id(first_path)
        print(f"\n[setup] copied {source_pdf.name} to {first_path}")
        print(f"[setup] doc_id = {doc_id[:12]}...")

        # 1. INGEST FOLDER -----------------------------------------------
        plan = await bulk_ops.ingest_folder_plan(corpus_folder, "Document", "Paper")
        _print_plan("ingest_folder_plan", plan)
        result = await bulk_ops.ingest_folder_execute(plan, config)
        _print_result("ingest_folder_execute", result)
        if result.n_failed:
            raise SystemExit(f"ingest_folder reported failures: {result.failures}")

        # 2. SYNC (UNCHANGED) --------------------------------------------
        plan = await bulk_ops.sync_plan(corpus_folder, "Document", "Paper")
        _print_plan("sync_plan (expect UNCHANGED)", plan)
        assert plan.n_unchanged == 1, (
            f"expected UNCHANGED=1, got new={plan.n_new} "
            f"moved={plan.n_moved} edited={plan.n_edited} "
            f"unchanged={plan.n_unchanged} orphan={plan.n_orphans}"
        )

        # 3. SYNC (MOVED) ------------------------------------------------
        moved_path = corpus_folder / f"moved-{source_pdf.name}"
        first_path.rename(moved_path)
        plan = await bulk_ops.sync_plan(corpus_folder, "Document", "Paper")
        _print_plan("sync_plan (expect MOVED after rename)", plan)
        assert plan.n_moved == 1, f"expected MOVED=1 after rename, got moved={plan.n_moved}"
        result = await bulk_ops.sync_execute(plan, config)
        _print_result("sync_execute", result)

        # 4. BULK RESOLVE OPENALEX ---------------------------------------
        plan = await bulk_ops.bulk_resolve_openalex_plan(skip_manual=True)
        _print_plan("bulk_resolve_openalex_plan", plan)
        result = await bulk_ops.bulk_resolve_openalex_execute(plan)
        _print_result("bulk_resolve_openalex_execute", result)

        # 5. DELETE DOC --------------------------------------------------
        plan = await bulk_ops.delete_doc_plan(doc_id)
        _print_plan(f"delete_doc_plan ({doc_id[:12]})", plan)
        result = await bulk_ops.delete_doc_execute(plan)
        _print_result("delete_doc_execute", result)
        assert result.ok, "delete_doc returned ok=False"

    # 6. ONTOLOGY LIFECYCLE -----------------------------------------------
    print("\n[ontology_lifecycle] inspecting MeSH state")
    mesh_plan = await ontology_lifecycle.delete_ontology_plan("mesh")
    _print_plan("delete_ontology_plan('mesh')", mesh_plan)

    # 7. PAUSE FOR INSPECTION + OPTIONAL CLEANUP -------------------------
    print("\n---")
    print("Smoke complete. Doc is already deleted. MeSH is still imported in Neo4j.")
    try:
        input(
            "Press Enter to delete MeSH and finish cleanup "
            "(Ctrl+C to keep MeSH for inspection / faster next run) > "
        )
    except KeyboardInterrupt:
        print("\n[interrupt] keeping MeSH imported for inspection")
        return

    if mesh_plan.is_imported:
        result = await ontology_lifecycle.delete_ontology_execute(mesh_plan)
        _print_result("delete_ontology_execute", result)


if __name__ == "__main__":
    asyncio.run(main())
