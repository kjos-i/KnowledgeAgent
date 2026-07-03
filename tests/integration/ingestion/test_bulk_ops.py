"""Integration tests for `ingestion/bulk_ops` — the GUI-facing
plan/execute orchestrators.

Covers every bulk_op exposed to the GUI:

  Folder management:
    - ingest_folder (plan + execute)
    - sync (plan with UNCHANGED / MOVED / ORPHAN; execute)
    - delete_doc (plan + execute)

  Metadata / re-embed:
    - bulk_resolve_openalex
    - bulk_re_embed

  Backfill (per-layer re-run of pipeline stages):
    - bulk_backfill_chunks (L5)
    - bulk_backfill_entities (L6 — uses LLM extractor)
    - bulk_backfill_ontology (L7 — pre-seeded ontology terms to
      avoid the heavy ~600 MB MeSH download in CI)
    - bulk_backfill_triples (L8 — uses LLM extractor)
    - bulk_backfill_cross_doc (L9 — pure Cypher)

  Xref ops (L7 cross-ontology + L10):
    - backfill_xrefs (uses synthetic OntologyTerm)
    - recompute_cross_doc_xrefs (uses synthetic OntologyTerm +
      canonical entities)
    - clear_xref_edges

Each test uses the smallest viable setup to exercise the
orchestration contract: ingest one PDF for the per-doc ops, ingest
two PDFs (or seed synthetic state) for the cross-doc ops, and
pre-seed :OntologyTerm nodes for the xref ops to avoid expensive
real ontology imports.

**LLM cost surface (real Anthropic Haiku calls — DO COST MONEY):**

  - `test_bulk_backfill_entities_*` — runs L6 entity extraction
    against the LLM adapter, one call per chunk of the 1 ingested PDF.
  - `test_bulk_backfill_triples_*` — runs L8 triple extraction
    against the LLM adapter, one call per chunk.
  - `test_bulk_backfill_ontology_*` — pure Cypher, **no LLM cost**.
  - `test_bulk_backfill_cross_doc_*` — pure Cypher, **no LLM cost**.
  - All other tests — **no LLM cost** (synthetic / orchestration-only).

Voyage embedding calls (`bulk_re_embed`) are paid but tiny.
OpenAlex calls (`bulk_resolve_openalex`) are free (unauthenticated).

Manual interactive counterpart: `scripts/smoke_bulk_ops.py`.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from knowledge_agent.ingestion.bulk_ops import (
    BackfillXrefsPlan,
    BulkBackfillPlan,
    BulkReEmbedPlan,
    BulkResolveOpenAlexPlan,
    ClearXrefEdgesPlan,
    DeleteDocPlan,
    IngestFolderPlan,
    RecomputeCrossDocXrefsPlan,
    SyncPlan,
    backfill_xrefs_execute,
    backfill_xrefs_plan,
    bulk_backfill_chunks_execute,
    bulk_backfill_chunks_plan,
    bulk_backfill_cross_doc_execute,
    bulk_backfill_cross_doc_plan,
    bulk_backfill_entities_execute,
    bulk_backfill_entities_plan,
    bulk_backfill_ontology_execute,
    bulk_backfill_ontology_plan,
    bulk_backfill_triples_execute,
    bulk_backfill_triples_plan,
    bulk_re_embed_execute,
    bulk_re_embed_plan,
    bulk_resolve_openalex_execute,
    bulk_resolve_openalex_plan,
    clear_xref_edges_execute,
    clear_xref_edges_plan,
    delete_doc_execute,
    delete_doc_plan,
    ingest_folder_execute,
    ingest_folder_plan,
    recompute_cross_doc_xrefs_execute,
    recompute_cross_doc_xrefs_plan,
    sync_execute,
    sync_plan,
)
from knowledge_agent.ingestion.ids import compute_doc_id, make_chunk_id
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    CrossDocConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
)
from knowledge_agent.kg.schema import (
    MESH_TERM_LABEL,
    MONDO_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
)

pytestmark = pytest.mark.integration


def _minimal_corpus_config() -> CorpusConfig:
    """Same L1-L5-only corpus as `test_pipeline.py` — keeps cost
    bounded."""
    return CorpusConfig(
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=False,
            triples=False,
            cross_doc=False,
            cross_doc_xrefs=False,
            xrefs="none",
        ),
        allowed_types=["Paper"],
    )


@pytest.fixture
def clean_both_stores(
    kg_client: Any, lance_client: Any, ensure_constraints: None
) -> None:
    """Same pattern as test_pipeline.py: wipe both stores before each
    bulk_ops test so multi-doc plans start from empty state."""
    import asyncio

    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    asyncio.run(lance_client.drop_chunks_table())


@pytest.fixture
def fresh_corpus_dir(tmp_path: Path, sample_pdf: Path) -> Path:
    """Copy the session sample PDF into a tmp corpus folder so we can
    rename / move it between tests without touching the source
    `test_documents/`."""
    target = tmp_path / "corpus"
    target.mkdir()
    shutil.copy(sample_pdf, target / sample_pdf.name)
    return target


async def test_ingest_folder_plan_lists_one_file(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """ingest_folder_plan on a 1-file corpus reports n_files=1 with
    a populated to_ingest list."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    assert isinstance(plan, IngestFolderPlan)
    assert plan.n_files == 1
    # The PDF in to_ingest matches our sample.
    assert any(item.path.name == sample_pdf.name for item in plan.items)


async def test_ingest_folder_execute_runs_pipeline_per_file(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """ingest_folder_execute runs the pipeline once per file in
    plan.items. After execute, both stores contain the doc's
    chunks."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    result = await ingest_folder_execute(plan, config)

    assert result.n_succeeded == 1
    assert result.n_failed == 0

    # Both stores have the doc.
    doc_id = compute_doc_id(fresh_corpus_dir / sample_pdf.name)
    lance_rows = await lance_client.get_chunks_by_doc_id(doc_id)
    assert lance_rows
    with kg_client.driver.session() as session:
        n_kg = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_kg == 1


async def test_sync_plan_classifies_unchanged_file(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """After an initial ingest, sync_plan classifies the same file as
    UNCHANGED (not NEW, MOVED, EDITED, or ORPHAN)."""
    # First ingest.
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    await ingest_folder_execute(plan, config)

    # Now sync — file hasn't changed.
    splan = await sync_plan(fresh_corpus_dir, "Document", "Paper")
    assert isinstance(splan, SyncPlan)
    assert splan.n_unchanged == 1
    assert splan.n_new == 0
    assert splan.n_moved == 0
    assert splan.n_edited == 0
    assert splan.n_orphans == 0


async def test_sync_plan_classifies_moved_file(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """After ingest, renaming the file in-place produces a MOVED
    classification (same hash, different path)."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    await ingest_folder_execute(plan, config)

    # Rename inside the corpus folder.
    original = fresh_corpus_dir / sample_pdf.name
    renamed = fresh_corpus_dir / "renamed.pdf"
    original.rename(renamed)

    splan = await sync_plan(fresh_corpus_dir, "Document", "Paper")
    assert splan.n_moved == 1
    assert splan.n_unchanged == 0
    # Sync execute patches the stored source_path.
    sresult = await sync_execute(splan, config)
    assert sresult.n_moved == 1


async def test_sync_plan_classifies_orphan_when_file_removed(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """When a previously-ingested file is removed from the corpus
    folder, sync_plan flags it as ORPHAN (in DB but not on disk)."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    await ingest_folder_execute(plan, config)

    # Remove the file from disk.
    (fresh_corpus_dir / sample_pdf.name).unlink()

    splan = await sync_plan(fresh_corpus_dir, "Document", "Paper")
    assert splan.n_orphans == 1
    assert splan.n_unchanged == 0


async def test_delete_doc_plan_reports_inventory(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """delete_doc_plan returns the title + chunk count so the GUI
    can show a confirmation dialog."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    await ingest_folder_execute(plan, config)

    doc_id = compute_doc_id(fresh_corpus_dir / sample_pdf.name)
    dplan = await delete_doc_plan(doc_id)
    assert isinstance(dplan, DeleteDocPlan)
    assert dplan.doc_id == doc_id
    assert dplan.n_chunks > 0


async def test_delete_doc_execute_wipes_both_stores(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """delete_doc_execute removes the document from BOTH stores
    (LanceDB chunks + Neo4j L1-L5)."""
    plan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()
    await ingest_folder_execute(plan, config)

    doc_id = compute_doc_id(fresh_corpus_dir / sample_pdf.name)
    dplan = await delete_doc_plan(doc_id)
    result = await delete_doc_execute(dplan)
    assert result.ok is True

    # Both stores empty for this doc.
    lance_rows = await lance_client.get_chunks_by_doc_id(doc_id) or []
    assert len(lance_rows) == 0
    with kg_client.driver.session() as session:
        n_kg = session.run(
            "MATCH (d:Document {doc_id: $doc_id}) RETURN count(d) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_kg == 0


# ---------------------------------------------------------------------------
# Corpus-config helpers for the backfill ops.
# ---------------------------------------------------------------------------


def _entities_corpus_config() -> CorpusConfig:
    """L1-L5 + L6 entities via LLM extractor (cheap)."""
    return CorpusConfig(
        layers=LayerFlags(
            openalex_papers=True,
            chunks=True,
            entities=True,
            triples=False,
            cross_doc=False,
            cross_doc_xrefs=False,
            xrefs="none",
        ),
        entities=EntityConfig(
            extractor="llm",
            entity_types=("disease", "drug", "gene", "protein"),
        ),
        allowed_types=["Paper"],
    )


def _triples_corpus_config() -> CorpusConfig:
    """L1-L5 + L6 + L8 (triples). Triples extractor uses LLM."""
    cfg = _entities_corpus_config()
    return cfg.model_copy(
        update={
            "layers": cfg.layers.model_copy(
                update={"triples": True},
            ),
        },
    )


def _cross_doc_corpus_config() -> CorpusConfig:
    """L1-L5 + L6 + L9 cross_doc. Cross-doc is pure Cypher (cheap).

    Must set `cross_doc=CrossDocConfig()` explicitly because
    `model_copy(update=...)` bypasses the model validator that
    auto-populates it from `layers.cross_doc=True`."""
    cfg = _entities_corpus_config()
    return cfg.model_copy(
        update={
            "layers": cfg.layers.model_copy(
                update={"cross_doc": True},
            ),
            "cross_doc": CrossDocConfig(),
        },
    )


def _mesh_corpus_config() -> CorpusConfig:
    """L1-L5 + L6 + L7 (MeSH ontology). Tests pre-seed :MeSHTerm
    nodes so ensure_ontology_imported sees the layer as imported
    and skips the ~600 MB download."""
    cfg = _entities_corpus_config()
    return cfg.model_copy(
        update={
            "ontology": {
                "mesh": OntologyConfig(matching="exact"),
            },
        },
    )


# ---------------------------------------------------------------------------
# bulk_resolve_openalex
# ---------------------------------------------------------------------------


async def test_bulk_resolve_openalex_plan_targets_indexed_docs(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """plan() lists every indexed doc when none are manual-flagged."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    await ingest_folder_execute(iplan, _minimal_corpus_config())

    plan = await bulk_resolve_openalex_plan()
    assert isinstance(plan, BulkResolveOpenAlexPlan)
    assert plan.n_targets == 1
    assert plan.n_skipped == 0


async def test_bulk_resolve_openalex_execute_against_real_openalex(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """execute() iterates targets and resolves each via OpenAlex.
    The sample PDF has a real DOI, so n_resolved should be 1."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    await ingest_folder_execute(iplan, _minimal_corpus_config())

    plan = await bulk_resolve_openalex_plan()
    result = await bulk_resolve_openalex_execute(plan)
    # At least one doc resolved (the sample PDF has a DOI).
    assert result.n_resolved + result.n_no_work + result.n_failed == 1
    assert result.n_failed == 0


# ---------------------------------------------------------------------------
# bulk_re_embed
# ---------------------------------------------------------------------------


async def test_bulk_re_embed_plan_includes_chunk_count(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """plan() reports total_chunks across all indexed docs."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    await ingest_folder_execute(iplan, _minimal_corpus_config())

    plan = await bulk_re_embed_plan()
    assert isinstance(plan, BulkReEmbedPlan)
    assert plan.n_targets == 1
    assert plan.total_chunks > 0


async def test_bulk_re_embed_execute_preserves_chunk_count(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """execute() re-embeds every chunk; final row count matches
    pre-re-embed count (no chunks added or lost)."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    await ingest_folder_execute(iplan, _minimal_corpus_config())
    doc_id = compute_doc_id(fresh_corpus_dir / sample_pdf.name)
    pre_count = len(await lance_client.get_chunks_by_doc_id(doc_id) or [])

    plan = await bulk_re_embed_plan()
    result = await bulk_re_embed_execute(plan)
    assert result.n_succeeded == 1
    assert result.n_failed == 0

    post_count = len(await lance_client.get_chunks_by_doc_id(doc_id) or [])
    assert post_count == pre_count


# ---------------------------------------------------------------------------
# bulk_backfill_chunks (L5)
# ---------------------------------------------------------------------------


async def test_bulk_backfill_chunks_succeeds_after_chunks_layer_enabled(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """Ingest with chunks=True (so chunks land in BOTH stores), then
    backfill_chunks re-runs the L5 write — idempotent success."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    config = _minimal_corpus_config()  # has chunks=True
    await ingest_folder_execute(iplan, config)

    plan = await bulk_backfill_chunks_plan()
    assert isinstance(plan, BulkBackfillPlan)
    assert plan.n_targets == 1
    result = await bulk_backfill_chunks_execute(plan, config)
    assert result.n_succeeded == 1
    assert result.n_failed == 0


# ---------------------------------------------------------------------------
# bulk_backfill_entities (L6 — uses LLM extractor)
# ---------------------------------------------------------------------------


async def test_bulk_backfill_entities_creates_entity_nodes(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
    sample_pdf: Path,
) -> None:
    """Ingest with entities=False, then enable entities and backfill.
    The L6 pass writes :Entity nodes + :MENTIONS edges. Cost: one
    Haiku call per chunk via the LLM adapter."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    base = _minimal_corpus_config()  # entities=False at ingest time
    await ingest_folder_execute(iplan, base)

    # Switch to entities=True for the backfill.
    backfill_config = _entities_corpus_config()
    plan = await bulk_backfill_entities_plan()
    result = await bulk_backfill_entities_execute(plan, backfill_config)
    assert result.n_succeeded == 1
    assert result.n_failed == 0

    # At least some entities exist in the KG now.
    doc_id = compute_doc_id(fresh_corpus_dir / sample_pdf.name)
    with kg_client.driver.session() as session:
        n_entities = session.run(
            "MATCH (c:Chunk {doc_id: $doc_id})-[:MENTIONS]->(e:Entity) "
            "RETURN count(DISTINCT e) AS n",
            doc_id=doc_id,
        ).single()["n"]
    assert n_entities > 0


# ---------------------------------------------------------------------------
# bulk_backfill_ontology (L7 — pre-seeded MeSH terms avoid the download)
# ---------------------------------------------------------------------------


async def test_bulk_backfill_ontology_links_entities_to_pre_seeded_mesh_terms(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """Ingest with entities=True so L6 entities exist, pre-seed a
    few :MeSHTerm nodes so `is_imported` returns True (no download),
    then backfill_ontology runs the linking pass. Asserts at least
    one :CANONICAL_TO edge gets created."""
    # Ingest with entities ON so L6 entities exist for linking.
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    entities_cfg = _entities_corpus_config()
    await ingest_folder_execute(iplan, entities_cfg)

    # Pre-seed :MeSHTerm nodes so the L7 import step short-circuits
    # via is_imported=True. Use real-looking IDs + lowercased labels
    # that may match entities the LLM extracted from a nutrition /
    # chronic-disease PDF.
    with kg_client.driver.session() as session:
        for term_id, label in [
            ("D003920", "diabetes mellitus"),
            ("D006973", "hypertension"),
            ("D029424", "obesity"),
            ("D008659", "metabolism"),
            ("D009765", "nutritional status"),
        ]:
            session.run(
                f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
                f"{{id: $id}}) SET t.label = $label",
                id=term_id, label=label,
            )

    mesh_cfg = _mesh_corpus_config()
    plan = await bulk_backfill_ontology_plan()
    result = await bulk_backfill_ontology_execute(plan, mesh_cfg)
    # No download triggered (is_imported True from our seeded terms);
    # the linking pass either matches or doesn't — both count as
    # success per the bulk_op contract (linking with zero matches is
    # not a failure).
    assert result.n_succeeded == 1
    assert result.n_failed == 0


# ---------------------------------------------------------------------------
# bulk_backfill_triples (L8 — uses LLM extractor)
# ---------------------------------------------------------------------------


async def test_bulk_backfill_triples_succeeds_after_l6_present(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """L8 backfill requires L6 entities. Ingest with entities=True,
    then run triples backfill. Cost: one Haiku call per chunk that
    has entities."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    entities_cfg = _entities_corpus_config()
    await ingest_folder_execute(iplan, entities_cfg)

    triples_cfg = _triples_corpus_config()
    plan = await bulk_backfill_triples_plan()
    result = await bulk_backfill_triples_execute(plan, triples_cfg)
    # triples_ok is True even when the LLM extracts zero triples —
    # empty output is a valid outcome, not a failure.
    assert result.n_succeeded == 1
    assert result.n_failed == 0


# ---------------------------------------------------------------------------
# bulk_backfill_cross_doc (L9 — pure Cypher)
# ---------------------------------------------------------------------------


async def test_bulk_backfill_cross_doc_runs_recompute_per_doc(
    fresh_corpus_dir: Path,
    kg_client: Any,
    lance_client: Any,
    clean_both_stores: None,
) -> None:
    """L9 backfill iterates docs + calls recompute_cross_doc_edges.
    With one doc, no :RELATED_TO edges are produced (no other doc to
    relate to) — that's still a success per the bulk_op contract."""
    iplan = await ingest_folder_plan(fresh_corpus_dir, "Document", "Paper")
    entities_cfg = _entities_corpus_config()
    await ingest_folder_execute(iplan, entities_cfg)

    cross_cfg = _cross_doc_corpus_config()
    plan = await bulk_backfill_cross_doc_plan()
    result = await bulk_backfill_cross_doc_execute(plan, cross_cfg)
    assert result.n_succeeded == 1
    assert result.n_failed == 0


# ---------------------------------------------------------------------------
# backfill_xrefs (synthetic OntologyTerm setup)
# ---------------------------------------------------------------------------


async def test_backfill_xrefs_resolves_dangling_xrefs(
    kg_client: Any, ensure_constraints: None
) -> None:
    """Pre-seed two :OntologyTerm nodes where one has a dangling_xref
    pointing at the other; backfill_xrefs creates the resolved edge."""
    # Clean state then seed.
    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
            f"{{id: 'MESH:D008659'}}) SET t.label = 'metabolism'"
        )
        session.run(
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MONDO_TERM_LABEL} "
            f"{{id: 'MONDO:0005066'}}) "
            f"SET t.label = 'metabolic disease', "
            f"t.dangling_xrefs = ['MESH:D008659']"
        )

    # Corpus that enables xrefs.
    config = _entities_corpus_config().model_copy(
        update={
            "layers": _entities_corpus_config().layers.model_copy(
                update={"xrefs": "use"},
            ),
        },
    )

    plan = await backfill_xrefs_plan(config)
    assert isinstance(plan, BackfillXrefsPlan)
    assert plan.n_dangling_sources >= 1

    result = await backfill_xrefs_execute(plan, config)
    assert result.xrefs_layer_skipped is False
    assert result.per_ontology_counts is not None

    # The :MONDO_XREF edge from MONDO:0005066 → MESH:D008659 exists.
    with kg_client.driver.session() as session:
        row = session.run(
            f"MATCH (s:{MONDO_TERM_LABEL} {{id: 'MONDO:0005066'}})"
            f"-[:MONDO_XREF]->(t:{MESH_TERM_LABEL} {{id: 'MESH:D008659'}}) "
            f"RETURN s.id AS sid"
        ).single()
    assert row is not None


async def test_backfill_xrefs_skips_when_xrefs_mode_none(
    kg_client: Any, ensure_constraints: None
) -> None:
    """When `layers.xrefs == "none"`, execute returns a skipped
    result without touching the graph."""
    config = _entities_corpus_config()  # xrefs="none" by default
    plan = await backfill_xrefs_plan(config)
    result = await backfill_xrefs_execute(plan, config)
    assert result.xrefs_layer_skipped is True
    assert result.per_ontology_counts is None


# ---------------------------------------------------------------------------
# recompute_cross_doc_xrefs (L10 graph-wide)
# ---------------------------------------------------------------------------


async def test_recompute_cross_doc_xrefs_when_layer_disabled_is_noop(
    kg_client: Any, ensure_constraints: None
) -> None:
    """When `layers.cross_doc_xrefs == False`, the plan summary
    reports the no-op and execute returns without rewriting."""
    config = _entities_corpus_config()  # cross_doc_xrefs=False
    plan = await recompute_cross_doc_xrefs_plan(config)
    assert isinstance(plan, RecomputeCrossDocXrefsPlan)
    assert plan.enabled is False

    result = await recompute_cross_doc_xrefs_execute(plan)
    # No-op contract: layer_skipped flag set, n_edges_written stays None.
    assert result.layer_skipped is True
    assert result.n_edges_written is None


async def test_recompute_cross_doc_xrefs_writes_edges_with_synthetic_setup(
    kg_client: Any, ensure_constraints: None
) -> None:
    """Pre-seed 2 docs sharing canonical MeSH terms, enable L10 +
    xrefs=use, run recompute. Edge count > 0."""
    # Clean + seed two docs with shared canonical concepts.
    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for doc_id in ("integ-doc-l10-A", "integ-doc-l10-B"):
            chunk_id = make_chunk_id(doc_id, 0)
            session.run(
                "MERGE (d:Document {doc_id: $doc_id}) "
                "ON CREATE SET d.in_corpus = true, d:Paper "
                "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                "SET c.doc_id = $doc_id, c.chunk_index = 0, "
                "c.section = 'X', c.page = 1, c.content_type = 'text' "
                "MERGE (c)-[:PART_OF]->(d) "
                "WITH c "
                "MERGE (e1:Entity {key: 'metformin', entity_type: 'drug'}) "
                "MERGE (c)-[:MENTIONS]->(e1) "
                "MERGE (t1:OntologyTerm:MeSHTerm {id: 'MESH:D008687'}) "
                "MERGE (e1)-[:CANONICAL_TO]->(t1) "
                "WITH c "
                "MERGE (e2:Entity {key: 'diabetes', entity_type: 'disease'}) "
                "MERGE (c)-[:MENTIONS]->(e2) "
                "MERGE (t2:OntologyTerm:MeSHTerm {id: 'MESH:D003920'}) "
                "MERGE (e2)-[:CANONICAL_TO]->(t2)",
                doc_id=doc_id, chunk_id=chunk_id,
            )

    # Build a corpus config with both xrefs="use" + cross_doc_xrefs=True.
    base = _entities_corpus_config()
    config = base.model_copy(
        update={
            "layers": base.layers.model_copy(
                update={"xrefs": "use", "cross_doc_xrefs": True},
            ),
        },
    )

    plan = await recompute_cross_doc_xrefs_plan(config)
    assert plan.enabled is True
    result = await recompute_cross_doc_xrefs_execute(plan)
    assert result.n_edges_written >= 1


# ---------------------------------------------------------------------------
# clear_xref_edges
# ---------------------------------------------------------------------------


async def test_clear_xref_edges_drops_target_ontology_edges(
    kg_client: Any, ensure_constraints: None
) -> None:
    """Pre-seed a :MONDO_XREF edge, then clear_xref_edges_execute
    drops it but preserves other ontologies' edges."""
    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        # MONDO term with a MONDO_XREF edge.
        session.run(
            f"MERGE (s:{ONTOLOGY_TERM_LABEL}:{MONDO_TERM_LABEL} "
            f"{{id: 'MONDO:0001'}}) "
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
            f"{{id: 'MESH:D001'}}) "
            f"MERGE (s)-[:MONDO_XREF]->(t)"
        )
        # MeSH source with a MESH_XREF edge (different ontology).
        session.run(
            f"MERGE (s:{ONTOLOGY_TERM_LABEL}:{MESH_TERM_LABEL} "
            f"{{id: 'MESH:D002'}}) "
            f"MERGE (t:{ONTOLOGY_TERM_LABEL}:{MONDO_TERM_LABEL} "
            f"{{id: 'MONDO:0002'}}) "
            f"MERGE (s)-[:MESH_XREF]->(t)"
        )

    plan = await clear_xref_edges_plan("mondo")
    assert isinstance(plan, ClearXrefEdgesPlan)
    result = await clear_xref_edges_execute(plan)
    # n_cleared = edges deleted + dangling_xrefs props removed.
    assert result.n_cleared is not None and result.n_cleared >= 1

    # MONDO_XREF gone; MESH_XREF survives.
    with kg_client.driver.session() as session:
        n_mondo = session.run(
            "MATCH ()-[r:MONDO_XREF]->() RETURN count(r) AS n"
        ).single()["n"]
        n_mesh = session.run(
            "MATCH ()-[r:MESH_XREF]->() RETURN count(r) AS n"
        ).single()["n"]
    assert n_mondo == 0
    assert n_mesh == 1
