"""Tests for ingestion.bulk_ops - Layer 3 UI-facing ops + Plan/Result contract.

End-to-end behaviour against real LanceDB / Neo4j is exercised by the
forthcoming `scripts/smoke_bulk_ops.py`. These unit tests verify the
plan/execute split + delegation to Layer 2 (`pipeline.py`) with all
heavy deps mocked.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.corpus_config import (
    CorpusConfig,
    CrossDocConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.ingestion.bulk_ops import (
    AddPlan,
    AddResult,
    BulkBackfillPlan,
    BulkBackfillResult,
    BulkReEmbedPlan,
    BulkReEmbedResult,
    BulkResolveOpenAlexPlan,
    BulkResolveOpenAlexResult,
    ClearXrefEdgesPlan,
    CrossDocBackfillResult,
    DeleteDocPlan,
    DeleteDocResult,
    IngestFolderItem,
    IngestFolderPlan,
    IngestFolderResult,
    MaterializeXrefEdgesPlan,
    RebuildVectorIndexPlan,
    RebuildVectorIndexResult,
    RecomputeCrossDocXrefsPlan,
    StripAllXrefsPlan,
    StripMaterializedXrefsPlan,
    SyncPlan,
    SyncResult,
    add_execute,
    add_plan,
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
    bulk_rebuild_vector_index_execute,
    bulk_rebuild_vector_index_plan,
    bulk_resolve_openalex_execute,
    bulk_resolve_openalex_plan,
    clear_xref_edges_execute,
    clear_xref_edges_plan,
    delete_doc_execute,
    delete_doc_plan,
    ingest_folder_execute,
    ingest_folder_plan,
    materialize_xref_edges_execute,
    materialize_xref_edges_plan,
    recompute_cross_doc_xrefs_execute,
    recompute_cross_doc_xrefs_plan,
    strip_all_xrefs_execute,
    strip_all_xrefs_plan,
    strip_materialized_xrefs_execute,
    strip_materialized_xrefs_plan,
    sync_execute,
    sync_plan,
)
from knowledge_agent.ingestion.sync_diff import (
    DiskFile,
    IndexedDoc,
    SyncBuckets,
)

# ---- DeleteDocPlan dataclass + summary string ----


def test_delete_doc_plan_summary_uses_title_when_present():
    plan = DeleteDocPlan(
        doc_id="abc123",
        title="My Paper",
        n_chunks=42,
        source_path="/tmp/paper.pdf",
    )
    assert "My Paper" in plan.summary
    assert "42 chunks" in plan.summary


def test_delete_doc_plan_summary_falls_back_to_source_path():
    plan = DeleteDocPlan(
        doc_id="abc123def456",
        title=None,
        n_chunks=10,
        source_path="/tmp/loose-file.md",
    )
    # No title -> use path so the user still recognises the doc.
    assert "/tmp/loose-file.md" in plan.summary


def test_delete_doc_plan_summary_falls_back_to_doc_id_when_nothing_else():
    plan = DeleteDocPlan(doc_id="abc123def456", title=None, n_chunks=0, source_path=None)
    # No title, no path -> show truncated doc_id (first 12 chars).
    assert "abc123def456" in plan.summary


def test_delete_doc_plan_is_frozen():
    """Plan is immutable - the dialog can't accidentally mutate it
    between display and execute."""
    plan = DeleteDocPlan(doc_id="abc", title=None, n_chunks=0, source_path=None)
    with pytest.raises(Exception):
        # dataclasses.FrozenInstanceError - too specific to type-check.
        plan.doc_id = "different"  # type: ignore[misc]


# ---- delete_doc_plan ----


async def test_delete_doc_plan_empty_doc_id_raises():
    """Empty doc_id is a programming error, not a returnable result."""
    with pytest.raises(ValueError):
        await delete_doc_plan("")


async def test_delete_doc_plan_returns_empty_plan_when_doc_not_in_lancedb():
    """No LanceDB rows -> n_chunks=0, None metadata. execute still valid."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=None)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await delete_doc_plan("missing-doc")

    assert plan.doc_id == "missing-doc"
    assert plan.n_chunks == 0
    assert plan.title is None
    assert plan.source_path is None


async def test_delete_doc_plan_empty_chunk_list_also_returns_empty_plan():
    """get_chunks_by_doc_id returning [] (read OK, no rows) - same shape."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await delete_doc_plan("missing-doc")

    assert plan.n_chunks == 0


async def test_delete_doc_plan_populates_from_first_chunk_row():
    """Title + source_path come from row[0]; n_chunks = len(rows)."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(
        return_value=[
            {"title": "Found Paper", "source_path": "/data/found.pdf"},
            {"title": "Found Paper", "source_path": "/data/found.pdf"},
            {"title": "Found Paper", "source_path": "/data/found.pdf"},
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await delete_doc_plan("docZ")

    assert plan.doc_id == "docZ"
    assert plan.title == "Found Paper"
    assert plan.source_path == "/data/found.pdf"
    assert plan.n_chunks == 3


async def test_delete_doc_plan_handles_missing_optional_metadata_fields():
    """Row without title / source_path -> plan fields are None, not KeyError."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(
        return_value=[
            {"chunk_id": "doc#0"},  # no title, no source_path
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await delete_doc_plan("docZ")

    assert plan.title is None
    assert plan.source_path is None
    assert plan.n_chunks == 1


# ---- delete_doc_execute ----


async def test_delete_doc_execute_delegates_to_pipeline_delete_doc():
    """execute should call pipeline.delete_doc with the plan's doc_id."""
    plan = DeleteDocPlan(doc_id="docZ", title=None, n_chunks=5, source_path=None)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=True,
    ) as pdd:
        result = await delete_doc_execute(plan)

    pdd.assert_called_once_with("docZ")
    assert result.doc_id == "docZ"
    assert result.ok is True


async def test_delete_doc_execute_propagates_failure():
    """pipeline.delete_doc returning False -> result.ok = False."""
    plan = DeleteDocPlan(doc_id="docZ", title=None, n_chunks=5, source_path=None)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=False,
    ):
        result = await delete_doc_execute(plan)

    assert result.ok is False


async def test_delete_doc_execute_returns_result_dataclass():
    """Result is a typed dataclass, not a raw bool - matches the UI contract."""
    plan = DeleteDocPlan(doc_id="docZ", title=None, n_chunks=0, source_path=None)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=True,
    ):
        result = await delete_doc_execute(plan)

    assert isinstance(result, DeleteDocResult)


# ---- IngestFolderPlan dataclass + summary string ----


def _ifi(
    path: str = "/tmp/p.pdf",
    doc_id: str = "doc1",
    size: int = 0,
    exists: bool = False,
    status: str | None = None,
) -> IngestFolderItem:
    return IngestFolderItem(
        path=Path(path),
        doc_id=doc_id,
        size_bytes=size,
        exists_in_db=exists,
        metadata_status=status,
    )


def test_ingest_folder_plan_counts_aggregate_over_items():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"),
        main_label="Document",
        sub_label="Paper",
        items=(
            _ifi("/tmp/x/a.pdf", "d1", 1_000_000, exists=False),
            _ifi("/tmp/x/b.pdf", "d2", 2_000_000, exists=True, status="enriched"),
            _ifi("/tmp/x/c.pdf", "d3", 3_000_000, exists=True, status="manual"),
        ),
    )
    assert plan.n_files == 3
    assert plan.n_overwrites == 2
    assert plan.n_manual == 1
    assert plan.total_bytes == 6_000_000


def test_ingest_folder_plan_summary_mentions_file_count_and_size():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"),
        main_label="Document",
        sub_label=None,
        items=(_ifi(size=5_000_000),),
    )
    assert "1 files" in plan.summary
    assert "MB" in plan.summary


def test_ingest_folder_plan_summary_mentions_overwrites_when_any():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"),
        main_label="Document",
        sub_label=None,
        items=(_ifi(exists=True),),
    )
    assert "overwrite" in plan.summary.lower()


def test_ingest_folder_plan_summary_mentions_manual_when_any():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"),
        main_label="Document",
        sub_label=None,
        items=(_ifi(exists=True, status="manual"),),
    )
    assert "manual" in plan.summary.lower()


def test_ingest_folder_plan_summary_no_overwrite_no_manual_when_clean():
    """Pure new ingest (no DB matches) -> simple summary, no warnings."""
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"),
        main_label="Document",
        sub_label=None,
        items=(_ifi(exists=False),),
    )
    assert "overwrite" not in plan.summary.lower()
    assert "manual" not in plan.summary.lower()


# ---- ingest_folder_plan ----


async def test_ingest_folder_plan_raises_when_folder_is_none():
    with pytest.raises(ValueError):
        await ingest_folder_plan(None, "Document")  # type: ignore[arg-type]


async def test_ingest_folder_plan_raises_when_path_is_not_directory(tmp_path):
    not_a_dir = tmp_path / "missing"
    with pytest.raises(ValueError):
        await ingest_folder_plan(not_a_dir, "Document")


async def test_ingest_folder_plan_skips_unsupported_extensions(tmp_path):
    """Only files whose extension matches `parse.supported_extensions()`
    show up in items - others (random .png, .exe) are ignored."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "random.exe").write_bytes(b"MZ")

    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await ingest_folder_plan(tmp_path, "Document", "Paper")

    paths = [i.path.name for i in plan.items]
    assert paths == ["paper.pdf"]


async def test_ingest_folder_plan_recurses_into_subdirectories(tmp_path):
    """Deeply nested files are picked up - relies on plan dialog to surface
    surprise scope (e.g. accidentally selecting ~/Documents)."""
    sub1 = tmp_path / "subA"
    sub1.mkdir()
    sub2 = sub1 / "subB"
    sub2.mkdir()
    (tmp_path / "top.pdf").write_bytes(b"top")
    (sub1 / "mid.pdf").write_bytes(b"mid")
    (sub2 / "deep.pdf").write_bytes(b"deep")

    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await ingest_folder_plan(tmp_path, "Document", "Paper")

    names = sorted(i.path.name for i in plan.items)
    assert names == ["deep.pdf", "mid.pdf", "top.pdf"]


async def test_ingest_folder_plan_marks_existing_doc_with_metadata_status(tmp_path):
    """A file whose hash matches an existing LanceDB doc -> exists_in_db True,
    metadata_status pulled from row[0]."""
    (tmp_path / "a.pdf").write_bytes(b"content-A")
    (tmp_path / "b.pdf").write_bytes(b"content-B")

    search_mock = MagicMock()
    # First file's get_chunks_by_doc_id returns an existing row.
    # Second file returns []. The mock just returns "existing" for the
    # first call, [] for subsequent - simulates "one doc already in DB".
    search_mock.get_chunks_by_doc_id = AsyncMock(
        side_effect=[
            [{"metadata_status": "manual"}],
            [],
        ]
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await ingest_folder_plan(tmp_path, "Document", "Paper")

    # Sorted by path: a.pdf is first by filename.
    assert plan.items[0].exists_in_db is True
    assert plan.items[0].metadata_status == "manual"
    assert plan.items[1].exists_in_db is False
    assert plan.items[1].metadata_status is None


async def test_ingest_folder_plan_returns_empty_plan_for_empty_folder(tmp_path):
    """No matching files -> plan.items = empty tuple, n_files = 0."""
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await ingest_folder_plan(tmp_path, "Document", "Paper")

    assert plan.n_files == 0
    assert plan.items == ()


# ---- ingest_folder_execute ----


def _config() -> CorpusConfig:
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),
    )


async def test_ingest_folder_execute_calls_pipeline_ingest_document_per_item():
    plan = IngestFolderPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        items=(
            _ifi("/tmp/a.pdf", "d1"),
            _ifi("/tmp/b.pdf", "d2"),
            _ifi("/tmp/c.pdf", "d3"),
        ),
    )
    with patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock:
        result = await ingest_folder_execute(plan, _config())

    assert id_mock.call_count == 3
    # Each call passed path + config + main_label + sub_label.
    for call in id_mock.call_args_list:
        args, _ = call
        assert args[2] == "Document"  # main_label
        assert args[3] == "Paper"  # sub_label
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    assert result.failures == ()


async def test_ingest_folder_execute_failsoft_per_file():
    """One file raising must NOT abort the loop - the rest still ingest."""
    plan = IngestFolderPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        items=(
            _ifi("/tmp/a.pdf", "d1"),
            _ifi("/tmp/bad.pdf", "d2"),
            _ifi("/tmp/c.pdf", "d3"),
        ),
    )
    side = [None, RuntimeError("docling exploded"), None]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document",
        side_effect=side,
    ):
        result = await ingest_folder_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert len(result.failures) == 1
    failed_name, failed_repr = result.failures[0]
    assert failed_name == "bad.pdf"
    assert "docling exploded" in failed_repr


async def test_ingest_folder_execute_returns_result_dataclass():
    plan = IngestFolderPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        items=(),
    )
    result = await ingest_folder_execute(plan, _config())
    assert isinstance(result, IngestFolderResult)
    assert result.n_succeeded == 0
    assert result.n_failed == 0


async def test_ingest_folder_execute_empty_plan_is_noop():
    """No items -> result reports zeros, ingest_document never called."""
    plan = IngestFolderPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        items=(),
    )
    with patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock:
        result = await ingest_folder_execute(plan, _config())

    id_mock.assert_not_called()
    assert result.n_succeeded == 0


# ---- SyncPlan dataclass + summary string ----


def _disk(path: str, doc_id: str) -> DiskFile:
    return DiskFile(path=Path(path), doc_id=doc_id)


def _ix(
    doc_id: str,
    stored_path: str | None = None,
    title: str | None = None,
    metadata_status: str | None = None,
    n_chunks: int = 1,
) -> IndexedDoc:
    return IndexedDoc(
        doc_id=doc_id,
        stored_path=stored_path,
        title=title,
        metadata_status=metadata_status,
        n_chunks=n_chunks,
    )


def _buckets(
    new=(),
    unchanged=(),
    moved=(),
    edited=(),
    orphan=(),
) -> SyncBuckets:
    return SyncBuckets(
        new=tuple(new),
        unchanged=tuple(unchanged),
        moved=tuple(moved),
        edited=tuple(edited),
        orphan=tuple(orphan),
    )


def test_sync_plan_n_properties_aggregate_buckets():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(
            new=[_disk("/a.pdf", "d1")],
            unchanged=[_disk("/b.pdf", "d2"), _disk("/c.pdf", "d3")],
            moved=[(_disk("/new/m.pdf", "d4"), _ix("d4"))],
            edited=[(_disk("/e.pdf", "d5-new"), _ix("d5-old"))],
            orphan=[_ix("d6"), _ix("d7")],
        ),
    )
    assert plan.n_new == 1
    assert plan.n_unchanged == 2
    assert plan.n_moved == 1
    assert plan.n_edited == 1
    assert plan.n_orphans == 2


def test_sync_plan_summary_mentions_all_action_buckets():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(
            new=[_disk("/a.pdf", "d1")],
            moved=[(_disk("/b.pdf", "d2"), _ix("d2"))],
            edited=[(_disk("/c.pdf", "d3-new"), _ix("d3-old"))],
            orphan=[_ix("d4")],
        ),
    )
    s = plan.summary
    assert "1 new" in s
    assert "1 moved" in s
    assert "1 edited" in s
    # Orphan delete is the destructive action - must be loud.
    assert "orphans to DELETE" in s


def test_sync_plan_summary_nothing_to_do_when_all_unchanged():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(unchanged=[_disk("/a.pdf", "d1")]),
    )
    assert "Nothing to sync" in plan.summary


def test_sync_plan_orphan_display_names_falls_back_to_path_then_doc_id():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(
            orphan=[
                _ix("d1", title="Great Paper", stored_path="/x/g.pdf"),
                _ix("d2-abcdef1234567890", stored_path="/x/no-title.pdf"),
                _ix("d3-abcdef1234567890"),  # no title, no path
            ]
        ),
    )
    names = plan.orphan_display_names
    assert names[0] == "Great Paper"  # title wins
    assert names[1] == "/x/no-title.pdf"  # path fallback
    assert "d3-abcdef" in names[2]  # doc_id truncated fallback


# ---- sync_plan (file walking + indexed lookup + classify) ----


async def test_sync_plan_raises_when_folder_is_none():
    with pytest.raises(ValueError):
        await sync_plan(None, "Document")  # type: ignore[arg-type]


async def test_sync_plan_raises_when_lancedb_list_fails(tmp_path):
    """list_indexed_docs raising -> propagates so we don't silently
    treat a temporary LanceDB outage as 'whole corpus is orphan'."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(side_effect=RuntimeError("lance boom"))
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
        pytest.raises(RuntimeError, match="lance boom"),
    ):
        await sync_plan(tmp_path, "Document", "Paper")


async def test_sync_plan_runs_walk_list_classify_end_to_end(tmp_path):
    """Two-file folder + one indexed doc - check the three-step plumbing
    produces a correct SyncBuckets (NEW + ORPHAN)."""
    (tmp_path / "newfile.pdf").write_bytes(b"new-content")

    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            {
                "doc_id": "orphan-d",
                "source_path": "/old/orphan.pdf",
                "title": "Orphan Title",
                "metadata_status": "enriched",
                "n_chunks": 5,
            }
        ]
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await sync_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 1
    assert plan.n_orphans == 1
    assert plan.buckets.orphan[0].title == "Orphan Title"


async def test_sync_plan_passes_indexed_doc_fields_through_to_classifier(tmp_path):
    """Fields from list_indexed_docs land in IndexedDoc objects unchanged."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            {
                "doc_id": "abc",
                "source_path": "/path",
                "title": "T",
                "metadata_status": "manual",
                "n_chunks": 7,
            }
        ]
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await sync_plan(tmp_path, "Document", "Paper")

    orphan = plan.buckets.orphan[0]
    assert orphan.doc_id == "abc"
    assert orphan.stored_path == "/path"
    assert orphan.title == "T"
    assert orphan.metadata_status == "manual"
    assert orphan.n_chunks == 7


# ---- sync_execute ----


async def test_sync_execute_new_bucket_calls_ingest_document_per_file():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(
            new=[
                _disk("/a.pdf", "d1"),
                _disk("/b.pdf", "d2"),
            ]
        ),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock,
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
    ):
        result = await sync_execute(plan, _config())

    assert id_mock.call_count == 2
    assert result.n_new_ingested == 2
    assert result.n_new_failed == 0


async def test_sync_execute_moved_bucket_patches_source_path_via_lancedb():
    """MOVED items are NOT re-ingested - just the stored path is patched."""
    old = _ix("d1", stored_path="/old/p.pdf")
    disk = _disk("/new/p.pdf", "d1")
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(moved=[(disk, old)]),
    )
    search_mock = MagicMock()
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock,
    ):
        result = await sync_execute(plan, _config())

    id_mock.assert_not_called()  # MOVED never re-ingests
    search_mock.update_doc_metadata.assert_called_once_with(
        "d1",
        {"source_path": "/new/p.pdf"},
    )
    assert result.n_moved == 1


async def test_sync_execute_edited_bucket_deletes_old_then_ingests_new():
    old = _ix("d-old", stored_path="/p.pdf")
    disk = _disk("/p.pdf", "d-new")
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(edited=[(disk, old)]),
    )
    search_mock = MagicMock()
    kg_mock = MagicMock()
    kg_mock.get_focal_labels_by_doc_id = AsyncMock(return_value=(None, None))
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc") as del_mock,
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as ing_mock,
    ):
        result = await sync_execute(plan, _config())

    # Old doc_id wiped FIRST (matches the bulk_ops design: edited = delete + ingest).
    del_mock.assert_called_once_with("d-old")
    ing_mock.assert_called_once()
    assert result.n_edited_succeeded == 1


async def test_sync_execute_edited_preserves_labels_from_old_doc():
    """EDITED with preserve=True: read old labels from KG BEFORE the
    delete, carry them into the fresh ingest even though the new
    content has a new content-hash doc_id."""
    old = _ix("d-old", stored_path="/p.pdf")
    disk = _disk("/p.pdf", "d-new")
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(edited=[(disk, old)]),
    )
    search_mock = MagicMock()
    kg_mock = MagicMock()
    # Old doc was labelled :Document:Note. Sync-time dropdown says Paper.
    # Preserve should win.
    kg_mock.get_focal_labels_by_doc_id = AsyncMock(
        return_value=("Document", "Note"),
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.get_kg_client", return_value=kg_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc"),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as ing_mock,
    ):
        await sync_execute(plan, _config())

    ing_mock.assert_called_once()
    call_args = ing_mock.call_args
    assert call_args.args[2] == "Document"
    assert call_args.args[3] == "Note"
    # And the read happened BEFORE the passed dropdown values propagated.
    kg_mock.get_focal_labels_by_doc_id.assert_called_once_with("d-old")


async def test_sync_execute_edited_overwrites_labels_when_preserve_false():
    """EDITED with preserve=False: passed (main, sub) reach ingest,
    no KG label read happens (short-circuit)."""
    old = _ix("d-old", stored_path="/p.pdf")
    disk = _disk("/p.pdf", "d-new")
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(edited=[(disk, old)]),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
        ) as kg_factory_mock,
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc"),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as ing_mock,
    ):
        await sync_execute(plan, _config(), preserve_existing_labels=False)

    ing_mock.assert_called_once()
    call_args = ing_mock.call_args
    assert call_args.args[2] == "Document"
    assert call_args.args[3] == "Paper"
    # Short-circuit — KG client factory shouldn't even have been consulted.
    kg_factory_mock.assert_not_called()


async def test_sync_execute_orphan_bucket_deletes_each_doc_id():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(orphan=[_ix("d1"), _ix("d2"), _ix("d3")]),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc", return_value=True
        ) as del_mock,
    ):
        result = await sync_execute(plan, _config())

    assert del_mock.call_count == 3
    assert result.n_orphans_deleted == 3


async def test_sync_execute_new_failure_does_not_abort_loop():
    """One NEW raises -> other NEW + MOVED + ORPHAN buckets still process."""
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(
            new=[_disk("/a.pdf", "d1"), _disk("/bad.pdf", "d2")],
            orphan=[_ix("d-orphan")],
        ),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document",
            side_effect=[None, RuntimeError("boom")],
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc", return_value=True
        ) as del_mock,
    ):
        result = await sync_execute(plan, _config())

    assert result.n_new_ingested == 1
    assert result.n_new_failed == 1
    assert len(result.failures) == 1
    assert result.failures[0][0].startswith("NEW")
    # Orphan deletion ran despite the NEW failure.
    del_mock.assert_called_once_with("d-orphan")


async def test_sync_execute_returns_result_dataclass():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(),
    )
    with patch("knowledge_agent.ingestion.bulk_ops.get_search_client"):
        result = await sync_execute(plan, _config())
    assert isinstance(result, SyncResult)


async def test_add_execute_stops_on_cancel_at_file_boundary():
    """`should_cancel` returning True stops the loop at a clean document
    boundary — files already ingested stay done, the rest are never started."""
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=(
            _ifi("/tmp/a.pdf", "d1"),
            _ifi("/tmp/b.pdf", "d2"),
            _ifi("/tmp/c.pdf", "d3"),
        ),
        n_skipped=0,
    )
    ingested = {"n": 0}

    async def _fake_ingest(*_a, **_k):
        ingested["n"] += 1

    def _cancel() -> bool:
        # Checked at the TOP of each iteration: allow the first file through,
        # then request cancel so the loop breaks before the second.
        return ingested["n"] >= 1

    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document",
        side_effect=_fake_ingest,
    ):
        result = await add_execute(plan, _config(), should_cancel=_cancel)

    assert ingested["n"] == 1  # only the first file processed
    assert result.n_succeeded == 1
    assert result.n_failed == 0


# ---- AddPlan dataclass + summary string ----


def test_add_plan_summary_mentions_new_count_and_skipped():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        new_items=(_ifi(size=1_000_000),),
        n_skipped=4,
    )
    s = plan.summary
    assert "1 new" in s
    assert "4 already in DB" in s


def test_add_plan_summary_omits_skipped_when_zero():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=(_ifi(),),
        n_skipped=0,
    )
    assert "skipped" not in plan.summary.lower()
    assert "already in DB" not in plan.summary


def test_add_plan_aggregate_properties():
    items = (
        _ifi(size=1_000_000),
        _ifi(size=2_000_000),
    )
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=items,
        n_skipped=0,
    )
    assert plan.n_new == 2
    assert plan.total_bytes == 3_000_000


# ---- add_plan ----


async def test_add_plan_raises_on_invalid_folder():
    with pytest.raises(ValueError):
        await add_plan(None, "Document")  # type: ignore[arg-type]


async def test_add_plan_includes_only_files_not_already_in_db(tmp_path):
    """Existing-in-DB files are skipped (counted), not put into new_items."""
    (tmp_path / "new.pdf").write_bytes(b"new")
    (tmp_path / "old.pdf").write_bytes(b"old")

    search_mock = MagicMock()
    # First file's hash is NOT in DB (returns []), second IS (returns existing rows).
    search_mock.get_chunks_by_doc_id = AsyncMock(
        side_effect=[
            [],
            [{"metadata_status": "enriched"}],
        ]
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await add_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 1
    assert plan.n_skipped == 1
    # The new_items entry is the first (alphabetical) - new.pdf.
    assert plan.new_items[0].path.name == "new.pdf"


async def test_add_plan_empty_folder_returns_zero_counts(tmp_path):
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        plan = await add_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 0
    assert plan.n_skipped == 0


async def test_add_plan_does_not_call_list_indexed_docs(tmp_path):
    """Add is per-file, not a full diff - it must not enumerate the
    entire index (the whole point of being cheaper than Sync)."""
    (tmp_path / "a.pdf").write_bytes(b"a")
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id = AsyncMock(return_value=[])
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.parse.supported_extensions", return_value={"pdf"}
        ),
    ):
        await add_plan(tmp_path, "Document", "Paper")

    search_mock.list_indexed_docs.assert_not_called()


# ---- add_execute ----


async def test_add_execute_calls_ingest_document_for_each_new_item():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        new_items=(_ifi("/tmp/a.pdf", "d1"), _ifi("/tmp/b.pdf", "d2")),
        n_skipped=0,
    )
    with patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock:
        result = await add_execute(plan, _config())

    assert id_mock.call_count == 2
    assert result.n_succeeded == 2


async def test_add_execute_does_not_touch_skipped_items():
    """n_skipped is informational only - execute never sees those files."""
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=(_ifi("/tmp/new.pdf", "d-new"),),
        n_skipped=99,  # plan recorded 99 already-in-DB files
    )
    with patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock:
        result = await add_execute(plan, _config())

    # Only the one NEW item was ingested.
    assert id_mock.call_count == 1
    assert result.n_succeeded == 1


async def test_add_execute_failsoft_per_file():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=(
            _ifi("/tmp/a.pdf", "d1"),
            _ifi("/tmp/bad.pdf", "d2"),
            _ifi("/tmp/c.pdf", "d3"),
        ),
        n_skipped=0,
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document",
        side_effect=[None, RuntimeError("boom"), None],
    ):
        result = await add_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert "boom" in result.failures[0][1]


async def test_add_execute_returns_result_dataclass():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        new_items=(),
        n_skipped=0,
    )
    result = await add_execute(plan, _config())
    assert isinstance(result, AddResult)


# ---- bulk_resolve_openalex ----


def _indexed_dict(
    doc_id: str,
    metadata_status: str = "enriched",
    title: str | None = None,
    sub_label: str | None = "Paper",
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_path": f"/p/{doc_id}.pdf",
        "title": title or f"T-{doc_id}",
        "metadata_status": metadata_status,
        "sub_label": sub_label,
        "n_chunks": 3,
    }


async def test_bulk_resolve_openalex_plan_skips_manual_by_default():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            _indexed_dict("d1", "pending"),
            _indexed_dict("d2", "manual"),
            _indexed_dict("d3", "enriched"),
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_resolve_openalex_plan()

    assert set(plan.target_doc_ids) == {"d1", "d3"}
    assert len(plan.skipped_manual) == 1
    assert plan.skipped_manual[0].doc_id == "d2"
    assert plan.skip_manual is True


async def test_bulk_resolve_openalex_plan_skip_manual_false_includes_all():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            _indexed_dict("d1", "manual"),
            _indexed_dict("d2", "enriched"),
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_resolve_openalex_plan(skip_manual=False)

    assert set(plan.target_doc_ids) == {"d1", "d2"}
    assert plan.n_skipped == 0


async def test_bulk_resolve_openalex_plan_skips_non_paper_docs():
    """DOI/OpenAlex enrichment is Paper-only, so non-Paper docs are never
    targets — captured in `skipped_non_paper`, not resolved — regardless
    of metadata_status or skip_manual (mirrors the ingest-time gate)."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            _indexed_dict("paper1", "enriched", sub_label="Paper"),
            _indexed_dict("note1", "enriched", sub_label="Note"),
            _indexed_dict("untyped", "pending", sub_label=None),
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_resolve_openalex_plan()

    # Only the Paper is a target; the Note and the untyped doc are held
    # back as non-Paper, not silently dropped.
    assert set(plan.target_doc_ids) == {"paper1"}
    assert {d.doc_id for d in plan.skipped_non_paper} == {"note1", "untyped"}
    assert plan.n_skipped == 0  # neither non-Paper doc counts as a manual skip


async def test_bulk_resolve_openalex_plan_non_paper_skip_survives_skip_manual_false():
    """Even with skip_manual=False (overwrite manual), a non-Paper doc is
    still excluded — the Paper gate is not a manual-protection knob."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            _indexed_dict("paper1", "manual", sub_label="Paper"),
            _indexed_dict("note1", "manual", sub_label="Note"),
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_resolve_openalex_plan(skip_manual=False)

    assert set(plan.target_doc_ids) == {"paper1"}  # manual Paper still targeted
    assert {d.doc_id for d in plan.skipped_non_paper} == {"note1"}


async def test_bulk_resolve_openalex_plan_raises_on_lancedb_failure():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(side_effect=RuntimeError("lance boom"))
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_search_client",
            return_value=search_mock,
        ),
        pytest.raises(RuntimeError, match="lance boom"),
    ):
        await bulk_resolve_openalex_plan()


def test_bulk_resolve_openalex_plan_summary_mentions_skipped_when_present():
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1", "d2"),
        skipped_manual=(
            IndexedDoc(
                doc_id="d3",
                stored_path=None,
                title=None,
                metadata_status="manual",
                n_chunks=1,
            ),
        ),
        skip_manual=True,
    )
    s = plan.summary
    assert "2 docs" in s
    assert "1 manual" in s


def test_bulk_resolve_openalex_plan_summary_mentions_non_paper_skip():
    """The dialog must tell the user why non-Paper docs won't be resolved,
    not silently shrink the target count."""
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1",),
        skipped_manual=(),
        skip_manual=True,
        skipped_non_paper=(
            IndexedDoc(
                doc_id="note1",
                stored_path=None,
                title=None,
                metadata_status="baseline",
                n_chunks=1,
            ),
        ),
    )
    s = plan.summary
    assert "1 docs" in s
    assert "non-Paper" in s
    assert "Paper-only" in s


async def test_bulk_resolve_openalex_execute_counts_three_buckets():
    """Resolved / no work / failed buckets each get a count."""
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1", "d2", "d3", "d4"),
        skipped_manual=(),
        skip_manual=True,
    )
    # d1 resolves, d2 no work, d3 raises, d4 resolves.
    side = [
        {
            "work_resolved": True,
            "metadata_patched": True,
            "kg_l1_l4_ok": True,
            "new_status": "enriched",
        },
        {
            "work_resolved": False,
            "metadata_patched": False,
            "kg_l1_l4_ok": False,
            "new_status": None,
        },
        RuntimeError("boom"),
        {
            "work_resolved": True,
            "metadata_patched": True,
            "kg_l1_l4_ok": False,
            "new_status": "enriched",
        },
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.resolve_openalex",
        side_effect=side,
    ):
        result = await bulk_resolve_openalex_execute(plan)

    assert result.n_resolved == 2
    assert result.n_no_work == 1
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"


async def test_bulk_resolve_openalex_execute_per_doc_skip_manual_false():
    """Plan-side filter already chose targets; per-doc call must NOT
    re-apply skip_manual or some manual docs would be silently filtered
    twice (once at plan, once at per-doc)."""
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1",),
        skipped_manual=(),
        skip_manual=False,
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.resolve_openalex",
        return_value={"work_resolved": True, "new_status": "enriched"},
    ) as ro_mock:
        await bulk_resolve_openalex_execute(plan)

    # Bulk MUST call per-doc with skip_manual=False so the plan-side
    # filter is the ONLY skip gate.
    ro_mock.assert_called_once_with("d1", skip_manual=False)


# ---- bulk_re_embed ----


async def test_bulk_re_embed_plan_lists_all_indexed_docs():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(
        return_value=[
            {"doc_id": "d1", "n_chunks": 10},
            {"doc_id": "d2", "n_chunks": 20},
        ]
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_re_embed_plan()

    assert set(plan.target_doc_ids) == {"d1", "d2"}
    assert plan.total_chunks == 30


def test_bulk_re_embed_plan_summary_mentions_doc_and_chunk_counts():
    plan = BulkReEmbedPlan(target_doc_ids=("d1", "d2"), total_chunks=50)
    s = plan.summary
    assert "2 docs" in s
    assert "50 chunks" in s


async def test_bulk_re_embed_plan_raises_on_lancedb_failure():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(side_effect=RuntimeError("lance boom"))
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_search_client",
            return_value=search_mock,
        ),
        pytest.raises(RuntimeError, match="lance boom"),
    ):
        await bulk_re_embed_plan()


async def test_bulk_re_embed_execute_counts_successes_and_failures():
    plan = BulkReEmbedPlan(target_doc_ids=("d1", "d2", "d3"), total_chunks=0)
    side = [
        {"embed_ok": True, "lancedb_ok": True, "n_chunks": 5},
        {"embed_ok": False, "lancedb_ok": False, "n_chunks": 0},  # voyage fail
        RuntimeError("crash"),
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.re_embed",
        side_effect=side,
    ):
        result = await bulk_re_embed_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 2
    failed_ids = {f[0] for f in result.failures}
    assert failed_ids == {"d2", "d3"}


# ---- bulk_backfill_chunks / entities / ontology ----


async def test_bulk_backfill_chunks_plan_uses_layer_name_chunks():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_chunks_plan()
    assert plan.layer_name == "chunks"
    assert "chunks" in plan.summary


async def test_bulk_backfill_entities_plan_uses_layer_name_entities():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_entities_plan()
    assert plan.layer_name == "entities"


async def test_bulk_backfill_ontology_plan_uses_layer_name_ontology():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_ontology_plan()
    assert plan.layer_name == "ontology"


def _mock_downstream_reconciles():
    """Patch all 5 reconcile_*_to_config in bulk_ops as async no-ops.

    bulk_backfill_chunks_execute / bulk_backfill_entities_execute reconcile
    every downstream layer (L6-L10) to config before looping, which runs real
    Cypher. Unit tests must never touch a KG instance, so mock the reconciles
    out. (get_kg_client stays unmocked — it is lazy; the sibling ontology /
    triples / cross-doc execute tests leave it unmocked too.) SSOT: add a
    layer here if a 6th reconcile is introduced.
    """
    return patch.multiple(
        "knowledge_agent.ingestion.bulk_ops",
        reconcile_ontologies_to_config=AsyncMock(),
        reconcile_entities_to_config=AsyncMock(),
        reconcile_triples_to_config=AsyncMock(),
        reconcile_cross_doc_to_config=AsyncMock(),
        reconcile_cross_doc_xrefs_to_config=AsyncMock(),
    )


async def test_bulk_backfill_chunks_execute_counts_chunks_ok_as_success():
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"),
        layer_name="chunks",
    )
    side = [
        {"chunks_ok": True, "entities": {}},
        {"chunks_ok": False, "entities": {}},
        RuntimeError("boom"),
    ]
    with (
        _mock_downstream_reconciles(),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_chunks",
            side_effect=side,
        ),
    ):
        result = await bulk_backfill_chunks_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 2


async def test_bulk_backfill_entities_execute_counts_entities_ok_as_success():
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"),
        layer_name="entities",
    )
    side = [
        {"entities_ok": True, "n_mentions": 5, "ontology": {}},
        {"entities_ok": False, "n_mentions": 0, "ontology": {}},
    ]
    with (
        _mock_downstream_reconciles(),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_entities",
            side_effect=side,
        ),
    ):
        result = await bulk_backfill_entities_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1


async def test_bulk_backfill_ontology_execute_counts_any_import_ok_as_success():
    """One doc gets MeSH+GO results (one OK), another gets all failures."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"),
        layer_name="ontology",
    )
    side = [
        {
            "mesh": {"imported": False, "import_ok": True, "n_links": 5},
            "go": {"imported": False, "import_ok": False, "n_links": 0},
        },
        {
            "mesh": {"imported": False, "import_ok": False, "n_links": 0},
        },
        {},  # no ontologies enabled - no-op success
    ]
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_ontology",
            side_effect=side,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_ontologies_to_config",
            new_callable=AsyncMock,
        ),
    ):
        result = await bulk_backfill_ontology_execute(plan, _config())

    # d1 = success (MeSH worked), d2 = fail (nothing imported), d3 = no-op success.
    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d2"


async def test_bulk_backfill_ontology_execute_recomputes_l10_when_layer_on():
    """Re-linking changes :CANONICAL_TO, which L10 rides on, so the op
    rebuilds L10 :RELATED_BY_XREF at the end when the layer is on."""
    plan = BulkBackfillPlan(target_doc_ids=("d1",), layer_name="ontology")
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_ontologies_to_config",
            new_callable=AsyncMock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_ontology",
            return_value={"mesh": {"import_ok": True, "n_links": 3}},
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            return_value=7,
        ) as l10_recompute,
    ):
        result = await bulk_backfill_ontology_execute(
            plan,
            _config_xrefs("use", cross_doc_xrefs=True, cross_doc_xrefs_threshold=3),
        )
    l10_recompute.assert_called_once()
    # threshold flows through positionally.
    args, _ = l10_recompute.call_args
    assert args[1] == 3
    assert result.n_succeeded == 1


async def test_bulk_backfill_ontology_execute_no_l10_when_layer_off():
    """cross_doc_xrefs off -> no L10 recompute (L8/L9 never depend on
    canonical links, so nothing downstream needs rebuilding)."""
    plan = BulkBackfillPlan(target_doc_ids=("d1",), layer_name="ontology")
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_ontologies_to_config",
            new_callable=AsyncMock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_ontology",
            return_value={"mesh": {"import_ok": True, "n_links": 3}},
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await bulk_backfill_ontology_execute(plan, _config())
    l10_recompute.assert_not_called()
    assert result.n_succeeded == 1


async def test_bulk_backfill_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="chunks")
    with (
        _mock_downstream_reconciles(),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.backfill_chunks"),
    ):
        result = await bulk_backfill_chunks_execute(plan, _config())
    assert isinstance(result, BulkBackfillResult)


# ---- bulk_backfill_triples (L8) ----


async def test_bulk_backfill_triples_plan_uses_layer_name_triples():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_triples_plan()
    assert plan.layer_name == "triples"


async def test_bulk_backfill_triples_execute_counts_triples_ok_as_success():
    """triples_ok=True counts as success regardless of n_triples - the
    LLM finding zero qualifying relations is still a clean run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"),
        layer_name="triples",
    )
    side = [
        {"triples_ok": True, "n_triples": 5},  # wrote 5 edges
        {"triples_ok": True, "n_triples": 0},  # no relations found - still success
        {"triples_ok": False, "n_triples": 0},  # Cypher / LLM failure
    ]
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples",
            side_effect=side,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_triples_to_config",
            new_callable=AsyncMock,
        ),
    ):
        result = await bulk_backfill_triples_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"


async def test_bulk_backfill_triples_execute_catches_per_doc_exceptions():
    """A doc raising mid-iteration counts as a failure, others still run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"),
        layer_name="triples",
    )
    side = [
        RuntimeError("boom"),
        {"triples_ok": True, "n_triples": 3},
    ]
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples",
            side_effect=side,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_triples_to_config",
            new_callable=AsyncMock,
        ),
    ):
        result = await bulk_backfill_triples_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1
    # The failure carries the doc_id + the repr of the exception.
    assert result.failures[0][0] == "d1"


async def test_bulk_backfill_triples_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="triples")
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_triples_to_config",
            new_callable=AsyncMock,
        ),
    ):
        result = await bulk_backfill_triples_execute(plan, _config())
    assert isinstance(result, BulkBackfillResult)


# ---- bulk_backfill_cross_doc (L9) ----


def _kg_client_with_edge_count(n: int = 0) -> MagicMock:
    """A mock KG client whose count_related_to_edges returns `n` (the L9
    execute now queries the corpus-wide :RELATED_TO total for its report)."""
    client = MagicMock()
    client.count_related_to_edges = AsyncMock(return_value=n)
    return client


async def test_bulk_backfill_cross_doc_plan_uses_layer_name_cross_doc():
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}])
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_cross_doc_plan(_config())
    assert plan.layer_name == "cross_doc"


async def test_bulk_backfill_cross_doc_execute_counts_cross_doc_ok_as_success():
    """cross_doc_ok=True counts as success regardless of n_edges - a
    doc with no other doc meeting the threshold is still a clean run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"),
        layer_name="cross_doc",
    )
    side = [
        {"cross_doc_ok": True, "n_edges": 5},  # wrote 5 edges
        {"cross_doc_ok": True, "n_edges": 0},  # no overlap met threshold - still success
        {"cross_doc_ok": False, "n_edges": 0},  # Cypher failure
    ]
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc",
            side_effect=side,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_cross_doc_to_config",
            new_callable=AsyncMock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=_kg_client_with_edge_count(7),
        ),
    ):
        result = await bulk_backfill_cross_doc_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"
    assert result.n_edges == 7  # corpus-wide :RELATED_TO count, not summed per-doc


async def test_bulk_backfill_cross_doc_execute_catches_per_doc_exceptions():
    """A doc raising mid-iteration counts as a failure, others still run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"),
        layer_name="cross_doc",
    )
    side = [
        RuntimeError("boom"),
        {"cross_doc_ok": True, "n_edges": 2},
    ]
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc",
            side_effect=side,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_cross_doc_to_config",
            new_callable=AsyncMock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=_kg_client_with_edge_count(0),
        ),
    ):
        result = await bulk_backfill_cross_doc_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1
    assert result.failures[0][0] == "d1"


async def test_bulk_backfill_cross_doc_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="cross_doc")
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.reconcile_cross_doc_to_config",
            new_callable=AsyncMock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=_kg_client_with_edge_count(0),
        ),
    ):
        result = await bulk_backfill_cross_doc_execute(plan, _config())
    assert isinstance(result, CrossDocBackfillResult)


async def test_bulk_backfill_cross_doc_plan_summary_names_threshold():
    """#3a: the confirm summary states the threshold, so the user sees what
    'related' means for this run."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs = AsyncMock(return_value=[{"doc_id": "d1"}, {"doc_id": "d2"}])
    config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, cross_doc=True),
        entities=EntityConfig(extractors=["llm"]),
        cross_doc=CrossDocConfig(threshold=4),
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_backfill_cross_doc_plan(config)
    assert "Rebuild cross-doc links" in plan.summary
    assert "4" in plan.summary  # the configured threshold


async def test_bulk_re_embed_execute_returns_result_dataclass():
    plan = BulkReEmbedPlan(target_doc_ids=(), total_chunks=0)
    result = await bulk_re_embed_execute(plan, _config())
    assert isinstance(result, BulkReEmbedResult)


async def test_bulk_resolve_openalex_execute_returns_result_dataclass():
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=(),
        skipped_manual=(),
        skip_manual=True,
    )
    result = await bulk_resolve_openalex_execute(plan)
    assert isinstance(result, BulkResolveOpenAlexResult)


# ---- materialize_xref_edges (L7: resolve dangling xrefs into edges) ----


def _config_xrefs(
    xrefs_mode: str = "use",
    cross_doc_xrefs: bool = False,
    cross_doc_xrefs_threshold: int = 2,
) -> CorpusConfig:
    """Build a CorpusConfig with the xref-related layers set.

    Default: `xrefs="use"` + L10 off, with at least one ontology
    enabled so the config validator is happy (xrefs without any
    ontology is technically allowed; including ontology_mesh keeps
    the corpus "realistic")."""
    from knowledge_agent.corpus_config import (
        CrossDocXrefsConfig,
        EntityConfig,
        OntologyConfig,
    )

    flags_kwargs = {
        "chunks": True,
        "entities": True,
        "ontology_mesh": True,
        "xrefs": xrefs_mode,
        "cross_doc_xrefs": cross_doc_xrefs,
    }
    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(**flags_kwargs),
        entities=EntityConfig(extractor="llm"),
        ontology={"mesh": OntologyConfig(matching="exact")},
        cross_doc_xrefs=(
            CrossDocXrefsConfig(threshold=cross_doc_xrefs_threshold) if cross_doc_xrefs else None
        ),
    )


def test_materialize_xref_edges_plan_summary_when_layer_off():
    """xrefs="none" -> summary calls out the no-op state."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        xrefs_mode="none",
        n_dangling_sources=0,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    assert "no-op" in plan.summary
    assert '"none"' in plan.summary


def test_materialize_xref_edges_plan_summary_use_no_l10():
    """xrefs on, L10 off -> summary describes resolution but not L10, and
    names the selected ontologies."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        xrefs_mode="use",
        n_dangling_sources=42,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    s = plan.summary
    assert "42" in s
    assert "MERGEd" in s or "idempotent" in s
    assert "mesh" in s and "go" in s
    # No L10 mention when the layer is off.
    assert "RELATED_BY_XREF" not in s


def test_materialize_xref_edges_plan_summary_use_plus_l10_mentions_rebuild():
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        xrefs_mode="use",
        n_dangling_sources=100,
        will_recompute_l10=True,
        l10_threshold=3,
    )
    s = plan.summary
    assert "100" in s
    assert "RELATED_BY_XREF" in s
    assert "threshold=3" in s


async def test_materialize_xref_edges_plan_sums_dangling_across_selection():
    """The plan sums `count_dangling_xrefs` over the selected ontologies
    and maps names to term_labels."""
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_dangling_xrefs",
            side_effect=[5, 7],
        ),
    ):
        plan = await materialize_xref_edges_plan(["mesh", "go"], _config_xrefs("use"))
    assert plan.ontology_names == ("mesh", "go")
    assert plan.term_labels == ("MeSHTerm", "GOTerm")
    assert plan.n_dangling_sources == 12
    assert plan.xrefs_mode == "use"


async def test_materialize_xref_edges_plan_unknown_ontology_raises():
    with pytest.raises(ValueError):
        await materialize_xref_edges_plan(["not-a-real-ontology"], _config_xrefs("use"))


async def test_materialize_xref_edges_plan_flags_l10_when_layer_on():
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_dangling_xrefs",
            return_value=0,
        ),
    ):
        plan = await materialize_xref_edges_plan(
            ["mesh"],
            _config_xrefs("use", cross_doc_xrefs=True, cross_doc_xrefs_threshold=5),
        )
    assert plan.will_recompute_l10 is True
    assert plan.l10_threshold == 5


async def test_materialize_xref_edges_execute_skips_when_layer_off():
    """xrefs="none" -> execute returns skipped result, no client calls."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        xrefs_mode="none",
        n_dangling_sources=0,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
        ) as get_client,
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.materialize_xref_edges_for_ontology",
        ) as materialize,
    ):
        result = await materialize_xref_edges_execute(plan)
    assert result.xrefs_layer_skipped is True
    assert result.per_ontology_edges == {}
    assert result.l10_attempted is False
    get_client.assert_not_called()
    materialize.assert_not_called()


async def test_materialize_xref_edges_execute_resolves_each_selected_ontology():
    """Resolves each selected ontology; per_ontology_edges maps term_label
    to count; L10 not touched when the layer is off."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        xrefs_mode="use",
        n_dangling_sources=5,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.materialize_xref_edges_for_ontology",
            side_effect=[5, 7],
        ) as materialize,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await materialize_xref_edges_execute(plan)
    assert materialize.call_count == 2
    l10_recompute.assert_not_called()
    assert result.xrefs_layer_skipped is False
    assert result.per_ontology_edges == {"MeSHTerm": 5, "GOTerm": 7}
    assert result.l10_attempted is False


async def test_materialize_xref_edges_execute_fail_soft_per_ontology():
    """One ontology's resolve raising -> its entry is None, others still run."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        xrefs_mode="use",
        n_dangling_sources=5,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.materialize_xref_edges_for_ontology",
            side_effect=[RuntimeError("boom"), 7],
        ),
    ):
        result = await materialize_xref_edges_execute(plan)
    assert result.per_ontology_edges == {"MeSHTerm": None, "GOTerm": 7}


async def test_materialize_xref_edges_execute_recomputes_l10_when_planned():
    """When `plan.will_recompute_l10 is True`, the L10 global rebuild is
    invoked with the plan's threshold."""
    plan = MaterializeXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        xrefs_mode="use",
        n_dangling_sources=5,
        will_recompute_l10=True,
        l10_threshold=3,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.materialize_xref_edges_for_ontology",
            return_value=5,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            return_value=42,
        ) as l10_recompute,
    ):
        result = await materialize_xref_edges_execute(plan)
    l10_recompute.assert_called_once()
    # Verify threshold flowed through positionally.
    args, _ = l10_recompute.call_args
    assert args[1] == 3
    assert result.l10_attempted is True
    assert result.n_l10_edges_written == 42


# ---- recompute_cross_doc_xrefs (standalone L10 rebuild) ----


def test_recompute_cross_doc_xrefs_plan_summary_layer_off():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=False,
        n_existing_l10_edges=0,
        threshold=2,
    )
    assert "off" in plan.summary
    assert "no-op" in plan.summary


def test_recompute_cross_doc_xrefs_plan_summary_layer_on():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=True,
        n_existing_l10_edges=17,
        threshold=5,
    )
    s = plan.summary
    assert "17" in s
    assert "threshold=5" in s
    assert "wiped and rewritten" in s


async def test_recompute_cross_doc_xrefs_plan_layer_off_skips_edge_count():
    """Layer off -> plan still builds, but n_existing_l10_edges defaults
    to 0 (no Cypher query fired)."""
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_kg_client",
    ) as get_client:
        plan = await recompute_cross_doc_xrefs_plan(_config_xrefs("use"))
    assert plan.enabled is False
    assert plan.n_existing_l10_edges == 0
    get_client.assert_not_called()


async def test_recompute_cross_doc_xrefs_plan_layer_on_queries_existing_count():
    """Layer on -> plan queries the live L10 edge count via the client's
    directed-count helper (count_related_by_xref_edges), so the "existing
    edges" figure matches the true count (an undirected count would double it)."""
    kg_mock = MagicMock()
    kg_mock.count_related_by_xref_edges = AsyncMock(return_value=25)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_kg_client",
        return_value=kg_mock,
    ):
        plan = await recompute_cross_doc_xrefs_plan(
            _config_xrefs("use", cross_doc_xrefs=True, cross_doc_xrefs_threshold=4),
        )
    assert plan.enabled is True
    assert plan.n_existing_l10_edges == 25
    assert plan.threshold == 4
    kg_mock.count_related_by_xref_edges.assert_awaited_once()


async def test_recompute_cross_doc_xrefs_execute_skipped_when_layer_off():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=False,
        n_existing_l10_edges=0,
        threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
        ) as get_client,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await recompute_cross_doc_xrefs_execute(plan)
    assert result.layer_skipped is True
    assert result.n_edges_written is None
    get_client.assert_not_called()
    l10_recompute.assert_not_called()


async def test_recompute_cross_doc_xrefs_execute_calls_global_recompute_when_on():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=True,
        n_existing_l10_edges=12,
        threshold=3,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            return_value=42,
        ) as l10_recompute,
    ):
        result = await recompute_cross_doc_xrefs_execute(plan)
    l10_recompute.assert_called_once()
    args, _ = l10_recompute.call_args
    assert args[1] == 3  # threshold positional
    assert result.layer_skipped is False
    assert result.n_edges_written == 42


async def test_recompute_cross_doc_xrefs_execute_reraises_on_failure():
    """A global-recompute crash re-raises (surfaces as a 'failed' status)
    instead of being swallowed into a fake success."""
    plan = RecomputeCrossDocXrefsPlan(
        enabled=True,
        n_existing_l10_edges=0,
        threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await recompute_cross_doc_xrefs_execute(plan)


# ---- clear_xref_edges (L7: delete xref edges for the selection) ----


async def test_clear_xref_edges_plan_unknown_ontology_raises():
    with pytest.raises(ValueError):
        await clear_xref_edges_plan(["not-a-real-ontology"], _config())


async def test_clear_xref_edges_plan_sums_edges_and_maps_labels():
    """Plan sums the edge count across the selection and maps names to
    term_labels, so execute runs without re-querying."""
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_xref_edges",
            side_effect=[12, 3],
        ),
    ):
        plan = await clear_xref_edges_plan(["mesh", "go"], _config())
    assert plan.ontology_names == ("mesh", "go")
    assert plan.term_labels == ("MeSHTerm", "GOTerm")
    assert plan.n_existing_edges == 15
    # _config() has cross_doc_xrefs off -> no L10 recompute planned.
    assert plan.will_recompute_l10 is False


def test_clear_xref_edges_plan_summary_mentions_outbound_only():
    """The summary explicitly states inbound xrefs are left alone."""
    plan = ClearXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        n_existing_edges=10,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    s = plan.summary
    assert "outbound" in s.lower() or "outgoing" in s.lower()
    assert "INBOUND" in s
    assert "10" in s
    assert "mesh" in s


async def test_clear_xref_edges_execute_deletes_each_selected_ontology():
    """Execute deletes edges per selected ontology and maps term_label to
    the count; L10 not touched when the layer is off."""
    plan = ClearXrefEdgesPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_existing_edges=14,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.remove_xref_edges_for_ontology",
            side_effect=[10, 4],
        ) as remove_fn,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await clear_xref_edges_execute(plan)
    assert remove_fn.call_count == 2
    l10_recompute.assert_not_called()
    assert result.per_ontology_deleted == {"MeSHTerm": 10, "GOTerm": 4}
    assert result.l10_attempted is False


async def test_clear_xref_edges_execute_fail_soft_per_ontology():
    """One ontology's delete raising -> its entry is None, others still run."""
    plan = ClearXrefEdgesPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_existing_edges=0,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.remove_xref_edges_for_ontology",
            side_effect=[RuntimeError("boom"), 4],
        ),
    ):
        result = await clear_xref_edges_execute(plan)
    assert result.per_ontology_deleted == {"MeSHTerm": None, "GOTerm": 4}


async def test_clear_xref_edges_plan_flags_l10_when_layer_on():
    """Plan captures will_recompute_l10 + threshold from config so execute
    can refresh L10 after clearing."""
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_xref_edges",
            return_value=3,
        ),
    ):
        plan = await clear_xref_edges_plan(
            ["mesh"],
            _config_xrefs("use", cross_doc_xrefs=True, cross_doc_xrefs_threshold=4),
        )
    assert plan.will_recompute_l10 is True
    assert plan.l10_threshold == 4
    assert "L10" in plan.summary


async def test_clear_xref_edges_execute_recomputes_l10_when_planned():
    """will_recompute_l10=True -> clearing is followed by an L10 global
    rebuild with the plan's threshold, so L10 doesn't reference the
    now-deleted equivalences."""
    plan = ClearXrefEdgesPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        n_existing_edges=5,
        will_recompute_l10=True,
        l10_threshold=4,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.remove_xref_edges_for_ontology",
            return_value=5,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            return_value=9,
        ) as l10_recompute,
    ):
        result = await clear_xref_edges_execute(plan)
    l10_recompute.assert_called_once()
    args, _ = l10_recompute.call_args
    assert args[1] == 4
    assert result.per_ontology_deleted == {"MeSHTerm": 5}


# ---- strip_materialized_xrefs (L7: prune resolved dangling entries) ----


async def test_strip_materialized_xrefs_plan_sums_dangling_and_maps_labels():
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_dangling_xrefs",
            side_effect=[5, 2],
        ),
    ):
        plan = await strip_materialized_xrefs_plan(["mesh", "go"], _config())
    assert plan.ontology_names == ("mesh", "go")
    assert plan.term_labels == ("MeSHTerm", "GOTerm")
    assert plan.n_dangling_sources == 7


def test_strip_materialized_xrefs_plan_summary_no_edges_no_l10():
    plan = StripMaterializedXrefsPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        n_dangling_sources=4,
    )
    s = plan.summary
    assert "mesh" in s
    assert "no edges" in s.lower()
    assert "RELATED_BY_XREF" not in s


async def test_strip_materialized_xrefs_execute_tidies_each_ontology_no_l10():
    plan = StripMaterializedXrefsPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_dangling_sources=6,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.strip_materialized_xrefs_for_ontology",
            side_effect=[3, 1],
        ) as strip_fn,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await strip_materialized_xrefs_execute(plan)
    assert strip_fn.call_count == 2
    l10_recompute.assert_not_called()
    assert result.per_ontology_tidied == {"MeSHTerm": 3, "GOTerm": 1}


async def test_strip_materialized_xrefs_execute_fail_soft_per_ontology():
    plan = StripMaterializedXrefsPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_dangling_sources=6,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.strip_materialized_xrefs_for_ontology",
            side_effect=[RuntimeError("boom"), 1],
        ),
    ):
        result = await strip_materialized_xrefs_execute(plan)
    assert result.per_ontology_tidied == {"MeSHTerm": None, "GOTerm": 1}


# ---- strip_all_xrefs (L7: remove the dangling_xrefs property) ----


async def test_strip_all_xrefs_plan_sums_dangling_and_maps_labels():
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.count_dangling_xrefs",
            side_effect=[5, 2],
        ),
    ):
        plan = await strip_all_xrefs_plan(["mesh", "go"], _config())
    assert plan.ontology_names == ("mesh", "go")
    assert plan.term_labels == ("MeSHTerm", "GOTerm")
    assert plan.n_dangling_sources == 7


def test_strip_all_xrefs_plan_summary_warns_reimport_no_l10():
    plan = StripAllXrefsPlan(
        ontology_names=("mesh",),
        term_labels=("MeSHTerm",),
        n_dangling_sources=4,
    )
    s = plan.summary
    assert "mesh" in s
    assert "re-import" in s
    assert "no edges" in s.lower()
    assert "RELATED_BY_XREF" not in s


async def test_strip_all_xrefs_execute_removes_property_each_ontology_no_l10():
    plan = StripAllXrefsPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_dangling_sources=6,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.remove_dangling_xrefs_for_ontology",
            side_effect=[3, 1],
        ) as remove_fn,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = await strip_all_xrefs_execute(plan)
    assert remove_fn.call_count == 2
    l10_recompute.assert_not_called()
    assert result.per_ontology_cleared == {"MeSHTerm": 3, "GOTerm": 1}


async def test_strip_all_xrefs_execute_fail_soft_per_ontology():
    plan = StripAllXrefsPlan(
        ontology_names=("mesh", "go"),
        term_labels=("MeSHTerm", "GOTerm"),
        n_dangling_sources=6,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.xrefs.remove_dangling_xrefs_for_ontology",
            side_effect=[RuntimeError("boom"), 1],
        ),
    ):
        result = await strip_all_xrefs_execute(plan)
    assert result.per_ontology_cleared == {"MeSHTerm": None, "GOTerm": 1}


# ---- end-of-action index maintenance (deferred optimize / auto-rebuild /
#      delete-compaction) + the manual Rebuild-vector-index op ----------------


async def test_ingest_folder_execute_runs_end_of_action_maintenance():
    plan = IngestFolderPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        items=(_ifi("/tmp/a.pdf", "d1"), _ifi("/tmp/b.pdf", "d2")),
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.maintain_indexes_after_action",
            new_callable=AsyncMock,
        ) as maint,
    ):
        await ingest_folder_execute(plan, _config())

    maint.assert_awaited_once()
    assert maint.await_args.kwargs["n_written"] == 2


async def test_add_execute_runs_end_of_action_maintenance():
    plan = AddPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        new_items=(_ifi("/tmp/a.pdf", "d1"),),
        n_skipped=0,
    )
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.maintain_indexes_after_action",
            new_callable=AsyncMock,
        ) as maint,
    ):
        await add_execute(plan, _config())

    maint.assert_awaited_once()
    assert maint.await_args.kwargs["n_written"] == 1


async def test_sync_execute_maintenance_counts_writes_and_deletes():
    """NEW+EDITED feed n_written; ORPHAN deletes feed n_deleted; MOVED excluded."""
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label="Paper",
        buckets=_buckets(
            new=[_disk("/a.pdf", "d1"), _disk("/b.pdf", "d2")],
            orphan=[_ix("d-orphan")],
        ),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.maintain_indexes_after_action",
            new_callable=AsyncMock,
        ) as maint,
    ):
        await sync_execute(plan, _config(), preserve_existing_labels=False)

    maint.assert_awaited_once()
    assert maint.await_args.kwargs["n_written"] == 2
    assert maint.await_args.kwargs["n_deleted"] == 1


async def test_sync_execute_maintenance_move_only_is_no_write_no_delete():
    plan = SyncPlan(
        folder=Path("/tmp"),
        main_label="Document",
        sub_label=None,
        buckets=_buckets(moved=[(_disk("/new/p.pdf", "d1"), _ix("d1", stored_path="/old/p.pdf"))]),
    )
    search_mock = MagicMock()
    search_mock.update_doc_metadata = AsyncMock(return_value=True)
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client", return_value=search_mock),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.maintain_indexes_after_action",
            new_callable=AsyncMock,
        ) as maint,
    ):
        await sync_execute(plan, _config(), preserve_existing_labels=False)

    maint.assert_awaited_once()
    assert maint.await_args.kwargs["n_written"] == 0
    assert maint.await_args.kwargs["n_deleted"] == 0


async def test_bulk_re_embed_execute_rebuilds_vector_index_on_success():
    plan = BulkReEmbedPlan(target_doc_ids=("d1",), total_chunks=0)
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.re_embed",
            new_callable=AsyncMock,
            return_value={"embed_ok": True, "lancedb_ok": True, "n_chunks": 5},
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.rebuild_vector_index",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        await bulk_re_embed_execute(plan, _config())

    rebuild.assert_awaited_once()


async def test_bulk_re_embed_execute_skips_rebuild_when_none_succeed():
    plan = BulkReEmbedPlan(target_doc_ids=("d1",), total_chunks=0)
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.re_embed",
            new_callable=AsyncMock,
            return_value={"embed_ok": False, "lancedb_ok": False, "n_chunks": 0},
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.rebuild_vector_index",
            new_callable=AsyncMock,
        ) as rebuild,
    ):
        await bulk_re_embed_execute(plan, _config())

    rebuild.assert_not_awaited()


async def test_delete_doc_execute_compacts_after_delete():
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.compact_indexes",
            new_callable=AsyncMock,
        ) as compact,
    ):
        result = await delete_doc_execute(
            DeleteDocPlan(doc_id="d1", title="T", n_chunks=3, source_path="/p/d1.pdf")
        )

    assert result.ok is True
    compact.assert_awaited_once()


async def test_delete_doc_execute_skips_compact_when_delete_fails():
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.pipeline.compact_indexes",
            new_callable=AsyncMock,
        ) as compact,
    ):
        result = await delete_doc_execute(
            DeleteDocPlan(doc_id="d1", title="T", n_chunks=3, source_path="/p/d1.pdf")
        )

    assert result.ok is False
    compact.assert_not_awaited()


async def test_bulk_rebuild_vector_index_execute_calls_rebuild():
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.rebuild_vector_index",
        new_callable=AsyncMock,
    ) as rebuild:
        result = await bulk_rebuild_vector_index_execute(RebuildVectorIndexPlan(n_chunks=10))

    rebuild.assert_awaited_once()
    assert isinstance(result, RebuildVectorIndexResult)
    assert result.n_chunks == 10


async def test_bulk_rebuild_vector_index_plan_reads_chunk_count():
    search_mock = MagicMock()
    search_mock.count_chunks = AsyncMock(return_value=42)
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = await bulk_rebuild_vector_index_plan()

    assert plan.n_chunks == 42
    assert "42" in plan.summary
