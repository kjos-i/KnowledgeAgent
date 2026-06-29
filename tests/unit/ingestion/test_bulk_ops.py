"""Tests for ingestion.bulk_ops - Layer 3 UI-facing ops + Plan/Result contract.

End-to-end behaviour against real LanceDB / Neo4j is exercised by the
forthcoming `scripts/smoke_bulk_ops.py`. These unit tests verify the
plan/execute split + delegation to Layer 2 (`pipeline.py`) with all
heavy deps mocked.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from knowledge_agent.ingestion.bulk_ops import (
    AddPlan,
    AddResult,
    BackfillXrefsPlan,
    BackfillXrefsResult,
    BulkBackfillPlan,
    BulkBackfillResult,
    BulkReEmbedPlan,
    BulkReEmbedResult,
    BulkResolveOpenAlexPlan,
    BulkResolveOpenAlexResult,
    ClearXrefEdgesPlan,
    ClearXrefEdgesResult,
    DeleteDocPlan,
    DeleteDocResult,
    IngestFolderItem,
    IngestFolderPlan,
    IngestFolderResult,
    RecomputeCrossDocXrefsPlan,
    RecomputeCrossDocXrefsResult,
    SyncPlan,
    SyncResult,
    add_execute,
    add_plan,
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
from knowledge_agent.ingestion.sync_diff import (
    DiskFile,
    IndexedDoc,
    SyncBuckets,
)
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    LayerFlags,
)


# ---- DeleteDocPlan dataclass + summary string ----


def test_delete_doc_plan_summary_uses_title_when_present():
    plan = DeleteDocPlan(
        doc_id="abc123", title="My Paper", n_chunks=42,
        source_path="/tmp/paper.pdf",
    )
    assert "My Paper" in plan.summary
    assert "42 chunks" in plan.summary


def test_delete_doc_plan_summary_falls_back_to_source_path():
    plan = DeleteDocPlan(
        doc_id="abc123def456", title=None, n_chunks=10,
        source_path="/tmp/loose-file.md",
    )
    # No title -> use path so the user still recognises the doc.
    assert "/tmp/loose-file.md" in plan.summary


def test_delete_doc_plan_summary_falls_back_to_doc_id_when_nothing_else():
    plan = DeleteDocPlan(
        doc_id="abc123def456", title=None, n_chunks=0, source_path=None
    )
    # No title, no path -> show truncated doc_id (first 12 chars).
    assert "abc123def456" in plan.summary


def test_delete_doc_plan_is_frozen():
    """Plan is immutable - the dialog can't accidentally mutate it
    between display and execute."""
    plan = DeleteDocPlan(
        doc_id="abc", title=None, n_chunks=0, source_path=None
    )
    with pytest.raises(Exception):
        # dataclasses.FrozenInstanceError - too specific to type-check.
        plan.doc_id = "different"  # type: ignore[misc]


# ---- delete_doc_plan ----


def test_delete_doc_plan_empty_doc_id_raises():
    """Empty doc_id is a programming error, not a returnable result."""
    with pytest.raises(ValueError):
        delete_doc_plan("")


def test_delete_doc_plan_returns_empty_plan_when_doc_not_in_lancedb():
    """No LanceDB rows -> n_chunks=0, None metadata. execute still valid."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = None
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = delete_doc_plan("missing-doc")

    assert plan.doc_id == "missing-doc"
    assert plan.n_chunks == 0
    assert plan.title is None
    assert plan.source_path is None


def test_delete_doc_plan_empty_chunk_list_also_returns_empty_plan():
    """get_chunks_by_doc_id returning [] (read OK, no rows) - same shape."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = []
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = delete_doc_plan("missing-doc")

    assert plan.n_chunks == 0


def test_delete_doc_plan_populates_from_first_chunk_row():
    """Title + source_path come from row[0]; n_chunks = len(rows)."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = [
        {"title": "Found Paper", "source_path": "/data/found.pdf"},
        {"title": "Found Paper", "source_path": "/data/found.pdf"},
        {"title": "Found Paper", "source_path": "/data/found.pdf"},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = delete_doc_plan("docZ")

    assert plan.doc_id == "docZ"
    assert plan.title == "Found Paper"
    assert plan.source_path == "/data/found.pdf"
    assert plan.n_chunks == 3


def test_delete_doc_plan_handles_missing_optional_metadata_fields():
    """Row without title / source_path -> plan fields are None, not KeyError."""
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = [
        {"chunk_id": "doc#0"},  # no title, no source_path
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = delete_doc_plan("docZ")

    assert plan.title is None
    assert plan.source_path is None
    assert plan.n_chunks == 1


# ---- delete_doc_execute ----


def test_delete_doc_execute_delegates_to_pipeline_delete_doc():
    """execute should call pipeline.delete_doc with the plan's doc_id."""
    plan = DeleteDocPlan(
        doc_id="docZ", title=None, n_chunks=5, source_path=None
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=True,
    ) as pdd:
        result = delete_doc_execute(plan)

    pdd.assert_called_once_with("docZ")
    assert result.doc_id == "docZ"
    assert result.ok is True


def test_delete_doc_execute_propagates_failure():
    """pipeline.delete_doc returning False -> result.ok = False."""
    plan = DeleteDocPlan(
        doc_id="docZ", title=None, n_chunks=5, source_path=None
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=False,
    ):
        result = delete_doc_execute(plan)

    assert result.ok is False


def test_delete_doc_execute_returns_result_dataclass():
    """Result is a typed dataclass, not a raw bool - matches the UI contract."""
    plan = DeleteDocPlan(
        doc_id="docZ", title=None, n_chunks=0, source_path=None
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
        return_value=True,
    ):
        result = delete_doc_execute(plan)

    assert isinstance(result, DeleteDocResult)


# ---- IngestFolderPlan dataclass + summary string ----


def _ifi(
    path: str = "/tmp/p.pdf", doc_id: str = "doc1", size: int = 0,
    exists: bool = False, status: str | None = None,
) -> IngestFolderItem:
    return IngestFolderItem(
        path=Path(path), doc_id=doc_id, size_bytes=size,
        exists_in_db=exists, metadata_status=status,
    )


def test_ingest_folder_plan_counts_aggregate_over_items():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"), main_label="Document", sub_label="Paper",
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
        folder=Path("/tmp/x"), main_label="Document", sub_label=None,
        items=(_ifi(size=5_000_000),),
    )
    assert "1 files" in plan.summary
    assert "MB" in plan.summary


def test_ingest_folder_plan_summary_mentions_overwrites_when_any():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"), main_label="Document", sub_label=None,
        items=(_ifi(exists=True),),
    )
    assert "overwrite" in plan.summary.lower()


def test_ingest_folder_plan_summary_mentions_manual_when_any():
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"), main_label="Document", sub_label=None,
        items=(_ifi(exists=True, status="manual"),),
    )
    assert "manual" in plan.summary.lower()


def test_ingest_folder_plan_summary_no_overwrite_no_manual_when_clean():
    """Pure new ingest (no DB matches) -> simple summary, no warnings."""
    plan = IngestFolderPlan(
        folder=Path("/tmp/x"), main_label="Document", sub_label=None,
        items=(_ifi(exists=False),),
    )
    assert "overwrite" not in plan.summary.lower()
    assert "manual" not in plan.summary.lower()


# ---- ingest_folder_plan ----


def test_ingest_folder_plan_raises_when_folder_is_none():
    with pytest.raises(ValueError):
        ingest_folder_plan(None, "Document")  # type: ignore[arg-type]


def test_ingest_folder_plan_raises_when_path_is_not_directory(tmp_path):
    not_a_dir = tmp_path / "missing"
    with pytest.raises(ValueError):
        ingest_folder_plan(not_a_dir, "Document")


def test_ingest_folder_plan_skips_unsupported_extensions(tmp_path):
    """Only files whose extension matches `parse.supported_extensions()`
    show up in items - others (random .png, .exe) are ignored."""
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "random.exe").write_bytes(b"MZ")

    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = []
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = ingest_folder_plan(tmp_path, "Document", "Paper")

    paths = [i.path.name for i in plan.items]
    assert paths == ["paper.pdf"]


def test_ingest_folder_plan_recurses_into_subdirectories(tmp_path):
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
    search_mock.get_chunks_by_doc_id.return_value = []
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = ingest_folder_plan(tmp_path, "Document", "Paper")

    names = sorted(i.path.name for i in plan.items)
    assert names == ["deep.pdf", "mid.pdf", "top.pdf"]


def test_ingest_folder_plan_marks_existing_doc_with_metadata_status(tmp_path):
    """A file whose hash matches an existing LanceDB doc -> exists_in_db True,
    metadata_status pulled from row[0]."""
    (tmp_path / "a.pdf").write_bytes(b"content-A")
    (tmp_path / "b.pdf").write_bytes(b"content-B")

    search_mock = MagicMock()
    # First file's get_chunks_by_doc_id returns an existing row.
    # Second file returns []. The mock just returns "existing" for the
    # first call, [] for subsequent - simulates "one doc already in DB".
    search_mock.get_chunks_by_doc_id.side_effect = [
        [{"metadata_status": "manual"}],
        [],
    ]
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = ingest_folder_plan(tmp_path, "Document", "Paper")

    # Sorted by path: a.pdf is first by filename.
    assert plan.items[0].exists_in_db is True
    assert plan.items[0].metadata_status == "manual"
    assert plan.items[1].exists_in_db is False
    assert plan.items[1].metadata_status is None


def test_ingest_folder_plan_returns_empty_plan_for_empty_folder(tmp_path):
    """No matching files -> plan.items = empty tuple, n_files = 0."""
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = ingest_folder_plan(tmp_path, "Document", "Paper")

    assert plan.n_files == 0
    assert plan.items == ()


# ---- ingest_folder_execute ----


def _config() -> CorpusConfig:
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),
    )


def test_ingest_folder_execute_calls_pipeline_ingest_document_per_item():
    plan = IngestFolderPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
        items=(
            _ifi("/tmp/a.pdf", "d1"),
            _ifi("/tmp/b.pdf", "d2"),
            _ifi("/tmp/c.pdf", "d3"),
        ),
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"
    ) as id_mock:
        result = ingest_folder_execute(plan, _config())

    assert id_mock.call_count == 3
    # Each call passed path + config + main_label + sub_label.
    for call in id_mock.call_args_list:
        args, _ = call
        assert args[2] == "Document"  # main_label
        assert args[3] == "Paper"     # sub_label
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    assert result.failures == ()


def test_ingest_folder_execute_failsoft_per_file():
    """One file raising must NOT abort the loop - the rest still ingest."""
    plan = IngestFolderPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
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
        result = ingest_folder_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert len(result.failures) == 1
    failed_name, failed_repr = result.failures[0]
    assert failed_name == "bad.pdf"
    assert "docling exploded" in failed_repr


def test_ingest_folder_execute_returns_result_dataclass():
    plan = IngestFolderPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        items=(),
    )
    result = ingest_folder_execute(plan, _config())
    assert isinstance(result, IngestFolderResult)
    assert result.n_succeeded == 0
    assert result.n_failed == 0


def test_ingest_folder_execute_empty_plan_is_noop():
    """No items -> result reports zeros, ingest_document never called."""
    plan = IngestFolderPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        items=(),
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"
    ) as id_mock:
        result = ingest_folder_execute(plan, _config())

    id_mock.assert_not_called()
    assert result.n_succeeded == 0


# ---- SyncPlan dataclass + summary string ----


def _disk(path: str, doc_id: str) -> DiskFile:
    return DiskFile(path=Path(path), doc_id=doc_id)


def _ix(
    doc_id: str, stored_path: str | None = None, title: str | None = None,
    metadata_status: str | None = None, n_chunks: int = 1,
) -> IndexedDoc:
    return IndexedDoc(
        doc_id=doc_id, stored_path=stored_path, title=title,
        metadata_status=metadata_status, n_chunks=n_chunks,
    )


def _buckets(
    new=(), unchanged=(), moved=(), edited=(), orphan=(),
) -> SyncBuckets:
    return SyncBuckets(
        new=tuple(new), unchanged=tuple(unchanged),
        moved=tuple(moved), edited=tuple(edited), orphan=tuple(orphan),
    )


def test_sync_plan_n_properties_aggregate_buckets():
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
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
        folder=Path("/tmp"), main_label="Document", sub_label=None,
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
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(unchanged=[_disk("/a.pdf", "d1")]),
    )
    assert "Nothing to sync" in plan.summary


def test_sync_plan_orphan_display_names_falls_back_to_path_then_doc_id():
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(orphan=[
            _ix("d1", title="Great Paper", stored_path="/x/g.pdf"),
            _ix("d2-abcdef1234567890", stored_path="/x/no-title.pdf"),
            _ix("d3-abcdef1234567890"),  # no title, no path
        ]),
    )
    names = plan.orphan_display_names
    assert names[0] == "Great Paper"             # title wins
    assert names[1] == "/x/no-title.pdf"          # path fallback
    assert "d3-abcdef" in names[2]                # doc_id truncated fallback


# ---- sync_plan (file walking + indexed lookup + classify) ----


def test_sync_plan_raises_when_folder_is_none():
    with pytest.raises(ValueError):
        sync_plan(None, "Document")  # type: ignore[arg-type]


def test_sync_plan_raises_when_lancedb_list_fails(tmp_path):
    """list_indexed_docs raising -> propagates so we don't silently
    treat a temporary LanceDB outage as 'whole corpus is orphan'."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs.side_effect = RuntimeError("lance boom")
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        with pytest.raises(RuntimeError, match="lance boom"):
            sync_plan(tmp_path, "Document", "Paper")


def test_sync_plan_runs_walk_list_classify_end_to_end(tmp_path):
    """Two-file folder + one indexed doc - check the three-step plumbing
    produces a correct SyncBuckets (NEW + ORPHAN)."""
    (tmp_path / "newfile.pdf").write_bytes(b"new-content")

    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [
        {
            "doc_id": "orphan-d",
            "source_path": "/old/orphan.pdf",
            "title": "Orphan Title",
            "metadata_status": "enriched",
            "n_chunks": 5,
        }
    ]
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = sync_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 1
    assert plan.n_orphans == 1
    assert plan.buckets.orphan[0].title == "Orphan Title"


def test_sync_plan_passes_indexed_doc_fields_through_to_classifier(tmp_path):
    """Fields from list_indexed_docs land in IndexedDoc objects unchanged."""
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [
        {
            "doc_id": "abc",
            "source_path": "/path",
            "title": "T",
            "metadata_status": "manual",
            "n_chunks": 7,
        }
    ]
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = sync_plan(tmp_path, "Document", "Paper")

    orphan = plan.buckets.orphan[0]
    assert orphan.doc_id == "abc"
    assert orphan.stored_path == "/path"
    assert orphan.title == "T"
    assert orphan.metadata_status == "manual"
    assert orphan.n_chunks == 7


# ---- sync_execute ----


def test_sync_execute_new_bucket_calls_ingest_document_per_file():
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
        buckets=_buckets(new=[
            _disk("/a.pdf", "d1"), _disk("/b.pdf", "d2"),
        ]),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock,
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
    ):
        result = sync_execute(plan, _config())

    assert id_mock.call_count == 2
    assert result.n_new_ingested == 2
    assert result.n_new_failed == 0


def test_sync_execute_moved_bucket_patches_source_path_via_lancedb():
    """MOVED items are NOT re-ingested - just the stored path is patched."""
    old = _ix("d1", stored_path="/old/p.pdf")
    disk = _disk("/new/p.pdf", "d1")
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(moved=[(disk, old)]),
    )
    search_mock = MagicMock()
    search_mock.update_doc_metadata.return_value = True
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as id_mock,
    ):
        result = sync_execute(plan, _config())

    id_mock.assert_not_called()  # MOVED never re-ingests
    search_mock.update_doc_metadata.assert_called_once_with(
        "d1", {"source_path": "/new/p.pdf"},
    )
    assert result.n_moved == 1


def test_sync_execute_edited_bucket_deletes_old_then_ingests_new():
    old = _ix("d-old", stored_path="/p.pdf")
    disk = _disk("/p.pdf", "d-new")
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
        buckets=_buckets(edited=[(disk, old)]),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc") as del_mock,
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document") as ing_mock,
    ):
        result = sync_execute(plan, _config())

    # Old doc_id wiped FIRST (matches the bulk_ops design: edited = delete + ingest).
    del_mock.assert_called_once_with("d-old")
    ing_mock.assert_called_once()
    assert result.n_edited_succeeded == 1


def test_sync_execute_orphan_bucket_deletes_each_doc_id():
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(orphan=[_ix("d1"), _ix("d2"), _ix("d3")]),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
              return_value=True) as del_mock,
    ):
        result = sync_execute(plan, _config())

    assert del_mock.call_count == 3
    assert result.n_orphans_deleted == 3


def test_sync_execute_new_failure_does_not_abort_loop():
    """One NEW raises -> other NEW + MOVED + ORPHAN buckets still process."""
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(
            new=[_disk("/a.pdf", "d1"), _disk("/bad.pdf", "d2")],
            orphan=[_ix("d-orphan")],
        ),
    )
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document",
              side_effect=[None, RuntimeError("boom")]),
        patch("knowledge_agent.ingestion.bulk_ops.pipeline.delete_doc",
              return_value=True) as del_mock,
    ):
        result = sync_execute(plan, _config())

    assert result.n_new_ingested == 1
    assert result.n_new_failed == 1
    assert len(result.failures) == 1
    assert result.failures[0][0].startswith("NEW")
    # Orphan deletion ran despite the NEW failure.
    del_mock.assert_called_once_with("d-orphan")


def test_sync_execute_returns_result_dataclass():
    plan = SyncPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        buckets=_buckets(),
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client"
    ):
        result = sync_execute(plan, _config())
    assert isinstance(result, SyncResult)


# ---- AddPlan dataclass + summary string ----


def test_add_plan_summary_mentions_new_count_and_skipped():
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
        new_items=(_ifi(size=1_000_000),),
        n_skipped=4,
    )
    s = plan.summary
    assert "1 new" in s
    assert "4 already in DB" in s


def test_add_plan_summary_omits_skipped_when_zero():
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
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
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        new_items=items, n_skipped=0,
    )
    assert plan.n_new == 2
    assert plan.total_bytes == 3_000_000


# ---- add_plan ----


def test_add_plan_raises_on_invalid_folder():
    with pytest.raises(ValueError):
        add_plan(None, "Document")  # type: ignore[arg-type]


def test_add_plan_includes_only_files_not_already_in_db(tmp_path):
    """Existing-in-DB files are skipped (counted), not put into new_items."""
    (tmp_path / "new.pdf").write_bytes(b"new")
    (tmp_path / "old.pdf").write_bytes(b"old")

    search_mock = MagicMock()
    # First file's hash is NOT in DB (returns []), second IS (returns existing rows).
    search_mock.get_chunks_by_doc_id.side_effect = [
        [],
        [{"metadata_status": "enriched"}],
    ]
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = add_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 1
    assert plan.n_skipped == 1
    # The new_items entry is the first (alphabetical) - new.pdf.
    assert plan.new_items[0].path.name == "new.pdf"


def test_add_plan_empty_folder_returns_zero_counts(tmp_path):
    search_mock = MagicMock()
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        plan = add_plan(tmp_path, "Document", "Paper")

    assert plan.n_new == 0
    assert plan.n_skipped == 0


def test_add_plan_does_not_call_list_indexed_docs(tmp_path):
    """Add is per-file, not a full diff - it must not enumerate the
    entire index (the whole point of being cheaper than Sync)."""
    (tmp_path / "a.pdf").write_bytes(b"a")
    search_mock = MagicMock()
    search_mock.get_chunks_by_doc_id.return_value = []
    with (
        patch("knowledge_agent.ingestion.bulk_ops.get_search_client",
              return_value=search_mock),
        patch("knowledge_agent.ingestion.bulk_ops.parse.supported_extensions",
              return_value={"pdf"}),
    ):
        add_plan(tmp_path, "Document", "Paper")

    search_mock.list_indexed_docs.assert_not_called()


# ---- add_execute ----


def test_add_execute_calls_ingest_document_for_each_new_item():
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label="Paper",
        new_items=(_ifi("/tmp/a.pdf", "d1"), _ifi("/tmp/b.pdf", "d2")),
        n_skipped=0,
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"
    ) as id_mock:
        result = add_execute(plan, _config())

    assert id_mock.call_count == 2
    assert result.n_succeeded == 2


def test_add_execute_does_not_touch_skipped_items():
    """n_skipped is informational only - execute never sees those files."""
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        new_items=(_ifi("/tmp/new.pdf", "d-new"),),
        n_skipped=99,  # plan recorded 99 already-in-DB files
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.ingest_document"
    ) as id_mock:
        result = add_execute(plan, _config())

    # Only the one NEW item was ingested.
    assert id_mock.call_count == 1
    assert result.n_succeeded == 1


def test_add_execute_failsoft_per_file():
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
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
        result = add_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert "boom" in result.failures[0][1]


def test_add_execute_returns_result_dataclass():
    plan = AddPlan(
        folder=Path("/tmp"), main_label="Document", sub_label=None,
        new_items=(), n_skipped=0,
    )
    result = add_execute(plan, _config())
    assert isinstance(result, AddResult)


# ---- bulk_resolve_openalex ----


def _indexed_dict(
    doc_id: str, metadata_status: str = "enriched", title: str | None = None,
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "source_path": f"/p/{doc_id}.pdf",
        "title": title or f"T-{doc_id}",
        "metadata_status": metadata_status,
        "n_chunks": 3,
    }


def test_bulk_resolve_openalex_plan_skips_manual_by_default():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [
        _indexed_dict("d1", "pending"),
        _indexed_dict("d2", "manual"),
        _indexed_dict("d3", "enriched"),
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_resolve_openalex_plan()

    assert set(plan.target_doc_ids) == {"d1", "d3"}
    assert len(plan.skipped_manual) == 1
    assert plan.skipped_manual[0].doc_id == "d2"
    assert plan.skip_manual is True


def test_bulk_resolve_openalex_plan_skip_manual_false_includes_all():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [
        _indexed_dict("d1", "manual"),
        _indexed_dict("d2", "enriched"),
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_resolve_openalex_plan(skip_manual=False)

    assert set(plan.target_doc_ids) == {"d1", "d2"}
    assert plan.n_skipped == 0


def test_bulk_resolve_openalex_plan_raises_on_lancedb_failure():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.side_effect = RuntimeError("lance boom")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        with pytest.raises(RuntimeError, match="lance boom"):
            bulk_resolve_openalex_plan()


def test_bulk_resolve_openalex_plan_summary_mentions_skipped_when_present():
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1", "d2"),
        skipped_manual=(IndexedDoc(
            doc_id="d3", stored_path=None, title=None,
            metadata_status="manual", n_chunks=1,
        ),),
        skip_manual=True,
    )
    s = plan.summary
    assert "2 docs" in s
    assert "1 manual" in s


def test_bulk_resolve_openalex_execute_counts_three_buckets():
    """Resolved / no work / failed buckets each get a count."""
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1", "d2", "d3", "d4"),
        skipped_manual=(), skip_manual=True,
    )
    # d1 resolves, d2 no work, d3 raises, d4 resolves.
    side = [
        {"work_resolved": True, "metadata_patched": True,
         "kg_l1_l4_ok": True, "new_status": "enriched"},
        {"work_resolved": False, "metadata_patched": False,
         "kg_l1_l4_ok": False, "new_status": None},
        RuntimeError("boom"),
        {"work_resolved": True, "metadata_patched": True,
         "kg_l1_l4_ok": False, "new_status": "enriched"},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.resolve_openalex",
        side_effect=side,
    ):
        result = bulk_resolve_openalex_execute(plan)

    assert result.n_resolved == 2
    assert result.n_no_work == 1
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"


def test_bulk_resolve_openalex_execute_per_doc_skip_manual_false():
    """Plan-side filter already chose targets; per-doc call must NOT
    re-apply skip_manual or some manual docs would be silently filtered
    twice (once at plan, once at per-doc)."""
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=("d1",), skipped_manual=(), skip_manual=False,
    )
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.resolve_openalex",
        return_value={"work_resolved": True, "new_status": "enriched"},
    ) as ro_mock:
        bulk_resolve_openalex_execute(plan)

    # Bulk MUST call per-doc with skip_manual=False so the plan-side
    # filter is the ONLY skip gate.
    ro_mock.assert_called_once_with("d1", skip_manual=False)


# ---- bulk_re_embed ----


def test_bulk_re_embed_plan_lists_all_indexed_docs():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [
        {"doc_id": "d1", "n_chunks": 10},
        {"doc_id": "d2", "n_chunks": 20},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_re_embed_plan()

    assert set(plan.target_doc_ids) == {"d1", "d2"}
    assert plan.total_chunks == 30


def test_bulk_re_embed_plan_summary_mentions_doc_and_chunk_counts():
    plan = BulkReEmbedPlan(target_doc_ids=("d1", "d2"), total_chunks=50)
    s = plan.summary
    assert "2 docs" in s
    assert "50 chunks" in s


def test_bulk_re_embed_plan_raises_on_lancedb_failure():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.side_effect = RuntimeError("lance boom")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        with pytest.raises(RuntimeError, match="lance boom"):
            bulk_re_embed_plan()


def test_bulk_re_embed_execute_counts_successes_and_failures():
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
        result = bulk_re_embed_execute(plan)

    assert result.n_succeeded == 1
    assert result.n_failed == 2
    failed_ids = {f[0] for f in result.failures}
    assert failed_ids == {"d2", "d3"}


# ---- bulk_backfill_chunks / entities / ontology ----


def test_bulk_backfill_chunks_plan_uses_layer_name_chunks():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [{"doc_id": "d1"}]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_backfill_chunks_plan()
    assert plan.layer_name == "chunks"
    assert "chunks" in plan.summary


def test_bulk_backfill_entities_plan_uses_layer_name_entities():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [{"doc_id": "d1"}]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_backfill_entities_plan()
    assert plan.layer_name == "entities"


def test_bulk_backfill_ontology_plan_uses_layer_name_ontology():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [{"doc_id": "d1"}]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_backfill_ontology_plan()
    assert plan.layer_name == "ontology"


def test_bulk_backfill_chunks_execute_counts_chunks_ok_as_success():
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"), layer_name="chunks",
    )
    side = [
        {"chunks_ok": True, "entities": {}},
        {"chunks_ok": False, "entities": {}},
        RuntimeError("boom"),
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_chunks",
        side_effect=side,
    ):
        result = bulk_backfill_chunks_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 2


def test_bulk_backfill_entities_execute_counts_entities_ok_as_success():
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"), layer_name="entities",
    )
    side = [
        {"entities_ok": True, "n_mentions": 5, "ontology": {}},
        {"entities_ok": False, "n_mentions": 0, "ontology": {}},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_entities",
        side_effect=side,
    ):
        result = bulk_backfill_entities_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1


def test_bulk_backfill_ontology_execute_counts_any_import_ok_as_success():
    """One doc gets MeSH+GO results (one OK), another gets all failures."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"), layer_name="ontology",
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
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_ontology",
        side_effect=side,
    ):
        result = bulk_backfill_ontology_execute(plan, _config())

    # d1 = success (MeSH worked), d2 = fail (nothing imported), d3 = no-op success.
    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d2"


def test_bulk_backfill_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="chunks")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_chunks"
    ):
        result = bulk_backfill_chunks_execute(plan, _config())
    assert isinstance(result, BulkBackfillResult)


# ---- bulk_backfill_triples (L8) ----


def test_bulk_backfill_triples_plan_uses_layer_name_triples():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [{"doc_id": "d1"}]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_backfill_triples_plan()
    assert plan.layer_name == "triples"


def test_bulk_backfill_triples_execute_counts_triples_ok_as_success():
    """triples_ok=True counts as success regardless of n_triples - the
    LLM finding zero qualifying relations is still a clean run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"), layer_name="triples",
    )
    side = [
        {"triples_ok": True, "n_triples": 5},   # wrote 5 edges
        {"triples_ok": True, "n_triples": 0},   # no relations found - still success
        {"triples_ok": False, "n_triples": 0},  # Cypher / LLM failure
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples",
        side_effect=side,
    ):
        result = bulk_backfill_triples_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"


def test_bulk_backfill_triples_execute_catches_per_doc_exceptions():
    """A doc raising mid-iteration counts as a failure, others still run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"), layer_name="triples",
    )
    side = [
        RuntimeError("boom"),
        {"triples_ok": True, "n_triples": 3},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples",
        side_effect=side,
    ):
        result = bulk_backfill_triples_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1
    # The failure carries the doc_id + the repr of the exception.
    assert result.failures[0][0] == "d1"


def test_bulk_backfill_triples_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="triples")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_triples"
    ):
        result = bulk_backfill_triples_execute(plan, _config())
    assert isinstance(result, BulkBackfillResult)


# ---- bulk_backfill_cross_doc (L9) ----


def test_bulk_backfill_cross_doc_plan_uses_layer_name_cross_doc():
    search_mock = MagicMock()
    search_mock.list_indexed_docs.return_value = [{"doc_id": "d1"}]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_search_client",
        return_value=search_mock,
    ):
        plan = bulk_backfill_cross_doc_plan()
    assert plan.layer_name == "cross_doc"


def test_bulk_backfill_cross_doc_execute_counts_cross_doc_ok_as_success():
    """cross_doc_ok=True counts as success regardless of n_edges - a
    doc with no other doc meeting the threshold is still a clean run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2", "d3"), layer_name="cross_doc",
    )
    side = [
        {"cross_doc_ok": True, "n_edges": 5},   # wrote 5 edges
        {"cross_doc_ok": True, "n_edges": 0},   # no overlap met threshold - still success
        {"cross_doc_ok": False, "n_edges": 0},  # Cypher failure
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc",
        side_effect=side,
    ):
        result = bulk_backfill_cross_doc_execute(plan, _config())

    assert result.n_succeeded == 2
    assert result.n_failed == 1
    assert result.failures[0][0] == "d3"


def test_bulk_backfill_cross_doc_execute_catches_per_doc_exceptions():
    """A doc raising mid-iteration counts as a failure, others still run."""
    plan = BulkBackfillPlan(
        target_doc_ids=("d1", "d2"), layer_name="cross_doc",
    )
    side = [
        RuntimeError("boom"),
        {"cross_doc_ok": True, "n_edges": 2},
    ]
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc",
        side_effect=side,
    ):
        result = bulk_backfill_cross_doc_execute(plan, _config())

    assert result.n_succeeded == 1
    assert result.n_failed == 1
    assert result.failures[0][0] == "d1"


def test_bulk_backfill_cross_doc_execute_returns_result_dataclass():
    plan = BulkBackfillPlan(target_doc_ids=(), layer_name="cross_doc")
    with patch(
        "knowledge_agent.ingestion.bulk_ops.pipeline.backfill_cross_doc"
    ):
        result = bulk_backfill_cross_doc_execute(plan, _config())
    assert isinstance(result, BulkBackfillResult)


def test_bulk_re_embed_execute_returns_result_dataclass():
    plan = BulkReEmbedPlan(target_doc_ids=(), total_chunks=0)
    result = bulk_re_embed_execute(plan)
    assert isinstance(result, BulkReEmbedResult)


def test_bulk_resolve_openalex_execute_returns_result_dataclass():
    plan = BulkResolveOpenAlexPlan(
        target_doc_ids=(), skipped_manual=(), skip_manual=True,
    )
    result = bulk_resolve_openalex_execute(plan)
    assert isinstance(result, BulkResolveOpenAlexResult)


# ---- backfill_xrefs (L7 cross-ontology xref resolution) ----


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
    from knowledge_agent.kg.corpus_config import (
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
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(**flags_kwargs),
        entities=EntityConfig(extractor="llm"),
        ontology={"mesh": OntologyConfig(matching="exact")},
        cross_doc_xrefs=(
            CrossDocXrefsConfig(threshold=cross_doc_xrefs_threshold)
            if cross_doc_xrefs else None
        ),
    )


def test_backfill_xrefs_plan_summary_when_layer_off():
    """xrefs="none" -> summary calls out the no-op state."""
    plan = BackfillXrefsPlan(
        xrefs_mode="none",
        n_dangling_sources=0,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    assert "no-op" in plan.summary
    assert '"none"' in plan.summary


def test_backfill_xrefs_plan_summary_use_no_l10():
    """xrefs on, L10 off -> summary describes resolution but not L10."""
    plan = BackfillXrefsPlan(
        xrefs_mode="use",
        n_dangling_sources=42,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    s = plan.summary
    assert "42" in s
    assert "MERGEd" in s or "idempotent" in s
    # No L10 mention when the layer is off.
    assert "RELATED_BY_XREF" not in s


def test_backfill_xrefs_plan_summary_use_plus_l10_mentions_rebuild():
    plan = BackfillXrefsPlan(
        xrefs_mode="use",
        n_dangling_sources=100,
        will_recompute_l10=True,
        l10_threshold=3,
    )
    s = plan.summary
    assert "100" in s
    assert "RELATED_BY_XREF" in s
    assert "threshold=3" in s


def test_backfill_xrefs_plan_aggregates_dangling_across_all_sub_labels():
    """The factory sums `count_dangling_xrefs` per sub-label."""
    kg_mock = MagicMock()
    counts = {"MeSHTerm": 4, "GOTerm": 7, "ChEBITerm": 11}
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=kg_mock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.count_dangling_xrefs",
            side_effect=lambda c, lbl: counts.get(lbl, 0),
        ),
    ):
        plan = backfill_xrefs_plan(_config_xrefs(xrefs_mode="use"))
    assert plan.n_dangling_sources == 4 + 7 + 11
    assert plan.will_recompute_l10 is False
    assert plan.xrefs_mode == "use"


def test_backfill_xrefs_plan_flags_l10_when_layer_on():
    kg_mock = MagicMock()
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=kg_mock,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.count_dangling_xrefs",
            return_value=0,
        ),
    ):
        plan = backfill_xrefs_plan(
            _config_xrefs(
                xrefs_mode="use",
                cross_doc_xrefs=True,
                cross_doc_xrefs_threshold=5,
            )
        )
    assert plan.will_recompute_l10 is True
    assert plan.l10_threshold == 5


def test_backfill_xrefs_execute_skips_when_layer_off():
    """xrefs="none" -> execute returns skipped result, no client calls."""
    plan = BackfillXrefsPlan(
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
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.backfill_resolved_xrefs",
        ) as backfill,
    ):
        result = backfill_xrefs_execute(plan, _config_xrefs("none"))
    assert result.xrefs_layer_skipped is True
    assert result.per_ontology_counts is None
    assert result.l10_attempted is False
    get_client.assert_not_called()
    backfill.assert_not_called()


def test_backfill_xrefs_execute_calls_resolve_when_layer_on():
    plan = BackfillXrefsPlan(
        xrefs_mode="use",
        n_dangling_sources=5,
        will_recompute_l10=False,
        l10_threshold=2,
    )
    fake_counts = {"MeSHTerm": {"n_edges_attempted": 5, "n_sources_cleaned": 3}}
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.backfill_resolved_xrefs",
            return_value=fake_counts,
        ) as backfill,
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
        ) as l10_recompute,
    ):
        result = backfill_xrefs_execute(plan, _config_xrefs("use"))
    backfill.assert_called_once()
    l10_recompute.assert_not_called()
    assert result.xrefs_layer_skipped is False
    assert result.per_ontology_counts == fake_counts
    assert result.l10_attempted is False


def test_backfill_xrefs_execute_calls_l10_recompute_when_layer_on():
    """When `plan.will_recompute_l10 is True`, the L10 global rebuild
    is invoked with the plan's threshold."""
    plan = BackfillXrefsPlan(
        xrefs_mode="use",
        n_dangling_sources=5,
        will_recompute_l10=True,
        l10_threshold=3,
    )
    fake_counts = {"MeSHTerm": {"n_edges_attempted": 5, "n_sources_cleaned": 3}}
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.backfill_resolved_xrefs",
            return_value=fake_counts,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "cross_doc_xrefs_writes.recompute_cross_doc_xrefs_global",
            return_value=42,
        ) as l10_recompute,
    ):
        result = backfill_xrefs_execute(
            plan, _config_xrefs("use", cross_doc_xrefs=True),
        )
    l10_recompute.assert_called_once()
    # Verify threshold flowed through positionally.
    args, _ = l10_recompute.call_args
    assert args[1] == 3
    assert result.l10_attempted is True
    assert result.n_l10_edges_written == 42


# ---- recompute_cross_doc_xrefs (standalone L10 rebuild) ----


def test_recompute_cross_doc_xrefs_plan_summary_layer_off():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=False, n_existing_l10_edges=0, threshold=2,
    )
    assert "off" in plan.summary
    assert "no-op" in plan.summary


def test_recompute_cross_doc_xrefs_plan_summary_layer_on():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=True, n_existing_l10_edges=17, threshold=5,
    )
    s = plan.summary
    assert "17" in s
    assert "threshold=5" in s
    assert "wiped and rewritten" in s


def test_recompute_cross_doc_xrefs_plan_layer_off_skips_edge_count():
    """Layer off -> plan still builds, but n_existing_l10_edges defaults
    to 0 (no Cypher query fired)."""
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_kg_client",
    ) as get_client:
        plan = recompute_cross_doc_xrefs_plan(_config_xrefs("use"))
    assert plan.enabled is False
    assert plan.n_existing_l10_edges == 0
    get_client.assert_not_called()


def test_recompute_cross_doc_xrefs_plan_layer_on_queries_existing_count():
    """Layer on -> factory queries live L10 edge count via session."""
    kg_mock = MagicMock()
    sess = MagicMock()
    sess.__enter__ = MagicMock(return_value=sess)
    sess.__exit__ = MagicMock(return_value=None)
    fake_result = MagicMock()
    fake_result.single.return_value = {"n": 25}
    sess.run.return_value = fake_result
    kg_mock.driver.session.return_value = sess
    with patch(
        "knowledge_agent.ingestion.bulk_ops.get_kg_client",
        return_value=kg_mock,
    ):
        plan = recompute_cross_doc_xrefs_plan(
            _config_xrefs("use", cross_doc_xrefs=True, cross_doc_xrefs_threshold=4),
        )
    assert plan.enabled is True
    assert plan.n_existing_l10_edges == 25
    assert plan.threshold == 4


def test_recompute_cross_doc_xrefs_execute_skipped_when_layer_off():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=False, n_existing_l10_edges=0, threshold=2,
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
        result = recompute_cross_doc_xrefs_execute(plan)
    assert result.layer_skipped is True
    assert result.n_edges_written is None
    get_client.assert_not_called()
    l10_recompute.assert_not_called()


def test_recompute_cross_doc_xrefs_execute_calls_global_recompute_when_on():
    plan = RecomputeCrossDocXrefsPlan(
        enabled=True, n_existing_l10_edges=12, threshold=3,
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
        result = recompute_cross_doc_xrefs_execute(plan)
    l10_recompute.assert_called_once()
    args, _ = l10_recompute.call_args
    assert args[1] == 3  # threshold positional
    assert result.layer_skipped is False
    assert result.n_edges_written == 42


# ---- clear_xref_edges (per-ontology xref wipe) ----


def test_clear_xref_edges_plan_unknown_ontology_raises():
    with pytest.raises(ValueError):
        clear_xref_edges_plan("not-a-real-ontology")


def test_clear_xref_edges_plan_carries_counts_and_term_label():
    """Plan captures both edge + dangling counts so execute can run
    deterministically without re-querying."""
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.count_xref_edges",
            return_value=12,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.count_dangling_xrefs",
            return_value=7,
        ),
    ):
        plan = clear_xref_edges_plan("mesh")
    assert plan.ontology_name == "mesh"
    assert plan.term_label == "MeSHTerm"
    assert plan.n_existing_edges == 12
    assert plan.n_dangling_sources == 7


def test_clear_xref_edges_plan_summary_mentions_outbound_only():
    """The summary explicitly states inbound xrefs are left alone."""
    plan = ClearXrefEdgesPlan(
        ontology_name="mesh",
        term_label="MeSHTerm",
        n_existing_edges=10,
        n_dangling_sources=4,
    )
    s = plan.summary
    assert "outgoing" in s
    assert "INBOUND" in s
    assert "10" in s
    assert "4" in s


def test_clear_xref_edges_execute_delegates_to_kg_helper():
    """Execute calls `ontology_xrefs.clear_xref_edges_for_ontology` with
    the plan's term_label and returns the sum."""
    plan = ClearXrefEdgesPlan(
        ontology_name="mesh",
        term_label="MeSHTerm",
        n_existing_edges=10,
        n_dangling_sources=4,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.clear_xref_edges_for_ontology",
            return_value=14,
        ) as clear_fn,
    ):
        result = clear_xref_edges_execute(plan)
    clear_fn.assert_called_once()
    args, _ = clear_fn.call_args
    assert args[1] == "MeSHTerm"
    assert result.ontology_name == "mesh"
    assert result.n_cleared == 14


def test_clear_xref_edges_execute_fail_soft_when_helper_returns_none():
    """Helper returns None on Cypher error -> result.n_cleared is None."""
    plan = ClearXrefEdgesPlan(
        ontology_name="mesh",
        term_label="MeSHTerm",
        n_existing_edges=0,
        n_dangling_sources=0,
    )
    with (
        patch(
            "knowledge_agent.ingestion.bulk_ops.get_kg_client",
            return_value=MagicMock(),
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops."
            "ontology_xrefs.clear_xref_edges_for_ontology",
            return_value=None,
        ),
    ):
        result = clear_xref_edges_execute(plan)
    assert result.n_cleared is None
    assert result.ontology_name == "mesh"
