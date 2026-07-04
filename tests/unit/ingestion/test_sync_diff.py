"""Tests for ingestion.sync_diff.classify - pure 5-bucket classifier."""

from pathlib import Path

from knowledge_agent.ingestion.sync_diff import (
    DiskFile,
    IndexedDoc,
    SyncBuckets,
    classify,
)


def _disk(path: str, doc_id: str) -> DiskFile:
    return DiskFile(path=Path(path), doc_id=doc_id)


def _idx(
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


# ---- empty inputs ----


def test_classify_empty_inputs_returns_empty_buckets():
    result = classify([], [])
    assert result.new == ()
    assert result.unchanged == ()
    assert result.moved == ()
    assert result.edited == ()
    assert result.orphan == ()


def test_classify_returns_typed_buckets():
    result = classify([], [])
    assert isinstance(result, SyncBuckets)


# ---- NEW bucket ----


def test_classify_new_file_no_index_match():
    """File on disk, nothing in index -> NEW."""
    disk = [_disk("/data/a.pdf", "doc-A")]
    result = classify(disk, [])
    assert result.new == (disk[0],)
    assert result.unchanged == ()


def test_classify_new_file_index_has_others_unrelated():
    """File on disk with hash X, index has Y at a different path -> NEW
    and the index entry becomes ORPHAN."""
    disk = [_disk("/data/a.pdf", "doc-A")]
    indexed = [_idx("doc-Y", stored_path="/old/y.pdf")]
    result = classify(disk, indexed)
    assert result.new == (disk[0],)
    assert result.orphan == (indexed[0],)


# ---- UNCHANGED bucket ----


def test_classify_unchanged_when_doc_id_and_path_match():
    disk = [_disk("/data/a.pdf", "doc-A")]
    indexed = [_idx("doc-A", stored_path="/data/a.pdf")]
    result = classify(disk, indexed)
    assert result.unchanged == (disk[0],)
    assert result.new == ()
    assert result.moved == ()
    assert result.orphan == ()


# ---- MOVED bucket ----


def test_classify_moved_when_doc_id_matches_but_path_differs():
    disk = [_disk("/new/a.pdf", "doc-A")]
    indexed = [_idx("doc-A", stored_path="/old/a.pdf")]
    result = classify(disk, indexed)
    assert result.moved == ((disk[0], indexed[0]),)
    assert result.new == ()
    assert result.unchanged == ()
    assert result.orphan == ()


def test_classify_moved_when_indexed_stored_path_is_none():
    """An indexed doc with no stored_path can still be MOVED if doc_id matches."""
    disk = [_disk("/anywhere/a.pdf", "doc-A")]
    indexed = [_idx("doc-A", stored_path=None)]
    result = classify(disk, indexed)
    assert result.moved == ((disk[0], indexed[0]),)


# ---- EDITED bucket ----


def test_classify_edited_when_path_matches_but_hash_differs():
    """File at path X has new content (new hash) -> EDITED with the OLD
    indexed doc (different doc_id) so execute can delete the old."""
    disk = [_disk("/data/a.pdf", "doc-A-new")]
    indexed = [_idx("doc-A-old", stored_path="/data/a.pdf")]
    result = classify(disk, indexed)
    assert result.edited == ((disk[0], indexed[0]),)
    assert result.new == ()
    assert result.orphan == ()


def test_classify_edited_does_not_double_consume_indexed_doc():
    """If two disk files have the same path (impossible on real FS but
    asserting the algorithm's once-only consumption guarantee), only the
    first triggers EDITED; the second falls through to NEW."""
    disk = [
        _disk("/data/a.pdf", "doc-X"),
        _disk("/data/a.pdf", "doc-Y"),  # same path, different hash
    ]
    indexed = [_idx("doc-original", stored_path="/data/a.pdf")]
    result = classify(disk, indexed)
    # First disk file consumes the indexed doc via path match.
    assert len(result.edited) == 1
    assert result.edited[0] == (disk[0], indexed[0])
    # Second disk file with same path falls through to NEW.
    assert disk[1] in result.new


# ---- ORPHAN bucket ----


def test_classify_orphan_when_indexed_doc_has_no_matching_disk_file():
    """In index, not on disk by hash or path -> ORPHAN."""
    indexed = [_idx("doc-G", stored_path="/data/g.pdf")]
    result = classify([], indexed)
    assert result.orphan == (indexed[0],)


def test_classify_multiple_orphans_when_nothing_on_disk():
    indexed = [
        _idx("doc-1", stored_path="/x/1.pdf"),
        _idx("doc-2", stored_path="/x/2.pdf"),
        _idx("doc-3", stored_path="/x/3.pdf"),
    ]
    result = classify([], indexed)
    assert set(result.orphan) == set(indexed)


# ---- combined scenarios ----


def test_classify_mixed_scenario_all_buckets_populated():
    """One disk file in each non-empty bucket - sanity check the full algorithm."""
    disk = [
        _disk("/data/clean.pdf", "doc-clean"),  # UNCHANGED
        _disk("/new/moved.pdf", "doc-moved"),  # MOVED
        _disk("/data/edited.pdf", "doc-edited-new"),  # EDITED
        _disk("/data/brand-new.pdf", "doc-brand-new"),  # NEW
    ]
    indexed = [
        _idx("doc-clean", stored_path="/data/clean.pdf"),
        _idx("doc-moved", stored_path="/old/moved.pdf"),
        _idx("doc-edited-old", stored_path="/data/edited.pdf"),
        _idx("doc-orphan", stored_path="/data/orphan.pdf"),
    ]
    result = classify(disk, indexed)

    assert result.unchanged == (disk[0],)
    assert result.moved == ((disk[1], indexed[1]),)
    assert result.edited == ((disk[2], indexed[2]),)
    assert result.new == (disk[3],)
    assert len(result.orphan) == 1
    assert result.orphan[0].doc_id == "doc-orphan"


def test_classify_doc_id_match_takes_precedence_over_path_match():
    """A disk file whose doc_id is in the index gets UNCHANGED/MOVED even
    if its path also happens to match a DIFFERENT indexed doc."""
    disk = [_disk("/data/x.pdf", "doc-A")]
    indexed = [
        _idx("doc-A", stored_path="/different/path.pdf"),  # doc_id match -> MOVED
        _idx("doc-B", stored_path="/data/x.pdf"),  # path match (irrelevant now)
    ]
    result = classify(disk, indexed)

    # Tier 1 wins: doc-A becomes MOVED (doc_id match).
    assert result.moved == ((disk[0], indexed[0]),)
    # doc-B's path match is no longer relevant since the disk file is
    # already classified. doc-B becomes ORPHAN.
    assert indexed[1] in result.orphan


def test_classify_does_not_mutate_inputs():
    """classify is a pure function - input lists must be unchanged."""
    disk = [_disk("/a.pdf", "doc-A")]
    indexed = [_idx("doc-A", stored_path="/a.pdf")]
    disk_snapshot = list(disk)
    indexed_snapshot = list(indexed)
    classify(disk, indexed)
    assert disk == disk_snapshot
    assert indexed == indexed_snapshot
