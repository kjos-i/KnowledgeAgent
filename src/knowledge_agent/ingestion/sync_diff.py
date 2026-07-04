"""Pure-function 5-bucket diff classifier used by `bulk_ops.sync`.

Sync compares files on disk vs docs in the index and classifies into:

  - NEW       : file on disk, doc_id NOT in index            -> ingest
  - UNCHANGED : same doc_id + same path                       -> no-op
  - MOVED     : same doc_id, different path                   -> update stored path
  - EDITED    : same path, different doc_id                   -> delete old + ingest new
  - ORPHAN    : doc_id in index but doesn't match any disk file

Sync deletes orphans with explicit confirmation; Add ignores them.

This module is deliberately a pure function over plain dataclasses so
the algorithm is testable without any I/O setup. Callers in
`bulk_ops.sync_plan` wrap it with the file-walking + LanceDB
listing.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiskFile:
    """One file currently on disk in the user-selected folder.

    `doc_id` is the SHA-256 of the file's bytes (computed by the caller
    via `compute_doc_id`). `path` is the absolute path to the file as
    it exists on disk right now.
    """

    path: Path
    doc_id: str


@dataclass(frozen=True)
class IndexedDoc:
    """One doc currently in the index (LanceDB).

    `stored_path` is the path that was on the doc's row at last ingest -
    used to detect MOVED (same hash, new path) and EDITED (same path,
    new hash). It is allowed to be None (a doc indexed before the
    `source_path` column existed, or one inserted by a partial op that
    didn't have a path).
    """

    doc_id: str
    stored_path: str | None
    title: str | None
    metadata_status: str | None
    n_chunks: int


@dataclass(frozen=True)
class SyncBuckets:
    """Result of `classify`. Each tuple is processed independently by execute.

    `moved` and `edited` carry pairs `(disk_file, indexed_doc)` because
    execute needs both the new state AND the old state - MOVED to know
    which stored row to patch, EDITED to know which doc_id to delete
    before ingesting the new content.
    """

    new: tuple[DiskFile, ...]
    unchanged: tuple[DiskFile, ...]
    moved: tuple[tuple[DiskFile, IndexedDoc], ...]
    edited: tuple[tuple[DiskFile, IndexedDoc], ...]
    orphan: tuple[IndexedDoc, ...]


def classify(
    disk_files: list[DiskFile],
    indexed_docs: list[IndexedDoc],
) -> SyncBuckets:
    """Bucket disk files + indexed docs into the 5 sync categories.

    Algorithm (each disk file walked once, each indexed doc consumed at most once):

    For every disk file:
      1. doc_id match in the index -> either UNCHANGED (paths equal) or
         MOVED (paths differ). The indexed doc is consumed.
      2. No doc_id match -> try a path match (an indexed doc whose
         `stored_path` equals this disk file's path). If found, that's
         EDITED (content changed -> new hash, same path); the indexed
         doc is consumed.
      3. No doc_id or path match -> NEW.

    After all disk files are processed, any indexed doc still
    unconsumed is an ORPHAN (in index, no matching disk file by hash
    or by path).

    Notes:
      - Path comparison uses `path.as_posix()` so forward slashes are
        the wire format on every OS. The pipeline stores paths via
        `as_posix()` too, so stored vs current always compare cleanly.
      - Cross-platform case sensitivity (Windows vs Linux paths) is
        NOT normalized here.
      - EDITED is path-based; if the user moved a file AND edited it,
        the old `doc_id` will appear as ORPHAN (path no longer points
        at the same content) and the new content gets bucketed as
        NEW. That asymmetry is acceptable - a rare case, and ORPHAN
        confirmation still surfaces it to the user.
      - An indexed doc with `stored_path=None` can only match by
        doc_id - it can never trigger EDITED (no path to match against).

    Pure function: no side effects, no I/O.
    """
    by_doc_id: dict[str, IndexedDoc] = {d.doc_id: d for d in indexed_docs}
    by_path: dict[str, IndexedDoc] = {
        d.stored_path: d for d in indexed_docs if d.stored_path is not None
    }
    consumed: set[str] = set()

    new_bucket: list[DiskFile] = []
    unchanged_bucket: list[DiskFile] = []
    moved_bucket: list[tuple[DiskFile, IndexedDoc]] = []
    edited_bucket: list[tuple[DiskFile, IndexedDoc]] = []

    for disk in disk_files:
        disk_path_str = disk.path.as_posix()
        # Tier 1: doc_id match. Content already in the index somewhere.
        idx = by_doc_id.get(disk.doc_id)
        if idx is not None:
            consumed.add(idx.doc_id)
            if idx.stored_path == disk_path_str:
                unchanged_bucket.append(disk)
            else:
                moved_bucket.append((disk, idx))
            continue

        # Tier 2: path match against a DIFFERENT doc_id - file was edited.
        idx_by_path = by_path.get(disk_path_str)
        if idx_by_path is not None and idx_by_path.doc_id not in consumed:
            consumed.add(idx_by_path.doc_id)
            edited_bucket.append((disk, idx_by_path))
            continue

        # Tier 3: brand new.
        new_bucket.append(disk)

    orphan_bucket: list[IndexedDoc] = [d for d in indexed_docs if d.doc_id not in consumed]

    return SyncBuckets(
        new=tuple(new_bucket),
        unchanged=tuple(unchanged_bucket),
        moved=tuple(moved_bucket),
        edited=tuple(edited_bucket),
        orphan=tuple(orphan_bucket),
    )
