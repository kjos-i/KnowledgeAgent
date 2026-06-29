"""Stable identifiers - single source of truth for `doc_id` and `chunk_id`.

`doc_id` is the SHA-256 hex digest of the raw file bytes: file-level
identity that is rename-stable (renaming the file doesn't change its ID)
and gives free exact-duplicate detection (two paths with the same bytes
yield the same doc_id, so the ingestion pipeline can skip the second).

`chunk_id` is `{doc_id}{SEP}{index}` where `SEP` is a single character
declared as a constant here. Compute IDs only through these helpers -
never inline - so any change to the separator or hashing scheme is a
single-file edit.

The hashing is streamed in 1 MiB blocks so files of any size process
without loading the whole thing into memory.
"""

import hashlib
from pathlib import Path

# Block size for streamed hashing. 1 MiB is the conventional middle
# ground: small enough that very small files don't waste memory, large
# enough that the per-block overhead is negligible for large files.
_READ_BLOCK = 1 << 20

# Separator between doc_id and chunk index in a chunk_id. Single source of
# truth - never duplicate this character literal anywhere else.
CHUNK_ID_SEPARATOR = "#"


def compute_doc_id(path: Path) -> str:
    """SHA-256 hex digest of the raw file bytes - the document's `doc_id`.

    Streams the file in 1 MiB blocks, so this works for files of any size
    without loading them fully into memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_READ_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def make_chunk_id(doc_id: str, index: int) -> str:
    """Deterministic chunk id: `{doc_id}{SEP}{index}`.

    Re-running the ingestion on the same file with the same chunker config
    produces the same chunk_ids - that's what makes the LanceDB and KG
    writes idempotent at the row level.
    """
    return f"{doc_id}{CHUNK_ID_SEPARATOR}{index}"
