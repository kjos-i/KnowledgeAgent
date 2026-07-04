"""Tests for ingestion.ids - compute_doc_id, make_chunk_id.

Two test styles in this file:

  - Example-based tests at the top (the original suite) — pin
    specific known cases (hash of "hello world", empty-file SHA,
    cross-block streaming).
  - Property-based tests at the bottom (via `hypothesis`) — generate
    arbitrary inputs and assert invariants (output is hex of length
    64; chunk_id round-trips; different content → different ids).

Property-based tests catch edge cases the hand-picked examples
miss; example-based tests pin known-quantity behaviour. Both styles
serve the same module — keeping them in one file keeps the 1:1
mirror with `src/knowledge_agent/ingestion/ids.py`.
"""

import hashlib
import re
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from knowledge_agent.ingestion.ids import (
    CHUNK_ID_SEPARATOR,
    compute_doc_id,
    make_chunk_id,
)


def test_compute_doc_id_matches_sha256_of_file_contents(tmp_path: Path):
    """compute_doc_id returns the SHA-256 hex digest of the file bytes."""
    content = b"hello world"
    path = tmp_path / "test.txt"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert compute_doc_id(path) == expected


def test_compute_doc_id_same_content_same_id(tmp_path: Path):
    """Identical contents -> identical doc_id, regardless of filename or path."""
    content = b"some test bytes"
    a = tmp_path / "a.pdf"
    b = tmp_path / "subdir" / "b.txt"
    b.parent.mkdir()
    a.write_bytes(content)
    b.write_bytes(content)
    assert compute_doc_id(a) == compute_doc_id(b)


def test_compute_doc_id_different_content_different_id(tmp_path: Path):
    """A single-byte change produces a completely different doc_id."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"AAA")
    b.write_bytes(b"AAB")
    assert compute_doc_id(a) != compute_doc_id(b)


def test_compute_doc_id_empty_file(tmp_path: Path):
    """Empty files hash to the SHA-256 of empty bytes (known constant)."""
    path = tmp_path / "empty.txt"
    path.write_bytes(b"")
    assert (
        compute_doc_id(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_compute_doc_id_large_file_streams(tmp_path: Path):
    """Files larger than the read block process correctly (streamed hashing)."""
    # 3 MiB - bigger than the 1 MiB _READ_BLOCK, ensures the loop iterates.
    content = b"x" * (3 << 20)
    path = tmp_path / "big.bin"
    path.write_bytes(content)
    assert compute_doc_id(path) == hashlib.sha256(content).hexdigest()


def test_make_chunk_id_format():
    assert make_chunk_id("abc123", 0) == f"abc123{CHUNK_ID_SEPARATOR}0"
    assert make_chunk_id("abc123", 42) == f"abc123{CHUNK_ID_SEPARATOR}42"


def test_make_chunk_id_uses_separator_constant():
    """The format uses CHUNK_ID_SEPARATOR - one source of truth."""
    chunk_id = make_chunk_id("doc", 1)
    assert CHUNK_ID_SEPARATOR in chunk_id


# ---------------------------------------------------------------------------
# Property-based tests (hypothesis).
#
# Generated inputs catch edge cases the hand-picked examples miss:
# pathological byte sequences, large or zero chunk indices, etc.
# ---------------------------------------------------------------------------


_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


@given(content=st.binary(min_size=0, max_size=4096))
def test_compute_doc_id_always_returns_64_char_lowercase_hex(
    content: bytes,
) -> None:
    """compute_doc_id is hex-encoded SHA-256 — always 64 lowercase
    hex chars regardless of input bytes.

    Uses `tempfile.TemporaryDirectory()` inline rather than the
    `tmp_path` fixture: hypothesis runs `@given` many times per
    test invocation, and pytest fixtures don't reset between cases.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "x.bin"
        path.write_bytes(content)
        out = compute_doc_id(path)
    assert _HEX64.fullmatch(out) is not None


@given(content=st.binary(min_size=0, max_size=4096))
def test_compute_doc_id_deterministic(content: bytes) -> None:
    """Two calls on the same content give the same id (catches any
    accidental state leak inside the hasher)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.bin"
        b = Path(tmpdir) / "b.bin"
        a.write_bytes(content)
        b.write_bytes(content)
        assert compute_doc_id(a) == compute_doc_id(b)


@given(
    content_a=st.binary(min_size=0, max_size=512),
    content_b=st.binary(min_size=0, max_size=512),
)
def test_compute_doc_id_distinguishes_different_content(
    content_a: bytes,
    content_b: bytes,
) -> None:
    """Different bytes (whenever they are different) must produce
    different ids. Equal bytes must produce equal ids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.bin"
        b = Path(tmpdir) / "b.bin"
        a.write_bytes(content_a)
        b.write_bytes(content_b)
        if content_a == content_b:
            assert compute_doc_id(a) == compute_doc_id(b)
        else:
            assert compute_doc_id(a) != compute_doc_id(b)


@given(
    doc_id=st.text(
        alphabet=st.characters(
            min_codepoint=ord("0"),
            max_codepoint=ord("z"),
            blacklist_categories=("Cs",),
            blacklist_characters=(CHUNK_ID_SEPARATOR,),
        ),
        min_size=1,
        max_size=64,
    ),
    index=st.integers(min_value=0, max_value=10**6),
)
def test_make_chunk_id_format_invariants(doc_id: str, index: int) -> None:
    """For any valid doc_id (no separator inside) + any non-negative
    index, the chunk_id is `{doc_id}{SEP}{index}` exactly."""
    chunk_id = make_chunk_id(doc_id, index)
    assert chunk_id == f"{doc_id}{CHUNK_ID_SEPARATOR}{index}"
    # Round-trip: split on the separator should recover (doc_id, index_str).
    head, _, tail = chunk_id.rpartition(CHUNK_ID_SEPARATOR)
    assert head == doc_id
    assert tail == str(index)


@given(
    doc_id=st.text(
        alphabet=st.characters(
            min_codepoint=ord("0"),
            max_codepoint=ord("z"),
            blacklist_categories=("Cs",),
            blacklist_characters=(CHUNK_ID_SEPARATOR,),
        ),
        min_size=1,
        max_size=32,
    ),
    a=st.integers(min_value=0, max_value=10**4),
    b=st.integers(min_value=0, max_value=10**4),
)
def test_make_chunk_id_distinct_indices_distinct_chunk_ids(doc_id: str, a: int, b: int) -> None:
    """For a fixed doc_id, distinct indices produce distinct
    chunk_ids (injectivity in the index axis — the LanceDB/KG
    idempotency contract relies on this)."""
    if a != b:
        assert make_chunk_id(doc_id, a) != make_chunk_id(doc_id, b)
    else:
        assert make_chunk_id(doc_id, a) == make_chunk_id(doc_id, b)
