"""Integration tests for `ingestion/parsers/json_parser` — real
JSON / JSONL / NDJSON parsing.

Unit tests in `tests/unit/ingestion/test_json_parser.py` cover the
parsing logic with constructed inputs. This file exercises the same
paths but written to disk first, catching path-handling +
file-encoding regressions.

No external deps — uses stdlib `json`. No `pytest.mark.slow` (these
tests are fast).

Manual interactive counterpart: none. The `.jsonl` / `.json` paths
are exercised by any pipeline ingest of a JSON file.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from knowledge_agent.ingestion.parsers import json_parser

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


def test_jsonl_file_yields_one_chunk_per_non_blank_line(tmp_path: Path) -> None:
    """JSONL → one chunk per line. Blank lines are skipped."""
    src = tmp_path / "data.jsonl"
    src.write_text(
        '{"id": 1, "text": "first"}\n'
        '{"id": 2, "text": "second"}\n'
        "\n"  # blank line — should be skipped
        '{"id": 3, "text": "third"}\n'
    )
    chunks = json_parser.parse(src)
    assert len(chunks) == 3
    assert all(c.content_type == "json_object" for c in chunks)
    # Each chunk text is parsable JSON.
    for c in chunks:
        json.loads(c.text)


def test_ndjson_extension_same_path_as_jsonl(tmp_path: Path) -> None:
    """`.ndjson` routes through the same line-delimited path as
    `.jsonl`."""
    src = tmp_path / "data.ndjson"
    src.write_text('{"a": 1}\n{"a": 2}\n')
    chunks = json_parser.parse(src)
    assert len(chunks) == 2


def test_json_array_yields_one_chunk_per_element(tmp_path: Path) -> None:
    """`.json` whose top-level is a list → one chunk per element."""
    src = tmp_path / "list.json"
    src.write_text(
        json.dumps(
            [
                {"id": 1, "text": "one"},
                {"id": 2, "text": "two"},
                {"id": 3, "text": "three"},
            ]
        )
    )
    chunks = json_parser.parse(src)
    assert len(chunks) == 3
    for c in chunks:
        json.loads(c.text)


def test_json_object_yields_single_chunk_containing_whole_doc(
    tmp_path: Path,
) -> None:
    """`.json` whose top-level is an object → one chunk containing
    the whole document as a single JSON value."""
    src = tmp_path / "obj.json"
    payload = {"title": "doc", "body": "hello"}
    src.write_text(json.dumps(payload))
    chunks = json_parser.parse(src)
    assert len(chunks) == 1
    assert json.loads(chunks[0].text) == payload


def test_json_scalar_at_top_level_yields_single_chunk(tmp_path: Path) -> None:
    """`.json` whose top-level is a scalar (string / number / null) →
    single chunk containing the scalar."""
    src = tmp_path / "scalar.json"
    src.write_text(json.dumps("just a string"))
    chunks = json_parser.parse(src)
    assert len(chunks) == 1
    assert json.loads(chunks[0].text) == "just a string"


def test_malformed_json_raises(tmp_path: Path) -> None:
    """Per the dispatcher contract, parse failures are the caller's
    problem — JSONDecodeError bubbles up rather than being swallowed."""
    src = tmp_path / "bad.json"
    src.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        json_parser.parse(src)


def test_chunk_indices_are_sequential_in_jsonl(tmp_path: Path) -> None:
    """chunk_index is 0-based + sequential for downstream
    make_chunk_id() round-trips."""
    src = tmp_path / "data.jsonl"
    src.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n')
    chunks = json_parser.parse(src)
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
