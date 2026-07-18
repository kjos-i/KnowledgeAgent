"""Tests for the JSON / JSONL / NDJSON parser strategy.

Covers extension routing, shape-based chunking (array vs object vs
scalar), blank-line skipping in line-delimited variants, chunk_index
monotonicity, and error pass-through on malformed JSON.
"""

import json
from pathlib import Path

import pytest

from knowledge_agent.ingestion.parsers import json_parser

# ---- EXTENSIONS ----


def test_extensions_covers_three_variants():
    assert set(json_parser.EXTENSIONS) == {"json", "jsonl", "ndjson"}


# ---- .json: top-level array -> per-element ----


def test_json_array_emits_one_chunk_per_element(tmp_path: Path):
    payload = [
        {"id": "C001", "name": "Aspirin"},
        {"id": "C002", "name": "Caffeine"},
        {"id": "C003", "name": "Glucose"},
    ]
    path = tmp_path / "compounds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    chunks = json_parser.parse(path)
    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    # text is canonical JSON re-serialization, not Python repr
    assert json.loads(chunks[0].text) == {"id": "C001", "name": "Aspirin"}
    assert json.loads(chunks[2].text) == {"id": "C003", "name": "Glucose"}


def test_json_array_chunks_all_marked_json_object(tmp_path: Path):
    path = tmp_path / "arr.json"
    path.write_text(json.dumps([{"a": 1}, {"b": 2}]), encoding="utf-8")
    chunks = json_parser.parse(path)
    assert all(c.content_type == "json_object" for c in chunks)


def test_json_array_handles_unicode_without_escaping(tmp_path: Path):
    """`ensure_ascii=False` keeps non-ASCII chars intact in chunk text."""
    payload = [{"name": "ångström"}]
    path = tmp_path / "unicode.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    chunks = json_parser.parse(path)
    assert "ångström" in chunks[0].text
    assert "\\u00e5" not in chunks[0].text


def test_json_with_utf8_bom_parses(tmp_path: Path):
    """A .json file saved with a UTF-8 BOM (common from Windows editors) must
    parse. Regression: reading as plain utf-8 left the BOM as a leading
    char and json.loads crashed on it."""
    payload = [{"id": "C001", "name": "Aspirin"}]
    path = tmp_path / "bom.json"
    path.write_text(json.dumps(payload), encoding="utf-8-sig")  # writes a leading BOM
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # sanity: BOM present
    chunks = json_parser.parse(path)
    assert len(chunks) == 1
    assert json.loads(chunks[0].text) == {"id": "C001", "name": "Aspirin"}


def test_jsonl_with_utf8_bom_parses(tmp_path: Path):
    """A .jsonl file with a UTF-8 BOM must parse — the BOM must not corrupt the
    first line's JSON object (its raw text becomes the chunk)."""
    path = tmp_path / "bom.jsonl"
    path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8-sig")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    chunks = json_parser.parse(path)
    assert len(chunks) == 2
    assert json.loads(chunks[0].text) == {"a": 1}  # no BOM leaked into the text


def test_json_empty_array_emits_no_chunks(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    chunks = json_parser.parse(path)
    assert chunks == []


# ---- .json: top-level object -> whole-doc-as-chunk ----


def test_json_object_emits_one_chunk_with_full_text(tmp_path: Path):
    payload = {
        "experiment_id": "EXP-2026-042",
        "researcher": {"name": "Dr. Lee", "affiliation": "UiO"},
    }
    raw = json.dumps(payload, indent=2)
    path = tmp_path / "experiment.json"
    path.write_text(raw, encoding="utf-8")

    chunks = json_parser.parse(path)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    # whole-doc preserves original text (incl. indentation) verbatim
    assert chunks[0].text == raw
    assert chunks[0].content_type == "json_object"


def test_json_scalar_top_level_treated_as_whole_doc(tmp_path: Path):
    """Edge case: top level is a string / number / bool. Single chunk."""
    path = tmp_path / "scalar.json"
    path.write_text('"just a string"', encoding="utf-8")
    chunks = json_parser.parse(path)
    assert len(chunks) == 1
    assert chunks[0].text == '"just a string"'


# ---- .jsonl / .ndjson: per-line ----


def test_jsonl_emits_one_chunk_per_line(tmp_path: Path):
    lines = [
        '{"patient_id": "P1", "note": "fatigue"}',
        '{"patient_id": "P2", "note": "improvement"}',
        '{"patient_id": "P3", "note": "new symptoms"}',
    ]
    path = tmp_path / "notes.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chunks = json_parser.parse(path)
    assert len(chunks) == 3
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    # raw line text is preserved (no re-serialization)
    assert chunks[0].text == lines[0]
    assert chunks[2].text == lines[2]


def test_ndjson_extension_uses_same_path(tmp_path: Path):
    path = tmp_path / "data.ndjson"
    path.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    chunks = json_parser.parse(path)
    assert len(chunks) == 2


def test_jsonl_skips_blank_lines_without_gaps(tmp_path: Path):
    path = tmp_path / "sparse.jsonl"
    path.write_text(
        '{"a": 1}\n\n   \n{"a": 2}\n\n{"a": 3}\n',
        encoding="utf-8",
    )
    chunks = json_parser.parse(path)
    assert len(chunks) == 3
    # chunk_index must be contiguous - no holes from skipped blanks
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_jsonl_chunks_marked_json_object(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"k": "v"}\n', encoding="utf-8")
    chunks = json_parser.parse(path)
    assert chunks[0].content_type == "json_object"


# ---- malformed input pass-through ----


def test_malformed_json_raises_through_to_caller(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json_parser.parse(path)


# ---- dispatcher integration ----


def _dispatcher_config():
    """CorpusConfig instance whose fields the json parser ignores."""
    from knowledge_agent.kg.corpus_config import CorpusConfig, LayerFlags

    return CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True),
    )


def test_dispatcher_routes_json_to_json_parser(tmp_path: Path):
    """Verify the top-level dispatcher picks json_parser for .json files."""
    from knowledge_agent.ingestion.parse import parse_document

    path = tmp_path / "x.json"
    path.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    chunks = parse_document(path, _dispatcher_config())
    assert len(chunks) == 2
    assert all(c.content_type == "json_object" for c in chunks)


def test_dispatcher_routes_jsonl_to_json_parser(tmp_path: Path):
    from knowledge_agent.ingestion.parse import parse_document

    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    chunks = parse_document(path, _dispatcher_config())
    assert len(chunks) == 3


def test_supported_extensions_now_includes_json_family():
    from knowledge_agent.ingestion.parse import supported_extensions

    exts = supported_extensions()
    assert {"json", "jsonl", "ndjson"}.issubset(exts)
