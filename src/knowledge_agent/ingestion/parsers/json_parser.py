"""JSON / JSONL / NDJSON parser strategy.

Generic JSON data (NOT docling's own `DoclingDocument` schema, which is
JSON-serialized intermediate state for round-tripping docling output -
that variant is intentionally not registered here so user `.json` data
files route to this parser).

Chunking is dispatched by extension + top-level shape (locked in the
Phase 4 design):

  - `.jsonl` / `.ndjson` -> one chunk per non-blank line. Each line is
    already a complete JSON object; we hand it through as-is without
    re-serializing.
  - `.json` whose top level is a list -> one chunk per element.
  - `.json` whose top level is anything else (object, scalar) -> whole
    document as one chunk.

`content_type` is "json_object" on every emitted chunk so downstream can
distinguish JSON-derived chunks from docling-parsed prose, code, etc.

Malformed JSON bubbles up as `json.JSONDecodeError` - per the dispatcher
contract, parse failures are the caller's problem.

No external deps; uses stdlib `json`. Huge files (>100 MB) would benefit
from `ijson` streaming; deferred until a real case appears.
"""

import json
import logging
from pathlib import Path

from knowledge_agent.ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

# Parser dispatcher contract.
EXTENSIONS: tuple[str, ...] = ("json", "jsonl", "ndjson")

_LINE_DELIMITED_EXTS: frozenset[str] = frozenset({"jsonl", "ndjson"})


def parse(path: Path) -> list[ParsedChunk]:
    """Dispatch by extension then by top-level shape (for .json)."""
    ext = path.suffix.lower().lstrip(".")
    if ext in _LINE_DELIMITED_EXTS:
        return _parse_line_delimited(path)
    return _parse_json_document(path)


def _parse_line_delimited(path: Path) -> list[ParsedChunk]:
    """One chunk per non-blank line; each line is a complete JSON object.

    Lines are NOT re-parsed and re-serialized - the raw line text becomes
    the chunk text. Blank lines (common as trailing newlines) are skipped
    silently so they don't leave gaps in chunk_index.
    """
    chunks: list[ParsedChunk] = []
    # utf-8-sig strips a leading BOM if present (some editors add one); plain
    # utf-8 leaves it on the first line and corrupts that line's JSON object.
    text = path.read_text(encoding="utf-8-sig")
    # Split on newline ONLY (JSONL is newline-delimited). str.splitlines() also
    # breaks on U+2028 / U+2029 / U+0085 / vertical-tab / form-feed, which are
    # valid characters INSIDE a JSON string value and would wrongly shred a
    # record. `.strip()` below still drops any trailing \r from CRLF files.
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        chunks.append(
            ParsedChunk(
                chunk_index=len(chunks),
                text=stripped,
                content_type="json_object",
            )
        )
    logger.info("parsed %s (line-delimited) -> %d chunks", path.name, len(chunks))
    return chunks


def _parse_json_document(path: Path) -> list[ParsedChunk]:
    """For .json files: array-at-top-level -> per-element; else whole-doc.

    Array elements are re-serialized via `json.dumps(..., ensure_ascii=False)`
    so the chunk text is canonical JSON regardless of how the input was
    formatted (pretty-printed, minified, mixed-case keys, etc.). For the
    whole-doc case we keep the original file text verbatim - cheaper, and
    preserves any meaningful whitespace.
    """
    # utf-8-sig strips a leading BOM if present; plain utf-8 keeps it as a
    # U+FEFF char so json.loads then fails ("Expecting value: line 1 column 1").
    text = path.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    if isinstance(data, list):
        chunks = [
            ParsedChunk(
                chunk_index=i,
                text=json.dumps(item, ensure_ascii=False),
                content_type="json_object",
            )
            for i, item in enumerate(data)
        ]
        logger.info("parsed %s (array) -> %d chunks", path.name, len(chunks))
        return chunks
    # Object, scalar, or any other top-level shape: whole-doc-as-chunk.
    logger.info("parsed %s (whole-doc) -> 1 chunk", path.name)
    return [
        ParsedChunk(
            chunk_index=0,
            text=text,
            content_type="json_object",
        )
    ]
