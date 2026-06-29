"""Shared contract for parser strategy modules.

Every module in `knowledge_agent.ingestion.parsers` exports:

  - `EXTENSIONS: tuple[str, ...]` — lowercase file extensions (no leading dot)
    that this parser handles. The dispatcher (`parsers/__init__.py`) uses this
    to route an incoming path to the right module.
  - `parse(path: Path) -> list[ParsedChunk]` — the parser entry point.

`ParsedChunk` is the universal contract between any parser and the rest of
the ingestion pipeline; downstream code (embed, KG writes, LanceDB writes)
does not know or care which parser produced a chunk.
"""

from dataclasses import dataclass


@dataclass
class ParsedChunk:
    """One chunk produced by a parser: text + best-effort structural metadata.

    `content_type` distinguishes the source format for downstream consumers:
    "text" (docling-parsed prose), "row" (tabular row-slab from a CSV/XLSX
    table chunk), "json_object" (one element of a JSON array or one JSONL
    line), "code" (an AST-aware function/class chunk), "transcript" (audio
    or video ASR segment), "frame_caption" (video frame caption).
    """

    chunk_index: int
    text: str
    section: str | None = None
    page: int | None = None
    content_type: str = "text"


class UnsupportedFormatError(ValueError):
    """No parser is registered for the file's extension."""
