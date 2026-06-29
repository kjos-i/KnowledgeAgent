"""Public entry point for document parsing.

Thin re-export shim over the parser dispatcher in `parsers/`. Lets
callers (`pipeline.py`, `bulk_ops.py`, `metadata.py`, tests, smoke
scripts) keep their `from knowledge_agent.ingestion.parse
import ...` imports while the underlying strategies live in
`parsers/<format>_parser.py`.

The dispatcher (`parsers.parse_document`) routes by file extension to
the right strategy module. Today the only registered strategy is
`docling_parser`; future strategies (`json_parser`, `code_parser`)
slot in by appending their module name to
`parsers.PARSER_MODULES` - no change here.

Docling-specific symbols (`SUPPORTED_FORMATS`, `is_image_path`,
`_extract_section`, `_extract_page`) are re-exported from
`docling_parser` for callers and tests that pre-date the split.
"""

from knowledge_agent.ingestion.parsers import (
    PARSER_MODULES,
    ParsedChunk,
    UnsupportedFormatError,
    parse_document,
    supported_extensions,
)
from knowledge_agent.ingestion.parsers.docling_parser import (
    SUPPORTED_FORMATS,
    _extract_page,
    _extract_section,
    is_image_path,
)

__all__ = [
    "PARSER_MODULES",
    "SUPPORTED_FORMATS",
    "ParsedChunk",
    "UnsupportedFormatError",
    "_extract_page",
    "_extract_section",
    "is_image_path",
    "parse_document",
    "supported_extensions",
]
