"""Parser dispatcher.

Routes an incoming file path to the right strategy module based on its
extension. Each strategy module in this package exports `EXTENSIONS` and
`parse(path) -> list[ParsedChunk]` (see `base.py`).

`PARSER_MODULES` lists the registered strategies in dispatch-priority
order. Modules are imported lazily on first use so heavy parser deps
(whisper for ASR, tree-sitter for code) don't load until their format
is actually encountered.
"""

import importlib
from pathlib import Path
from types import ModuleType

from knowledge_agent.ingestion.parsers.base import (
    ParsedChunk,
    UnsupportedFormatError,
)

# Names of parser strategy modules within this package. Order is
# dispatch priority - first match wins. As new parsers ship, append
# their module name here.
PARSER_MODULES: tuple[str, ...] = (
    "docling_parser",
    "json_parser",
    "code_parser",
)


def _load(module_name: str) -> ModuleType:
    """Import a strategy module by name. Cached by Python's import system."""
    return importlib.import_module(
        f"knowledge_agent.ingestion.parsers.{module_name}"
    )


def supported_extensions() -> set[str]:
    """Lowercase file extensions (no leading dot) handled by any parser.

    Union of every registered module's `EXTENSIONS` constant. Eagerly
    imports each parser module to read the constant; this is fine in
    practice because callers needing this set are also about to do real
    ingest work that loads those modules anyway.
    """
    exts: set[str] = set()
    for module_name in PARSER_MODULES:
        module = _load(module_name)
        exts.update(module.EXTENSIONS)
    return exts


def parse_document(path: Path) -> list[ParsedChunk]:
    """Dispatch by file extension to the matching parser module.

    Raises `UnsupportedFormatError` if no registered parser handles the
    extension. Errors raised by the strategy itself (docling parse
    failures, etc.) bubble up unchanged - the caller (`pipeline.py`)
    decides what to do.
    """
    ext = path.suffix.lower().lstrip(".")
    for module_name in PARSER_MODULES:
        module = _load(module_name)
        if ext in module.EXTENSIONS:
            return module.parse(path)
    raise UnsupportedFormatError(
        f"No parser registered for extension {ext!r}. "
        f"Supported: {sorted(supported_extensions())}."
    )


__all__ = [
    "PARSER_MODULES",
    "ParsedChunk",
    "UnsupportedFormatError",
    "parse_document",
    "supported_extensions",
]
