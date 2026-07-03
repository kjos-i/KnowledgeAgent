"""Parser dispatcher.

Routes an incoming file path to the right strategy module based on its
extension. Each strategy module in this package exports `EXTENSIONS` and
`parse(path, ...) -> list[ParsedChunk]` (see `base.py`).

`PARSER_MODULES` lists the registered strategies in dispatch-priority
order. Modules are imported lazily on first use so heavy parser deps
(whisper for ASR, tree-sitter for code) don't load until their format
is actually encountered.

The docling parser is the only strategy that reads per-corpus config
today (OCR flags, chunker strategy, image scale). The dispatcher
accepts a `CorpusConfig` and passes the relevant slice through to
docling; other parsers ignore it.
"""

import importlib
from pathlib import Path
from types import ModuleType

from knowledge_agent.ingestion.parsers.base import (
    ParsedChunk,
    UnsupportedFormatError,
)
from knowledge_agent.kg.corpus_config import CorpusConfig

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


def parse_document(
    path: Path,
    config: CorpusConfig,
    *,
    figures_dir: Path | None = None,
) -> list[ParsedChunk]:
    """Dispatch by file extension to the matching parser module.

    `config` is the ingest's `CorpusConfig` — its parser fields
    (`enable_pdf_ocr`, `chunker_strategy`, ...) are threaded through
    to the docling parser. Other parsers ignore it.

    `figures_dir` is the per-doc directory where the docling parser
    saves extracted PNGs when `config.extract_figures=True`. The
    pipeline computes it as `<corpus folder>/figures/<doc_id>/` and
    passes it through. None disables all figure-related work
    (multimodal off). Non-docling parsers ignore this kwarg.

    Raises `UnsupportedFormatError` if no registered parser handles the
    extension. Errors raised by the strategy itself (docling parse
    failures, etc.) bubble up unchanged - the caller (`pipeline.py`)
    decides what to do.
    """
    ext = path.suffix.lower().lstrip(".")
    for module_name in PARSER_MODULES:
        module = _load(module_name)
        if ext in module.EXTENSIONS:
            if module_name == "docling_parser":
                docling_cfg = module.config_from_corpus(config)
                return module.parse(path, docling_cfg, figures_dir)
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
