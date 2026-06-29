"""Docling-backed parser strategy.

Wraps `DocumentConverter` + `HybridChunker` so the dispatcher sees only
`EXTENSIONS` + `parse(path) -> list[ParsedChunk]`. Single point of docling
contact; if we ever swap parsers for one of these formats, this is the
only file that changes.

Supported formats are declared in `SUPPORTED_FORMATS`. Concrete file
extensions accepted (from docling's format -> extension map, version-
pinned for docling 2.96):

  - PDF       pdf
  - IMAGE     bmp, jpeg, jpg, png, tif, tiff, webp
  - DOCX      docm, docx, dotm, dotx
  - PPTX      potm, potx, ppsm, ppsx, pptm, pptx
  - XLSX      xlsm, xlsx
  - HTML      htm, html, xhtml
  - MD        md, txt, text, Rmd, qmd, rmd
  - XML_JATS  nxml, xml
  - CSV       csv (plus tsv via spoofed-name routing, see `parse()`)
  - ASCIIDOC  adoc, asc, asciidoc
  - LATEX     latex, tex
  - VTT       vtt
  - AUDIO     aac, avi, flac, m4a, mov, mp3, mp4, ogg, wav
              (docling unifies audio + video under AUDIO - video files
              have ONLY their audio track extracted and transcribed by
              the Whisper ASR pipeline)

**Video files: audio-only today.** When a video file (avi / mov / mp4)
is passed in, docling extracts the audio track via ffmpeg and runs
Whisper ASR on it. Visual content — slides shown on screen, diagrams,
demonstrations, charts — is NOT captured. A silent video (or one with
music-only audio) produces zero useful chunks. See
[[deferred-video-frame-extraction]] for the planned frame-OCR /
multimodal-embedding extension that would close this gap; until that
ships, treat video as "transcript-only".

OCR behaviour follows `enable_pdf_ocr` and `enable_image_ocr` in Settings:
PDFs default to OCR off (papers are born-digital); image inputs default
to OCR on (no text otherwise). **Video frames are NOT OCRed.** Standalone
image files (`IMAGE` category) ARE OCRed; the path is separate from the
video / audio pipeline.

ASR (audio + video transcription) requires the `parsers-asr` install
extra (which transitively pulls `docling[asr]` + a Whisper Turbo model)
AND the `ffmpeg` system binary on PATH. Registering AUDIO here is cheap
and doesn't load whisper - the pipeline only initialises when an actual
audio file is converted. Preflight checks live in `parser_lifecycle.py`.

TSV is handled by routing through docling's CSV backend, whose
`csv.Sniffer` auto-detects the tab delimiter. We hand the bytes to
docling wrapped in a `DocumentStream` with the name spoofed to ".csv";
the file on disk is never touched. See `parse()` below.

HybridChunker is structure-aware (respects headings, paragraphs, tables)
and merges undersized adjacent chunks within the same section. For CSV/
XLSX tables, the header row is repeated at the top of every chunk (built
into docling's chunker, `repeat_table_header=True`).

DOI extraction is NOT done here - it reads chunk text in `metadata.py` so
parsing stays format-agnostic.
"""

import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_core.types.io import DocumentStream

from knowledge_agent.config import get_settings
from knowledge_agent.ingestion.parser_lifecycle import (
    ensure_bundled_ffmpeg_on_path,
)
from knowledge_agent.ingestion.parsers.base import ParsedChunk

logger = logging.getLogger(__name__)

# Make the bundled ffmpeg (shipped via `imageio-ffmpeg` under the
# `parsers-asr` extra) visible to docling / whisper subprocess calls.
# No-op when the asr extra isn't installed (returns None silently).
# Runs at import time — this module is lazy-imported by the parser
# dispatcher only when a file actually needs parsing, so the cost is
# paid once per process and only when ingestion runs.
_ = ensure_bundled_ffmpeg_on_path()

# Formats we accept for ingestion. Single source of truth - drives both the
# DocumentConverter's `allowed_formats` and the `EXTENSIONS` constant the
# dispatcher reads.
SUPPORTED_FORMATS: tuple[InputFormat, ...] = (
    InputFormat.PDF,
    InputFormat.IMAGE,
    InputFormat.DOCX,
    InputFormat.PPTX,
    InputFormat.XLSX,
    InputFormat.HTML,
    InputFormat.MD,
    InputFormat.XML_JATS,
    InputFormat.CSV,
    InputFormat.ASCIIDOC,
    InputFormat.LATEX,
    InputFormat.VTT,
    InputFormat.AUDIO,
)

_IMAGE_EXTENSIONS = frozenset(
    ext.lower() for ext in FormatToExtensions[InputFormat.IMAGE]
)

# Extensions we route through a docling backend that natively handles a
# different (but compatible) extension. `tsv` -> CSV backend (delimiter
# auto-detected). Add new aliases here when more sibling-of-CSV formats
# arrive (e.g. .psv pipe-separated, .scsv semicolon-separated).
_ALIAS_EXTENSIONS: dict[str, str] = {
    "tsv": "csv",
}


def _derive_extensions() -> tuple[str, ...]:
    """Lowercase file extensions (no leading dot) handled by this parser.

    Derived from docling's format -> extension map for `SUPPORTED_FORMATS`
    so the constant never drifts from the registered formats. Adds the
    manual `_ALIAS_EXTENSIONS` entries (e.g. `tsv` -> CSV backend) that
    docling does not register natively but we route ourselves in `parse()`.
    Sorted for deterministic test output.
    """
    exts: set[str] = set()
    for fmt in SUPPORTED_FORMATS:
        exts.update(ext.lower() for ext in FormatToExtensions[fmt])
    exts.update(_ALIAS_EXTENSIONS.keys())
    return tuple(sorted(exts))


# Parser dispatcher contract: each strategy module exports `EXTENSIONS`.
EXTENSIONS: tuple[str, ...] = _derive_extensions()


def is_image_path(path: Path) -> bool:
    """True if `path` is a standalone image-format input (by extension)."""
    return path.suffix.lower().lstrip(".") in _IMAGE_EXTENSIONS


# DocumentConverter and HybridChunker have non-trivial init cost (loading
# tokenizer / model artifacts). Cache as process-wide singletons - first
# parse() call pays the cost, later calls reuse them.


@lru_cache(maxsize=1)
def _get_converter() -> DocumentConverter:
    """Cached docling converter with per-format options applied.

    PDFs and image inputs each get their own pipeline options so OCR can be
    toggled independently. Other formats use docling's defaults.
    """
    settings = get_settings()

    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = settings.enable_pdf_ocr

    image_opts = PdfPipelineOptions()
    image_opts.do_ocr = settings.enable_image_ocr

    return DocumentConverter(
        allowed_formats=list(SUPPORTED_FORMATS),
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_opts),
        },
    )


@lru_cache(maxsize=1)
def _get_chunker() -> HybridChunker | HierarchicalChunker:
    """Cached chunker, picked by `settings.chunker_strategy`.

    HybridChunker reads `chunk_max_tokens`. HierarchicalChunker ignores
    token limits by design - chunks align to document structure only.
    """
    settings = get_settings()
    if settings.chunker_strategy == "hierarchical":
        return HierarchicalChunker()
    return HybridChunker(max_tokens=settings.chunk_max_tokens)


def parse(path: Path) -> list[ParsedChunk]:
    """Parse one document file via docling and return its chunks in order.

    For alias extensions (see `_ALIAS_EXTENSIONS`, e.g. `.tsv`), the file
    bytes are wrapped in a `DocumentStream` with the name spoofed to the
    target extension so docling routes to the right backend. The file on
    disk is never touched.

    Raises any error docling raises - parsing is a hard prerequisite for
    everything downstream, so failures should bubble up to the caller
    (`pipeline.py`) which decides what to do (skip the file, retry, etc.).
    """
    converter = _get_converter()
    chunker = _get_chunker()

    source: str | DocumentStream
    ext = path.suffix.lower().lstrip(".")
    if ext in _ALIAS_EXTENSIONS:
        target_ext = _ALIAS_EXTENSIONS[ext]
        source = DocumentStream(
            name=f"{path.stem}.{target_ext}",
            stream=BytesIO(path.read_bytes()),
        )
    else:
        source = str(path)

    result = converter.convert(source=source)
    chunks: list[ParsedChunk] = []
    for index, chunk in enumerate(chunker.chunk(result.document)):
        chunks.append(
            ParsedChunk(
                chunk_index=index,
                text=chunk.text,
                section=_extract_section(chunk),
                page=_extract_page(chunk),
                content_type="text",
            )
        )
    logger.info("parsed %s -> %d chunks", path.name, len(chunks))
    return chunks


def _extract_section(chunk: Any) -> str | None:
    """Best-effort: last heading in the chunk's heading chain, or None."""
    try:
        headings = chunk.meta.headings
        if headings:
            return headings[-1]
    except (AttributeError, IndexError, TypeError):
        pass
    return None


def _extract_page(chunk: Any) -> int | None:
    """Best-effort: page number from the first prov entry of the first doc_item."""
    try:
        doc_items = chunk.meta.doc_items
        for item in doc_items:
            prov = getattr(item, "prov", None) or []
            for p in prov:
                page_no = getattr(p, "page_no", None)
                if page_no is not None:
                    return int(page_no)
    except (AttributeError, TypeError):
        pass
    return None
