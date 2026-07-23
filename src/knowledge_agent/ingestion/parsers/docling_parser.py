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
music-only audio) produces zero useful chunks. A planned frame-OCR /
multimodal-embedding extension would close this gap; until that
ships, treat video as "transcript-only".

**XML: modern JATS only.** Docling parses the newer JATS `.xml` /
`.nxml` articles (Z39.96 / JATS 1.x archiving + publishing). The OLDER
"NLM Journal Publishing DTD v3.0" variant (used by some legacy PubMed
Central / PLoS exports) is NOT recognised — docling raises
`ConversionError: File format not allowed` (format None). It is handled
like any other unsupported file: that one document is skipped / flagged
failed while the rest of the batch ingests normally. Workaround: ingest
the article's PDF instead (docling parses those), which supplies the
body. The article DOI does NOT depend on this — it is read straight from
the XML's `<article-id>` by `metadata.doi_from_jats`, independent of
docling, so it still works on the old variant. A custom fallback parser
for the old variant was considered and deliberately NOT built (the PDF
covers the body); see the eval / benchmark notes.

OCR + chunker behaviour is driven by `DoclingConfig` passed into
`parse()` by the pipeline (values ultimately come from the corpus's
`CorpusConfig`). PDFs default to OCR off (papers are born-digital);
image inputs default to OCR on (no text otherwise). **Video frames
are NOT OCRed.** Standalone image files (`IMAGE` category) ARE
OCRed; the path is separate from the video / audio pipeline.

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

import contextlib
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from docling.chunking import HierarchicalChunker, HybridChunker
from docling.datamodel.base_models import FormatToExtensions, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)
from docling_core.types.io import DocumentStream

from knowledge_agent.corpus_config import CorpusConfig
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

_IMAGE_EXTENSIONS = frozenset(ext.lower() for ext in FormatToExtensions[InputFormat.IMAGE])

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


class DoclingConfig(NamedTuple):
    """Frozen, hashable slice of per-corpus fields the docling parser reads.

    Keeping the parser's inputs to a small NamedTuple lets us key the
    process-wide converter / chunker caches on their contents — parsing
    100 files in one corpus pays the docling init cost once, and
    switching corpora naturally invalidates via the cache key.
    """

    enable_pdf_ocr: bool
    enable_image_ocr: bool
    chunker_strategy: str
    chunk_max_tokens: int
    merge_peers: bool
    images_scale: float
    extract_figures: bool
    embed_images: bool
    min_figure_bytes: int


def config_from_corpus(cfg: CorpusConfig) -> DoclingConfig:
    """Build a `DoclingConfig` from a `CorpusConfig`.

    Small helper so callers don't have to remember which fields go in.
    """
    return DoclingConfig(
        enable_pdf_ocr=cfg.enable_pdf_ocr,
        enable_image_ocr=cfg.enable_image_ocr,
        chunker_strategy=cfg.chunker_strategy,
        chunk_max_tokens=cfg.chunk_max_tokens,
        merge_peers=cfg.merge_peers,
        images_scale=cfg.images_scale,
        extract_figures=cfg.extract_figures,
        embed_images=cfg.embed_images,
        min_figure_bytes=cfg.min_figure_bytes,
    )


# DocumentConverter and HybridChunker have non-trivial init cost
# (loading tokenizer / model artifacts). Cache keyed on the tuple of
# per-corpus fields so a run over 100 files pays the docling init cost
# once, and switching corpora invalidates naturally via the cache key.
# `maxsize=4` keeps a handful of recent corpora warm without unbounded
# growth if the user pivots frequently.


@lru_cache(maxsize=4)
def _get_converter(config: DoclingConfig) -> DocumentConverter:
    """Cached docling converter with per-format options applied.

    PDFs and image inputs each get their own pipeline options so OCR
    can be toggled independently. `images_scale` applies to both (it
    controls the render scale docling uses before OCR / figure
    extraction). Other formats use docling's defaults.
    """
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = config.enable_pdf_ocr
    pdf_opts.images_scale = config.images_scale
    # When extract_figures is on, tell Docling to keep decoded PIL
    # images on each PictureItem so `parse()` below can save them as
    # PNGs. Without this flag Docling still identifies picture
    # bounding boxes but does not retain the image bytes.
    pdf_opts.generate_picture_images = config.extract_figures

    image_opts = PdfPipelineOptions()
    image_opts.do_ocr = config.enable_image_ocr
    image_opts.images_scale = config.images_scale
    # Standalone image inputs also route through PdfPipelineOptions
    # inside Docling; extraction toggle mirrors PDFs so a corpus with
    # extract_figures=True treats both source types uniformly.
    image_opts.generate_picture_images = config.extract_figures

    return DocumentConverter(
        allowed_formats=list(SUPPORTED_FORMATS),
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=image_opts),
        },
    )


@lru_cache(maxsize=4)
def _get_chunker(config: DoclingConfig) -> HybridChunker | HierarchicalChunker:
    """Cached chunker, picked by `config.chunker_strategy`.

    HybridChunker reads `chunk_max_tokens` and `merge_peers`.
    HierarchicalChunker ignores both by design — chunks align to
    document structure only, no token cap, no merging.

    Note on the noisy transformers warning during ingest:
        `[transformers] Token indices sequence length is longer than the
        specified maximum sequence length for this model (N > 512).
        Running this sequence through the model will result in indexing
        errors`
    HybridChunker's default tokenizer is `sentence-transformers/all-MiniLM-L6-v2`
    (BertTokenizer, model_max_length=512). When a single doc item — a long
    paragraph, a big table row, etc. — exceeds 512 tokens, the tokenizer
    fires that stock HuggingFace warning at the token-COUNT step. It is
    misleading in our context: the sentence-transformer model is never
    actually run. HybridChunker uses the tokenizer only to count tokens,
    then hands over-length items to `semchunk` which correctly splits
    them down to `chunk_max_tokens`. Verified from Docling source
    ([docling_core/transforms/chunker/hybrid_chunker.py] _split_using_plain_text
    + segment). Emitted chunks ARE properly capped — no silent
    truncation, no quality loss. Not surfaced in the GUI. Left
    unsuppressed to keep the transformers logger untouched for future
    warnings that might actually matter.
    """
    if config.chunker_strategy == "hierarchical":
        return HierarchicalChunker()
    return HybridChunker(
        max_tokens=config.chunk_max_tokens,
        merge_peers=config.merge_peers,
    )


def parse(
    path: Path,
    config: DoclingConfig,
    figures_dir: Path | None = None,
) -> list[ParsedChunk]:
    """Parse one document file via docling and return its chunks in order.

    `config` carries the per-corpus fields docling actually reads
    (OCR flags, chunker strategy, image scale, figure toggles). Passed
    from the ingest pipeline, which has the `CorpusConfig` in scope.

    `figures_dir` is the per-doc directory where extracted PNGs land
    (`<corpus folder>/figures/<doc_id>/`). Required when
    `config.extract_figures=True` for PDF / Word / PPT / Office inputs;
    ignored for standalone image files (they reference the source path
    in place). None disables all figure-related work regardless of the
    config flags.

    Multimodal chunk emission (option A, locked 2026-07-03):
      - PDF / Word / PPT with `extract_figures=True` + `embed_images=True`
        + `figures_dir` given: after the normal text chunker, walk
        `doc.pictures`, save each as `<i>.png` in `figures_dir`, and
        emit one figure ParsedChunk per saved picture. The figure
        chunk's `text` is the PDF's structured caption (empty string
        if no caption element exists — image-region OCR text has
        already been merged into text chunks by Docling's native OCR
        pass, so we do NOT re-OCR the picture here).
      - Standalone image inputs (PNG / JPG / TIFF): the image IS the
        document. When `embed_images=True`, emit ONE figure ParsedChunk
        with `text = doc.export_to_text()` (Docling's whole-doc OCR
        output) and `image_ref = <absolute path to the source file>`
        (no copy). Standalone image chunks are produced regardless of
        `extract_figures` — the file is already on disk.

    For alias extensions (see `_ALIAS_EXTENSIONS`, e.g. `.tsv`), the file
    bytes are wrapped in a `DocumentStream` with the name spoofed to the
    target extension so docling routes to the right backend. The file on
    disk is never touched.

    Raises any error docling raises - parsing is a hard prerequisite for
    everything downstream, so failures should bubble up to the caller
    (`pipeline.py`) which decides what to do (skip the file, retry, etc.).
    """
    converter = _get_converter(config)
    chunker = _get_chunker(config)

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
    doc = result.document

    chunks: list[ParsedChunk] = []
    index = 0

    # ---- Standalone image input branch ----
    # The image IS the document. Docling's OCR already ran (per
    # image_opts.do_ocr = config.enable_image_ocr). No text chunker
    # pass — a standalone image has no paragraph structure to chunk;
    # `export_to_text()` returns the full OCR content as one string.
    # Emit exactly one figure ParsedChunk when `embed_images=True`;
    # otherwise emit one text chunk carrying the OCR text (matches the
    # pre-multimodal behavior).
    if is_image_path(path):
        ocr_text = doc.export_to_text().strip() if doc else ""
        if config.embed_images:
            chunks.append(
                ParsedChunk(
                    chunk_index=index,
                    text=ocr_text,
                    section=None,
                    page=None,
                    content_type="figure",
                    image_ref=str(path.resolve()),
                )
            )
        else:
            chunks.append(
                ParsedChunk(
                    chunk_index=index,
                    text=ocr_text,
                    section=None,
                    page=None,
                    content_type="text",
                )
            )
        logger.info(
            "parsed standalone image %s -> 1 chunk",
            path.name,
        )
        return chunks

    # ---- Structured document branch (PDF / Word / PPT / Office / HTML / MD / etc.) ----
    for chunk in chunker.chunk(doc):
        chunks.append(
            ParsedChunk(
                chunk_index=index,
                text=chunk.text,
                section=_extract_section(chunk),
                page=_extract_page(chunk),
                content_type="text",
            )
        )
        index += 1

    # ---- Figure extraction + figure chunks (multimodal opt-in) ----
    # `extract_figures` alone saves the PNGs (useful for later
    # backfill / display). `embed_images` gates figure ParsedChunk
    # emission — the two flags are independent so a corpus can hold
    # image files on disk without yet feeding them to the multimodal
    # embedder.
    if config.extract_figures and figures_dir is not None:
        pictures = list(getattr(doc, "pictures", []) or [])
        if pictures:
            figures_dir.mkdir(parents=True, exist_ok=True)
        for i, picture in enumerate(pictures):
            try:
                image = picture.get_image(doc)
            except Exception as exc:
                logger.warning(
                    "docling picture %d: get_image failed: %r; skipping",
                    i,
                    exc,
                )
                continue
            if image is None:
                continue
            out = figures_dir / f"{i}.png"
            try:
                image.save(out)
            except OSError as exc:
                logger.warning(
                    "docling picture %d: save to %s failed: %r; skipping",
                    i,
                    out,
                    exc,
                )
                continue
            if config.min_figure_bytes > 0:
                try:
                    size_bytes = out.stat().st_size
                except OSError:
                    size_bytes = 0
                if size_bytes < config.min_figure_bytes:
                    logger.info(
                        "docling picture %d: %d B < min_figure_bytes=%d; dropping",
                        i,
                        size_bytes,
                        config.min_figure_bytes,
                    )
                    with contextlib.suppress(OSError):
                        out.unlink()
                    continue
            if config.embed_images:
                caption_text = ""
                try:
                    caption_text = (picture.caption_text(doc) or "").strip()
                except Exception:
                    caption_text = ""
                page_no = None
                try:
                    prov = getattr(picture, "prov", None) or []
                    if prov:
                        page_no = int(getattr(prov[0], "page_no", None) or 0) or None
                except (AttributeError, TypeError, ValueError):
                    page_no = None
                chunks.append(
                    ParsedChunk(
                        chunk_index=index,
                        text=caption_text,
                        section=None,
                        page=page_no,
                        content_type="figure",
                        image_ref=str(out),
                    )
                )
                index += 1

    logger.info(
        "parsed %s -> %d chunks (%d figure)",
        path.name,
        len(chunks),
        sum(1 for c in chunks if c.content_type == "figure"),
    )
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
