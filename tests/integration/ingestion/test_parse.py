"""Integration tests for `ingestion/parse` — docling against real
files in `test_documents/`.

Exercises the parse dispatcher's routing + the docling backend
against a checked-in sample PDF + a JATS-XML article. First run is
slow (docling initialises tokenizers / models on first use);
subsequent runs reuse the cached models.

No network calls — all docling work is local. The cost is CPU + RAM
for the model loading, not LLM tokens.

Manual interactive counterpart: `scripts/smoke_parse.py`.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from knowledge_agent.corpus_config import CorpusConfig
from knowledge_agent.ingestion.parse import (
    UnsupportedFormatError,
    parse_document,
    supported_extensions,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

# `parse_document` requires a CorpusConfig — only the docling strategy reads
# its fields (OCR flags, chunker strategy, image scale); other parsers ignore
# it. The default config (PDF OCR off, hybrid chunker) is what these
# format-coverage tests want, so one shared instance covers every call below.
_CONFIG = CorpusConfig()


def test_supported_extensions_includes_pdf_docx_xml() -> None:
    """The dispatcher knows about the common research-paper formats."""
    ext = supported_extensions()
    assert "pdf" in ext
    assert "docx" in ext
    assert "xml" in ext


def test_parse_pdf_returns_non_empty_chunks(sample_pdf: Path) -> None:
    """A real research PDF parses into > 0 chunks with non-empty text."""
    chunks = parse_document(sample_pdf, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_pdf_chunks_have_sequential_chunk_index(sample_pdf: Path) -> None:
    """chunk_index is 0-based + sequential across the output."""
    chunks = parse_document(sample_pdf, _CONFIG)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_parse_pdf_chunks_have_content_type_text(sample_pdf: Path) -> None:
    """A PDF parse produces docling-style text chunks. Tables / figures
    may use other content_type values; the majority must be text."""
    chunks = parse_document(sample_pdf, _CONFIG)
    text_count = sum(1 for c in chunks if c.content_type == "text")
    assert text_count >= len(chunks) // 2


def test_parse_pdf_chunks_carry_page_info_for_at_least_some_chunks(
    sample_pdf: Path,
) -> None:
    """Docling annotates chunks with page numbers (1-indexed) when the
    source layout supports it. Not every chunk has a page (cover/TOC
    chunks may have None), but a research PDF should have at least
    some pages annotated."""
    chunks = parse_document(sample_pdf, _CONFIG)
    with_page = [c for c in chunks if c.page is not None]
    assert len(with_page) > 0
    assert all(c.page >= 1 for c in with_page)


def test_parse_jats_xml_returns_chunks(sample_jats_xml: Path) -> None:
    """JATS-XML scientific articles parse via docling too."""
    chunks = parse_document(sample_jats_xml, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_docx_returns_chunks(sample_docx: Path) -> None:
    """DOCX support: a Word abstract parses into chunks."""
    chunks = parse_document(sample_docx, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_unsupported_extension_raises(tmp_path: Path) -> None:
    """A file with an unknown extension raises `UnsupportedFormatError`
    rather than returning an empty list — catches misuse early."""
    bogus = tmp_path / "bogus.xyz"
    bogus.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        parse_document(bogus, _CONFIG)


# ---------------------------------------------------------------------------
# Format coverage: one test per docling-supported format the project
# advertises. Each picks the matching fixture from conftest, parses,
# and asserts at-least-one non-empty chunk. Failure here means the
# parse layer regressed for that format.
# ---------------------------------------------------------------------------


def test_parse_pptx_returns_chunks(sample_pptx: Path) -> None:
    """PowerPoint slides parse via docling's PPTX pipeline."""
    chunks = parse_document(sample_pptx, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_xlsx_returns_chunks(sample_xlsx: Path) -> None:
    """Excel workbooks parse via docling's XLSX pipeline."""
    chunks = parse_document(sample_xlsx, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_html_returns_chunks(sample_html: Path) -> None:
    """HTML files parse via docling's HTML pipeline."""
    chunks = parse_document(sample_html, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_md_returns_chunks(sample_md: Path) -> None:
    """Markdown files parse via docling's MD pipeline."""
    chunks = parse_document(sample_md, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_tex_returns_chunks(sample_tex: Path) -> None:
    """LaTeX files parse via docling's LaTeX pipeline."""
    chunks = parse_document(sample_tex, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_adoc_returns_chunks(sample_adoc: Path) -> None:
    """AsciiDoc files parse via docling's AsciiDoc pipeline."""
    chunks = parse_document(sample_adoc, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_csv_returns_chunks(sample_csv: Path) -> None:
    """CSV files parse via docling's CSV pipeline."""
    chunks = parse_document(sample_csv, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_vtt_returns_chunks(sample_vtt: Path) -> None:
    """WebVTT subtitle files parse via docling's VTT pipeline."""
    chunks = parse_document(sample_vtt, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


def test_parse_image_runs_ocr_pipeline_without_error(sample_image: Path) -> None:
    """Image inputs (PNG/JPG/etc) go through docling's OCR pipeline.

    The contract under test is: the OCR pipeline RUNS without raising
    and returns a list. The list may be empty — a graphics-only image
    (cartoon, schematic, EM micrograph) genuinely has no
    machine-readable text for OCR to extract. That's a valid outcome,
    not a regression. If the file happens to have text, chunks have
    non-empty `text` fields.
    """
    chunks = parse_document(sample_image, _CONFIG)
    # parse_document must always return a list, not None / exception.
    assert isinstance(chunks, list)
    # Any chunks present must carry non-empty text — empty-text chunks
    # would indicate a contract break in the OCR pipeline.
    assert all(c.text for c in chunks)


# ---- ASR-dependent tests. Audio + video go through the `parsers-asr`
# extra (Whisper + imageio-ffmpeg). Skipped cleanly when the extra
# isn't installed — same pattern as the lifecycle's is_installed check.


def _asr_available() -> bool:
    """Mirrors `parser_lifecycle._asr_is_installed` so the integration
    tests opt-out cleanly when the user hasn't installed parsers-asr."""
    try:
        import imageio_ffmpeg  # noqa: F401
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _asr_available(),
    reason="parsers-asr extra not installed (whisper + imageio-ffmpeg)",
)
@pytest.mark.slow
def test_parse_audio_returns_chunks(sample_audio: Path) -> None:
    """Audio files transcribe via Whisper (parsers-asr extra).

    Marked `slow` because Whisper Turbo's first call loads ~1.5 GB of
    model weights — wall-clock measured in tens of seconds even on
    short clips."""
    chunks = parse_document(sample_audio, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)


@pytest.mark.skipif(
    not _asr_available(),
    reason="parsers-asr extra not installed (whisper + imageio-ffmpeg)",
)
@pytest.mark.slow
def test_parse_video_returns_chunks(sample_video: Path) -> None:
    """Video files have their audio track extracted via ffmpeg
    (bundled by imageio-ffmpeg) then transcribed via Whisper.

    Visual frames are NOT OCRed today — slides shown in a recording
    are invisible to the parser."""
    chunks = parse_document(sample_video, _CONFIG)
    assert len(chunks) > 0
    assert all(c.text for c in chunks)
