"""Tests for ingestion.parse - format support helpers + chunk extractors.

`parse_document` itself touches docling models on disk and is exercised by
the smoke scripts. Here we cover the format/extension helpers and the
defensive metadata extractors that have well-defined inputs.
"""

from pathlib import Path
from types import SimpleNamespace

from docling.datamodel.base_models import InputFormat

from knowledge_agent.ingestion.parse import (
    SUPPORTED_FORMATS,
    ParsedChunk,
    _extract_page,
    _extract_section,
    is_image_path,
    supported_extensions,
)

# ---- SUPPORTED_FORMATS ----


def test_supported_formats_includes_paper_formats():
    expected = {
        InputFormat.PDF,
        InputFormat.DOCX,
        InputFormat.HTML,
        InputFormat.XML_JATS,
        InputFormat.IMAGE,
    }
    assert expected.issubset(set(SUPPORTED_FORMATS))


# ---- supported_extensions ----


def test_supported_extensions_includes_pdf():
    assert "pdf" in supported_extensions()


def test_supported_extensions_includes_image_formats():
    exts = supported_extensions()
    for img in ["png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"]:
        assert img in exts, f"expected {img!r} in supported extensions"


def test_supported_extensions_includes_docx_and_html():
    exts = supported_extensions()
    assert "docx" in exts
    assert "html" in exts


def test_supported_extensions_includes_csv_and_tsv():
    exts = supported_extensions()
    assert "csv" in exts
    # tsv is an alias - routed through docling's CSV backend via
    # csv.Sniffer auto-detecting the tab delimiter.
    assert "tsv" in exts


def test_supported_extensions_includes_audio_and_video_formats():
    exts = supported_extensions()
    for audio in ["mp3", "wav", "m4a", "flac", "aac", "ogg"]:
        assert audio in exts, f"expected audio {audio!r} in supported extensions"
    for video in ["mp4", "mov", "avi"]:
        assert video in exts, f"expected video {video!r} in supported extensions"


def test_supported_extensions_are_lowercase_no_leading_dot():
    for ext in supported_extensions():
        assert ext == ext.lower()
        assert not ext.startswith(".")


# ---- is_image_path ----


def test_is_image_path_recognises_common_images():
    for name in ["foo.png", "bar.jpg", "baz.JPEG", "qux.tiff", "wibble.webp"]:
        assert is_image_path(Path(name)) is True, name


def test_is_image_path_rejects_non_images():
    for name in ["foo.pdf", "bar.docx", "baz.html", "qux.txt"]:
        assert is_image_path(Path(name)) is False, name


# ---- _extract_section ----


def test_extract_section_returns_last_heading():
    chunk = SimpleNamespace(
        meta=SimpleNamespace(headings=["H1", "H2", "H3"])
    )
    assert _extract_section(chunk) == "H3"


def test_extract_section_returns_none_when_no_headings():
    chunk = SimpleNamespace(meta=SimpleNamespace(headings=[]))
    assert _extract_section(chunk) is None


def test_extract_section_returns_none_when_meta_missing():
    chunk = SimpleNamespace()
    assert _extract_section(chunk) is None


# ---- _extract_page ----


def test_extract_page_pulls_from_first_doc_item_prov():
    chunk = SimpleNamespace(
        meta=SimpleNamespace(
            doc_items=[
                SimpleNamespace(prov=[SimpleNamespace(page_no=7)])
            ]
        )
    )
    assert _extract_page(chunk) == 7


def test_extract_page_returns_none_when_no_doc_items():
    chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[]))
    assert _extract_page(chunk) is None


def test_extract_page_returns_none_when_prov_empty():
    chunk = SimpleNamespace(
        meta=SimpleNamespace(doc_items=[SimpleNamespace(prov=[])])
    )
    assert _extract_page(chunk) is None


def test_extract_page_returns_none_when_meta_missing():
    assert _extract_page(SimpleNamespace()) is None


# ---- ParsedChunk dataclass ----


def test_parsed_chunk_defaults():
    pc = ParsedChunk(chunk_index=0, text="hi")
    assert pc.chunk_index == 0
    assert pc.text == "hi"
    assert pc.section is None
    assert pc.page is None
    assert pc.content_type == "text"
