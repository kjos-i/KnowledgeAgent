"""Unit tests for `ingestion.parsers.docling_parser`.

Focused on the figure-extraction filter (`min_figure_bytes`). Docling
itself is expensive to invoke; these tests mock `_get_converter` and
`_get_chunker` so a fake document with fake `pictures` flows through
`parse()`. Each fake picture's `get_image().save(path)` writes a
controllable number of bytes so we can assert what the size filter
does at the boundary.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from knowledge_agent.ingestion.parsers import docling_parser


def _fake_picture(byte_count: int, caption: str = "cap"):
    """A stand-in for a docling PictureItem.

    `get_image(doc)` returns a mock image whose `save(path)` writes
    `byte_count` bytes so the filter's `out.stat().st_size` check reads
    exactly that number.
    """

    def save(path, *_a, **_kw):
        Path(path).write_bytes(b"x" * byte_count)

    fake_image = MagicMock()
    fake_image.save.side_effect = save

    pic = MagicMock()
    pic.get_image.return_value = fake_image
    pic.caption_text.return_value = caption
    pic.prov = [SimpleNamespace(page_no=1)]
    return pic


def _fake_doc(pictures):
    doc = MagicMock()
    doc.pictures = pictures
    return doc


def _fake_config(min_figure_bytes: int) -> docling_parser.DoclingConfig:
    return docling_parser.DoclingConfig(
        enable_pdf_ocr=False,
        enable_image_ocr=False,
        chunker_strategy="hybrid",
        chunk_max_tokens=512,
        merge_peers=True,
        images_scale=1.0,
        extract_figures=True,
        embed_images=True,
        min_figure_bytes=min_figure_bytes,
    )


def _run_parse(
    tmp_path: Path,
    pictures,
    min_figure_bytes: int,
):
    """Drive `parse()` with the given fake pictures + threshold and
    return `(chunks, figures_dir)`. Clears the lru_cache so each test
    gets fresh mocked converter/chunker instances."""
    docling_parser._get_converter.cache_clear()
    docling_parser._get_chunker.cache_clear()

    # `.pdf` so the structured-doc branch runs (not the standalone-image
    # branch that skips figure walking).
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-fake")
    figures_dir = tmp_path / "figures"

    fake_result = MagicMock()
    fake_result.document = _fake_doc(pictures)
    fake_converter = MagicMock()
    fake_converter.convert.return_value = fake_result

    fake_chunker = MagicMock()
    fake_chunker.chunk.return_value = []  # No text chunks in this test.

    with (
        patch.object(
            docling_parser,
            "_get_converter",
            return_value=fake_converter,
        ),
        patch.object(
            docling_parser,
            "_get_chunker",
            return_value=fake_chunker,
        ),
    ):
        chunks = docling_parser.parse(
            fake_pdf,
            _fake_config(min_figure_bytes),
            figures_dir=figures_dir,
        )
    return chunks, figures_dir


def test_min_figure_bytes_drops_below_threshold(tmp_path: Path) -> None:
    """A 500-byte picture with threshold 2048 is deleted from disk and
    yields no figure chunk. A 4000-byte picture passes."""
    tiny = _fake_picture(byte_count=500)
    big = _fake_picture(byte_count=4000)

    chunks, figures_dir = _run_parse(
        tmp_path,
        [tiny, big],
        min_figure_bytes=2048,
    )

    # Only the big picture survives on disk (index 1 was the second
    # picture).
    saved = sorted((figures_dir).glob("*.png"))
    assert [p.name for p in saved] == ["1.png"], saved

    # Only one figure chunk emitted — for the big picture.
    figs = [c for c in chunks if c.content_type == "figure"]
    assert len(figs) == 1
    assert figs[0].image_ref.endswith("1.png")


def test_min_figure_bytes_zero_disables_filter(tmp_path: Path) -> None:
    """min_figure_bytes=0 keeps every picture regardless of size."""
    tiny = _fake_picture(byte_count=100)
    also_tiny = _fake_picture(byte_count=500)

    chunks, figures_dir = _run_parse(
        tmp_path,
        [tiny, also_tiny],
        min_figure_bytes=0,
    )

    saved = sorted((figures_dir).glob("*.png"))
    assert [p.name for p in saved] == ["0.png", "1.png"], saved

    figs = [c for c in chunks if c.content_type == "figure"]
    assert len(figs) == 2


def test_min_figure_bytes_equal_to_size_is_dropped(tmp_path: Path) -> None:
    """Boundary: file size exactly equal to `min_figure_bytes` still
    fails the `< min_figure_bytes` check → kept. One byte below the
    threshold is dropped."""
    at_threshold = _fake_picture(byte_count=2048)
    below = _fake_picture(byte_count=2047)

    chunks, figures_dir = _run_parse(
        tmp_path,
        [at_threshold, below],
        min_figure_bytes=2048,
    )

    saved = sorted((figures_dir).glob("*.png"))
    # index 0 = at_threshold (kept), index 1 = below (dropped)
    assert [p.name for p in saved] == ["0.png"], saved

    figs = [c for c in chunks if c.content_type == "figure"]
    assert len(figs) == 1
    assert figs[0].image_ref.endswith("0.png")
