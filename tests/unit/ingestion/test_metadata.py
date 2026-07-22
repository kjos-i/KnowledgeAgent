"""Tests for ingestion.metadata - DOI extraction + OpenAlex resolution.

DOI extraction is pure (regex on text), tested directly. OpenAlex calls
go through the central `_http_client` and are stubbed by patching
`knowledge_agent.ingestion.metadata._http_client.get` with AsyncMock.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from knowledge_agent.ingestion.metadata import (
    collect_doi_candidates,
    doi_from_jats,
    extract_doi_candidates,
    is_doi_eligible,
    resolve_doi,
    resolve_metadata,
)
from knowledge_agent.ingestion.parse import ParsedChunk


def _write_jats(tmp_path: Path, article_meta_inner: str, *, name: str = "article.xml") -> Path:
    """Write a minimal JATS file whose <article-meta> holds `article_meta_inner`."""
    p = tmp_path / name
    p.write_text(
        "<?xml version='1.0'?>\n"
        "<article><front><article-meta>\n"
        f"{article_meta_inner}\n"
        "<title-group><article-title>T</article-title></title-group>\n"
        "</article-meta></front><body><p>body text</p></body></article>",
        encoding="utf-8",
    )
    return p


def _chunk(index: int, text: str) -> ParsedChunk:
    return ParsedChunk(chunk_index=index, text=text)


def test_doi_from_jats_rejects_xml_entity_expansion(tmp_path: Path):
    """C19: a hostile JATS file that builds its DOI from an XML ENTITY must not
    be expanded. stdlib ElementTree expands ``&x;`` and would return the DOI;
    with defusedxml the entity is forbidden, so we return None (treated as
    'no DOI') rather than expanding it or crashing ingestion."""
    p = tmp_path / "evil.xml"
    p.write_text(
        "<?xml version='1.0'?>\n"
        "<!DOCTYPE article [<!ENTITY x '10.1234/evil'>]>\n"
        "<article><front><article-meta>"
        "<article-id pub-id-type='doi'>&x;</article-id>"
        "</article-meta></front></article>",
        encoding="utf-8",
    )
    assert doi_from_jats(p) is None


# ---- extract_doi_candidates ----


def test_extract_doi_finds_bare_doi():
    chunks = [_chunk(0, "see 10.1234/abc for details")]
    assert extract_doi_candidates(chunks) == ["10.1234/abc"]


def test_extract_doi_strips_url_prefix():
    chunks = [_chunk(0, "https://doi.org/10.1234/abc")]
    assert extract_doi_candidates(chunks) == ["10.1234/abc"]


def test_extract_doi_strips_trailing_punctuation():
    chunks = [_chunk(0, "10.1234/abc.")]
    assert extract_doi_candidates(chunks) == ["10.1234/abc"]


def test_extract_doi_lowercases():
    chunks = [_chunk(0, "DOI: 10.1234/ABC")]
    assert extract_doi_candidates(chunks) == ["10.1234/abc"]


def test_extract_doi_dedupes_keeping_first_occurrence_order():
    chunks = [
        _chunk(0, "10.1234/abc"),
        _chunk(1, "10.5678/xyz, then 10.1234/abc again"),
    ]
    assert extract_doi_candidates(chunks) == ["10.1234/abc", "10.5678/xyz"]


def test_extract_doi_respects_max_chunks_to_search():
    chunks = [
        _chunk(0, "no doi here"),
        _chunk(1, "still nothing"),
        _chunk(2, "10.1234/abc"),
        _chunk(3, "10.9999/zzz"),
    ]
    # Default max=3: scans 0,1,2 (catches abc) but stops before chunk 3.
    result = extract_doi_candidates(chunks)
    assert "10.1234/abc" in result
    assert "10.9999/zzz" not in result


def test_extract_doi_no_text_no_matches():
    chunks = [_chunk(0, "nothing resembling a DOI here")]
    assert extract_doi_candidates(chunks) == []


def test_extract_doi_empty_chunks_returns_empty():
    assert extract_doi_candidates([]) == []


# ---- resolve_doi ----


_HTTP_GET_PATCH = "knowledge_agent.ingestion.metadata._http_client.request"


def _http_response(status_code: int, json_data: Any = None) -> Mock:
    """Build a minimal httpx-response-like mock."""
    resp = Mock()
    resp.status_code = status_code
    resp.json = Mock(return_value=json_data)
    return resp


async def test_resolve_doi_returns_work_on_200():
    work = {"id": "https://openalex.org/W1", "title": "Test"}
    with patch(
        _HTTP_GET_PATCH,
        new_callable=AsyncMock,
        return_value=_http_response(200, work),
    ):
        assert await resolve_doi("10.1234/abc") == work


async def test_resolve_doi_returns_none_on_404():
    with patch(
        _HTTP_GET_PATCH,
        new_callable=AsyncMock,
        return_value=_http_response(404),
    ):
        assert await resolve_doi("10.1234/abc") is None


async def test_resolve_doi_raises_on_5xx():
    """Non-200, non-404 is a real API failure under typed-errors:
    raise so the orchestrator (resolve_metadata) can catch and try
    the next candidate."""
    with (
        patch(
            _HTTP_GET_PATCH,
            new_callable=AsyncMock,
            return_value=_http_response(500),
        ),
        pytest.raises(RuntimeError, match="status 500"),
    ):
        await resolve_doi("10.1234/abc")


async def test_resolve_doi_propagates_network_error():
    """Network failure (DNS, connection) propagates as the original
    httpx exception so the orchestrator boundary can distinguish."""
    with (
        patch(
            _HTTP_GET_PATCH,
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("boom"),
        ),
        pytest.raises(httpx.ConnectError, match="boom"),
    ):
        await resolve_doi("10.1234/abc")


async def test_resolve_doi_propagates_invalid_json():
    """Malformed 200 body propagates as ValueError so the boundary
    distinguishes a real OpenAlex bug from a legitimate 404 miss."""
    resp = Mock()
    resp.status_code = 200
    resp.json = Mock(side_effect=ValueError("not json"))
    with (
        patch(
            _HTTP_GET_PATCH,
            new_callable=AsyncMock,
            return_value=resp,
        ),
        pytest.raises(ValueError, match="not json"),
    ):
        await resolve_doi("10.1234/abc")


async def test_resolve_metadata_skips_candidate_on_api_failure():
    """resolve_metadata catches resolve_doi raises per-candidate so
    one transient outage doesn't kill the walk of the rest."""
    chunks = [_chunk(0, "10.1234/abc and 10.5678/xyz")]
    work = {"id": "W2"}
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock,
        side_effect=[RuntimeError("transient"), work],
    ) as mock_resolve:
        assert await resolve_metadata(chunks) == work
        assert mock_resolve.call_count == 2


# ---- resolve_metadata ----


async def test_resolve_metadata_returns_none_when_no_candidates():
    chunks = [_chunk(0, "no doi text here")]
    assert await resolve_metadata(chunks) is None


async def test_resolve_metadata_resolves_first_candidate():
    chunks = [_chunk(0, "10.1234/abc")]
    work = {"id": "W1"}
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock,
        return_value=work,
    ) as mock_resolve:
        assert await resolve_metadata(chunks) == work
        mock_resolve.assert_called_once_with("10.1234/abc")


async def test_resolve_metadata_tries_next_when_first_fails():
    chunks = [_chunk(0, "10.1234/abc and 10.5678/xyz")]
    work = {"id": "W2"}
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock,
        side_effect=[None, work],
    ) as mock_resolve:
        assert await resolve_metadata(chunks) == work
        assert mock_resolve.call_count == 2


async def test_resolve_metadata_returns_none_when_all_candidates_fail():
    chunks = [_chunk(0, "10.1234/abc")]
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assert await resolve_metadata(chunks) is None


# ---- doi_from_jats (structured DOI straight from JATS XML) ----


def test_doi_from_jats_reads_article_id_doi(tmp_path: Path):
    p = _write_jats(
        tmp_path, '<article-id pub-id-type="doi">10.1371/journal.pone.0001</article-id>'
    )
    assert doi_from_jats(p) == "10.1371/journal.pone.0001"


def test_doi_from_jats_lowercases(tmp_path: Path):
    p = _write_jats(tmp_path, '<article-id pub-id-type="doi">10.1371/ABC</article-id>')
    assert doi_from_jats(p) == "10.1371/abc"


def test_doi_from_jats_none_when_only_non_doi_ids(tmp_path: Path):
    p = _write_jats(tmp_path, '<article-id pub-id-type="pmid">123456</article-id>')
    assert doi_from_jats(p) is None


def test_doi_from_jats_ignores_reference_pub_ids(tmp_path: Path):
    """A cited reference's <pub-id> DOI must NOT be picked — only the
    article's own <article-id>, so a bibliography can't hijack the DOI."""
    p = tmp_path / "refs.xml"
    p.write_text(
        "<article><front><article-meta>"
        '<article-id pub-id-type="doi">10.1371/self</article-id>'
        "</article-meta></front>"
        "<back><ref-list><ref><element-citation>"
        '<pub-id pub-id-type="doi">10.9999/cited</pub-id>'
        "</element-citation></ref></ref-list></back></article>",
        encoding="utf-8",
    )
    assert doi_from_jats(p) == "10.1371/self"


def test_doi_from_jats_namespace_agnostic(tmp_path: Path):
    p = tmp_path / "ns.xml"
    p.write_text(
        '<article xmlns="http://jats.nlm.nih.gov"><front><article-meta>'
        '<article-id pub-id-type="doi">10.1371/ns</article-id>'
        "</article-meta></front></article>",
        encoding="utf-8",
    )
    assert doi_from_jats(p) == "10.1371/ns"


def test_doi_from_jats_none_on_unparseable(tmp_path: Path):
    p = tmp_path / "bad.xml"
    p.write_text("<not valid xml", encoding="utf-8")
    assert doi_from_jats(p) is None


# ---- collect_doi_candidates (structured-first, deduped) ----


def test_collect_doi_candidates_regex_only_without_source_path():
    chunks = [_chunk(0, "10.1234/abc")]
    assert collect_doi_candidates(chunks) == ["10.1234/abc"]


def test_collect_doi_candidates_structured_doi_first_for_xml(tmp_path: Path):
    p = _write_jats(tmp_path, '<article-id pub-id-type="doi">10.1371/structured</article-id>')
    chunks = [_chunk(0, "in-text mention 10.9999/intext")]
    assert collect_doi_candidates(chunks, source_path=p) == [
        "10.1371/structured",
        "10.9999/intext",
    ]


def test_collect_doi_candidates_dedupes_structured_with_text(tmp_path: Path):
    p = _write_jats(tmp_path, '<article-id pub-id-type="doi">10.1371/same</article-id>')
    chunks = [_chunk(0, "also printed 10.1371/same")]
    assert collect_doi_candidates(chunks, source_path=p) == ["10.1371/same"]


def test_collect_doi_candidates_non_xml_source_ignores_structured(tmp_path: Path):
    """A non-XML source path never triggers JATS reading (suffix-gated)."""
    pdf = tmp_path / "paper.pdf"
    chunks = [_chunk(0, "10.1234/frompdftext")]
    assert collect_doi_candidates(chunks, source_path=pdf) == ["10.1234/frompdftext"]


async def test_resolve_metadata_uses_structured_jats_doi_when_text_has_none(tmp_path: Path):
    """The whole point: an XML article whose DOI is NOT in the chunk text
    still resolves, via the structured <article-id>."""
    p = _write_jats(tmp_path, '<article-id pub-id-type="doi">10.1371/xonly</article-id>')
    chunks = [_chunk(0, "body text with no doi at all")]
    work = {"id": "W1"}
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock,
        return_value=work,
    ) as mock_resolve:
        assert await resolve_metadata(chunks, source_path=p) == work
        mock_resolve.assert_called_once_with("10.1371/xonly")


# ---- is_doi_eligible (Paper-only DOI/OpenAlex gate) ----


def test_is_doi_eligible_true_for_paper():
    from knowledge_agent.kg.schema import PAPER_LABEL

    assert is_doi_eligible(PAPER_LABEL) is True


def test_is_doi_eligible_false_for_generic_article_and_note():
    # A generic Article (news/blog/magazine) and a Note carry no DOI.
    assert is_doi_eligible("Article") is False
    assert is_doi_eligible("Note") is False


def test_is_doi_eligible_false_for_untyped_doc():
    # A doc ingested with no sub_label is not DOI-eligible.
    assert is_doi_eligible(None) is False


def test_is_doi_eligible_false_for_artifact_types():
    assert is_doi_eligible("Dataset") is False
    assert is_doi_eligible("Code") is False
