"""Tests for ingestion.metadata - DOI extraction + OpenAlex resolution.

DOI extraction is pure (regex on text), tested directly. OpenAlex calls
go through the central `_http_client` and are stubbed by patching
`knowledge_agent.ingestion.metadata._http_client.get` with AsyncMock.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from knowledge_agent.ingestion.metadata import (
    extract_doi_candidates,
    resolve_doi,
    resolve_metadata,
)
from knowledge_agent.ingestion.parse import ParsedChunk


def _chunk(index: int, text: str) -> ParsedChunk:
    return ParsedChunk(chunk_index=index, text=text)


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
        _HTTP_GET_PATCH, new_callable=AsyncMock,
        return_value=_http_response(200, work),
    ):
        assert await resolve_doi("10.1234/abc") == work


async def test_resolve_doi_returns_none_on_404():
    with patch(
        _HTTP_GET_PATCH, new_callable=AsyncMock,
        return_value=_http_response(404),
    ):
        assert await resolve_doi("10.1234/abc") is None


async def test_resolve_doi_raises_on_5xx():
    """Non-200, non-404 is a real API failure under typed-errors:
    raise so the orchestrator (resolve_metadata) can catch and try
    the next candidate."""
    with patch(
        _HTTP_GET_PATCH, new_callable=AsyncMock,
        return_value=_http_response(500),
    ):
        with pytest.raises(RuntimeError, match="status 500"):
            await resolve_doi("10.1234/abc")


async def test_resolve_doi_propagates_network_error():
    """Network failure (DNS, connection) propagates as the original
    httpx exception so the orchestrator boundary can distinguish."""
    with patch(
        _HTTP_GET_PATCH, new_callable=AsyncMock,
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(httpx.ConnectError, match="boom"):
            await resolve_doi("10.1234/abc")


async def test_resolve_doi_propagates_invalid_json():
    """Malformed 200 body propagates as ValueError so the boundary
    distinguishes a real OpenAlex bug from a legitimate 404 miss."""
    resp = Mock()
    resp.status_code = 200
    resp.json = Mock(side_effect=ValueError("not json"))
    with patch(
        _HTTP_GET_PATCH, new_callable=AsyncMock, return_value=resp,
    ):
        with pytest.raises(ValueError, match="not json"):
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
        new_callable=AsyncMock, return_value=work,
    ) as mock_resolve:
        assert await resolve_metadata(chunks) == work
        mock_resolve.assert_called_once_with("10.1234/abc")


async def test_resolve_metadata_tries_next_when_first_fails():
    chunks = [_chunk(0, "10.1234/abc and 10.5678/xyz")]
    work = {"id": "W2"}
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock, side_effect=[None, work],
    ) as mock_resolve:
        assert await resolve_metadata(chunks) == work
        assert mock_resolve.call_count == 2


async def test_resolve_metadata_returns_none_when_all_candidates_fail():
    chunks = [_chunk(0, "10.1234/abc")]
    with patch(
        "knowledge_agent.ingestion.metadata.resolve_doi",
        new_callable=AsyncMock, return_value=None,
    ):
        assert await resolve_metadata(chunks) is None
