"""Integration tests for failure paths across the major boundaries.

Most other integration tests cover the HAPPY path ("when everything
works, the result is X"). This file covers the OTHER side: what
happens when things FAIL. Each test deliberately injects a failure
at one boundary client (Voyage, OpenAlex, Neo4j driver, LanceDB,
Anthropic) and asserts the layer above handles it according to the
documented contract.

Contract today: every
layer fails soft with sentinels — embed returns None, KG writes
return False, OpenAlex resolution returns None, agent nodes set
state fields without raising. The typed-errors refactor (a parked
plan) will eventually change this contract — when that ships, these
tests need updating
to assert the typed `ErrorDetail` carries the right message instead
of just the sentinel.

Pattern: monkeypatch the boundary client at the appropriate level
(`voyageai.Client.multimodal_embed`, `neo4j.Driver.session`,
`httpx.get`, etc.) so the real test infrastructure (test Neo4j,
test LanceDB) isn't actually corrupted by the failure injection.

**Cost: NO LLM cost.** All Anthropic / LLM-adapter calls in this
file are MOCKED via monkeypatch — the whole point of failure-path
testing is to inject failures without hitting paid APIs. Only the
test Neo4j and (when reached) test LanceDB get touched.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Voyage embedding failures.
# ---------------------------------------------------------------------------


async def test_embed_texts_propagates_voyage_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Voyage client raises → `embed_texts` propagates the exception
    under the typed-errors contract. The pipeline orchestrator's
    try/except is what flips `embed_ok=False` and records the typed
    error on `IngestResult.embed_error`."""
    from knowledge_agent import embedder_factory
    from knowledge_agent.ingestion import embed

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated Voyage outage")

    monkeypatch.setattr(
        embedder_factory._build_voyage_client(),
        "multimodal_embed",
        boom,
    )

    with pytest.raises(RuntimeError, match="simulated Voyage outage"):
        await embed.embed_texts(["any text"])


async def test_embed_texts_propagates_voyage_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network-level exception (e.g. timeout) from the Voyage
    client also propagates under typed-errors."""
    from knowledge_agent import embedder_factory
    from knowledge_agent.ingestion import embed

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("simulated connection reset")

    monkeypatch.setattr(
        embedder_factory._build_voyage_client(),
        "multimodal_embed",
        timeout,
    )

    with pytest.raises(httpx.ConnectError, match="simulated connection reset"):
        await embed.embed_texts(["any text"])


async def test_embed_texts_empty_input_does_not_call_voyage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: an empty input list short-circuits BEFORE calling
    Voyage. Even if Voyage would raise, we return [] without ever
    touching the client. Protects against accidental API charges
    for no-op calls."""
    from knowledge_agent import embedder_factory
    from knowledge_agent.ingestion import embed

    called: list[Any] = []

    def track(*args: Any, **kwargs: Any) -> Any:
        called.append((args, kwargs))
        raise RuntimeError("should not be called")

    monkeypatch.setattr(
        embedder_factory._build_voyage_client(),
        "multimodal_embed",
        track,
    )

    result = await embed.embed_texts([])
    assert result == []
    assert called == []


# ---------------------------------------------------------------------------
# OpenAlex HTTP failures.
# ---------------------------------------------------------------------------


async def test_resolve_doi_propagates_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection error during the OpenAlex request propagates under
    typed-errors. `resolve_metadata` (the per-candidate orchestrator)
    catches per-candidate so the walk continues; the pipeline marks
    the doc 'pending' when no candidate succeeds."""
    from knowledge_agent import _http_client
    from knowledge_agent.ingestion import metadata

    async def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise httpx.ConnectError("simulated connection reset")

    monkeypatch.setattr(_http_client, "request", boom)

    with pytest.raises(httpx.ConnectError, match="simulated connection reset"):
        await metadata.resolve_doi("10.1234/anything")


async def test_resolve_doi_raises_on_500_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-200, non-404 response from OpenAlex raises under
    typed-errors so the orchestrator can distinguish 'API down' from
    'DOI not found'."""
    from knowledge_agent import _http_client
    from knowledge_agent.ingestion import metadata

    class _FakeResponse:
        status_code = 503

        def json(self) -> Any:
            raise AssertionError("should not be called on non-200")

    async def fake_get(*_args: Any, **_kwargs: Any) -> Any:
        return _FakeResponse()

    monkeypatch.setattr(_http_client, "request", fake_get)

    with pytest.raises(RuntimeError, match="status 503"):
        await metadata.resolve_doi("10.1234/anything")


async def test_resolve_doi_propagates_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 response with malformed JSON propagates as ValueError —
    typed-errors distinguishes 'real OpenAlex bug' from 404 miss."""
    from knowledge_agent import _http_client
    from knowledge_agent.ingestion import metadata

    class _FakeResponse:
        status_code = 200

        def json(self) -> Any:
            raise ValueError("malformed JSON from openalex")

    async def fake_get(*_args: Any, **_kwargs: Any) -> Any:
        return _FakeResponse()

    monkeypatch.setattr(_http_client, "request", fake_get)

    with pytest.raises(ValueError, match="malformed JSON from openalex"):
        await metadata.resolve_doi("10.1234/anything")


# ---------------------------------------------------------------------------
# Neo4j driver failures — KG leaves RAISE; orchestrator boundary
# catches and stores on the matching `_error: ErrorDetail | None`
# field of the user-facing result dataclass. These tests pin the
# leaf-raises half of the contract; `test_pipeline.py` /
# `test_bulk_ops.py` cover the orchestrator-catches half.
# ---------------------------------------------------------------------------


async def test_kg_chunk_write_propagates_when_session_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the Neo4j driver's session.run raises (connection drop,
    Cypher syntax error, etc.) the KG write propagates the exception
    — the orchestrator boundary catches it."""
    from knowledge_agent.kg import chunk_writes

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def run(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated Neo4j connection drop")

    class _FakeDriver:
        def session(self) -> _FakeSession:
            return _FakeSession()

    class _FakeClient:
        driver = _FakeDriver()

    class _Chunk:
        chunk_index = 0
        section = "Intro"
        page = 1
        content_type = "text"

    with pytest.raises(RuntimeError, match="simulated Neo4j connection drop"):
        await chunk_writes.write_chunks(
            _FakeClient(),
            "doc-1",
            [_Chunk()],
            "Document",
            "Paper",
        )


async def test_kg_delete_chunks_propagates_when_session_raises() -> None:
    """delete_chunks_by_doc_id propagates the exception on driver
    failure; the orchestrator boundary catches."""
    from knowledge_agent.kg import chunk_writes

    class _FakeSession:
        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def run(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("simulated drop")

    class _FakeDriver:
        def session(self) -> _FakeSession:
            return _FakeSession()

    class _FakeClient:
        driver = _FakeDriver()

    with pytest.raises(RuntimeError, match="simulated drop"):
        await chunk_writes.delete_chunks_by_doc_id(_FakeClient(), "doc-1")


async def test_kg_writes_reject_empty_doc_id_with_value_error() -> None:
    """Invalid input (empty doc_id) raises ValueError — catches caller
    bugs at the leaf with a clear message instead of failing silently
    or returning False."""
    from knowledge_agent.kg import chunk_writes

    class _FakeClient:
        driver = None  # never reached — empty doc_id short-circuits

    with pytest.raises(ValueError, match="no doc_id"):
        await chunk_writes.delete_chunks_by_doc_id(_FakeClient(), "")
    with pytest.raises(ValueError, match="no doc_id"):
        await chunk_writes.write_chunks(
            _FakeClient(),
            "",
            [],
            "Document",
            "Paper",
        )


# ---------------------------------------------------------------------------
# KG client cypher reads — read_query is the ONLY KG path that raises
# (per the docstring contract). The neo4j_retriever node catches.
# ---------------------------------------------------------------------------


async def test_read_query_raises_on_cypher_error(
    kg_client: Any,
    ensure_constraints: None,
) -> None:
    """`read_query` deliberately does NOT fail-soft (per its
    docstring). A bad Cypher string raises, which the caller
    (neo4j_retriever) wraps. This pin keeps that contract from
    drifting under refactors."""
    from neo4j.exceptions import Neo4jError

    with pytest.raises(Neo4jError):
        await kg_client.read_query("THIS IS NOT VALID CYPHER")


# ---------------------------------------------------------------------------
# LanceDB failures — writes propagate the table-missing error; reads
# treat missing-table as an empty result.
# ---------------------------------------------------------------------------


async def test_lance_write_chunks_raises_when_table_missing(
    lance_client: Any,
    clean_lance: None,
) -> None:
    """If the chunks table doesn't exist (no ensure_schema yet),
    write_chunks propagates LanceDB's table-missing error rather than
    auto-creating the table — auto-creating would mask schema drift.

    Under the typed-errors contract, this surfaces as a raise from the
    leaf; the pipeline orchestrator's try/except is what flips
    `lancedb_ok=False` and records the `lancedb_error` field.
    """
    # clean_lance dropped the table; we deliberately skip ensure_schema.
    with pytest.raises(Exception):
        await lance_client.write_chunks(
            [
                {
                    "chunk_id": "x#0",
                    "doc_id": "x",
                    "chunk_index": 0,
                    "section": None,
                    "page": None,
                    "char_start": 0,
                    "char_end": 0,
                    "content_type": "text",
                    "image_ref": None,
                    "text": "hello",
                    "embedding": [0.0] * 1024,
                    "main_label": "Document",
                    "sub_label": "Paper",
                    "doi": None,
                    "openalex_id": None,
                    "title": None,
                    "year": None,
                    "authors_display": None,
                    "venue": None,
                    "source_url": None,
                    "metadata_status": "baseline",
                    "language": "en",
                    "source_path": "x",
                    "ingested_at": __import__("datetime").datetime.now(),
                }
            ]
        )


async def test_lance_get_chunks_by_doc_id_on_missing_table_returns_empty(
    lance_client: Any,
    clean_lance: None,
) -> None:
    """`get_chunks_by_doc_id` on a missing table returns an empty list
    (not a raise). Sync/backfill workflows rely on this when the corpus
    is brand-new."""
    rows = await lance_client.get_chunks_by_doc_id("does-not-exist")
    assert rows == []


# ---------------------------------------------------------------------------
# OpenAlex pipeline path — bogus DOI returns None metadata, ingest
# would still continue (state would say metadata_status=baseline /
# pending). Covered by test_metadata's known-404 test; we add the
# composition path here.
# ---------------------------------------------------------------------------


async def test_resolve_metadata_returns_none_when_all_candidates_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If extract_doi_candidates returns several DOIs but EVERY one
    404s at OpenAlex, resolve_metadata returns None."""
    from knowledge_agent import _http_client
    from knowledge_agent.ingestion import metadata
    from knowledge_agent.ingestion.parsers import base

    class _FakeResponse:
        status_code = 404

        def json(self) -> Any:
            return {}

    async def fake_get(*_args: Any, **_kwargs: Any) -> Any:
        return _FakeResponse()

    monkeypatch.setattr(_http_client, "request", fake_get)

    chunks = [
        base.ParsedChunk(
            chunk_index=0,
            text="DOI: 10.1234/fake-a and 10.1234/fake-b",
        ),
    ]
    result = await metadata.resolve_metadata(chunks)
    assert result is None


# ---------------------------------------------------------------------------
# Pipeline composition — a missing extension is rejected at the
# input-validation boundary, NOT mid-pipeline.
# ---------------------------------------------------------------------------


async def test_ingest_document_rejects_unsupported_extension(tmp_path: Any) -> None:
    """ingest_document validates extension BEFORE side effects
    (parse, embed, write). Catches misuse before any cost is paid."""
    from knowledge_agent.ingestion.pipeline import ingest_document
    from knowledge_agent.kg.corpus_config import (
        CorpusConfig,
        LayerFlags,
    )

    bogus = tmp_path / "data.zzz"
    bogus.write_text("hello")
    config = CorpusConfig(
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    with pytest.raises(ValueError, match="No parser available"):
        await ingest_document(bogus, config, "Document", None)


async def test_ingest_document_rejects_invalid_main_label(tmp_path: Any) -> None:
    """Validates main_label up front. Same rationale as the extension
    check — fail before parsing."""
    from knowledge_agent.ingestion.pipeline import ingest_document
    from knowledge_agent.kg.corpus_config import (
        CorpusConfig,
        LayerFlags,
    )

    fake = tmp_path / "data.pdf"
    fake.write_text("not really a pdf, but the extension check passes")
    config = CorpusConfig(
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )
    with pytest.raises(ValueError, match="main_label"):
        await ingest_document(fake, config, "NotALabel", "Paper")
