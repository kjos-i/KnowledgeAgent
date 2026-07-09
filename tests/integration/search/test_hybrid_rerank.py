"""Integration test for the hybrid-search RRFReranker path — real LanceDB.

The unit tests in `tests/unit/search/test_search_client.py` prove that
`hybrid_search` *wires* an explicit `RRFReranker(K=rrf_rank_constant)`
and a `.limit(num_candidates)` onto the query builder (against a recording
stub). This file proves that the wired chain actually RUNS against a live
LanceDB table — the dead-knob fix changed the live query plan, so a stub
alone can't confirm LanceDB accepts the explicit reranker + candidate pool.

Self-contained: builds a real LanceDB table in a `tmp_path` directory with
5 rows carrying fixed fake embedding vectors + text, monkeypatches
`_embed_query` to a fixed vector (no Voyage key needed), builds the FTS
index (required for the hybrid BM25 leg), then runs `hybrid_search` and
asserts it returns at most `top_k` rows without raising.

LanceDB is embedded — no server / port. Uses `tmp_path` rather than the
`.env.test` `LANCEDB_PATH` so the test needs no external corpus and cleans
up automatically.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

import knowledge_agent.search.client as client_mod
from knowledge_agent.config import Settings
from knowledge_agent.search.client import LanceClient

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

_DOC_ID = "integ-hybrid-rerank-001"


def _settings(lancedb_path: Path) -> Settings:
    """Minimal Settings pointing at a throwaway LanceDB dir, with a
    non-default `rrf_rank_constant` + `num_candidates` so the assertions
    can't accidentally match LanceDB's built-in `RRFReranker(K=60)`."""
    return Settings(
        _env_file=None,
        anthropic_api_key="fake-anthropic",  # pragma: allowlist secret
        voyage_api_key="fake-voyage",  # pragma: allowlist secret
        neo4j_uri="neo4j://127.0.0.1:7687",
        neo4j_user="neo4j",
        neo4j_password="fake-password",  # pragma: allowlist secret
        lancedb_path=lancedb_path,
        rrf_rank_constant=17,
        num_candidates=20,
        top_k=3,
    )


def _synthetic_chunk(index: int, text: str, dims: int) -> dict[str, Any]:
    """One schema-conformant chunk row with a fixed one-hot embedding
    (distinct per row so the vector leg has something to rank)."""
    vector = [0.0] * dims
    vector[index % dims] = 1.0
    return {
        "chunk_id": f"{_DOC_ID}#{index}",
        "doc_id": _DOC_ID,
        "chunk_index": index,
        "section": "Introduction",
        "page": 1,
        "char_start": index * 100,
        "char_end": index * 100 + len(text),
        "content_type": "text",
        "image_ref": None,
        "text": text,
        "embedding": vector,
        "main_label": "Document",
        "sub_label": "Paper",
        "doi": None,
        "openalex_id": None,
        "title": "A Synthetic Paper",
        "year": 2026,
        "authors_display": "Integration, J.",
        "venue": None,
        "source_url": None,
        "metadata_status": "enriched",
        "language": "en",
        "source_path": "/tmp/integration-hybrid.pdf",
        "ingested_at": datetime.now(),
    }


async def test_hybrid_search_runs_explicit_rrf_reranker_against_real_lancedb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full hybrid path (BM25 + vector, fused by an EXPLICIT
    `RRFReranker(K=rrf_rank_constant)`, pool=`num_candidates`) executes
    against a live LanceDB table and returns <= top_k rows.

    This is the coverage the unit stub can't give: it confirms LanceDB
    accepts the explicit reranker + candidate-pool limit the dead-knob
    fix now attaches, and that the `_relevance_score` round-trips into
    `RetrievedChunk.score`.
    """
    settings = _settings(tmp_path / "lancedb")
    dims = settings.embedding_dims
    client = LanceClient(settings=settings)

    await client.ensure_schema()
    await client.write_chunks(
        [
            _synthetic_chunk(0, "membrane biology and the ESCRT-III spiral assembly", dims),
            _synthetic_chunk(
                1, "ESCRT machinery constricts the membrane neck during fission", dims
            ),
            _synthetic_chunk(2, "general discussion of cellular transport pathways", dims),
            _synthetic_chunk(3, "spiral filament assembly on the inner membrane surface", dims),
            _synthetic_chunk(4, "unrelated notes on crystallography methods", dims),
        ]
    )
    # FTS index is required for the hybrid BM25 leg. ensure_indexes builds
    # it regardless of row count (the vector index is skipped below the
    # training threshold, which is fine — brute-force scan handles the
    # 5-row vector leg).
    await client.ensure_indexes()

    # Fixed query vector — no Voyage call. One-hot aligned with row 0 so
    # the vector leg has a definite nearest neighbour.
    fixed_vector = [0.0] * dims
    fixed_vector[0] = 1.0

    async def _fake_embed(_text: str) -> list[float]:
        return fixed_vector

    monkeypatch.setattr(client_mod, "_embed_query", _fake_embed)

    # Explicit rrf_k (the fixed knob) — this is the path the eval case
    # will exercise. Must not raise, and must truncate the pool to top_k.
    hits = await client.hybrid_search(
        "ESCRT membrane spiral",
        top_k=settings.top_k,
        num_candidates=settings.num_candidates,
        rrf_k=settings.rrf_rank_constant,
    )

    # The pool held 5 rows; top_k=3 must cap the visible result.
    assert len(hits) <= settings.top_k
    assert hits, "hybrid search should surface at least one row for a matching query"
    # RRF fused score (`_relevance_score`) round-tripped into `score`.
    assert hits[0].score is not None
    # No duplicate chunk_ids in the returned slice.
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids))


async def test_hybrid_search_falls_back_to_settings_num_candidates_and_rrf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no explicit num_candidates / rrf_k, hybrid_search resolves
    both from settings and still runs end-to-end against real LanceDB
    (the default global-behaviour path callers use today)."""
    settings = _settings(tmp_path / "lancedb")
    dims = settings.embedding_dims
    client = LanceClient(settings=settings)

    await client.ensure_schema()
    await client.write_chunks(
        [_synthetic_chunk(i, f"membrane escrt spiral chunk {i}", dims) for i in range(5)]
    )
    await client.ensure_indexes()

    fixed_vector = [0.0] * dims
    fixed_vector[0] = 1.0

    async def _fake_embed(_text: str) -> list[float]:
        return fixed_vector

    monkeypatch.setattr(client_mod, "_embed_query", _fake_embed)

    hits = await client.hybrid_search("membrane escrt")  # settings supply the rest

    assert len(hits) <= settings.top_k
    assert hits
