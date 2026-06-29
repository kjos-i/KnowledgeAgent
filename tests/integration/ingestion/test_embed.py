"""Integration tests for `ingestion/embed` — real Voyage API calls.

`embed_texts` is the Voyage wrapper used at both ingest time
(`pipeline.py`) and query time (search/client `_embed_query`).
This file exercises it directly against the real Voyage multimodal
endpoint.

Cost: one Voyage call per test (~$0.001). Network required.

The function is also covered transitively by `test_pipeline.py` and
`test_search_client.py`; this file pins the wrapper contract
directly (input shapes, dimension match, empty-input optimization).

No manual interactive counterpart — `smoke_metadata.py` and
`smoke_pipeline.py` exercise it indirectly.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

import pytest

from knowledge_agent.config import get_settings
from knowledge_agent.ingestion.embed import embed_texts

pytestmark = pytest.mark.integration


def test_embed_texts_returns_one_vector_per_input() -> None:
    """N inputs in → N vectors out, in the same order."""
    inputs = [
        "Aspirin reduces the risk of heart attack.",
        "BRCA1 mutations increase breast cancer risk.",
        "Metformin is a first-line treatment for type 2 diabetes.",
    ]
    vectors = embed_texts(inputs)
    assert vectors is not None
    assert len(vectors) == len(inputs)


def test_embed_texts_returns_correct_dimension() -> None:
    """Each vector matches `settings.embedding_dims` — the dimension
    is contractually pinned to the Voyage model + the LanceDB schema
    expects exactly this size."""
    expected_dim = get_settings().embedding_dims
    vectors = embed_texts(["one short input"])
    assert vectors is not None
    assert len(vectors) == 1
    assert len(vectors[0]) == expected_dim


def test_embed_texts_empty_input_returns_empty_list_no_api_call() -> None:
    """Empty input is the fast-path optimization — no Voyage call."""
    vectors = embed_texts([])
    assert vectors == []


def test_embed_texts_document_vs_query_input_type_both_succeed() -> None:
    """The wrapper supports both index-side (`input_type="document"`)
    and query-side (`input_type="query"`) calls. Both should succeed
    for the same input."""
    text = "What does the article say about nutrition?"
    doc_vec = embed_texts([text], input_type="document")
    query_vec = embed_texts([text], input_type="query")
    assert doc_vec is not None
    assert query_vec is not None
    assert len(doc_vec[0]) == len(query_vec[0])


def test_embed_texts_vectors_are_floats_in_finite_range() -> None:
    """Sanity: every vector component is a finite float — catches
    NaN / Inf corruption from a misconfigured model."""
    vectors = embed_texts(["sanity-check text"])
    assert vectors is not None
    import math
    for v in vectors:
        for x in v:
            assert isinstance(x, float)
            assert math.isfinite(x)
