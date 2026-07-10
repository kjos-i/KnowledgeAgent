"""Unit tests for the LLM eval-case generator.

`generate_cases` is tested with an INJECTED fake LLM (a `RunnableLambda`
returning a canned `GeneratedCase`), so no real provider/network is
touched. `with_retry` is patched to identity module-wide so the fake
runs directly (no backoff, no `get_settings`). `sample_passages` is
tested with a fake search client. `generate_from_corpus` is live glue
(sample + generate) — exercised only in integration.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from knowledge_agent.evaluation.generator import (
    EvalGenerationConnectionError,
    GeneratedCase,
    Passage,
    generate_cases,
    sample_passages,
)
from knowledge_agent.evaluation.models import validate_case

# A valid, fully-pinned global retrieval config the generator reads to stamp
# each case. Isolated from the real Settings / user .env.
_FAKE_SETTINGS = SimpleNamespace(
    mode_classifier_model="fake-model",
    default_retrieval_mode="lancedb_only",
    lancedb_search_mode="hybrid",
    top_k=5,
    num_candidates=40,
    rrf_rank_constant=60,
    mmr_lambda=0.5,
    default_use_mmr=False,
    kg_max_rows=50,
)


@pytest.fixture(autouse=True)
def _identity_with_retry(monkeypatch):
    """Make `with_retry` a pass-through so the fake runnable runs directly
    (no backoff loop) and stub `get_settings` so the generator pins retrieval
    from a fake config — keeps these unit tests fast + isolated from the real
    Settings / user .env."""
    import knowledge_agent.evaluation.generator as gen_mod

    monkeypatch.setattr(gen_mod, "with_retry", lambda r: r)
    monkeypatch.setattr("knowledge_agent.config.get_settings", lambda: _FAKE_SETTINGS)


def _fake_llm(structured_result):
    """A fake chat model whose `with_structured_output(...)` returns a
    runnable yielding `structured_result` (a GeneratedCase, or a callable
    to run for its side effect)."""
    fn = structured_result if callable(structured_result) else (lambda _m: structured_result)
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(fn)
    return llm


# ---- generate_cases ----


async def test_generate_cases_builds_llm_origin_cases():
    llm = _fake_llm(
        GeneratedCase(
            question="What powers the lander?",
            answer_points=["A radioisotope thermoelectric generator"],
            keywords=["radioisotope", "generator"],
        )
    )
    cases = await generate_cases([Passage("doc-a", "x" * 300)], llm=llm)

    assert len(cases) == 1
    c = cases[0]
    assert c.origin == "llm"
    assert c.expected_sources == ["doc-a"]
    assert c.question == "What powers the lander?"
    assert c.required_keywords == ["radioisotope", "generator"]
    assert c.expected_answer_points == ["A radioisotope thermoelectric generator"]
    assert c.category == "generated"
    assert "review" in c.notes.lower()
    assert c.id.startswith("gen-00-")


async def test_generate_cases_one_case_per_passage_with_indexed_ids():
    llm = _fake_llm(GeneratedCase(question="Q?", answer_points=[], keywords=[]))
    cases = await generate_cases([Passage("d1", "a" * 300), Passage("d2", "b" * 300)], llm=llm)
    assert [c.expected_sources for c in cases] == [["d1"], ["d2"]]
    assert cases[0].id.startswith("gen-00-")
    assert cases[1].id.startswith("gen-01-")


async def test_generate_cases_skips_empty_question():
    llm = _fake_llm(GeneratedCase(question="   ", answer_points=[], keywords=[]))
    cases = await generate_cases([Passage("d1", "x" * 300)], llm=llm)
    assert cases == []


async def test_generate_cases_skips_passage_on_llm_error():
    def _boom(_messages):
        raise RuntimeError("llm down")

    cases = await generate_cases([Passage("d1", "x" * 300)], llm=_fake_llm(_boom))
    assert cases == []


async def test_generated_cases_are_runnable_with_pinned_retrieval():
    """Each case pins every retrieval knob from the global defaults, so it
    passes `validate_case` (the runner requires pinned knobs). Regression for
    the bug where generated cases left num_candidates/rrf_rank_constant None
    and the runner refused the whole dataset."""
    llm = _fake_llm(GeneratedCase(question="Q?", answer_points=[], keywords=[]))
    cases = await generate_cases([Passage("d1", "x" * 300)], llm=llm)
    assert cases
    r = cases[0].retrieval
    assert r.retrieval_mode == _FAKE_SETTINGS.default_retrieval_mode
    assert r.num_candidates == _FAKE_SETTINGS.num_candidates
    assert r.rrf_rank_constant == _FAKE_SETTINGS.rrf_rank_constant
    assert validate_case(cases[0]) == []  # runnable out of the box


async def test_generate_cases_aborts_on_connection_error():
    """A connection/network failure aborts the batch with a clear, retryable
    error — NOT a silent per-passage skip (which shrank the batch mysteriously)."""

    def _conn_boom(_messages):
        raise ConnectionError("Connection error.")

    with pytest.raises(EvalGenerationConnectionError, match="network"):
        await generate_cases([Passage("d1", "x" * 300)], llm=_fake_llm(_conn_boom))


# ---- sample_passages ----


def _fake_client(docs, chunks_by_doc):
    client = MagicMock()
    client.list_indexed_docs = AsyncMock(return_value=docs)
    client.get_chunks_by_doc_id = AsyncMock(side_effect=lambda doc_id: chunks_by_doc[doc_id])
    return client


async def test_sample_passages_one_per_doc_over_min_chars():
    client = _fake_client(
        [{"doc_id": "d1"}, {"doc_id": "d2"}],
        {
            "d1": [{"text": "too short"}, {"text": "y" * 300}],  # skip the short one
            "d2": [{"text": "z" * 300}],
        },
    )
    passages = await sample_passages(5, client=client, min_chars=200)
    assert [p.doc_id for p in passages] == ["d1", "d2"]
    assert all(len(p.text) >= 200 for p in passages)


async def test_sample_passages_caps_at_n():
    client = _fake_client(
        [{"doc_id": "d1"}, {"doc_id": "d2"}],
        {"d1": [{"text": "a" * 300}], "d2": [{"text": "b" * 300}]},
    )
    passages = await sample_passages(1, client=client, min_chars=200)
    assert len(passages) == 1


async def test_sample_passages_zero_returns_empty():
    # n <= 0 short-circuits before any client call.
    assert await sample_passages(0) == []
