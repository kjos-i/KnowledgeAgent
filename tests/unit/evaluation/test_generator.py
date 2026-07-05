"""Unit tests for the LLM eval-case generator.

`generate_cases` is tested with an INJECTED fake LLM (a `RunnableLambda`
returning a canned `GeneratedCase`), so no real provider/network is
touched. `with_retry` is patched to identity module-wide so the fake
runs directly (no backoff, no `get_settings`). `sample_passages` is
tested with a fake search client. `generate_from_corpus` is live glue
(sample + generate) — exercised only in integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from knowledge_agent.evaluation.generator import (
    GeneratedCase,
    Passage,
    generate_cases,
    sample_passages,
)


@pytest.fixture(autouse=True)
def _identity_with_retry(monkeypatch):
    """Make `with_retry` a pass-through so the fake runnable runs directly
    (no backoff loop, no `get_settings`) — keeps these unit tests fast +
    isolated from real Settings."""
    import knowledge_agent.evaluation.generator as gen_mod

    monkeypatch.setattr(gen_mod, "with_retry", lambda r: r)


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
