"""Tests for the LLM triples extractor (L8).

Mocks the ChatAnthropic chain - same pattern as
test_entity_extractors_llm.py. We verify:

  - Fast-path: empty entity_vocab -> [] without calling the LLM.
  - Prompt construction: vocab is rendered into the system prompt.
  - Vocab constraint: LLM output referencing keys not in the vocab is
    dropped post-hoc.
  - Predicate constraint: LLM output with predicates outside the 15 is
    dropped.
  - Vocab lookup: subject_entity_type / object_entity_type are filled
    from the vocab (the LLM only returns keys, not types).
  - Defensive fallback when structured output returns something
    unexpected.

Model + temperature are per-corpus kwargs threaded through the
extractor call — 2026-07-02 refactor removed the get_settings() global
read. Tests pass them explicitly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.ingestion import triples_extractor
from knowledge_agent.ingestion.triples_extractor import (
    _LLMTriple,
    _LLMTriples,
)
from knowledge_agent.kg.triples_writes import ExtractedTriple

# Constants for the model + temperature kwargs. Value doesn't matter
# except in `test_extract_propagates_model_and_temperature_to_get_llm`.
_TEST_MODEL = "claude-haiku"
_TEST_TEMPERATURE = 0.0


def _mock_llm_returning(structured_output):
    """Build the mock chain: llm.with_structured_output(...).invoke(...) -> output.

    `with_retry` is wired so that the production code's `_with_retry(structured)`
    call resolves to the same mock — i.e. the retry wrapper is the identity in
    tests. We test contract, not retry semantics; retry behaviour is covered
    by `llm_factory.with_retry` tests in isolation.
    """
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=structured_output)
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


# ---- Fast-path: empty vocab ----


async def test_extract_empty_vocab_returns_empty_without_llm_call():
    """No L6 entities = no possible triples. Don't waste an LLM call."""
    mock_llm = _mock_llm_returning(_LLMTriples(triples=[]))
    mock_structured = mock_llm.with_structured_output.return_value
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "some text",
            [],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []
    mock_structured.ainvoke.assert_not_called()


# ---- Prompt construction ----


def test_build_system_prompt_renders_entity_vocab():
    prompt = triples_extractor._build_system_prompt([("brca1", "GENE"), ("tp53", "GENE")])
    assert "brca1" in prompt
    assert "tp53" in prompt
    assert "GENE" in prompt


def test_build_system_prompt_lists_allowed_predicates():
    prompt = triples_extractor._build_system_prompt([("brca1", "GENE")])
    # Sample predicates - the prompt should mention these.
    assert "INHIBITS" in prompt
    assert "ACTIVATES" in prompt
    assert "BINDS_TO" in prompt
    assert "COMPARED_WITH" in prompt


def test_build_system_prompt_handles_no_vocab_gracefully():
    """Defensive - empty vocab shouldn't blow up the template render
    even though the extractor short-circuits before calling it."""
    prompt = triples_extractor._build_system_prompt([])
    assert "(none)" in prompt


# ---- End-to-end extract() with mocked LLM ----


async def test_extract_maps_valid_triple_to_extracted_triple():
    output = _LLMTriples(
        triples=[
            _LLMTriple(
                subject_key="brca1",
                predicate="INHIBITS",
                object_key="tp53",
                evidence_span="BRCA1 inhibits TP53 expression.",
            )
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "BRCA1 inhibits TP53 expression.",
            [("brca1", "GENE"), ("tp53", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == [
        ExtractedTriple(
            subject_key="brca1",
            subject_entity_type="GENE",
            predicate="INHIBITS",
            object_key="tp53",
            object_entity_type="GENE",
            evidence_span="BRCA1 inhibits TP53 expression.",
        )
    ]


async def test_extract_drops_triple_with_subject_not_in_vocab():
    """LLM hallucinated an entity - filter at extractor boundary."""
    output = _LLMTriples(
        triples=[
            _LLMTriple(
                subject_key="palb2",  # not in vocab
                predicate="INHIBITS",
                object_key="tp53",
                evidence_span="...",
            )
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "text",
            [("brca1", "GENE"), ("tp53", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []


async def test_extract_drops_triple_with_object_not_in_vocab():
    output = _LLMTriples(
        triples=[
            _LLMTriple(
                subject_key="brca1",
                predicate="INHIBITS",
                object_key="palb2",  # not in vocab
                evidence_span="...",
            )
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "text",
            [("brca1", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []


async def test_extract_drops_triple_with_unknown_predicate():
    """LLM ignored the constrained vocabulary - filter."""
    output = _LLMTriples(
        triples=[
            _LLMTriple(
                subject_key="brca1",
                predicate="BOGUS_VERB",  # not in TRIPLE_PREDICATE_RELS
                object_key="tp53",
                evidence_span="...",
            )
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "text",
            [("brca1", "GENE"), ("tp53", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []


async def test_extract_fills_entity_types_from_vocab_lookup():
    """LLM returns only keys; the extractor fills types from the vocab."""
    output = _LLMTriples(
        triples=[
            _LLMTriple(
                subject_key="aspirin",
                predicate="TREATS",
                object_key="headache",
                evidence_span="Aspirin treats headache.",
            )
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "Aspirin treats headache.",
            [("aspirin", "CHEMICAL"), ("headache", "DISEASE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result[0].subject_entity_type == "CHEMICAL"
    assert result[0].object_entity_type == "DISEASE"


async def test_extract_propagates_model_and_temperature_to_get_llm():
    """Extractor forwards the per-corpus model + temperature kwargs
    verbatim to `_get_llm` (which builds the ChatAnthropic client)."""
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ) as mock_get_llm:
        await triples_extractor.extract(
            "text",
            [("a", "GENE"), ("b", "GENE")],
            model="test-model",
            temperature=0.4,
        )

    mock_get_llm.assert_called_once_with("test-model", 0.4)


async def test_extract_binds_llmtriples_as_structured_output_schema():
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        await triples_extractor.extract(
            "text",
            [("a", "GENE"), ("b", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    mock_llm.with_structured_output.assert_called_once_with(_LLMTriples)


async def test_extract_defensive_fallback_on_non_llmtriples_result():
    """If structured output drifts (e.g. returns a dict), fail closed."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value={"triples": []})
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "text",
            [("a", "GENE"), ("b", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []


async def test_extract_returns_empty_when_llm_returns_no_triples():
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract(
            "Plain text with no relations.",
            [("brca1", "GENE"), ("tp53", "GENE")],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == []


# ---- extract_batched (window of N consecutive chunks) ----------------------


def _et(subject: str, predicate: str, obj: str, evidence: str) -> ExtractedTriple:
    return ExtractedTriple(
        subject_key=subject,
        subject_entity_type="T",
        predicate=predicate,
        object_key=obj,
        object_entity_type="T",
        evidence_span=evidence,
    )


async def test_extract_batched_size_1_is_per_chunk():
    """batch_size=1 makes one call per chunk and attributes each triple to
    its own chunk (the original per-chunk behaviour)."""
    chunks = [
        ("c0", "alpha inhibits beta.", [("alpha", "T"), ("beta", "T")]),
        ("c1", "gamma activates delta.", [("gamma", "T"), ("delta", "T")]),
    ]
    calls: list[str] = []

    async def fake_extract(text, vocab, *, model, temperature):
        calls.append(text)
        if "alpha" in text:
            return [_et("alpha", "INHIBITS", "beta", "alpha inhibits beta")]
        return [_et("gamma", "ACTIVATES", "delta", "gamma activates delta")]

    with patch("knowledge_agent.ingestion.triples_extractor.extract", side_effect=fake_extract):
        out = await triples_extractor.extract_batched(
            chunks, batch_size=1, model="m", temperature=0.0
        )

    assert len(calls) == 2  # one call per chunk
    assert [cid for cid, _ in out] == ["c0", "c1"]
    assert out[0][1][0].predicate == "INHIBITS"
    assert out[1][1][0].predicate == "ACTIVATES"


async def test_extract_batched_groups_and_attributes_by_evidence():
    """batch_size=2 groups consecutive chunks into one call over combined text
    + pooled vocab, and attributes a triple to the chunk its evidence lives in."""
    chunks = [
        ("c0", "alpha is here.", [("alpha", "T")]),
        ("c1", "beta appears and alpha inhibits beta downstream.", [("beta", "T")]),
        ("c2", "gamma solo.", [("gamma", "T")]),
    ]
    calls: list[dict[str, str]] = []

    async def fake_extract(text, vocab, *, model, temperature):
        calls.append(dict(vocab))
        if "alpha is here" in text:  # the c0+c1 batch
            return [_et("alpha", "INHIBITS", "beta", "alpha inhibits beta downstream")]
        return []

    with patch("knowledge_agent.ingestion.triples_extractor.extract", side_effect=fake_extract):
        out = await triples_extractor.extract_batched(
            chunks, batch_size=2, model="m", temperature=0.0
        )

    assert len(calls) == 2  # [c0,c1] and [c2] -> 2 non-overlapping batches
    assert set(calls[0].keys()) == {"alpha", "beta"}  # pooled vocab of the batch
    assert [cid for cid, _ in out] == ["c0", "c1", "c2"]  # one entry per chunk, in order
    by = dict(out)
    assert by["c0"] == []
    assert len(by["c1"]) == 1 and by["c1"][0].predicate == "INHIBITS"  # matched to c1
    assert by["c2"] == []


async def test_extract_batched_fallback_to_first_chunk_when_no_evidence_match():
    """When the evidence span matches no single chunk in the batch, the triple
    falls back to the batch's first chunk (verbatim span still stored)."""
    chunks = [
        ("c0", "alpha text.", [("alpha", "T")]),
        ("c1", "beta text.", [("beta", "T")]),
    ]

    async def fake_extract(text, vocab, *, model, temperature):
        return [_et("alpha", "INHIBITS", "beta", "paraphrased span not present verbatim")]

    with patch("knowledge_agent.ingestion.triples_extractor.extract", side_effect=fake_extract):
        out = await triples_extractor.extract_batched(
            chunks, batch_size=2, model="m", temperature=0.0
        )

    by = dict(out)
    assert len(by["c0"]) == 1  # fallback = first chunk of the batch
    assert by["c1"] == []


async def test_extract_batched_skips_empty_vocab_batch():
    """A batch whose chunks have no entities makes no LLM call."""
    chunks = [("c0", "no entities here.", [])]
    called = False

    async def fake_extract(*a, **k):
        nonlocal called
        called = True
        return []

    with patch("knowledge_agent.ingestion.triples_extractor.extract", side_effect=fake_extract):
        out = await triples_extractor.extract_batched(
            chunks, batch_size=1, model="m", temperature=0.0
        )

    assert called is False
    assert out == [("c0", [])]
