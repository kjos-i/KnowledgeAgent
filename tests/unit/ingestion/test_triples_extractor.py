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
"""

from unittest.mock import MagicMock, patch, AsyncMock

from knowledge_agent.ingestion import triples_extractor
from knowledge_agent.ingestion.triples_extractor import (
    _LLMTriple,
    _LLMTriples,
)
from knowledge_agent.kg.triples_writes import ExtractedTriple


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


def _patch_llm_and_settings(mock_llm):
    """Common patch setup - returns the patcher context managers as a tuple."""
    return (
        patch(
            "knowledge_agent.ingestion.triples_extractor._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.ingestion.triples_extractor.get_settings"
        ),
    )


# ---- Fast-path: empty vocab ----


async def test_extract_empty_vocab_returns_empty_without_llm_call():
    """No L6a entities = no possible triples. Don't waste an LLM call."""
    mock_llm = _mock_llm_returning(_LLMTriples(triples=[]))
    mock_structured = mock_llm.with_structured_output.return_value
    with patch(
        "knowledge_agent.ingestion.triples_extractor._get_llm",
        return_value=mock_llm,
    ):
        result = await triples_extractor.extract("some text", [])

    assert result == []
    mock_structured.ainvoke.assert_not_called()


# ---- Prompt construction ----


def test_build_system_prompt_renders_entity_vocab():
    prompt = triples_extractor._build_system_prompt(
        [("brca1", "GENE"), ("tp53", "GENE")]
    )
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
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "BRCA1 inhibits TP53 expression.",
            [("brca1", "GENE"), ("tp53", "GENE")],
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
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "text", [("brca1", "GENE"), ("tp53", "GENE")]
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
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract("text", [("brca1", "GENE")])

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
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "text", [("brca1", "GENE"), ("tp53", "GENE")]
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
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "Aspirin treats headache.",
            [("aspirin", "CHEMICAL"), ("headache", "DISEASE")],
        )

    assert result[0].subject_entity_type == "CHEMICAL"
    assert result[0].object_entity_type == "DISEASE"


async def test_extract_uses_settings_model_and_temperature():
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.ingestion.triples_extractor._get_llm",
            return_value=mock_llm,
        ) as mock_get_llm,
        patch(
            "knowledge_agent.ingestion.triples_extractor.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.triples_extractor_model = "test-model"
        mock_settings.return_value.triples_extractor_temperature = 0.4
        await triples_extractor.extract("text", [("a", "GENE"), ("b", "GENE")])

    mock_get_llm.assert_called_once_with("test-model", 0.4)


async def test_extract_binds_llmtriples_as_structured_output_schema():
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        await triples_extractor.extract("text", [("a", "GENE"), ("b", "GENE")])

    mock_llm.with_structured_output.assert_called_once_with(_LLMTriples)


async def test_extract_defensive_fallback_on_non_llmtriples_result():
    """If structured output drifts (e.g. returns a dict), fail closed."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value={"triples": []})
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "text", [("a", "GENE"), ("b", "GENE")]
        )

    assert result == []


async def test_extract_returns_empty_when_llm_returns_no_triples():
    output = _LLMTriples(triples=[])
    mock_llm = _mock_llm_returning(output)
    a, b = _patch_llm_and_settings(mock_llm)
    with a, b as mock_settings:
        mock_settings.return_value.triples_extractor_model = "claude-haiku"
        mock_settings.return_value.triples_extractor_temperature = 0.0
        result = await triples_extractor.extract(
            "Plain text with no relations.",
            [("brca1", "GENE"), ("tp53", "GENE")],
        )

    assert result == []
