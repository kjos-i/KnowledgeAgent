"""Tests for the LLM entity-extractor adapter.

Mocks the ChatAnthropic chain (same pattern as test_nodes.py) so these
tests don't hit the network. We verify:

  - Prompt construction: open mode (empty entity_types) vs closed mode
    (with types) builds different system prompts.
  - Result mapping: structured-output result is mapped to Mention with
    offset=None, confidence=None.
  - Defensive fallback: a non-_ExtractedMentions result returns [].

Model + temperature are per-corpus kwargs threaded through the
extractor call — 2026-07-02 refactor removed the get_settings() global
read. Tests pass them explicitly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.entity_extractors import llm
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.entity_extractors.llm import (
    _ExtractedMention,
    _ExtractedMentions,
)

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


# ---- prompt construction ----


def test_open_mode_prompt_when_entity_types_empty():
    """Empty entity_types -> open-mode system prompt (LLM categorises freely)."""
    prompt = llm._build_system_prompt([])
    assert "ONLY extract entities" not in prompt
    assert "any named entities" in prompt.lower()


def test_closed_mode_prompt_lists_types():
    """Non-empty entity_types -> closed-mode prompt enumerating them."""
    prompt = llm._build_system_prompt(["GENE", "DISEASE"])
    assert "ONLY extract entities" in prompt
    assert "GENE" in prompt
    assert "DISEASE" in prompt


def test_closed_mode_prompt_preserves_user_order():
    """The types list is rendered verbatim into the prompt - keeps the
    user's chosen ordering visible to the LLM."""
    prompt = llm._build_system_prompt(["DISEASE", "GENE"])
    # the types substring should appear in user order
    assert "DISEASE, GENE" in prompt


# ---- end-to-end extract() with mocked LLM ----


async def test_extract_returns_mentions_from_structured_output():
    """Happy path: LLM returns two mentions, adapter maps each to a Mention."""
    output = _ExtractedMentions(
        mentions=[
            _ExtractedMention(raw_text="BRCA1", entity_type="GENE"),
            _ExtractedMention(raw_text="breast cancer", entity_type="DISEASE"),
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        result = await llm.extract(
            "BRCA1 mutations cause breast cancer.",
            ["GENE", "DISEASE"],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result == [
        Mention(raw_text="BRCA1", entity_type="GENE", offset=None, confidence=None),
        Mention(
            raw_text="breast cancer",
            entity_type="DISEASE",
            offset=None,
            confidence=None,
        ),
    ]


async def test_extract_returns_empty_when_no_mentions_found():
    """LLM returns empty mentions list -> adapter returns []."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        assert (
            await llm.extract(
                "Just a plain sentence.",
                ["GENE"],
                model=_TEST_MODEL,
                temperature=_TEST_TEMPERATURE,
            )
            == []
        )


async def test_extract_propagates_model_and_temperature_to_get_llm():
    """Adapter forwards the per-corpus model + temperature kwargs
    verbatim to `_get_llm` (which builds the ChatAnthropic client)."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ) as mock_get_llm:
        await llm.extract("text", [], model="test-model", temperature=0.3)

    mock_get_llm.assert_called_once_with("test-model", 0.3)


async def test_extract_uses_structured_output_with_extracted_mentions_schema():
    """Adapter binds `_ExtractedMentions` as the structured-output schema -
    same single-source-of-truth pattern as the rest of the agent."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        await llm.extract(
            "text",
            [],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    mock_llm.with_structured_output.assert_called_once_with(_ExtractedMentions)


async def test_extract_passes_chunk_text_in_human_message():
    """The chunk text reaches the LLM verbatim as the user message."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    mock_structured = mock_llm.with_structured_output.return_value
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        await llm.extract(
            "This text mentions BRCA1.",
            [],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    # invoke is called with a list of [SystemMessage, HumanMessage].
    messages = mock_structured.ainvoke.call_args.args[0]
    human = messages[1]
    assert human.content == "This text mentions BRCA1."


async def test_extract_offset_and_confidence_always_none_for_llm():
    """Adapter never populates offset / confidence - those are NER-only
    fields. Even if the underlying output had them, the adapter strips."""
    output = _ExtractedMentions(mentions=[_ExtractedMention(raw_text="TP53", entity_type="GENE")])
    mock_llm = _mock_llm_returning(output)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        result = await llm.extract(
            "TP53 is a tumour suppressor.",
            [],
            model=_TEST_MODEL,
            temperature=_TEST_TEMPERATURE,
        )

    assert result[0].offset is None
    assert result[0].confidence is None


async def test_extract_defensive_fallback_on_non_extracted_mentions_result():
    """If the structured-output chain returns something other than
    _ExtractedMentions (shape drift, SDK change), the adapter fails
    closed with [] rather than crashing the pipeline mid-batch."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value={"mentions": []})  # dict, not model
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    with patch(
        "knowledge_agent.entity_extractors.llm._get_llm",
        return_value=mock_llm,
    ):
        assert (
            await llm.extract(
                "text",
                [],
                model=_TEST_MODEL,
                temperature=_TEST_TEMPERATURE,
            )
            == []
        )
