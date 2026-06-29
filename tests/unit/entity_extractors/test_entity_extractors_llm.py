"""Tests for the LLM entity-extractor adapter.

Mocks the ChatAnthropic chain (same pattern as test_nodes.py) so these
tests don't hit the network. We verify:

  - Prompt construction: open mode (empty entity_types) vs closed mode
    (with types) builds different system prompts.
  - Result mapping: structured-output result is mapped to Mention with
    offset=None, confidence=None.
  - Defensive fallback: a non-_ExtractedMentions result returns [].
"""

from unittest.mock import MagicMock, patch

from knowledge_agent.entity_extractors import llm
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.entity_extractors.llm import (
    _ExtractedMention,
    _ExtractedMentions,
)


def _mock_llm_returning(structured_output):
    """Build the mock chain: llm.with_structured_output(...).invoke(...) -> output.

    Sync invoke (not ainvoke), matching the adapter's call site.

    `with_retry` is wired so that the production code's `_with_retry(structured)`
    call resolves to the same mock — i.e. the retry wrapper is the identity in
    tests. We test contract, not retry semantics; retry behaviour is covered
    by `llm_factory.with_retry` tests in isolation.
    """
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value=structured_output)
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


def test_extract_returns_mentions_from_structured_output():
    """Happy path: LLM returns two mentions, adapter maps each to a Mention."""
    output = _ExtractedMentions(
        mentions=[
            _ExtractedMention(raw_text="BRCA1", entity_type="GENE"),
            _ExtractedMention(raw_text="breast cancer", entity_type="DISEASE"),
        ]
    )
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        result = llm.extract(
            "BRCA1 mutations cause breast cancer.",
            ["GENE", "DISEASE"],
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


def test_extract_returns_empty_when_no_mentions_found():
    """LLM returns empty mentions list -> adapter returns []."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        assert llm.extract("Just a plain sentence.", ["GENE"]) == []


def test_extract_uses_settings_model_and_temperature():
    """Adapter pulls model + temperature from settings (single source of truth)."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ) as mock_get_llm,
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "test-model"
        mock_settings.return_value.entity_extractor_temperature = 0.3
        llm.extract("text", [])

    mock_get_llm.assert_called_once_with("test-model", 0.3)


def test_extract_uses_structured_output_with_extracted_mentions_schema():
    """Adapter binds `_ExtractedMentions` as the structured-output schema -
    same single-source-of-truth pattern as the rest of the agent."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        llm.extract("text", [])

    mock_llm.with_structured_output.assert_called_once_with(_ExtractedMentions)


def test_extract_passes_chunk_text_in_human_message():
    """The chunk text reaches the LLM verbatim as the user message."""
    output = _ExtractedMentions(mentions=[])
    mock_llm = _mock_llm_returning(output)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        llm.extract("This text mentions BRCA1.", [])

    # invoke is called with a list of [SystemMessage, HumanMessage].
    messages = mock_structured.invoke.call_args.args[0]
    human = messages[1]
    assert human.content == "This text mentions BRCA1."


def test_extract_offset_and_confidence_always_none_for_llm():
    """Adapter never populates offset / confidence - those are NER-only
    fields. Even if the underlying output had them, the adapter strips."""
    output = _ExtractedMentions(
        mentions=[_ExtractedMention(raw_text="TP53", entity_type="GENE")]
    )
    mock_llm = _mock_llm_returning(output)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        result = llm.extract("TP53 is a tumour suppressor.", [])

    assert result[0].offset is None
    assert result[0].confidence is None


def test_extract_defensive_fallback_on_non_extracted_mentions_result():
    """If the structured-output chain returns something other than
    _ExtractedMentions (shape drift, SDK change), the adapter fails
    closed with [] rather than crashing the pipeline mid-batch."""
    mock_structured = MagicMock()
    mock_structured.invoke = MagicMock(return_value={"mentions": []})  # dict, not model
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    with (
        patch(
            "knowledge_agent.entity_extractors.llm._get_llm",
            return_value=mock_llm,
        ),
        patch(
            "knowledge_agent.entity_extractors.llm.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.entity_extractor_model = "claude-haiku"
        mock_settings.return_value.entity_extractor_temperature = 0.0
        assert llm.extract("text", []) == []
