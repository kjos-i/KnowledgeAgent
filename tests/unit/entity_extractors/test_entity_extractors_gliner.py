"""Tests for the GLiNER zero-shot entity-extractor adapter.

Mocks `_get_model()` to return a fake GLiNER model so the tests don't
need real PyTorch + GLiNER weights installed. Verifies:

  - KNOWN_LABELS is None (open vocab, zero-shot — accepts any label).
  - DEFAULT_LABELS is non-empty (used when corpus passes empty types).
  - MODEL_NAME + MODEL_REVISION are pinned (not `main`).
  - extract() maps GLiNER's dict output to Mention with offset and
    confidence populated (unlike SciSpaCy + LLM which leave them None).
  - entity_types semantics: empty → DEFAULT_LABELS; non-empty → passed
    through verbatim to GLiNER.

The fake model exposes `predict_entities(text, labels, threshold)`
returning a hand-rolled list of dicts matching GLiNER's real shape.
"""

from unittest.mock import MagicMock, patch, AsyncMock

from knowledge_agent.entity_extractors import gliner
from knowledge_agent.entity_extractors.base import Mention


# ---- module constants ----


def test_known_labels_is_none_open_vocabulary():
    """GLiNER is zero-shot — dispatcher must NOT cross-check
    entity_types against any closed set."""
    assert gliner.KNOWN_LABELS is None


def test_default_labels_is_non_empty_tuple_of_strings():
    """DEFAULT_LABELS used when corpus.toml provides no entity_types.
    Empty would crash GLiNER (which requires non-empty labels)."""
    assert isinstance(gliner.DEFAULT_LABELS, tuple)
    assert len(gliner.DEFAULT_LABELS) > 0
    assert all(isinstance(label, str) for label in gliner.DEFAULT_LABELS)


def test_default_labels_covers_general_ner_categories():
    """Pin the published default — PERSON / ORGANIZATION / LOCATION
    are universally useful and what most general-NER consumers expect."""
    assert "PERSON" in gliner.DEFAULT_LABELS
    assert "ORGANIZATION" in gliner.DEFAULT_LABELS
    assert "LOCATION" in gliner.DEFAULT_LABELS


def test_model_name_is_pinned_urchade_multi_v21():
    """First-ship pin — bump only after re-verifying provenance +
    safetensors + trust_remote_code on the HF model page."""
    assert gliner.MODEL_NAME == "urchade/gliner_multi-v2.1"


def test_model_revision_is_a_40_char_sha_not_a_branch():
    """Never `main` or a branch name — those auto-update and bypass
    our verification. Always a 40-char commit SHA."""
    assert len(gliner.MODEL_REVISION) == 40
    assert gliner.MODEL_REVISION != "main"
    # Hex chars only.
    assert all(c in "0123456789abcdef" for c in gliner.MODEL_REVISION)


# ---- extract() filter behaviour ----


def _fake_model_returning(predictions: list[dict]):
    """Build a MagicMock standing in for a GLiNER model.

    Real GLiNER exposes `predict_entities(text, labels, threshold)`
    returning a list of dicts with keys `text`, `label`, `start`,
    `end`, `score`. The mock returns whatever the test provides.
    """
    mock = MagicMock()
    mock.predict_entities.return_value = predictions
    return mock


async def test_extract_returns_empty_when_model_predicts_nothing():
    """No predictions → empty list, not error."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        assert await gliner.extract("some text", ["PERSON"]) == []


async def test_extract_maps_each_prediction_to_mention():
    """Every dict GLiNER returns becomes a Mention with offset +
    confidence populated from GLiNER's start / score fields."""
    predictions = [
        {"text": "Albert Einstein", "label": "PERSON",
         "start": 0, "end": 15, "score": 0.95},
        {"text": "Princeton", "label": "ORGANIZATION",
         "start": 30, "end": 39, "score": 0.88},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        result = await gliner.extract("text", ["PERSON", "ORGANIZATION"])

    assert result == [
        Mention(
            raw_text="Albert Einstein", entity_type="PERSON",
            offset=0, confidence=0.95,
        ),
        Mention(
            raw_text="Princeton", entity_type="ORGANIZATION",
            offset=30, confidence=0.88,
        ),
    ]


async def test_extract_preserves_offset_from_start_field():
    """Mention.offset comes from GLiNER's `start` (char position in
    chunk text), NOT a different field."""
    predictions = [
        {"text": "Berlin", "label": "LOCATION",
         "start": 42, "end": 48, "score": 0.91},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        result = await gliner.extract("text", ["LOCATION"])

    assert result[0].offset == 42


async def test_extract_preserves_confidence_from_score_field():
    """GLiNER exposes per-mention scores — surface them on Mention.
    Unlike SciSpaCy + LLM which both leave confidence=None."""
    predictions = [
        {"text": "Marie Curie", "label": "PERSON",
         "start": 0, "end": 11, "score": 0.73},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        result = await gliner.extract("text", ["PERSON"])

    assert result[0].confidence == 0.73


# ---- entity_types semantics ----


async def test_extract_passes_non_empty_entity_types_verbatim():
    """Non-empty entity_types flows into GLiNER's labels arg as-is.
    GLiNER is zero-shot so whatever the user puts in corpus.toml is
    what GLiNER predicts against."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        await gliner.extract("text", ["GENE", "PROTEIN", "DISEASE"])

    # Confirm the model was called with the labels we passed.
    args, kwargs = fake_model.predict_entities.call_args
    # GLiNER signature: predict_entities(text, labels, threshold=...)
    assert args[1] == ["GENE", "PROTEIN", "DISEASE"]


async def test_extract_uses_default_labels_when_entity_types_empty():
    """Empty entity_types → DEFAULT_LABELS passed to GLiNER. GLiNER
    requires non-empty labels; without this fallback the call would
    raise."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        await gliner.extract("text", [])

    args, kwargs = fake_model.predict_entities.call_args
    assert args[1] == list(gliner.DEFAULT_LABELS)


async def test_extract_passes_threshold_to_model():
    """The score threshold lives as a module constant — confirm it
    actually reaches GLiNER's predict_entities call."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        await gliner.extract("text", ["PERSON"])

    args, kwargs = fake_model.predict_entities.call_args
    assert kwargs.get("threshold") == gliner._SCORE_THRESHOLD


async def test_extract_forwards_text_unchanged_to_model():
    """The chunk text is passed verbatim — no preprocessing.
    GLiNER tokenises internally."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        await gliner.extract("Mixed-case Text  with  weird whitespace.", ["X"])

    args, kwargs = fake_model.predict_entities.call_args
    assert args[0] == "Mixed-case Text  with  weird whitespace."


# ---- raw_text preservation ----


async def test_extract_preserves_original_spelling_in_raw_text():
    """raw_text is the verbatim span from the source — do NOT
    lowercase or normalise. Lowercasing happens in entity_writes when
    computing the :Entity node's key for the MERGE."""
    predictions = [
        {"text": "CRISPR-Cas9", "label": "TECHNOLOGY",
         "start": 0, "end": 11, "score": 0.85},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner._get_model",
        return_value=fake_model,
    ):
        result = await gliner.extract("text", ["TECHNOLOGY"])

    # Original case preserved — NOT crispr-cas9.
    assert result[0].raw_text == "CRISPR-Cas9"
