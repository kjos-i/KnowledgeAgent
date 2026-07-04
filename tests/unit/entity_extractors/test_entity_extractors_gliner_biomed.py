"""Tests for the GLiNER-BioMed zero-shot entity-extractor adapter.

Mirrors `test_entity_extractors_gliner.py` but pins the biomedical
specifics: model name, biomedical DEFAULT_LABELS, and the explicit
acknowledgement that this checkpoint has no .safetensors (pickle
format only — flagged in provenance).
"""

from unittest.mock import MagicMock, patch

from knowledge_agent.entity_extractors import gliner_biomed
from knowledge_agent.entity_extractors.base import Mention

# ---- module constants ----


def test_known_labels_is_none_open_vocabulary():
    """Zero-shot — dispatcher must NOT cross-check entity_types
    against any closed set, same as the general gliner adapter."""
    assert gliner_biomed.KNOWN_LABELS is None


def test_default_labels_is_non_empty_tuple_of_strings():
    """DEFAULT_LABELS used when corpus.toml provides no entity_types."""
    assert isinstance(gliner_biomed.DEFAULT_LABELS, tuple)
    assert len(gliner_biomed.DEFAULT_LABELS) > 0
    assert all(isinstance(label, str) for label in gliner_biomed.DEFAULT_LABELS)


def test_default_labels_cover_core_biomedical_categories():
    """Pin the published default — biomedical research papers expect
    DISEASE / CHEMICAL / GENE / PROTEIN canonical NER coverage."""
    assert "DISEASE" in gliner_biomed.DEFAULT_LABELS
    assert "CHEMICAL" in gliner_biomed.DEFAULT_LABELS
    assert "GENE" in gliner_biomed.DEFAULT_LABELS
    assert "PROTEIN" in gliner_biomed.DEFAULT_LABELS


def test_model_name_is_pinned_ihor_biomed_bi_large():
    """First-ship pin — bi-large variant chosen for BC5CDR + smaller
    footprint. Bump only after re-verifying provenance on HF."""
    assert gliner_biomed.MODEL_NAME == "Ihor/gliner-biomed-bi-large-v1.0"


def test_model_revision_is_a_40_char_sha_not_a_branch():
    """Never `main` or a branch name — always a 40-char commit SHA."""
    assert len(gliner_biomed.MODEL_REVISION) == 40
    assert gliner_biomed.MODEL_REVISION != "main"
    assert all(c in "0123456789abcdef" for c in gliner_biomed.MODEL_REVISION)


# ---- extract() behaviour (mirrors gliner adapter contract) ----


def _fake_model_returning(predictions: list[dict]):
    """MagicMock standing in for a GLiNER model.

    Real GLiNER exposes `predict_entities(text, labels, threshold)`
    returning a list of dicts with keys `text`, `label`, `start`,
    `end`, `score`."""
    mock = MagicMock()
    mock.predict_entities.return_value = predictions
    return mock


async def test_extract_returns_empty_when_model_predicts_nothing():
    """No predictions → empty list, not error."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        assert await gliner_biomed.extract("biomedical text", ["DISEASE"]) == []


async def test_extract_maps_each_prediction_to_mention_with_score_and_offset():
    """Every dict GLiNER returns becomes a Mention with offset +
    confidence populated from GLiNER's start / score fields."""
    predictions = [
        {"text": "type 2 diabetes", "label": "DISEASE", "start": 0, "end": 15, "score": 0.94},
        {"text": "metformin", "label": "CHEMICAL", "start": 25, "end": 34, "score": 0.89},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        result = await gliner_biomed.extract("text", ["DISEASE", "CHEMICAL"])

    assert result == [
        Mention(
            raw_text="type 2 diabetes",
            entity_type="DISEASE",
            offset=0,
            confidence=0.94,
        ),
        Mention(
            raw_text="metformin",
            entity_type="CHEMICAL",
            offset=25,
            confidence=0.89,
        ),
    ]


async def test_extract_passes_non_empty_entity_types_verbatim():
    """Non-empty entity_types flows into GLiNER's labels arg as-is."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        await gliner_biomed.extract("text", ["GENE", "PROTEIN"])

    args, _ = fake_model.predict_entities.call_args
    assert args[1] == ["GENE", "PROTEIN"]


async def test_extract_uses_biomedical_default_labels_when_entity_types_empty():
    """Empty entity_types → DEFAULT_LABELS (biomedical categories)
    passed to GLiNER. GLiNER requires non-empty labels; without this
    fallback the call would raise."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        await gliner_biomed.extract("text", [])

    args, _ = fake_model.predict_entities.call_args
    assert args[1] == list(gliner_biomed.DEFAULT_LABELS)


async def test_extract_passes_threshold_to_model():
    """Score threshold module constant reaches GLiNER's call."""
    fake_model = _fake_model_returning([])
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        await gliner_biomed.extract("text", ["DISEASE"])

    _, kwargs = fake_model.predict_entities.call_args
    assert kwargs.get("threshold") == gliner_biomed._SCORE_THRESHOLD


async def test_extract_preserves_original_spelling_in_raw_text():
    """raw_text preserves verbatim span — lowercasing happens in
    entity_writes at the :Entity MERGE step."""
    predictions = [
        {"text": "TP53", "label": "GENE", "start": 0, "end": 4, "score": 0.95},
    ]
    fake_model = _fake_model_returning(predictions)
    with patch(
        "knowledge_agent.entity_extractors.gliner_biomed._get_model",
        return_value=fake_model,
    ):
        result = await gliner_biomed.extract("text", ["GENE"])

    # Original case preserved — NOT tp53.
    assert result[0].raw_text == "TP53"
