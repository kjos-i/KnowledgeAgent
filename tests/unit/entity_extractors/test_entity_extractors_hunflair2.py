"""Tests for the HunFlair2 biomedical NER adapter.

Mocks `_get_model()` to return a fake Flair tagger so tests don't
need real Flair + PyTorch installed. Pins:

  - KNOWN_LABELS is None (open vocab from the dispatcher's perspective
    even though HunFlair2 has a fixed internal label set — the user
    has no narrowing surface, so the dispatcher doesn't validate
    against the 5 labels).
  - extract() IGNORES its `entity_types` arg — always returns every
    HunFlair2 emission. This is the locked UX (single on/off option).
  - MODEL_NAME + MODEL_REVISION are pinned (not `main`).
  - Flair's title-case tags are normalised to our UPPERCASE_SNAKE
    convention via `_FLAIR_TO_OUR_LABELS`.

The fake tagger exposes `predict(sentence)` and stuffs span objects
into the sentence so `sentence.get_spans('ner')` returns them.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from knowledge_agent.entity_extractors import hunflair2
from knowledge_agent.entity_extractors.base import Mention


@dataclass
class _FakeSpan:
    """Stand-in for a Flair Span - duck-typed to what extract() uses."""

    text: str
    tag: str
    start_position: int
    score: float


# ---- module constants ----


def test_known_labels_is_none_user_has_no_narrowing_surface():
    """HunFlair2 is exposed as an all-or-nothing adapter — dispatcher
    must NOT cross-check user `entity_types` against any closed set."""
    assert hunflair2.KNOWN_LABELS is None


def test_model_name_is_pinned_hunflair2_ner():
    """First-ship pin — the unified hunflair2-ner model (NOT the older
    HunFlair v1 sub-tagger shape)."""
    assert hunflair2.MODEL_NAME == "hunflair/hunflair2-ner"


def test_model_revision_is_a_40_char_sha_not_a_branch():
    """Never `main` or a branch name — always a 40-char commit SHA."""
    assert len(hunflair2.MODEL_REVISION) == 40
    assert hunflair2.MODEL_REVISION != "main"
    assert all(c in "0123456789abcdef" for c in hunflair2.MODEL_REVISION)


def test_flair_to_our_labels_covers_five_biomedical_categories():
    """Pin the published label mapping. HunFlair2 emits Disease,
    Chemical, Gene, Species, CellLine — normalised to our
    UPPERCASE_SNAKE convention for downstream consistency."""
    assert hunflair2._FLAIR_TO_OUR_LABELS == {
        "Disease": "DISEASE",
        "Chemical": "CHEMICAL",
        "Gene": "GENE",
        "Species": "SPECIES",
        "CellLine": "CELL_LINE",
    }


# ---- extract() label normalisation ----


class _FakeSentence:
    """Stand-in for `flair.data.Sentence` - duck-typed to what
    extract() uses. The model's `predict(sentence)` mutation is
    simulated by `_fake_model_emitting` setting `_spans` on the
    fake sentence directly."""

    def __init__(self, text: str):
        self.text = text
        self._spans: list[_FakeSpan] = []

    def get_spans(self, layer: str):  # noqa: ARG002
        return list(self._spans)


def _fake_model_emitting(spans: list[_FakeSpan]):
    """Build a MagicMock standing in for a Flair tagger.

    Real Flair: `model.predict(sentence)` mutates `sentence` in
    place, attaching label annotations; `sentence.get_spans('ner')`
    returns the spans. The mock writes the canned spans onto the
    fake Sentence's `_spans` attribute when predict() is called.
    """
    model = MagicMock()

    def _predict_side_effect(sentence):
        sentence._spans = spans

    model.predict.side_effect = _predict_side_effect
    return model


def _run_extract(spans: list[_FakeSpan], entity_types: list[str] | None = None) -> list:
    """Run hunflair2.extract() with mocked dependencies.

    Returns the resulting Mention list. Tests compose by passing
    canned spans and asserting on the returned Mentions. Avoids
    repeating the same `with patch(...): with patch(...):` block
    in every test.
    """
    fake_model = _fake_model_emitting(spans)
    with patch(
        "knowledge_agent.entity_extractors.hunflair2._get_model",
        return_value=fake_model,
    ), patch(
        "knowledge_agent.entity_extractors.hunflair2._build_sentence",
        side_effect=lambda text: _FakeSentence(text),
    ):
        return hunflair2.extract(
            "input text",
            entity_types if entity_types is not None else [],
        )


def test_extract_returns_empty_when_model_predicts_nothing():
    """No spans → empty list, not error."""
    assert _run_extract([]) == []


def test_extract_normalises_flair_title_case_to_uppercase_snake():
    """Disease → DISEASE; CellLine → CELL_LINE etc. so :Entity nodes
    have consistent label casing across all adapters."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=10, score=0.92),
        _FakeSpan(text="HeLa", tag="CellLine",
                  start_position=30, score=0.88),
        _FakeSpan(text="aspirin", tag="Chemical",
                  start_position=50, score=0.95),
        _FakeSpan(text="TP53", tag="Gene",
                  start_position=70, score=0.91),
        _FakeSpan(text="E. coli", tag="Species",
                  start_position=90, score=0.85),
    ]
    result = _run_extract(spans)
    labels = [m.entity_type for m in result]
    assert labels == ["DISEASE", "CELL_LINE", "CHEMICAL", "GENE", "SPECIES"]


def test_extract_preserves_offset_from_start_position():
    """Mention.offset comes from Flair's `start_position` (char
    position in chunk text)."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=42, score=0.92),
    ]
    result = _run_extract(spans)
    assert result[0].offset == 42


def test_extract_preserves_confidence_from_score():
    """Flair exposes per-span scores — surface on Mention.confidence."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=0, score=0.73),
    ]
    result = _run_extract(spans)
    assert result[0].confidence == 0.73


def test_extract_skips_unknown_flair_tags():
    """If Flair upstream adds a new tag not in _FLAIR_TO_OUR_LABELS,
    the adapter skips rather than emit unmapped. Catches Flair
    upstream changes — bump the dict when this triggers in practice."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=0, score=0.92),
        _FakeSpan(text="future_label", tag="NotAKnownTag",
                  start_position=20, score=0.85),
    ]
    result = _run_extract(spans)
    assert len(result) == 1
    assert result[0].entity_type == "DISEASE"


def test_extract_preserves_original_spelling_in_raw_text():
    """raw_text preserves verbatim span — lowercasing happens at
    :Entity MERGE in entity_writes."""
    spans = [
        _FakeSpan(text="Fragile X Syndrome", tag="Disease",
                  start_position=0, score=0.95),
    ]
    result = _run_extract(spans)
    assert result[0].raw_text == "Fragile X Syndrome"


# ---- entity_types is IGNORED (locked UX: all-or-nothing) ----


def test_extract_ignores_entity_types_arg_empty():
    """entity_types=[] → returns everything HunFlair2 emits (full set)."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=0, score=0.92),
        _FakeSpan(text="aspirin", tag="Chemical",
                  start_position=20, score=0.95),
    ]
    result = _run_extract(spans, entity_types=[])
    assert len(result) == 2
    assert {m.entity_type for m in result} == {"DISEASE", "CHEMICAL"}


def test_extract_ignores_entity_types_arg_non_empty():
    """entity_types=["DISEASE"] is IGNORED — HunFlair2 still returns
    every emission, including non-DISEASE ones. Pinned UX: user has
    no narrowing surface."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=0, score=0.92),
        _FakeSpan(text="aspirin", tag="Chemical",
                  start_position=20, score=0.95),
    ]
    # User tried to narrow to DISEASE only — adapter should IGNORE
    # and return everything anyway.
    result = _run_extract(spans, entity_types=["DISEASE"])
    assert len(result) == 2
    assert {m.entity_type for m in result} == {"DISEASE", "CHEMICAL"}


def test_extract_ignores_entity_types_arg_garbage():
    """Even nonsense entity_types — silently ignored, no error."""
    spans = [
        _FakeSpan(text="diabetes", tag="Disease",
                  start_position=0, score=0.92),
    ]
    # Garbage in - no raise.
    result = _run_extract(spans, entity_types=["NONSENSE_TYPE", "asdf"])
    assert len(result) == 1
