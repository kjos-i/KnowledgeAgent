"""Tests for entity_extractors dispatcher (get_extractor, get_known_labels,
validate_entity_types).

These tests exercise the public API in `entity_extractors/__init__.py`.
The dispatcher should resolve known adapter names without loading any
heavy ML dependencies up-front (lazy imports), and validate_entity_types
should fail fast on typo'd labels for closed-vocabulary adapters.

We don't actually CALL extract() here - that's covered per-adapter in
test_entity_extractors_llm.py and the per-adapter test modules.
"""

import pytest

from knowledge_agent.entity_extractors import (
    Mention,
    get_extractor,
    get_known_labels,
    validate_entity_types,
)


# ---- public re-exports ----


def test_mention_reexported_from_package():
    """`Mention` is re-exported at package level for convenience."""
    assert Mention is not None
    m = Mention(raw_text="x", entity_type="Y")
    assert m.raw_text == "x"


# ---- get_extractor ----


def test_get_extractor_returns_llm_module():
    """`'llm'` resolves to the LLM adapter module."""
    mod = get_extractor("llm")
    assert hasattr(mod, "extract")
    assert hasattr(mod, "KNOWN_LABELS")
    assert mod.KNOWN_LABELS is None


def test_get_extractor_returns_gliner_module():
    """`'gliner'` resolves to the GLiNER adapter module.

    Importing the module is safe even when gliner/torch aren't
    installed - the `from gliner import GLiNER` inside the adapter is
    intentionally deferred to `_get_model()`, called only on
    `extract()`."""
    mod = get_extractor("gliner")
    assert hasattr(mod, "extract")
    assert hasattr(mod, "KNOWN_LABELS")
    # GLiNER is zero-shot (open vocab) so KNOWN_LABELS is None, same
    # shape as the LLM adapter.
    assert mod.KNOWN_LABELS is None


def test_get_extractor_returns_gliner_biomed_module():
    """`'gliner_biomed'` resolves to the GLiNER-BioMed adapter.

    Same lazy-import discipline as the general gliner adapter — gliner
    library is imported only inside `_get_model()`."""
    mod = get_extractor("gliner_biomed")
    assert hasattr(mod, "extract")
    assert hasattr(mod, "KNOWN_LABELS")
    # Also zero-shot.
    assert mod.KNOWN_LABELS is None


def test_get_extractor_returns_hunflair2_module():
    """`'hunflair2'` resolves to the HunFlair2 adapter module.

    Same lazy-import discipline — Flair + PyTorch imports live inside
    `_get_model()`, not at module top-level."""
    mod = get_extractor("hunflair2")
    assert hasattr(mod, "extract")
    assert hasattr(mod, "KNOWN_LABELS")
    # HunFlair2 is exposed as all-or-nothing — the user has NO
    # narrowing surface, so dispatcher treats it as open vocab
    # (no validation against the internal label set).
    assert mod.KNOWN_LABELS is None


def test_get_extractor_unknown_name_raises_with_valid_set():
    """Unknown adapter name surfaces a ValueError listing what IS known
    so the user can correct a typo without grepping the source."""
    with pytest.raises(ValueError) as excinfo:
        get_extractor("not_a_real_adapter")
    msg = str(excinfo.value)
    assert "not_a_real_adapter" in msg
    assert "llm" in msg
    assert "gliner" in msg
    assert "gliner_biomed" in msg
    assert "hunflair2" in msg


# ---- get_known_labels ----


def test_get_known_labels_llm_is_none():
    """LLM = open vocabulary."""
    assert get_known_labels("llm") is None


def test_get_known_labels_gliner_is_none():
    """GLiNER is zero-shot - no closed label set, accepts any string
    via the inference call. Same shape as the LLM adapter."""
    assert get_known_labels("gliner") is None


def test_get_known_labels_gliner_biomed_is_none():
    """GLiNER-BioMed is also zero-shot."""
    assert get_known_labels("gliner_biomed") is None


def test_get_known_labels_hunflair2_is_none():
    """HunFlair2 is exposed as all-or-nothing — no user-facing
    narrowing surface, so dispatcher treats it as open vocab."""
    assert get_known_labels("hunflair2") is None


# ---- validate_entity_types ----


def test_validate_entity_types_llm_accepts_anything():
    """LLM open vocab -> validator is a no-op even for nonsense labels."""
    validate_entity_types("llm", ["GENE", "MADE_UP", "asdf"])  # no raise


def test_validate_entity_types_llm_empty_list_is_fine():
    """Empty list is the default ('use adapter default behaviour'); no raise."""
    validate_entity_types("llm", [])


def test_validate_entity_types_gliner_accepts_anything():
    """GLiNER open vocab - validator is a no-op for any label list,
    same contract as the LLM adapter."""
    validate_entity_types("gliner", ["GENE", "MADE_UP", "asdf"])  # no raise


def test_validate_entity_types_gliner_empty_list_is_fine():
    """Empty list = adapter falls back to DEFAULT_LABELS internally;
    validator stays a no-op."""
    validate_entity_types("gliner", [])


def test_validate_entity_types_gliner_biomed_accepts_anything():
    """GLiNER-BioMed open vocab — validator no-op for any labels."""
    validate_entity_types(
        "gliner_biomed", ["GENE", "DISEASE", "MADE_UP"],
    )  # no raise


def test_validate_entity_types_gliner_biomed_empty_list_is_fine():
    """Empty list = adapter falls back to biomedical DEFAULT_LABELS."""
    validate_entity_types("gliner_biomed", [])


def test_validate_entity_types_hunflair2_accepts_anything():
    """HunFlair2's adapter IGNORES entity_types entirely (locked UX:
    all-or-nothing). Dispatcher validator is a no-op so no surprise
    error if user accidentally sets entity_types."""
    validate_entity_types(
        "hunflair2", ["DISEASE", "MADE_UP", "asdf"],
    )  # no raise


def test_validate_entity_types_hunflair2_empty_list_is_fine():
    """Empty list — also fine; adapter ignores it either way."""
    validate_entity_types("hunflair2", [])


def test_validate_entity_types_unknown_adapter_propagates_value_error():
    """Asking validate for an unknown adapter falls through to
    get_extractor's ValueError - same error contract as the rest of
    the API."""
    with pytest.raises(ValueError, match="not_a_real_adapter"):
        validate_entity_types("not_a_real_adapter", ["X"])
