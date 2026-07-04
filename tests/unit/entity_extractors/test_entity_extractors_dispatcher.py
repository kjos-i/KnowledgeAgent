"""Tests for entity_extractors dispatcher (get_extractor, get_known_labels,
validate_entity_types).

These tests exercise the public API in `entity_extractors/__init__.py`.
The dispatcher should resolve known adapter names without loading any
heavy ML dependencies up-front (lazy imports), and validate_entity_types
should fail fast on typo'd labels for closed-vocabulary adapters.

We don't actually CALL extract() here - that's covered per-adapter in
test_entity_extractors_llm.py and the per-adapter test modules.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from knowledge_agent.entity_extractors import (
    Mention,
    effective_entity_types,
    extract_union,
    get_extractor,
    get_known_labels,
    validate_entity_types,
)

_GET_EXTRACTOR = "knowledge_agent.entity_extractors.get_extractor"


def _fake(mentions=(), default_labels=None):
    """A stand-in extractor module: async `extract` returning `mentions`,
    plus an optional `DEFAULT_LABELS` constant."""
    ns = SimpleNamespace(extract=AsyncMock(return_value=list(mentions)))
    if default_labels is not None:
        ns.DEFAULT_LABELS = tuple(default_labels)
    return ns


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
        "gliner_biomed",
        ["GENE", "DISEASE", "MADE_UP"],
    )  # no raise


def test_validate_entity_types_gliner_biomed_empty_list_is_fine():
    """Empty list = adapter falls back to biomedical DEFAULT_LABELS."""
    validate_entity_types("gliner_biomed", [])


def test_validate_entity_types_hunflair2_accepts_anything():
    """HunFlair2's adapter IGNORES entity_types entirely (locked UX:
    all-or-nothing). Dispatcher validator is a no-op so no surprise
    error if user accidentally sets entity_types."""
    validate_entity_types(
        "hunflair2",
        ["DISEASE", "MADE_UP", "asdf"],
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


# ---- effective_entity_types (entity_types_mode) ----


def test_effective_entity_types_replace_is_passthrough():
    """'replace' mode returns the user's list unchanged."""
    assert effective_entity_types("gliner", ["GENE"], "replace") == ["GENE"]


def test_effective_entity_types_empty_is_passthrough_regardless_of_mode():
    """Empty list -> adapter falls back to its own defaults, so mode is
    irrelevant and we pass the empty list straight through."""
    assert effective_entity_types("gliner", [], "add") == []
    assert effective_entity_types("gliner", [], "replace") == []


def test_effective_entity_types_add_merges_adapter_defaults(monkeypatch):
    """'add' mode = user's list + the adapter's DEFAULT_LABELS, deduped,
    user labels first."""
    monkeypatch.setattr(
        _GET_EXTRACTOR,
        lambda n: _fake(default_labels=("PERSON", "ORG")),
    )
    assert effective_entity_types("gliner", ["GENE"], "add") == [
        "GENE",
        "PERSON",
        "ORG",
    ]


def test_effective_entity_types_add_dedupes(monkeypatch):
    """A user label already in the defaults isn't duplicated."""
    monkeypatch.setattr(
        _GET_EXTRACTOR,
        lambda n: _fake(default_labels=("GENE", "ORG")),
    )
    assert effective_entity_types("gliner", ["GENE"], "add") == ["GENE", "ORG"]


def test_effective_entity_types_add_no_defaults_equals_replace(monkeypatch):
    """An adapter with no DEFAULT_LABELS (LLM, HunFlair2): 'add' == the
    user's list (nothing to merge)."""
    monkeypatch.setattr(_GET_EXTRACTOR, lambda n: _fake())  # no DEFAULT_LABELS
    assert effective_entity_types("llm", ["GENE"], "add") == ["GENE"]


# ---- extract_union (priority-ordered union) ----


async def test_extract_union_empty_names_returns_empty():
    """No extractors selected -> no work, empty result."""
    assert await extract_union("text", [], []) == []


async def test_extract_union_single_extractor_stamps_its_source(monkeypatch):
    """One extractor: every surviving mention records that one adapter."""
    monkeypatch.setattr(
        _GET_EXTRACTOR,
        lambda n: _fake([Mention(raw_text="Aspirin", entity_type="CHEMICAL")]),
    )
    out = await extract_union("t", ["ner"], [])
    assert len(out) == 1
    assert out[0].sources == ("ner",)


async def test_extract_union_base_owns_overlapping_span(monkeypatch):
    """Same span from two adapters -> base (index 0) keeps its type;
    every finder is recorded in sources."""
    ner = _fake([Mention(raw_text="Aspirin", entity_type="CHEMICAL")])
    llm = _fake([Mention(raw_text="aspirin", entity_type="DRUG")])
    mods = {"ner": ner, "llm": llm}
    monkeypatch.setattr(_GET_EXTRACTOR, lambda n: mods[n])

    out = await extract_union("t", ["ner", "llm"], [])
    assert len(out) == 1
    assert out[0].entity_type == "CHEMICAL"  # base wins the type
    assert out[0].sources == ("ner", "llm")  # both recorded


async def test_extract_union_later_extractor_adds_new_spans(monkeypatch):
    """The gap-filler contributes spans the base didn't find."""
    ner = _fake([Mention(raw_text="BRCA1", entity_type="GENE")])
    llm = _fake([Mention(raw_text="EGFR", entity_type="GENE")])
    mods = {"ner": ner, "llm": llm}
    monkeypatch.setattr(_GET_EXTRACTOR, lambda n: mods[n])

    out = await extract_union("t", ["ner", "llm"], [])
    by_key = {m.raw_text: m for m in out}
    assert set(by_key) == {"BRCA1", "EGFR"}
    assert by_key["BRCA1"].sources == ("ner",)
    assert by_key["EGFR"].sources == ("llm",)


async def test_extract_union_normalizes_span_for_overlap(monkeypatch):
    """Overlap detection is case + whitespace insensitive, so surface
    variants of the same span merge onto one node."""
    ner = _fake([Mention(raw_text="T cells", entity_type="CELL")])
    llm = _fake([Mention(raw_text="  t   cells ", entity_type="CELLTYPE")])
    mods = {"ner": ner, "llm": llm}
    monkeypatch.setattr(_GET_EXTRACTOR, lambda n: mods[n])

    out = await extract_union("t", ["ner", "llm"], [])
    assert len(out) == 1
    assert out[0].entity_type == "CELL"  # base owner
    assert out[0].sources == ("ner", "llm")


async def test_extract_union_forwards_llm_kwargs_only_to_llm(monkeypatch):
    """model/temperature go ONLY to the 'llm' adapter; other adapters
    are called with just (text, effective_types)."""
    ner = _fake([])
    llm = _fake([])
    mods = {"ner": ner, "llm": llm}
    monkeypatch.setattr(_GET_EXTRACTOR, lambda n: mods[n])

    await extract_union(
        "t",
        ["ner", "llm"],
        ["GENE"],
        llm_kwargs={"model": "m", "temperature": 0.0},
    )
    ner.extract.assert_awaited_once_with("t", ["GENE"])
    llm.extract.assert_awaited_once_with(
        "t",
        ["GENE"],
        model="m",
        temperature=0.0,
    )
