"""Tests for entity_extractors.base - the Mention contract.

The Mention dataclass is the public output type that every adapter
produces. These tests pin down its shape so that adapters and the
write path can rely on it.
"""

import pytest

from knowledge_agent.entity_extractors.base import Mention


def test_mention_required_fields():
    """raw_text and entity_type are required; offset / confidence default None."""
    m = Mention(raw_text="BRCA1", entity_type="GENE")
    assert m.raw_text == "BRCA1"
    assert m.entity_type == "GENE"
    assert m.offset is None
    assert m.confidence is None


def test_mention_full_fields():
    """All four fields populated (NER-style)."""
    m = Mention(
        raw_text="TP53",
        entity_type="GENE",
        offset=42,
        confidence=0.95,
    )
    assert m.raw_text == "TP53"
    assert m.entity_type == "GENE"
    assert m.offset == 42
    assert m.confidence == 0.95


def test_mention_is_frozen():
    """Frozen dataclass: in-place mutation raises FrozenInstanceError.

    Frozen guarantees the same Mention can flow through batching and
    write paths without aliasing surprises - no caller can mutate it
    after extraction.
    """
    from dataclasses import FrozenInstanceError

    m = Mention(raw_text="BRCA1", entity_type="GENE")
    with pytest.raises(FrozenInstanceError):
        m.raw_text = "different"  # type: ignore[misc]


def test_mention_equality_by_value():
    """Frozen dataclass uses field-wise equality, useful for set/dict ops."""
    a = Mention(raw_text="BRCA1", entity_type="GENE")
    b = Mention(raw_text="BRCA1", entity_type="GENE")
    assert a == b
    assert hash(a) == hash(b)


def test_mention_distinguishes_by_entity_type():
    """'Apple' as GENE vs COMPANY are distinct - drives the composite
    NODE KEY in the :Entity write path."""
    gene = Mention(raw_text="Apple", entity_type="GENE")
    company = Mention(raw_text="Apple", entity_type="COMPANY")
    assert gene != company


def test_mention_preserves_original_casing():
    """raw_text is NOT lowercased by the dataclass - that's the write
    path's job. 'BRCA1' stays 'BRCA1' here so NER offsets stay valid
    against the chunk text."""
    m = Mention(raw_text="BRCA1", entity_type="GENE")
    assert m.raw_text == "BRCA1"  # not "brca1"
