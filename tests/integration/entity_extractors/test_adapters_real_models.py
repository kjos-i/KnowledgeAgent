"""Integration tests for the NER adapters — exercises each against
the REAL Hugging Face model (no mocks).

Three adapters covered (one test per):
  - GLiNER (`urchade/gliner_multi-v2.1`) — general-domain zero-shot NER
  - GLiNER-BioMed (`Ihor/gliner-biomed-bi-large-v1.0`) — biomedical
  - HunFlair2 (`hunflair/hunflair2-ner`) — biomedical Flair tagger

Each test downloads the model on first run (~1-2 GB per adapter,
cached under `~/.cache/huggingface/`). Subsequent runs reuse the
cache. Total disk on first full integration run: ~3-4 GB.

The LLM adapter is exercised indirectly via the pipeline + agent
integration tests (`test_pipeline.py` runs L6 with `extractor=llm`
when entities are enabled); it doesn't need a separate test here.

Manual interactive counterpart: `scripts/smoke_kg_l6_entities.py
--adapter <name>` exercises each adapter against the synthetic
biomedical text + writes via `write_entities`.

Skipped by default; opt in via `pytest -m "integration and slow"`
(the `slow` marker keeps these out of the default integration
run so the fast integration tests stay fast).
"""

from __future__ import annotations

import pytest

from knowledge_agent.entity_extractors import get_extractor

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Biomedical text rich in entities — exercises each adapter's
# vocabulary. Aspirin / metformin / EGFR / BRCA1 are well-known
# tokens that every biomedical NER catches; "type 2 diabetes
# mellitus" hits the disease class.
BIOMED_TEXT = (
    "Aspirin reduces the risk of myocardial infarction in patients "
    "with type 2 diabetes mellitus, according to a 2021 meta-analysis. "
    "The effect was observed across multiple ethnic groups, including "
    "BRCA1 mutation carriers. EGFR-positive lung cancers also showed "
    "reduced inflammation markers. Metformin remains the first-line "
    "treatment for hyperglycemia."
)


def test_gliner_extracts_real_entities_with_default_labels() -> None:
    """GLiNER zero-shot — pass empty entity_types tuple so it uses
    the adapter's DEFAULT_LABELS. Asserts > 0 mentions returned with
    valid `raw_text` + `entity_type`."""
    extractor = get_extractor("gliner")
    mentions = extractor.extract(BIOMED_TEXT, ())
    assert len(mentions) > 0
    # Every mention has the required attributes.
    for m in mentions:
        assert m.raw_text  # non-empty
        assert m.entity_type  # non-empty


def test_gliner_biomed_extracts_real_entities_with_default_labels() -> None:
    """GLiNER-BioMed — biomedical zero-shot. Asserts entity types
    include at least one of DISEASE / CHEMICAL / GENE / PROTEIN
    (the adapter's biomedical DEFAULT_LABELS)."""
    extractor = get_extractor("gliner_biomed")
    mentions = extractor.extract(BIOMED_TEXT, ())
    assert len(mentions) > 0

    types = {m.entity_type.upper() for m in mentions}
    biomedical = {"DISEASE", "CHEMICAL", "GENE", "PROTEIN",
                  "SPECIES", "CELL_LINE", "CELL_TYPE", "ANATOMY"}
    # At least one biomedical label was extracted.
    assert types & biomedical


def test_hunflair2_extracts_real_entities_via_locked_label_set() -> None:
    """HunFlair2 — Flair PrefixedSequenceTagger with locked 5 labels
    (DISEASE / CHEMICAL / GENE / SPECIES / CELL_LINE). The adapter
    ignores `entity_types` since the model emits only its trained
    set; we just verify mentions come back with one of those labels."""
    extractor = get_extractor("hunflair2")
    mentions = extractor.extract(BIOMED_TEXT, ())
    assert len(mentions) > 0

    types = {m.entity_type.upper() for m in mentions}
    hunflair2_set = {"DISEASE", "CHEMICAL", "GENE", "SPECIES", "CELL_LINE"}
    # Every detected type is in the locked HunFlair2 vocabulary.
    assert types.issubset(hunflair2_set)


def test_each_adapter_returns_mentions_with_offsets_when_supported() -> None:
    """NER adapters set the `offset` field when their underlying
    model exposes token-level position info. GLiNER + HunFlair2
    both support this; LLM does not. This test asserts that
    GLiNER-BioMed (the simplest case) sets offsets on every mention."""
    extractor = get_extractor("gliner_biomed")
    mentions = extractor.extract(BIOMED_TEXT, ())
    with_offset = [m for m in mentions if m.offset is not None]
    # Every GLiNER mention should have an offset.
    assert len(with_offset) == len(mentions)
    # Offsets fall within the text bounds.
    for m in with_offset:
        assert 0 <= m.offset < len(BIOMED_TEXT)
