"""Tests for the shared baseline-vs-draft config diff (gui/library/config_diff).

Focus: the entities subsection, where the diff must reflect the FULL extractor
list and entity_types_mode — not just the primary extractor — so it can never
contradict is_dirty() (B13).
"""

from __future__ import annotations

from knowledge_agent.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
)
from knowledge_agent.gui.library.config_diff import config_diff


def _config(*, extractors: list[str], mode: str = "replace") -> CorpusConfig:
    return CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        allowed_types=["Paper"],
        entities=EntityConfig(extractors=extractors, entity_types_mode=mode),
    )


def test_config_diff_detects_secondary_extractor_change():
    """B13: adding a SECONDARY extractor must show as a change. The old diff
    compared only extractors[0], so ['llm'] vs ['llm','gliner'] read as 'no
    changes' while is_dirty() disagreed — and a re-ingest silently applied it."""
    base = _config(extractors=["llm"])
    draft = _config(extractors=["llm", "gliner"])
    diffs = config_diff(base, draft)
    assert any(d[0] == "extractors" for d in diffs), diffs


def test_config_diff_detects_entity_types_mode_change():
    """B13: flipping entity_types_mode (replace<->add) must show as a change;
    the old diff never compared it."""
    base = _config(extractors=["llm"], mode="replace")
    draft = _config(extractors=["llm"], mode="add")
    diffs = config_diff(base, draft)
    assert any(d[0] == "entity_types_mode" for d in diffs), diffs


def test_config_diff_detects_ontology_matching_change():
    """A matching-mode flip (exact<->fuzzy) must show as a change; the old diff
    compared only the ontology_* enable flags, so it lit ● + enabled Discard
    while the card read 'no changes' — and a re-ingest silently applied it."""
    base = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True, ontology_mesh=True),
        entities=EntityConfig(extractors=["llm"]),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )
    draft = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True, ontology_mesh=True),
        entities=EntityConfig(extractors=["llm"]),
        ontology={"mesh": OntologyConfig(matching="fuzzy")},
    )
    assert any("matching" in d[0].lower() for d in config_diff(base, draft))
