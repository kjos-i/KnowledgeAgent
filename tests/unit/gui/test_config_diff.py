"""Tests for the shared baseline-vs-draft config diff
(`gui.library.config_diff`).

Pure function over two `CorpusConfig`s — no Flet. The Ingest tab's summary
card and the Select tab's corpus card both render this, so pinning it here
guards both against silent divergence.
"""

from __future__ import annotations

from knowledge_agent.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.gui.library.config_diff import config_diff


def _cfg(**overrides) -> CorpusConfig:
    base = {
        "allowed_types": ["Paper"],
        "layers": LayerFlags(chunks=True, entities=True),
        "entities": EntityConfig(extractors=["llm"]),
    }
    base.update(overrides)
    return CorpusConfig(**base)


def test_identical_configs_have_no_diff():
    cfg = _cfg()
    assert config_diff(cfg, cfg.model_copy(deep=True)) == []


def test_layer_flag_change_renders_on_off():
    baseline = _cfg(layers=LayerFlags(chunks=True, entities=True))
    draft = _cfg(layers=LayerFlags(chunks=True, entities=True, triples=True))
    assert ("triples", "off", "on") in config_diff(baseline, draft)


def test_scalar_change_renders_string_values():
    diffs = config_diff(_cfg(chunk_max_tokens=512), _cfg(chunk_max_tokens=800))
    assert ("chunk_max_tokens", "512", "800") in diffs


def test_xrefs_three_state_change_uses_raw_values():
    baseline = _cfg(layers=LayerFlags(chunks=True, entities=True, xrefs="none"))
    draft = _cfg(layers=LayerFlags(chunks=True, entities=True, xrefs="use"))
    assert ("xrefs", "none", "use") in config_diff(baseline, draft)


def test_multiple_changes_all_reported():
    baseline = _cfg(
        chunk_max_tokens=512,
        layers=LayerFlags(chunks=True, entities=True),
    )
    draft = _cfg(
        chunk_max_tokens=800,
        layers=LayerFlags(chunks=True, entities=True, triples=True),
    )
    fields = {name for name, _, _ in config_diff(baseline, draft)}
    assert {"chunk_max_tokens", "triples"} <= fields


def test_diff_is_directional_baseline_then_draft():
    """The triple is (label, baseline_value, draft_value) — order matters so
    the card can render 'was → now'."""
    diffs = config_diff(_cfg(chunk_max_tokens=512), _cfg(chunk_max_tokens=800))
    _name, old, new = next(d for d in diffs if d[0] == "chunk_max_tokens")
    assert (old, new) == ("512", "800")
