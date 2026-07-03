"""Tests for the multi-extractor (priority-ordered union) UI logic in
`CorpusConfigEditor`.

These drive the pure state-manipulation handlers directly (toggle /
move / mode-change) rather than rendering Flet — the fake app + page
from `conftest.py` supply a callable `.update()` so the mutation +
refresh path runs without launching a real UI.

Covers:
  - loading a config populates `_selected_extractors` in order
  - toggle add appends (lowest priority); toggle remove drops
  - the last extractor can't be deselected
  - up/down reorder swaps priority
  - the Replace/Add mode radio persists onto the config
  - entity_types edits preserve the extractors list + mode
"""

from __future__ import annotations

from knowledge_agent.gui.library.corpus_config_editor import CorpusConfigEditor
from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)


def _editor(fake_app):
    ed = CorpusConfigEditor(fake_app)
    ed._create_controls()
    return ed


def _load(ed, extractors, entity_types=None, mode="replace"):
    cfg = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(
            extractors=list(extractors),
            entity_types=list(entity_types or []),
            entity_types_mode=mode,
        ),
    )
    ed._corpus_config = cfg
    ed._baseline_config = cfg.model_copy(deep=True)
    ed._populate_controls()
    return ed


def test_populate_sets_selected_extractors_in_order(fake_app):
    ed = _load(_editor(fake_app), ["hunflair2", "llm"])
    assert ed._selected_extractors == ["hunflair2", "llm"]


def test_toggle_add_appends_as_lowest_priority(fake_app):
    ed = _load(_editor(fake_app), ["llm"])
    ed._on_extractor_toggle("hunflair2")
    assert ed._selected_extractors == ["llm", "hunflair2"]
    assert ed._corpus_config.entities.extractors == ["llm", "hunflair2"]


def test_toggle_remove_drops_extractor(fake_app):
    ed = _load(_editor(fake_app), ["llm", "hunflair2"])
    ed._on_extractor_toggle("hunflair2")
    assert ed._selected_extractors == ["llm"]
    assert ed._corpus_config.entities.extractors == ["llm"]


def test_cannot_deselect_the_last_extractor(fake_app):
    ed = _load(_editor(fake_app), ["llm"])
    ed._on_extractor_toggle("llm")
    # Still exactly one — the last can't be removed.
    assert ed._selected_extractors == ["llm"]
    assert ed._corpus_config.entities.extractors == ["llm"]


def test_move_up_raises_priority(fake_app):
    ed = _load(_editor(fake_app), ["llm", "hunflair2"])
    ed._on_extractor_move("hunflair2", -1)
    assert ed._selected_extractors == ["hunflair2", "llm"]
    assert ed._corpus_config.entities.extractors == ["hunflair2", "llm"]


def test_move_down_lowers_priority(fake_app):
    ed = _load(_editor(fake_app), ["hunflair2", "llm"])
    ed._on_extractor_move("hunflair2", 1)
    assert ed._selected_extractors == ["hunflair2", "llm"][::-1]


def test_move_beyond_boundary_is_noop(fake_app):
    ed = _load(_editor(fake_app), ["llm", "hunflair2"])
    ed._on_extractor_move("llm", -1)  # already at top
    assert ed._selected_extractors == ["llm", "hunflair2"]


def test_mode_radio_change_persists(fake_app):
    ed = _load(_editor(fake_app), ["gliner"], entity_types=["GENE"])
    ed.entity_types_mode_radio.value = "add"
    ed._on_entity_types_mode_changed(None)
    assert ed._corpus_config.entities.entity_types_mode == "add"
    # extractors + types preserved.
    assert ed._corpus_config.entities.extractors == ["gliner"]
    assert ed._corpus_config.entities.entity_types == ["GENE"]


def test_entity_types_edit_preserves_extractors_and_mode(fake_app):
    ed = _load(
        _editor(fake_app), ["hunflair2", "llm"],
        entity_types=["GENE"], mode="add",
    )
    ed.entity_types_field.value = "DISEASE, CHEMICAL"
    ed._on_entity_types_blur(None)
    ent = ed._corpus_config.entities
    assert ent.entity_types == ["DISEASE", "CHEMICAL"]
    assert ent.extractors == ["hunflair2", "llm"]
    assert ent.entity_types_mode == "add"


def test_info_banner_visible_from_third_extractor(fake_app):
    ed = _load(_editor(fake_app), ["llm"])
    assert ed.extractor_info_banner.visible is False
    ed._on_extractor_toggle("hunflair2")
    assert ed.extractor_info_banner.visible is False
    ed._on_extractor_toggle("gliner")
    assert ed.extractor_info_banner.visible is True


def test_entity_types_field_disabled_when_hunflair2_only(fake_app):
    ed = _load(_editor(fake_app), ["hunflair2"])
    assert ed.entity_types_field.disabled is True
    # Adding the LLM re-enables the shared types field.
    ed._on_extractor_toggle("llm")
    assert ed.entity_types_field.disabled is False


def test_mode_radio_visible_only_when_gliner_selected(fake_app):
    ed = _load(_editor(fake_app), ["llm"])
    assert ed.entity_types_mode_radio.visible is False
    ed._on_extractor_toggle("gliner_biomed")
    assert ed.entity_types_mode_radio.visible is True
