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

from unittest.mock import patch

from knowledge_agent.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
)
from knowledge_agent.gui.library.corpus_config_editor import CorpusConfigEditor


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


def test_ontology_checkbox_hard_disabled_when_not_downloaded(fake_app):
    """An un-downloaded ontology can't be enabled — its checkbox is
    `disabled` (a hard block, not just greyed) with an Installs hint, while
    a downloaded one stays enabled. Ingest never auto-downloads."""
    ed = _editor(fake_app)
    with patch(
        "knowledge_agent.kg.ontologies.lifecycle.is_ontology_downloaded",
        side_effect=lambda key: key == "go",  # only GO is on disk
    ):
        _load(ed, extractors=["llm"])  # entities layer on
    assert ed.ontology_checkboxes["go"].disabled is False
    assert ed.ontology_checkboxes["mesh"].disabled is True
    assert "Installs" in (ed.ontology_checkboxes["mesh"].tooltip or "")


def test_extractor_temp_slider_greyed_for_sampling_free_model(fake_app):
    """The entity/triples temperature sliders grey out when their selected
    LLM model dropped temperature (e.g. Opus 4.8), and stay active for a
    temp-taking model (Haiku 4.5)."""
    fake_app.gui_config.llm_provider = "anthropic"
    ed = _load(_editor(fake_app), extractors=["llm"])
    ed._corpus_config = ed._corpus_config.model_copy(
        update={
            "entity_extractor_model": "claude-opus-4-8",  # sampling-free
            "triples_extractor_model": "claude-haiku-4-5",  # takes temperature
        },
    )
    ed._sync_extractor_temp_enabled()
    assert ed.entity_extractor_temperature_slider.disabled is True
    assert "temperature" in (ed.entity_extractor_temperature_slider.tooltip or "")
    assert ed.triples_extractor_temperature_slider.disabled is False
    assert ed.triples_extractor_temperature_slider.tooltip is None


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
        _editor(fake_app),
        ["hunflair2", "llm"],
        entity_types=["GENE"],
        mode="add",
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


def test_toggling_entities_on_seeds_a_valid_section(fake_app):
    """Ticking the entities layer ON when there is no [entities] section
    seeds a default EntityConfig, so the config stays valid.

    Regression: `_on_layer_toggle` used to flip only `layers.entities`,
    leaving `entities=None` — an invalid combo the model validator rejects,
    which sat silently in the draft until the next ingest / bulk-op
    pre-flight blew up ("layers.entities=true requires an [entities]
    section")."""
    ed = _editor(fake_app)
    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=False),  # off → no [entities] section
    )
    ed._baseline_config = ed._corpus_config.model_copy(deep=True)
    ed._populate_controls()

    ed.entities_checkbox.value = True  # user ticks the entities layer
    ed._on_layer_toggle("entities")

    assert ed._corpus_config.layers.entities is True
    assert ed._corpus_config.entities is not None
    assert ed._corpus_config.entities.extractors  # >= 1 extractor seeded
    # The whole config validates now (this raised before the fix).
    CorpusConfig.model_validate(ed._corpus_config.model_dump(mode="json"))


def test_toggling_entities_off_clears_the_seeded_section(fake_app):
    """Regression: turning entities OFF on a corpus whose baseline had no
    [entities] section drops the section the ON path seeded, so a toggle
    on→off leaves no phantom pending change (entities back to None, not dirty)."""
    ed = _editor(fake_app)
    ed._corpus_config = CorpusConfig(
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=False),  # off → no [entities] section
    )
    ed._baseline_config = ed._corpus_config.model_copy(deep=True)
    ed._populate_controls()

    ed.entities_checkbox.value = True
    ed._on_layer_toggle("entities")
    assert ed._corpus_config.entities is not None  # seeded on ON

    ed.entities_checkbox.value = False
    ed._on_layer_toggle("entities")
    assert ed._corpus_config.layers.entities is False
    assert ed._corpus_config.entities is None  # seed cleared on OFF
    assert ed.is_dirty() is False  # draft == baseline: no phantom pending change


def test_model_blur_ignores_bare_to_composite_normalization(fake_app):
    """Regression: the model picker shows a bare stored model in its composite
    provider:model form; a blur that only echoes that form must NOT be recorded
    as a change (no phantom pending edit)."""
    fake_app.gui_config.llm_provider = "anthropic"
    ed = _load(_editor(fake_app), ["llm"])
    ed._corpus_config = ed._corpus_config.model_copy(
        update={"entity_extractor_model": "claude-haiku-4-5"},  # bare default
    )
    # The field displays the composite form (as _populate_controls sets it).
    ed.entity_extractor_model_field.value = "anthropic:claude-haiku-4-5"
    ed._on_entity_extractor_model_blur(None)
    assert ed._corpus_config.entity_extractor_model == "claude-haiku-4-5"  # unchanged


def test_model_blur_commits_a_genuinely_different_model(fake_app):
    """A real model pick still commits (the no-op skip only ignores same-model
    bare↔composite echoes)."""
    fake_app.gui_config.llm_provider = "anthropic"
    ed = _load(_editor(fake_app), ["llm"])
    ed.entity_extractor_model_field.value = "anthropic:claude-sonnet-4-6"
    ed._on_entity_extractor_model_blur(None)
    assert ed._corpus_config.entity_extractor_model == "anthropic:claude-sonnet-4-6"


def test_entities_off_refused_while_dependent_layer_on(fake_app):
    """Regression: turning entities OFF is refused while a layer that requires
    it (triples / cross_doc / cross_doc_xrefs / any ontology) is still on."""
    ed = _load(_editor(fake_app), ["llm"])  # entities on, nothing depends on it
    assert ed._dependency_error_for("entities", False) is None  # allowed

    # Triples on → refused, message names it.
    ed._corpus_config = ed._corpus_config.model_copy(
        update={"layers": ed._corpus_config.layers.model_copy(update={"triples": True})},
    )
    err = ed._dependency_error_for("entities", False)
    assert err is not None
    assert "Triples" in err

    # An ontology on (no triples) is also a dependent.
    ed._corpus_config = ed._corpus_config.model_copy(
        update={
            "layers": ed._corpus_config.layers.model_copy(
                update={"triples": False, "ontology_mesh": True},
            ),
        },
    )
    err2 = ed._dependency_error_for("entities", False)
    assert err2 is not None
    assert "Ontology" in err2
