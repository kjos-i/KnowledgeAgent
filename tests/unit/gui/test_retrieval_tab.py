"""Tests for the Retrieval tab's input-mode radio (R4c).

The radio (Conversational / Direct query / Direct Cypher) persists the
user's choice to `GuiConfig.input_mode` via the standard commit-with-
rollback pattern. The downstream *consumer*
(`app._invoke_state_for_input_mode`) is covered in test_app_input_mode;
here we pin the *setter* — a radio change writes the config and rolls
back cleanly on a save failure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import ConfigError, GuiConfig
from knowledge_agent.gui.settings.retrieval_tab import RetrievalTab

_SAVE = "knowledge_agent.gui.settings.retrieval_tab.save_config"
_APPLY = "knowledge_agent.gui.settings.retrieval_tab.apply_retrieval_to_env"
_RESET = "knowledge_agent.gui.settings.retrieval_tab.reset_after_settings_change"


def _tab() -> RetrievalTab:
    app = MagicMock()
    app.gui_config = GuiConfig()
    app.page = MagicMock()
    return RetrievalTab(app)


def test_input_mode_radio_reflects_config_default():
    tab = _tab()
    assert tab.input_mode_radio is not None
    assert tab.input_mode_radio.value == "conversational"


def test_input_mode_change_persists():
    tab = _tab()
    tab.input_mode_radio.value = "direct_cypher"
    with patch(_SAVE), patch(_APPLY), patch(_RESET):
        tab.on_input_mode_changed(MagicMock())
    assert tab.app.gui_config.input_mode == "direct_cypher"


def test_input_mode_change_rolls_back_on_save_failure():
    tab = _tab()
    tab.input_mode_radio.value = "direct_query"
    # save_config raising ConfigError → _commit returns False → revert.
    with (
        patch(_SAVE, side_effect=ConfigError("disk full")),
        patch(_APPLY),
        patch(_RESET),
    ):
        tab.on_input_mode_changed(MagicMock())
    assert tab.app.gui_config.input_mode == "conversational"
    assert tab.input_mode_radio.value == "conversational"


# ---- mode-based gray-out (a disabled control is inert AND not fed to search) ----


def _sync_with(tab: RetrievalTab, **cfg: object) -> None:
    for key, value in cfg.items():
        setattr(tab.app.gui_config, key, value)
    tab._sync_enabled_state()


def test_neo4j_only_grays_the_whole_lancedb_block():
    tab = _tab()
    _sync_with(tab, retrieval_mode="neo4j_only")
    assert tab.lancedb_mode_radio.disabled is True
    # Flet doesn't cascade the group's disabled, so each radio must be set.
    assert all(r.disabled for r in tab._lancedb_radios)
    # ...and doesn't recolor disabled radios, so the block is faded too.
    assert tab._lancedb_mode_box.opacity == 0.4
    assert tab.num_candidates_field.disabled is True
    assert tab.rrf_constant_field.disabled is True
    assert tab.use_mmr_checkbox.disabled is True
    assert tab.mmr_lambda_slider.disabled is True
    assert tab.kg_max_rows_field.disabled is False  # the Neo4j leg runs


def test_lancedb_only_grays_kg_max_rows():
    tab = _tab()
    _sync_with(tab, retrieval_mode="lancedb_only", lancedb_search_mode="hybrid")
    assert tab.kg_max_rows_field.disabled is True  # no Neo4j leg
    assert tab.lancedb_mode_radio.disabled is False
    assert not any(r.disabled for r in tab._lancedb_radios)
    assert tab._lancedb_mode_box.opacity == 1.0  # LanceDB block fully visible
    assert tab.num_candidates_field.disabled is False
    assert tab.rrf_constant_field.disabled is False  # hybrid → rrf fuses


def test_fts_grays_rrf_and_mmr_but_not_the_pool():
    tab = _tab()
    _sync_with(tab, retrieval_mode="lancedb_only", lancedb_search_mode="fts")
    assert tab.num_candidates_field.disabled is False  # pool still used
    assert tab.rrf_constant_field.disabled is True  # rrf only fuses in hybrid
    assert tab.use_mmr_checkbox.disabled is True  # FTS has no vectors
    assert tab.mmr_lambda_slider.disabled is True


def test_mmr_off_grays_only_the_lambda_slider():
    tab = _tab()
    _sync_with(tab, retrieval_mode="lancedb_only", lancedb_search_mode="hybrid", use_mmr=False)
    assert tab.use_mmr_checkbox.disabled is False  # can flip on
    assert tab.mmr_lambda_slider.disabled is True  # off → its parameter is irrelevant


def test_both_leg_mode_enables_lance_and_kg():
    tab = _tab()
    _sync_with(tab, retrieval_mode="parallel_fused", lancedb_search_mode="hybrid")
    assert tab.num_candidates_field.disabled is False
    assert tab.rrf_constant_field.disabled is False
    assert tab.kg_max_rows_field.disabled is False
