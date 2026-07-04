"""Tests for `knowledge_agent.gui.right_panel.RightPanel`.

Verifies the mode-switcher state machine + that each mode dispatches
to the right view builder.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import flet as ft

from knowledge_agent.gui.right_panel import (
    MODE_FILE,
    MODE_INFO,
    MODE_LATEST,
    MODE_SETTINGS,
    RightPanel,
)


def test_build_starts_on_latest_mode(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    ctl = panel.build()
    assert isinstance(ctl, ft.Container)
    assert panel.current_mode == MODE_LATEST
    # All four mode buttons are registered (Latest / File via Open Result /
    # Settings / Info).
    assert set(panel.mode_buttons) == {
        MODE_LATEST, MODE_FILE, MODE_SETTINGS, MODE_INFO,
    }


def test_switch_mode_updates_current_mode_and_highlights(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    panel.build()

    panel.switch_mode(MODE_SETTINGS)
    assert panel.current_mode == MODE_SETTINGS
    # The settings button is the active one — its bgcolor is the ACTIVE_BG.
    from knowledge_agent.gui._styles import ACTIVE_BG

    assert panel.mode_buttons[MODE_SETTINGS].bgcolor == ACTIVE_BG
    # Other buttons are NOT tinted.
    assert panel.mode_buttons[MODE_LATEST].bgcolor is None


def test_switch_to_file_mode_with_no_loaded_file_falls_through_to_latest(
    fake_app: MagicMock,
):
    """Defensive: Clear may wipe loaded_file while MODE_FILE is active.
    The view dispatch silently falls back to Latest instead of crashing."""
    panel = RightPanel(fake_app)
    panel.build()
    fake_app.loaded_file = None
    panel.switch_mode(MODE_FILE)
    # The view container's content should be a Latest-style column
    # (the empty-state because last_answer is also None).
    container = panel.view_container
    assert container is not None
    # We can't easily introspect deeper than this without rendering,
    # but the build mustn't raise.


def test_switch_to_settings_renders_settings_view(fake_app: MagicMock):
    """Slice 2 wired the real SettingsView into MODE_SETTINGS.

    Verify the rendered tree IS the sub-tab Tabs widget (not the old
    'coming soon' stub Column). The actual sub-tab content is covered
    by each sub-tab's own tests."""
    fake_app.gui_config = MagicMock()
    fake_app.gui_config.llm_provider = "anthropic"
    fake_app.gui_config.embedding_provider = "voyage"
    # Numeric / string defaults the sub-tabs read at init time.
    for attr, val in (
        ("top_k", 5),
        ("retrieval_mode", "auto"),
        ("lancedb_search_mode", "hybrid"),
        ("num_candidates", 100),
        ("rrf_rank_constant", 60),
        ("rrf_rank_window_size", 50),
        ("mmr_lambda", 0.6),
        ("mmr_candidate_multiplier", 4),
        ("kg_max_rows", 50),
        ("chat_router_temperature", 0.0),
        ("skip_query_builder", False),
        ("direct_retrieve", False),
        ("use_mmr", False),
        ("keep_loaded_file_on_clear", True),
        ("restore_last_corpus", True),
        ("debug_mode", False),
        ("neo4j_uri", "neo4j://localhost:7687"),
        ("neo4j_user", "neo4j"),
        ("lancedb_path", None),
        ("ollama_base_url", "http://localhost:11434"),
        ("mode_classifier_model", "claude-haiku-4-5-20251001"),
        ("mode_classifier_temperature", 0.0),
        ("query_builder_model", "claude-haiku-4-5-20251001"),
        ("query_builder_temperature", 0.0),
        ("cypher_builder_model", "claude-sonnet-4-6"),
        ("cypher_builder_temperature", 0.0),
        ("synthesizer_model", "claude-sonnet-4-6"),
        ("synthesizer_temperature", 0.0),
        ("anthropic_requests_per_second", None),
        ("openai_requests_per_second", None),
        ("google_requests_per_second", None),
        ("ollama_requests_per_second", None),
        ("llm_max_retries", 3),
        ("embedding_model", "voyage-multimodal-3"),
        ("voyage_embedding_model", "voyage-multimodal-3"),
        ("openai_embedding_model", "text-embedding-3-small"),
        ("google_embedding_model", "models/text-embedding-004"),
        ("hf_embedding_model", "BAAI/bge-m3"),
        ("voyage_requests_per_second", None),
    ):
        setattr(fake_app.gui_config, attr, val)
    panel = RightPanel(fake_app)
    panel.build()
    panel.switch_mode(MODE_SETTINGS)
    content = panel.view_container.content
    # SettingsView returns a Tabs widget (the 5 sub-tabs).
    assert isinstance(content, ft.Tabs)
    assert content.length == 5


def test_switch_to_info_renders_stub_message(fake_app: MagicMock):
    panel = RightPanel(fake_app)
    panel.build()
    panel.switch_mode(MODE_INFO)
    content = panel.view_container.content
    body = content.controls[1]
    # Info panel is a placeholder until it's filled last (the how-to guide).
    assert "coming soon" in body.content.value.lower()


def test_button_row_contains_all_five_buttons_with_view_result_label(
    fake_app: MagicMock,
):
    """All five buttons live on one row inside the panel:
    View Result, Save Result, Open Result, Settings, Info. The
    MODE_LATEST mode button is labelled "View Result" (renamed for
    clarity — "Result" matches the file artifacts the agent produces)."""
    panel = RightPanel(fake_app)
    container = panel.build()
    # The outer Container wraps an inner Column [view_container, button_row].
    column = container.content
    button_row = column.controls[1]
    assert isinstance(button_row, ft.Row)
    assert len(button_row.controls) == 5
    # The MODE_LATEST button now reads "View Result". The button's
    # content is a centred Text widget (so two-line wrapping stays
    # centred), so we read `.content.value`.
    assert panel.mode_buttons[MODE_LATEST].content.value == "View Result"
