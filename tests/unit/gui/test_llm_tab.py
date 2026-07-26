"""Tests for the LLM settings tab — installed-providers display, per-node
model pickers that span every installed provider, and per-node temperature
greying.

Provider install/uninstall lives in the Installs tab (see test_installs_tab).
There is no single "active provider": each node picks a 'provider:model' ref
from any installed provider (see `available_models`). The router is GUI-only
but reuses the same per-node model+temp machinery (keyed 'chat_router').
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.settings.llm_tab import LlmTab, available_models, model_options


def _tab() -> LlmTab:
    app = MagicMock()
    app.gui_config = GuiConfig()  # real defaults (anthropic)
    app.page = MagicMock()
    return LlmTab(app)  # _create_controls() runs in __init__


def test_llm_tab_builds_chat_router_field():
    tab = _tab()
    assert "chat_router" in tab.node_model_fields
    assert tab.node_model_fields["chat_router"].value == tab.app.gui_config.chat_router_model
    assert "chat_router" in tab.node_temp_sliders


def test_node_pickers_migrate_bare_to_composite():
    """On build, each node's stored bare model is normalized to a
    'provider:model' composite (wrapped with the global provider), and the
    dropdown value tracks it — so legacy configs display + dispatch correctly."""
    tab = _tab()  # GuiConfig defaults are bare, global provider = anthropic
    for node in ("mode_classifier", "synthesizer", "chat_router"):
        stored = getattr(tab.app.gui_config, f"{node}_model")
        assert stored.startswith("anthropic:")  # bare → wrapped with global provider
        assert tab.node_model_fields[node].value == stored


def test_available_models_only_installed_providers():
    """available_models() yields (provider, model) pairs only for providers
    whose adapter is installed — so pickers span exactly what's set up."""
    from knowledge_agent.gui.settings import llm_tab

    fake_registry = {
        "anthropic": {"is_installed_fn": lambda: True},
        "openai": {"is_installed_fn": lambda: True},
        "google": {"is_installed_fn": lambda: False},
        "ollama": {"is_installed_fn": lambda: False},
    }
    with patch.object(llm_tab, "LLM_PROVIDER_REGISTRY", fake_registry):
        models = available_models()
    assert {p for p, _ in models} == {"anthropic", "openai"}  # not google/ollama
    assert ("anthropic", "claude-opus-4-8") in models
    assert ("openai", "gpt-4o") in models


def test_model_options_keys_are_composite_refs():
    """model_options() keys are stored 'provider:model' refs (not bare names)."""
    from knowledge_agent.gui.settings import llm_tab

    fake_registry = {
        "anthropic": {"is_installed_fn": lambda: True},
        "openai": {"is_installed_fn": lambda: False},
        "google": {"is_installed_fn": lambda: False},
        "ollama": {"is_installed_fn": lambda: False},
    }
    with patch.object(llm_tab, "LLM_PROVIDER_REGISTRY", fake_registry):
        keys = {o.key for o in model_options()}
    assert "anthropic:claude-opus-4-8" in keys
    assert all(k.startswith("anthropic:") for k in keys)  # only the installed provider


def test_installed_providers_display_lists_names():
    """The Installed-providers box shows installed adapters' display names
    (read-only) — no single-select radio any more."""
    tab = _tab()
    tab._installed_state = {"anthropic": True, "openai": True, "google": False, "ollama": False}
    tab._sync_installed_providers_display()
    shown = tab.installed_providers_box.content.value
    assert "Anthropic Claude" in shown and "OpenAI GPT" in shown
    assert "Gemini" not in shown  # google not installed


def test_installed_providers_display_empty_state():
    tab = _tab()
    tab._installed_state = dict.fromkeys(("anthropic", "openai", "google", "ollama"), False)
    tab._sync_installed_providers_display()
    assert "No LLM providers installed" in tab.installed_providers_box.content.value


# ---- rate limits (includes the Voyage embedder) ----


def test_rate_limit_fields_span_llm_providers_plus_voyage():
    """The rate-limit section covers the four LLM providers AND the Voyage
    embedder (its native client honours a cap). Voyage is embedding-only, so
    it never appears in the LLM installed-providers list."""
    tab = _tab()
    assert set(tab.rate_limit_fields) == {
        "anthropic",
        "openai",
        "google",
        "ollama",
        "voyage",
    }
    tab._installed_state = dict.fromkeys(("anthropic", "openai", "google", "ollama"), True)
    tab._sync_installed_providers_display()
    assert "voyage" not in tab.installed_providers_box.content.value.lower()


def test_voyage_rate_blur_commits_and_bridges():
    """Editing the Voyage rate writes GuiConfig and bridges it via
    apply_voyage_rate_to_env (its own helper, alongside apply_llm_to_env)."""
    from knowledge_agent.gui.settings import llm_tab

    tab = _tab()
    tab.rate_limit_fields["voyage"].value = "5"
    with (
        patch.object(llm_tab, "save_config"),
        patch.object(llm_tab, "apply_llm_to_env"),
        patch.object(llm_tab, "apply_voyage_rate_to_env") as rate_bridge,
        patch.object(llm_tab, "reset_after_settings_change"),
    ):
        tab.on_rate_limit_blur("voyage")
    assert tab.app.gui_config.voyage_requests_per_second == 5.0
    rate_bridge.assert_called_once()


# ---- temperature-slider greying (sampling-free models) ----


def test_temp_slider_greys_out_for_sampling_free_model():
    """A node whose model dropped temperature (e.g. Opus 4.8) has its temp
    slider disabled with a tooltip; switching to a temp-taking model
    (Haiku 4.5) re-enables it. The backend omits temperature for those
    models regardless — this just surfaces that."""
    tab = _tab()
    tab.app.gui_config.synthesizer_model = "claude-opus-4-8"
    tab._sync_temp_enabled("synthesizer")
    assert tab.node_temp_sliders["synthesizer"].disabled is True
    assert "temperature" in (tab.node_temp_sliders["synthesizer"].tooltip or "")

    tab.app.gui_config.synthesizer_model = "claude-haiku-4-5"
    tab._sync_temp_enabled("synthesizer")
    assert tab.node_temp_sliders["synthesizer"].disabled is False
    assert tab.node_temp_sliders["synthesizer"].tooltip is None

    # A composite ref: the provider comes from the ref, not the global one.
    tab.app.gui_config.synthesizer_model = "openai:gpt-4o"
    tab._sync_temp_enabled("synthesizer")
    assert tab.node_temp_sliders["synthesizer"].disabled is False  # openai takes temperature


# ---- 3-tier section info text + chat-router fold ----


def test_all_llm_sections_have_info_text():
    """Every LLM-tab section's icon text is present across all three tiers
    (regression against a tier silently going blank, or a constant being
    renamed away from the section_header that uses it)."""
    from knowledge_agent.gui.settings import llm_tab as lt

    for prefix in ("_INSTALLED", "_MODELS", "_RATES"):
        for tier in ("STANDARD", "BEGINNER", "TECHNICAL"):
            value = getattr(lt, f"{prefix}_INFO_{tier}")
            assert value and value.strip(), f"{prefix}_INFO_{tier} is empty"


def test_render_node_block_has_no_info_param():
    """The chat-router help was folded into the models section (i), so the
    per-node `info` icon parameter is removed (dead code gone)."""
    import inspect

    params = inspect.signature(LlmTab._render_node_block).parameters
    assert "info" not in params  # no per-node icon hook any more


def test_model_options_label_uses_plain_separator():
    """The picker label uses a plain ASCII separator, not an em-dash, per the
    house no-em-dash rule for user-facing text."""
    from knowledge_agent.gui.settings import llm_tab

    fake_registry = {
        "anthropic": {"is_installed_fn": lambda: True},
        "openai": {"is_installed_fn": lambda: False},
        "google": {"is_installed_fn": lambda: False},
        "ollama": {"is_installed_fn": lambda: False},
    }
    with patch.object(llm_tab, "LLM_PROVIDER_REGISTRY", fake_registry):
        labels = [o.text for o in model_options()]
    assert labels  # non-empty
    assert all("—" not in t for t in labels)  # no em-dash
    assert any(" / " in t for t in labels)  # plain separator
