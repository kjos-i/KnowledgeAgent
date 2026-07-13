"""Tests for the Embedding settings tab — the active-provider switch + its
dimension-change guard.

Provider install/uninstall moved to the Installs tab (covered in
test_installs_tab); this tab now owns only the CHOICE of embedder + its model
+ rate. The switch handler asks the backend whether moving to a new provider
would strand existing chunks at a mismatched vector dim, and gates a
destructive switch behind a hard confirm.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import GuiConfig
from knowledge_agent.gui.settings.embedding_tab import EmbeddingTab


def _tab() -> EmbeddingTab:
    app = MagicMock()
    app.gui_config = GuiConfig()
    app.page = MagicMock()
    return EmbeddingTab(app)


def _plan(**kw) -> MagicMock:
    plan = MagicMock(**kw)
    plan.summary = kw.get("summary", "SUMMARY")
    return plan


# ---- C3: active-provider switch dimension guard ----
#
# The switch handler asks the backend `switch_embedder_plan` whether moving to
# the new provider would strand existing chunks at a mismatched vector dim. A
# mismatch is DESTRUCTIVE (LanceDB pins the dim at table creation) → a hard
# confirm gates it. No mismatch (or a plan error) → the switch applies straight
# through. `_apply_provider_switch` is patched in these tests: it owns the
# commit machinery (disk + env), which is exercised elsewhere; here we test only
# the guard decision.

_SWITCH = "knowledge_agent.gui.settings.embedding_tab.switch_embedder_plan"


def test_switch_same_provider_is_noop_no_plan_call():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="voyage")
    with (
        patch(_SWITCH) as plan_fn,
        patch.object(tab, "_apply_provider_switch") as apply,
    ):
        tab.on_active_provider_changed(MagicMock())
    plan_fn.assert_not_called()
    apply.assert_not_called()


def test_switch_no_dim_mismatch_applies_without_dialog():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="openai")
    plan = _plan(dim_mismatch=False)
    with (
        patch(_SWITCH, return_value=plan),
        patch.object(tab, "_apply_provider_switch") as apply,
    ):
        tab.on_active_provider_changed(MagicMock())
    apply.assert_called_once_with("openai")
    tab.app.page.show_dialog.assert_not_called()


def test_switch_dim_mismatch_shows_confirm_and_defers_apply():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="huggingface")
    plan = _plan(dim_mismatch=True, summary="DESTRUCTIVE switch: 1024-dim → 384-dim")
    with (
        patch(_SWITCH, return_value=plan),
        patch.object(tab, "_apply_provider_switch") as apply,
    ):
        tab.on_active_provider_changed(MagicMock())
    tab.app.page.show_dialog.assert_called_once()
    apply.assert_not_called()


def test_switch_dim_mismatch_confirm_applies_switch():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="huggingface")
    plan = _plan(dim_mismatch=True)
    captured: dict = {}
    with (
        patch(_SWITCH, return_value=plan),
        patch.object(tab, "_show_confirm", side_effect=lambda **kw: captured.update(kw)),
        patch.object(tab, "_apply_provider_switch") as apply,
    ):
        tab.on_active_provider_changed(MagicMock())
        captured["on_confirm"]()
    apply.assert_called_once_with("huggingface")


def test_switch_dim_mismatch_cancel_reverts_radio():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="huggingface")
    plan = _plan(dim_mismatch=True)
    captured: dict = {}
    with (
        patch(_SWITCH, return_value=plan),
        patch.object(tab, "_show_confirm", side_effect=lambda **kw: captured.update(kw)),
    ):
        tab.on_active_provider_changed(MagicMock())
        captured["on_cancel"]()
    assert tab.active_provider_radio.value == "voyage"


def test_switch_plan_error_falls_through_to_plain_switch():
    tab = _tab()
    tab.app.gui_config.embedding_provider = "voyage"
    tab.active_provider_radio = MagicMock(value="openai")
    with (
        patch(_SWITCH, side_effect=RuntimeError("lancedb down")),
        patch.object(tab, "_apply_provider_switch") as apply,
    ):
        tab.on_active_provider_changed(MagicMock())
    apply.assert_called_once_with("openai")
    tab.app.page.show_dialog.assert_not_called()
