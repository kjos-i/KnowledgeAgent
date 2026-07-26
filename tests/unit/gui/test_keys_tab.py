"""Tests for the Settings → Keys sub-tab.

Keyring is mocked at the `keys_tab` import site (never a real OS keyring),
so construction + save-on-blur are exercised without touching credentials.
Covers the save-only-if-changed logic and that the optional LangSmith key
is a first-class field alongside the provider keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.config_store import API_KEY_NAMES
from knowledge_agent.gui.settings.keys_tab import KeysTab

_M = "knowledge_agent.gui.settings.keys_tab"


def _tab(fake_app: MagicMock, existing: dict[str, str] | None = None) -> KeysTab:
    """Build a KeysTab with `get_api_key` mocked so no keyring is touched."""
    existing = existing or {}
    with patch(f"{_M}.get_api_key", side_effect=lambda n: existing.get(n)):
        tab = KeysTab(fake_app)
        tab.build()
    return tab


def test_keys_tab_has_a_field_per_registered_key(fake_app: MagicMock):
    """One field per keyring identifier — including the optional LangSmith
    key added for eval tracing."""
    tab = _tab(fake_app)
    assert set(tab.key_fields) == set(API_KEY_NAMES)
    assert "langsmith" in tab.key_fields


def test_blur_saves_a_changed_key(fake_app: MagicMock):
    tab = _tab(fake_app)
    tab.key_fields["langsmith"].value = "ls-new-value"  # pragma: allowlist secret
    with (
        patch(f"{_M}.get_api_key", return_value=None),
        patch(f"{_M}.set_api_key") as set_key,
        patch(f"{_M}.apply_keys_to_env") as apply_env,
        patch(f"{_M}.reset_after_settings_change"),
    ):
        tab.on_key_blur("langsmith")
    set_key.assert_called_once_with("langsmith", "ls-new-value")  # pragma: allowlist secret
    apply_env.assert_called_once()  # new key bridged to env after save


def test_blur_empty_field_is_noop(fake_app: MagicMock):
    """An empty field must not wipe a stored key (explicit-delete isn't a
    GUI flow yet)."""
    tab = _tab(fake_app)
    tab.key_fields["anthropic"].value = ""
    with (
        patch(f"{_M}.get_api_key", return_value=None),
        patch(f"{_M}.set_api_key") as set_key,
    ):
        tab.on_key_blur("anthropic")
    set_key.assert_not_called()


def test_blur_unchanged_key_is_noop(fake_app: MagicMock):
    """Tabbing in and out without an edit is a no-op (field pre-loaded with
    the keyring value)."""
    tab = _tab(fake_app)
    tab.key_fields["openai"].value = "already-stored"  # pragma: allowlist secret
    with (
        patch(f"{_M}.get_api_key", return_value="already-stored"),  # pragma: allowlist secret
        patch(f"{_M}.set_api_key") as set_key,
    ):
        tab.on_key_blur("openai")
    set_key.assert_not_called()
