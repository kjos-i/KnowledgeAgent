"""Tests for the Settings → Keys sub-tab.

Keyring is mocked at the `keys_tab` import site (never a real OS keyring),
so construction + save-on-blur are exercised without touching credentials.
Covers the save-only-if-changed logic and that the optional LangSmith key
is a first-class field alongside the provider keys.
"""

from __future__ import annotations

import os
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
    """An empty field must not wipe a stored key; deletion is a separate,
    explicit action via the per-key delete button."""
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


# ---- delete flow (per-key delete button, option i: clears only its env var) ----


def test_delete_shows_confirm_dialog_when_key_exists(fake_app: MagicMock):
    """The delete button opens a confirm dialog (delete acts on confirm only)."""
    tab = _tab(fake_app, existing={"openai": "stored-key"})  # pragma: allowlist secret
    with patch(f"{_M}.get_api_key", return_value="stored-key"):  # pragma: allowlist secret
        tab._on_key_delete("openai")
    fake_app.page.show_dialog.assert_called_once()


def test_delete_noop_when_key_not_set(fake_app: MagicMock):
    """Deleting a key that isn't stored just reports it, with no dialog."""
    tab = _tab(fake_app)
    with patch(f"{_M}.get_api_key", return_value=None):
        tab._on_key_delete("google")
    fake_app.page.show_dialog.assert_not_called()
    assert "not set" in tab.status.value.lower()


def test_delete_key_removes_and_clears_only_its_env_var(fake_app: MagicMock, monkeypatch):
    """_delete_key deletes the keyring entry (set_api_key with empty), clears
    ONLY this key's env var (option i, not a shell-wide clobber), resets the
    caches, and empties the field."""
    tab = _tab(fake_app, existing={"anthropic": "sk-old"})  # pragma: allowlist secret
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-old")  # pragma: allowlist secret
    monkeypatch.setenv("OPENAI_API_KEY", "keep-me")  # pragma: allowlist secret
    with (
        patch(f"{_M}.get_api_key", return_value=None),  # key is gone after delete
        patch(f"{_M}.set_api_key") as set_key,
        patch(f"{_M}.reset_after_settings_change") as reset,
    ):
        tab._delete_key("anthropic")
    set_key.assert_called_once_with("anthropic", "")  # empty value deletes
    reset.assert_called_once()
    assert "ANTHROPIC_API_KEY" not in os.environ  # deleted key's env var cleared (option i)
    assert os.environ["OPENAI_API_KEY"] == "keep-me"  # pragma: allowlist secret
    assert tab.key_fields["anthropic"].value == ""  # field emptied
