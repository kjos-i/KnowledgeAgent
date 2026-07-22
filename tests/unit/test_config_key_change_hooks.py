"""B15: reset_after_key_change() runs GUI-registered cache clearers.

The chat-router lru_cache lives in gui/ (which config.py must not import, per
layering), so the GUI registers a clearer via register_key_change_hook. This
pins that the hook actually fires on every reset — otherwise a fixed / rotated
API key stays served from the stale cached router until app restart.
"""

from __future__ import annotations

from knowledge_agent import config as cfg


def test_reset_after_key_change_runs_registered_hooks():
    called: list[bool] = []
    saved = list(cfg._KEY_CHANGE_HOOKS)
    try:
        cfg.register_key_change_hook(lambda: called.append(True))
        cfg.reset_after_key_change()
        assert called == [True]
    finally:
        cfg._KEY_CHANGE_HOOKS[:] = saved


def test_register_key_change_hook_is_idempotent():
    saved = list(cfg._KEY_CHANGE_HOOKS)
    try:

        def _hook() -> None:
            pass

        cfg.register_key_change_hook(_hook)
        cfg.register_key_change_hook(_hook)
        assert cfg._KEY_CHANGE_HOOKS.count(_hook) == 1
    finally:
        cfg._KEY_CHANGE_HOOKS[:] = saved
