"""Unit tests for `logging_crash` — crash-file retention, the allowlisted
system-info dump (secrets must NEVER leak), crash-file writes, and hook
install.

Global-state safety: the install test saves + restores `sys.excepthook` /
`threading.excepthook` and patches out `enable_faulthandler`, so no test
leaves the process's hooks or faulthandler mutated.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from knowledge_agent import logging_crash as LC
from knowledge_agent.logging_ring_buffer import RingBufferHandler

if TYPE_CHECKING:
    from pathlib import Path


def _make_crash(crash_dir: Path, name: str, age_days: float = 0) -> Path:
    p = crash_dir / name
    p.write_text("x", encoding="utf-8")
    if age_days:
        old = time.time() - age_days * 86400
        os.utime(p, (old, old))
    return p


# ---- cleanup_old_crashes ----


def test_cleanup_keeps_only_newest_max_files(tmp_path: Path):
    for i in range(5):
        _make_crash(tmp_path, f"crash_2026-01-0{i}.log")
    LC.cleanup_old_crashes(tmp_path, retention_days=3650, max_files=2)
    assert len(list(tmp_path.glob("crash_*.log"))) == 2


def test_cleanup_deletes_aged_out_files(tmp_path: Path):
    fresh = _make_crash(tmp_path, "crash_fresh.log", age_days=0)
    old = _make_crash(tmp_path, "crash_old.log", age_days=30)
    LC.cleanup_old_crashes(tmp_path, retention_days=7, max_files=100)
    assert fresh.exists()
    assert not old.exists()


def test_cleanup_ignores_non_crash_files(tmp_path: Path):
    keep = tmp_path / "notes.txt"
    keep.write_text("x", encoding="utf-8")
    LC.cleanup_old_crashes(tmp_path, retention_days=0, max_files=0)
    assert keep.exists()


def test_cleanup_missing_dir_is_noop(tmp_path: Path):
    LC.cleanup_old_crashes(tmp_path / "nope", retention_days=1, max_files=1)  # must not raise


# ---- capture_system_info: the allowlist is the security boundary ----


def test_capture_system_info_excludes_secrets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NEO4J_URI", "neo4j://host:7687")  # allowlisted
    monkeypatch.setenv("NEO4J_PASSWORD", "sup3rsecret")  # NOT allowlisted
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")  # NOT allowlisted
    info = LC.capture_system_info()
    env = info["environment"]
    assert env.get("NEO4J_URI") == "neo4j://host:7687"
    assert "NEO4J_PASSWORD" not in env
    assert "ANTHROPIC_API_KEY" not in env
    # The secret values must not appear anywhere in the captured dump.
    blob = repr(info)
    assert "sup3rsecret" not in blob
    assert "sk-secret" not in blob


# ---- write_crash_file ----


def test_write_crash_file_has_traceback_ring_and_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("NEO4J_PASSWORD", "leaky")  # not allowlisted → must not appear
    ring = RingBufferHandler()
    ring.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "ring line here", None, None))
    try:
        raise ValueError("boom happened")
    except ValueError as exc:
        path = LC.write_crash_file(tmp_path, type(exc), exc, exc.__traceback__, ring, source="test")
    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "=== Traceback ===" in text
    assert "boom happened" in text
    assert "ring line here" in text  # ring tail embedded
    assert "leaky" not in text  # secret env excluded from the dump


# ---- make_sys_excepthook (call the returned hook; original patched for isolation) ----


def test_sys_excepthook_writes_crash_then_chains(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chained: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))
    hook = LC.make_sys_excepthook(tmp_path, ring=None)
    try:
        raise ValueError("hooked")
    except ValueError as exc:
        hook(type(exc), exc, exc.__traceback__)
    assert list(tmp_path.glob("crash_*.log"))  # crash file written
    assert chained  # chained to the (patched) original hook


def test_sys_excepthook_passes_keyboard_interrupt_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    chained: list[tuple] = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))
    hook = LC.make_sys_excepthook(tmp_path, ring=None)
    hook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert not list(tmp_path.glob("crash_*.log"))  # no crash file for Ctrl-C
    assert chained  # passed straight through to original


# ---- install_crash_handlers (global hooks saved + restored; faulthandler patched out) ----


@pytest.fixture
def _restore_global_hooks():
    saved_sys, saved_thread = sys.excepthook, threading.excepthook
    yield
    sys.excepthook = saved_sys
    threading.excepthook = saved_thread


def test_install_crash_handlers_sets_hooks_and_returns_asyncio_handler(
    tmp_path: Path, _restore_global_hooks
):
    with patch.object(LC, "enable_faulthandler"):  # never touch the real faulthandler
        handler = LC.install_crash_handlers(tmp_path, ring=None, retention_days=7, max_files=5)
    assert callable(handler)  # asyncio handler returned for the caller to wire onto the loop
    assert sys.excepthook.__module__ == LC.__name__  # our crash hook, not the default
    assert threading.excepthook.__module__ == LC.__name__
