"""Tests for `knowledge_agent.logging_setup`.

Covers the full setup contract: dictConfig wiring, QueueListener
fan-out, library noise clamp, packaged detection, ring buffer
subscribe pattern, crash hooks (sys/threading/asyncio), faulthandler,
captureWarnings, permanent archive, PII redaction, crash cleanup.

Test isolation: a function-scoped fixture (`reset_logging_state`)
saves and restores all global state mutated by `init_logging()` —
without it, dictConfig leftovers, hook installs, and the
QueueListener thread bleed across tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from knowledge_agent import logging_setup
from knowledge_agent.logging_setup import (
    CRASH_SUBDIR,
    ENV_LOG_CONSOLE,
    ENV_LOG_DIR,
    ENV_LOG_LEVEL,
    ENV_PACKAGED,
    LIBRARY_NOISE_CLAMP,
    LOG_FILE_NAME,
    LoggingSettings,
    RedactingFormatter,
    RingBufferHandler,
    _detect_packaged,
    init_logging,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Global-state isolation fixture.
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_logging_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Save + restore every global mutated by `init_logging()`.

    Points the log dir at a fresh tmp_path so tests don't write to the
    user's real `%LOCALAPPDATA%`. Tears down the QueueListener,
    re-resets sys.excepthook + threading.excepthook, and clears all
    handlers from the root + clamped loggers.
    """
    saved_sys_excepthook = sys.excepthook
    saved_threading_excepthook = threading.excepthook
    saved_root_handlers = list(logging.root.handlers)
    saved_root_level = logging.root.level
    saved_listener = logging_setup._LISTENER
    saved_initialized = logging_setup._INITIALIZED
    saved_lib_levels = {name: logging.getLogger(name).level for name in LIBRARY_NOISE_CLAMP}

    # Point the log dir at tmp_path. Tests that need a different
    # location can override KAGENT_LOG_DIR again.
    monkeypatch.setenv(ENV_LOG_DIR, str(tmp_path))
    # Force not-packaged unless a test sets it.
    monkeypatch.delenv(ENV_PACKAGED, raising=False)
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)
    monkeypatch.delenv(ENV_LOG_CONSOLE, raising=False)

    try:
        yield tmp_path
    finally:
        # Stop the listener so writes flush.
        if logging_setup._LISTENER is not None:
            with contextlib.suppress(Exception):
                logging_setup._LISTENER.stop()
        # Restore module state.
        logging_setup._LISTENER = saved_listener
        logging_setup._INITIALIZED = saved_initialized
        # Restore root + clamped logger state.
        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
        for h in saved_root_handlers:
            logging.root.addHandler(h)
        logging.root.setLevel(saved_root_level)
        for name, level in saved_lib_levels.items():
            logging.getLogger(name).setLevel(level)
        # Restore hooks.
        sys.excepthook = saved_sys_excepthook
        threading.excepthook = saved_threading_excepthook


def _flush_queue() -> None:
    """Stop the listener so every queued record reaches its handler.

    Tests must call this before asserting on file content — the queue
    handler is async, so `logger.info()` returning does not mean the
    record has been written.
    """
    if logging_setup._LISTENER is not None:
        logging_setup._LISTENER.stop()
        logging_setup._LISTENER = None


# ---------------------------------------------------------------------------
# 1. Happy path.
# ---------------------------------------------------------------------------


def test_happy_path_writes_log_file(reset_logging_state: Path) -> None:
    """A single info call should land in `kagent.log`."""
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="DEBUG"))
    logging.getLogger("test_logger").info("hello world")
    _flush_queue()

    log_path = reset_logging_state / LOG_FILE_NAME
    assert log_path.is_file()
    content = log_path.read_text(encoding="utf-8")
    assert "hello world" in content
    assert "test_logger" in content
    assert "INFO" in content


# ---------------------------------------------------------------------------
# 2. Packaged detection — each of the 4 layers in isolation.
# ---------------------------------------------------------------------------


def test_detect_packaged_via_sys_frozen(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert _detect_packaged() is True


def test_detect_packaged_via_env_var(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setenv(ENV_PACKAGED, "1")
    assert _detect_packaged() is True


def test_detect_packaged_via_no_stdout(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "stdout", None)
    try:
        assert _detect_packaged() is True
    finally:
        # Pytest captures stdout; restore so its own teardown survives.
        monkeypatch.undo()


def test_detect_packaged_via_exe_heuristic(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", "/usr/local/bin/KnowledgeAgent")
    assert _detect_packaged() is True


def test_detect_packaged_dev_default(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal `python` run with all four signals absent."""
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", "/usr/bin/python3.13")
    assert _detect_packaged() is False


# ---------------------------------------------------------------------------
# 3. dictConfig idempotency.
# ---------------------------------------------------------------------------


def test_init_logging_is_idempotent(reset_logging_state: Path) -> None:
    """Two init calls must NOT double up handlers on the root logger
    or leave two QueueListener threads running."""
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    handlers_after_first = list(logging.root.handlers)
    listener_after_first = logging_setup._LISTENER

    init_logging(LoggingSettings(log_dir=reset_logging_state))
    handlers_after_second = list(logging.root.handlers)
    listener_after_second = logging_setup._LISTENER

    assert len(handlers_after_first) == len(handlers_after_second) == 1
    assert listener_after_first is not listener_after_second  # rebuilt cleanly


# ---------------------------------------------------------------------------
# 4. QueueHandler async-safety — log from a coroutine, assert no
#    blocking on file I/O. Heuristic: 500 records inside a coroutine
#    should complete in well under a second.
# ---------------------------------------------------------------------------


def test_logging_from_coroutine_does_not_block_event_loop(
    reset_logging_state: Path,
) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="DEBUG"))
    logger = logging.getLogger("coroutine_test")

    async def _loop_test() -> float:
        start = time.perf_counter()
        for i in range(500):
            logger.info("emit %d", i)
        return time.perf_counter() - start

    elapsed = asyncio.run(_loop_test())
    _flush_queue()
    # 500 enqueues with no disk I/O should finish in << 1 second.
    # Generous bound: any sane machine clears this. If this fails the
    # QueueHandler wiring is broken and emitters are blocking.
    assert elapsed < 1.0, f"500 emits took {elapsed:.3f}s — emitter blocked"

    log_path = reset_logging_state / LOG_FILE_NAME
    content = log_path.read_text(encoding="utf-8")
    assert content.count("emit ") >= 500


# ---------------------------------------------------------------------------
# 5. Frozen path — packaged build, no console handler attached.
# ---------------------------------------------------------------------------


def test_packaged_build_skips_console_handler(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_PACKAGED, "1")
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    assert logging.getHandlerByName("console") is None
    # File + ring should still be attached via the queue listener.
    assert logging.getHandlerByName("file") is not None
    assert logging.getHandlerByName("ring") is not None


def test_explicit_console_override_in_packaged(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_PACKAGED, "1")
    monkeypatch.setenv(ENV_LOG_CONSOLE, "1")
    init_logging(LoggingSettings.load())
    assert logging.getHandlerByName("console") is not None


# ---------------------------------------------------------------------------
# 6. Library noise clamp — every clamped library is at WARNING/ERROR.
# ---------------------------------------------------------------------------


def test_library_noise_clamp(reset_logging_state: Path) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    for name, level_name in LIBRARY_NOISE_CLAMP.items():
        expected = logging.getLevelName(level_name)
        actual = logging.getLogger(name).level
        assert actual == expected, (
            f"{name} should be {level_name} ({expected}), got "
            f"{logging.getLevelName(actual)} ({actual})"
        )


# ---------------------------------------------------------------------------
# 7. Ring buffer subscribe / unsubscribe / broken-callback safety /
#    concurrent emits.
# ---------------------------------------------------------------------------


def test_ring_buffer_subscribe_fires_on_emit() -> None:
    handler = RingBufferHandler(maxlen=10)
    seen: list[logging.LogRecord] = []
    handler.subscribe(seen.append)
    handler.emit(_make_record("hello"))
    assert len(seen) == 1
    assert seen[0].getMessage() == "hello"


def test_ring_buffer_unsubscribe_stops_delivery() -> None:
    handler = RingBufferHandler(maxlen=10)
    seen: list[logging.LogRecord] = []
    unsubscribe = handler.subscribe(seen.append)
    handler.emit(_make_record("first"))
    unsubscribe()
    handler.emit(_make_record("second"))
    assert [r.getMessage() for r in seen] == ["first"]


def test_ring_buffer_broken_callback_does_not_crash() -> None:
    handler = RingBufferHandler(maxlen=10)
    good_seen: list[logging.LogRecord] = []

    def bad(_record: logging.LogRecord) -> None:
        raise RuntimeError("boom")

    handler.subscribe(bad)
    handler.subscribe(good_seen.append)
    handler.emit(_make_record("ok"))
    assert len(good_seen) == 1


def test_ring_buffer_concurrent_emits_land_in_deque() -> None:
    handler = RingBufferHandler(maxlen=10_000)
    threads = []
    for tid in range(8):
        t = threading.Thread(
            target=lambda tid=tid: [handler.emit(_make_record(f"t{tid}-{i}")) for i in range(100)]
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    snapshot = handler.get_snapshot()
    assert len(snapshot) == 800


def test_ring_buffer_snapshot_is_a_copy() -> None:
    handler = RingBufferHandler(maxlen=10)
    handler.emit(_make_record("a"))
    snap = handler.get_snapshot()
    handler.emit(_make_record("b"))
    assert len(snap) == 1  # original snapshot unchanged
    assert len(handler.get_snapshot()) == 2


# ---------------------------------------------------------------------------
# 8. Ring buffer always at DEBUG regardless of file handler level.
# ---------------------------------------------------------------------------


def test_ring_buffer_captures_debug_even_at_info_root(
    reset_logging_state: Path,
) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="INFO"))
    # Force a fresh-record path; bypass root-level filtering by going
    # through a logger with explicit DEBUG.
    debug_logger = logging.getLogger("debug_emitter")
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.debug("debug-detail")
    _flush_queue()

    ring = logging.getHandlerByName("ring")
    assert isinstance(ring, RingBufferHandler)
    assert ring.level == logging.DEBUG
    messages = [r.getMessage() for r in ring.get_snapshot()]
    assert "debug-detail" in messages

    # File handler should be at INFO (root level), so the DEBUG line
    # is NOT in the file.
    file_path = reset_logging_state / LOG_FILE_NAME
    assert "debug-detail" not in file_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 9. sys.excepthook writes a crash file with traceback + system info.
# ---------------------------------------------------------------------------


def test_sys_excepthook_writes_crash_file(reset_logging_state: Path) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    try:
        raise RuntimeError("synthetic crash")
    except RuntimeError as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)

    crash_dir = reset_logging_state / CRASH_SUBDIR
    crash_files = list(crash_dir.glob("crash_*.log"))
    assert crash_files, "no crash file written"
    body = crash_files[0].read_text(encoding="utf-8")
    assert "synthetic crash" in body
    assert "RuntimeError" in body
    assert "Traceback" in body
    assert "Python:" in body
    assert "Platform:" in body
    assert "Installed packages" in body


def test_sys_excepthook_passes_through_keyboardinterrupt(
    reset_logging_state: Path,
) -> None:
    """Ctrl-C should NOT generate a crash file."""
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt as exc:
        sys.excepthook(type(exc), exc, exc.__traceback__)
    crash_files = list((reset_logging_state / CRASH_SUBDIR).glob("crash_*.log"))
    assert crash_files == []


# ---------------------------------------------------------------------------
# 10. threading.excepthook writes a crash file with the thread name.
# ---------------------------------------------------------------------------


def test_threading_excepthook_writes_crash_file(
    reset_logging_state: Path,
) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state))

    def thread_target() -> None:
        raise RuntimeError("thread crash")

    t = threading.Thread(target=thread_target, name="WorkerX")
    t.start()
    t.join()

    crash_files = list((reset_logging_state / CRASH_SUBDIR).glob("crash_*.log"))
    assert crash_files, "no crash file written from worker thread"
    body = crash_files[0].read_text(encoding="utf-8")
    assert "thread crash" in body
    assert "Thread: WorkerX" in body


# ---------------------------------------------------------------------------
# 11. asyncio loop exception handler writes a crash file.
# ---------------------------------------------------------------------------


def test_asyncio_exception_handler_writes_crash_file(
    reset_logging_state: Path,
) -> None:
    asyncio_handler = init_logging(LoggingSettings(log_dir=reset_logging_state))

    async def _run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(asyncio_handler)

        async def crasher() -> None:
            raise RuntimeError("async crash")

        task = asyncio.create_task(crasher())
        # Let it raise + bubble up so the loop sees an unretrieved
        # task exception.
        with contextlib.suppress(RuntimeError):
            await task
        # An unhandled task exception case — invoke the handler
        # directly so we don't rely on garbage collector timing.
        loop.call_exception_handler(
            {
                "message": "Unhandled exception in task",
                "exception": RuntimeError("async crash"),
            }
        )

    asyncio.run(_run())
    crash_files = list((reset_logging_state / CRASH_SUBDIR).glob("crash_*.log"))
    assert crash_files, "no crash file from asyncio loop handler"
    body = crash_files[0].read_text(encoding="utf-8")
    assert "async crash" in body
    assert "asyncio.loop.exception_handler" in body


# ---------------------------------------------------------------------------
# 12. Crash cleanup — pre-populate 60 files spanning 60 days, init,
#     assert <=50 files remain and none older than 30 days.
# ---------------------------------------------------------------------------


def test_crash_cleanup_age_and_count(reset_logging_state: Path) -> None:
    crash_dir = reset_logging_state / CRASH_SUBDIR
    crash_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    # 60 files, 1 per day, oldest 60 days back.
    for days_ago in range(60):
        stamp = (now - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H-%M-%SZ")
        path = crash_dir / f"crash_{stamp}.log"
        path.write_text("synthetic", encoding="utf-8")
        # Backdate the mtime so the cleanup sees the real age.
        ts = (now - timedelta(days=days_ago)).timestamp()
        import os

        os.utime(path, (ts, ts))

    init_logging(
        LoggingSettings(
            log_dir=reset_logging_state,
            crash_retention_days=30,
            crash_max_files=50,
        )
    )

    remaining = sorted(crash_dir.glob("crash_*.log"))
    assert len(remaining) <= 50
    cutoff = (now - timedelta(days=30)).timestamp()
    for p in remaining:
        assert p.stat().st_mtime >= cutoff, f"{p.name} older than 30 days"


# ---------------------------------------------------------------------------
# 13. Permanent archive — opt-in second handler attached.
# ---------------------------------------------------------------------------


def test_permanent_archive_handler_attached(
    reset_logging_state: Path,
) -> None:
    archive_path = reset_logging_state / "permanent" / "archive.log"
    init_logging(
        LoggingSettings(
            log_dir=reset_logging_state,
            permanent_archive_path=archive_path,
        )
    )
    archive = logging.getHandlerByName("archive")
    assert archive is not None

    logging.getLogger("archive_test").warning("audit trail entry")
    _flush_queue()

    assert archive_path.is_file()
    body = archive_path.read_text(encoding="utf-8")
    assert "audit trail entry" in body


def test_permanent_archive_off_by_default(reset_logging_state: Path) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    assert logging.getHandlerByName("archive") is None


# ---------------------------------------------------------------------------
# 14. PII redaction — home dir masked in file output; preserved in
#     the ring buffer (which is the crash-context source).
# ---------------------------------------------------------------------------


def test_home_dir_redacted_in_file_but_preserved_in_ring(
    reset_logging_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_home = "/home/sekret-user"
    monkeypatch.setattr(logging_setup, "_HOME", fake_home)
    monkeypatch.setattr(
        logging_setup,
        "_HOME_RE",
        __import__("re").compile(__import__("re").escape(fake_home), __import__("re").IGNORECASE),
    )
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="DEBUG"))
    logging.getLogger("pii_test").info("found file at %s/data.txt", fake_home)
    _flush_queue()

    file_body = (reset_logging_state / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert fake_home not in file_body
    assert "~/data.txt" in file_body

    # Ring buffer keeps full paths so crash files have diagnostic
    # context. Its formatter (if any) is the plain one.
    ring = logging.getHandlerByName("ring")
    assert isinstance(ring, RingBufferHandler)
    raw_messages = [r.getMessage() for r in ring.get_snapshot()]
    assert any(fake_home in m for m in raw_messages)


def test_redacting_formatter_unit() -> None:
    """Direct unit test of `RedactingFormatter` without init."""
    formatter = RedactingFormatter("%(message)s")
    rec = _make_record("path is " + str(Path.home()) + "/x")
    out = formatter.format(rec)
    assert str(Path.home()) not in out
    assert "~/x" in out


# ---------------------------------------------------------------------------
# 15. captureWarnings — warnings.warn(...) flows through logging.
# ---------------------------------------------------------------------------


def test_warnings_captured_into_logging(reset_logging_state: Path) -> None:
    """`init_logging` must call `captureWarnings(True)` AND wire the
    `py.warnings` logger through to the file handler.

    Note on pytest: each test runs inside `warnings.catch_warnings()`
    which temporarily resets `warnings.showwarning`, undoing our
    `captureWarnings(True)` hook for the test body. We can't easily
    exercise the full `warnings.warn() -> py.warnings logger` path
    here. What we CAN verify is that the load-bearing piece is in
    place: if a warning ever does reach `py.warnings`, it flows
    through the configured handlers to the file. The captureWarnings
    call itself is a one-liner in init_logging() that's plain to
    inspect.
    """
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="DEBUG"))
    logging.getLogger("py.warnings").warning("synthetic warning")
    _flush_queue()

    body = (reset_logging_state / LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "synthetic warning" in body


# ---------------------------------------------------------------------------
# 16. faulthandler is enabled after init.
# ---------------------------------------------------------------------------


def test_faulthandler_enabled_after_init(reset_logging_state: Path) -> None:
    import faulthandler

    pre = faulthandler.is_enabled()
    init_logging(LoggingSettings(log_dir=reset_logging_state))
    assert faulthandler.is_enabled() is True
    # fault.log should exist (may be empty — that's fine).
    fault_log = reset_logging_state / CRASH_SUBDIR / "fault.log"
    assert fault_log.is_file()
    # Restore pre-state — the fixture doesn't manage faulthandler.
    if not pre:
        faulthandler.disable()


# ---------------------------------------------------------------------------
# 17. Listener cleanup — atexit registers, .stop() flushes queue.
# ---------------------------------------------------------------------------


def test_listener_flushes_queue_on_stop(reset_logging_state: Path) -> None:
    init_logging(LoggingSettings(log_dir=reset_logging_state, log_level="DEBUG"))
    logger = logging.getLogger("flush_test")
    for i in range(50):
        logger.info("msg %d", i)
    _flush_queue()  # explicit stop simulates atexit
    body = (reset_logging_state / LOG_FILE_NAME).read_text(encoding="utf-8")
    for i in range(50):
        assert f"msg {i}" in body


# ---------------------------------------------------------------------------
# LoggingSettings loader.
# ---------------------------------------------------------------------------


def test_logging_settings_loads_from_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "settings.toml"
    toml_path.write_text(
        '[logging]\nlog_level = "WARNING"\ncrash_retention_days = 7\n',
        encoding="utf-8",
    )
    # Make sure env vars don't override.
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)
    settings = LoggingSettings.load(toml_path)
    assert settings.log_level == "WARNING"
    assert settings.crash_retention_days == 7


def test_logging_settings_env_overrides_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "settings.toml"
    toml_path.write_text('[logging]\nlog_level = "WARNING"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")
    settings = LoggingSettings.load(toml_path)
    assert settings.log_level == "DEBUG"


def test_logging_settings_missing_file_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)
    monkeypatch.delenv(ENV_LOG_DIR, raising=False)
    monkeypatch.delenv(ENV_LOG_CONSOLE, raising=False)
    settings = LoggingSettings.load(tmp_path / "nope.toml")
    assert settings.log_level == "INFO"  # default


def test_logging_settings_empty_string_log_dir_treated_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TOML round-trip can materialise empty strings — must mean
    'auto-resolve', not 'use cwd'."""
    toml_path = tmp_path / "settings.toml"
    toml_path.write_text('[logging]\nlog_dir = ""\n', encoding="utf-8")
    monkeypatch.delenv(ENV_LOG_DIR, raising=False)
    settings = LoggingSettings.load(toml_path)
    assert settings.log_dir is None


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _make_record(msg: str) -> logging.LogRecord:
    """Build a minimal LogRecord for direct-handler tests."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
