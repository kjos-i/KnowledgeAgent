"""Crash handling: sys / threading / asyncio exception hooks +
crash-file writes + faulthandler + crash-file retention.

Single concern: capturing uncaught exceptions across every Python
context (main thread, worker threads, async callbacks, native code)
and writing one self-contained crash file per incident with full
diagnostic context.

The companion module is `logging_setup.py` (which builds the
dictConfig and starts the QueueListener) and `logging_ring_buffer.py`
(which holds the in-memory log tail that gets embedded in each crash
file). This module depends on both but neither depends on it — the
crash handlers are wired by `init_logging()` AFTER the ring buffer
+ file handlers are running.

Crash file layout (`crash_<UTC>.log`):

  === Crash report ===
  Timestamp, source, thread, app version, platform, Python, exe

  === Traceback ===
  Full Python traceback

  === Environment (allowlisted) ===
  Subset of os.environ — see ALLOWLISTED_ENV_KEYS. NEVER includes
  secrets (passwords, API keys).

  === Torch / hardware ===
  Torch version + CUDA availability if torch is already loaded.

  === Installed packages ===
  pip freeze equivalent for the running interpreter.

  === Last N log records ===
  Tail from the in-memory ring buffer — full DEBUG verbosity.

faulthandler is enabled separately; native crashes (PyTorch /
neo4j-driver / lancedb segfaults) land in `crashes/fault.log`.

Retention policy: cleaned at startup by `cleanup_old_crashes`.
Two limits stack (both must be satisfied — older OR newer):
  - age > `crash_retention_days` → delete
  - count > `crash_max_files` (newest-first) → delete extras

A crash loop fills the folder faster than age cleanup catches up, so
both limits are needed.
"""

from __future__ import annotations

import contextlib
import faulthandler
import importlib.metadata
import json
import logging
import os
import platform
import sys
import threading
import traceback
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from knowledge_agent import DISTRIBUTION_NAME

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from knowledge_agent.logging_ring_buffer import RingBufferHandler

# ---------------------------------------------------------------------------
# Format constants shared with `logging_setup` (kept in sync — the
# ring-tail dump in crash files uses the same format as the live
# handlers so a developer can read them interchangeably).
# ---------------------------------------------------------------------------


LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


FAULT_LOG_NAME = "fault.log"
"""Native-crash log written by `faulthandler`. Sits alongside the
per-incident `crash_*.log` files in the crashes subdir."""


ALLOWLISTED_ENV_KEYS: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "LANG",
    "LC_ALL",
    "KAGENT_LOG_DIR",
    "KAGENT_LOG_LEVEL",
    "KAGENT_LOG_CONSOLE",
    "KAGENT_PACKAGED",
    "NEO4J_URI",
    "NEO4J_USER",
    "LANCEDB_PATH",
    "OS",
    "PROCESSOR_ARCHITECTURE",
)
"""Keys included in crash-file environment dumps. NEVER include
secrets (NEO4J_PASSWORD, ANTHROPIC_API_KEY, VOYAGE_API_KEY, etc.).
If you need a new key here, add it explicitly — the allowlist is
the security boundary."""


# ---------------------------------------------------------------------------
# Crash file retention.
# ---------------------------------------------------------------------------


def cleanup_old_crashes(crash_dir: Path, retention_days: int, max_files: int) -> None:
    """Delete crash files older than `retention_days` AND keep only
    the newest `max_files`. Either limit alone is insufficient — a
    crash loop fills the folder faster than age-cleanup catches up.

    Idempotent + best-effort: I/O errors are logged at WARNING and
    don't propagate (logging init must not crash startup).
    """
    if not crash_dir.is_dir():
        return
    now = datetime.now(UTC).timestamp()
    cutoff = now - retention_days * 86400
    crash_files = sorted(
        (p for p in crash_dir.glob("crash_*.log") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in crash_files[max_files:]:
        with contextlib.suppress(OSError):
            path.unlink()
    for path in crash_files[:max_files]:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# System-info capture for crash file headers.
# ---------------------------------------------------------------------------


def capture_system_info() -> dict[str, Any]:
    """Build the system-info dict that appears in every crash file.

    Includes app version, Python version, platform, allowlisted env
    vars, installed package list, and torch device info if torch is
    already loaded (no eager import).
    """
    try:
        app_version = importlib.metadata.version(DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        app_version = "dev"

    env_subset = {k: os.environ[k] for k in ALLOWLISTED_ENV_KEYS if k in os.environ}

    packages: list[str] = []
    try:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name") or "unknown"
            version = dist.version
            packages.append(f"{name}=={version}")
    except Exception:
        pass
    packages.sort()

    torch_info: dict[str, Any] = {}
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            torch_info = {
                "version": getattr(torch_mod, "__version__", "unknown"),
                "cuda_available": bool(torch_mod.cuda.is_available()),
                "cuda_device_count": int(torch_mod.cuda.device_count()),
                "cuda_device_name": (
                    torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else None
                ),
            }
        except Exception:
            torch_info = {"version": "introspection_failed"}

    return {
        "app_version": app_version,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "environment": env_subset,
        "packages": packages,
        "torch": torch_info,
    }


def write_crash_file(
    crash_dir: Path,
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: Any,
    ring: RingBufferHandler | None,
    thread_name: str = "main",
    source: str = "sys.excepthook",
) -> Path | None:
    """Write one crash file with traceback + system info + ring tail.

    Returns the written path, or None on I/O failure (caller logs a
    breadcrumb but the program continues to die / unwind normally).
    """
    try:
        crash_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        stamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
        path = crash_dir / f"crash_{stamp}.log"

        sysinfo = capture_system_info()
        tb_lines = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        ring_lines: list[str] = []
        if ring is not None:
            plain_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
            for rec in ring.get_snapshot():
                with contextlib.suppress(Exception):
                    ring_lines.append(plain_formatter.format(rec))

        with path.open("w", encoding="utf-8") as fh:
            fh.write("=== Crash report ===\n")
            fh.write(f"Timestamp: {now.isoformat()}\n")
            fh.write(f"Source: {source}\n")
            fh.write(f"Thread: {thread_name}\n")
            fh.write(f"App version: {sysinfo['app_version']}\n")
            fh.write(f"Platform: {sysinfo['platform']}\n")
            fh.write(f"Python: {sysinfo['python']}\n")
            fh.write(f"Executable: {sysinfo['executable']}\n")
            fh.write("\n=== Traceback ===\n")
            fh.write(tb_lines)
            fh.write("\n=== Environment (allowlisted) ===\n")
            for k, v in sorted(sysinfo["environment"].items()):
                fh.write(f"{k}={v}\n")
            fh.write("\n=== Torch / hardware ===\n")
            fh.write(json.dumps(sysinfo["torch"], indent=2, default=str))
            fh.write("\n\n=== Installed packages ===\n")
            for pkg in sysinfo["packages"]:
                fh.write(f"{pkg}\n")
            fh.write(f"\n=== Last {len(ring_lines)} log records ===\n")
            for line in ring_lines:
                fh.write(line)
                fh.write("\n")
        return path
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Hook factories.
# ---------------------------------------------------------------------------


def make_sys_excepthook(crash_dir: Path, ring: RingBufferHandler | None) -> Callable[..., None]:
    """Return a `sys.excepthook` that writes a crash file then chains
    to the original Python hook so the interpreter still prints the
    traceback to stderr (visible in dev; harmless when packaged)."""
    original = sys.excepthook

    def hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: Any,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            original(exc_type, exc_value, exc_traceback)
            return
        path = write_crash_file(
            crash_dir,
            exc_type,
            exc_value,
            exc_traceback,
            ring,
            thread_name=threading.current_thread().name,
            source="sys.excepthook",
        )
        if path is not None:
            logging.getLogger(__name__).critical("Crash report written to %s", path)
        original(exc_type, exc_value, exc_traceback)

    return hook


def make_threading_excepthook(
    crash_dir: Path, ring: RingBufferHandler | None
) -> Callable[[threading.ExceptHookArgs], None]:
    """Return a `threading.excepthook` for worker-thread crashes
    (e.g. Flet button handlers that spawn threads)."""

    def hook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, SystemExit):
            return
        path = write_crash_file(
            crash_dir,
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
            ring,
            thread_name=args.thread.name if args.thread else "unknown",
            source="threading.excepthook",
        )
        if path is not None:
            logging.getLogger(__name__).critical("Crash report written to %s", path)

    return hook


def make_asyncio_exception_handler(
    crash_dir: Path, ring: RingBufferHandler | None
) -> Callable[[Any, dict[str, Any]], None]:
    """Return a handler for `loop.set_exception_handler(...)`.

    Caller (Flet entry point or async smoke) must wire this onto the
    running loop — `logging_setup`'s docstring shows the exact
    pattern.
    """

    def handler(loop: Any, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        message = context.get("message", "asyncio unhandled exception")
        if exc is None:
            logging.getLogger(__name__).error("asyncio: %s", message)
            return
        path = write_crash_file(
            crash_dir,
            type(exc),
            exc,
            exc.__traceback__,
            ring,
            thread_name=threading.current_thread().name,
            source="asyncio.loop.exception_handler",
        )
        if path is not None:
            logging.getLogger(__name__).critical(
                "Crash report written to %s (asyncio: %s)", path, message
            )

    return handler


# ---------------------------------------------------------------------------
# faulthandler — native crashes.
# ---------------------------------------------------------------------------


# Strong reference to the file object the faulthandler writes into.
# Python doesn't keep the wrapper alive on its own (faulthandler holds
# the OS fd, not the Python file object), so without this reference
# the file gets garbage-collected and writes silently fail.
_FAULT_FILE: Any = None


def enable_faulthandler(crash_dir: Path) -> None:
    """Open the fault log and install `faulthandler` to write into it.

    Best-effort: an OSError opening the file is swallowed (the rest
    of the logging system still works). Re-calling is safe — the
    previous file handle is dropped + replaced.
    """
    global _FAULT_FILE
    fault_log_path = crash_dir / FAULT_LOG_NAME
    try:
        _FAULT_FILE = fault_log_path.open("a", encoding="utf-8")
        faulthandler.enable(file=_FAULT_FILE)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public install entry point — called by `init_logging()` after the
# ring buffer + file handlers are already running.
# ---------------------------------------------------------------------------


def install_crash_handlers(
    crash_dir: Path,
    ring: RingBufferHandler | None,
    *,
    retention_days: int,
    max_files: int,
) -> Callable[[Any, dict[str, Any]], None]:
    """Install all four crash hooks (sys / threading / asyncio /
    faulthandler) + run startup retention cleanup.

    Returns the asyncio exception handler. The caller MUST install it
    onto the running event loop with
    `loop.set_exception_handler(handler)` — asyncio binds handlers to
    specific loop instances, and the loop usually doesn't exist yet at
    the time logging is initialised. `logging_setup.init_logging`'s
    docstring shows the pattern.
    """
    cleanup_old_crashes(crash_dir, retention_days, max_files)

    sys.excepthook = make_sys_excepthook(crash_dir, ring)
    threading.excepthook = make_threading_excepthook(crash_dir, ring)
    asyncio_handler = make_asyncio_exception_handler(crash_dir, ring)
    enable_faulthandler(crash_dir)

    return asyncio_handler
