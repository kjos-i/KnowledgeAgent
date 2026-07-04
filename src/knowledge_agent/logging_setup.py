"""Application logging system — dictConfig + QueueListener + entry point.

Three-file split (refactor 2026-06-29):
  - `logging_setup.py` (THIS FILE) — dictConfig + LoggingSettings +
    RedactingFormatter + rotating gzip + the public `init_logging()`
    entry point that wires everything together.
  - `logging_crash.py` — sys / threading / asyncio crash hooks +
    crash-file writes + faulthandler.
  - `logging_ring_buffer.py` — `RingBufferHandler` (in-memory log
    tail with pub/sub) for crash files + the GUI's Recent Activity
    panel.

For backward compatibility, this module re-exports `RingBufferHandler`
+ `RedactingFormatter` so the dictConfig dotted-path references
(`knowledge_agent.logging_setup.RingBufferHandler`) keep working.

Single entry point: `init_logging()`. Call once at process start from
your entry point (Flet `main.py` or a smoke script) BEFORE
`ft.app(...)` / `asyncio.run(...)`. NOT from library modules — those
just do `logger = logging.getLogger(__name__)` and inherit the
configured root.

Architecture
------------
- `dictConfig`-driven setup (idempotent — safe to call twice).
- Emitters log through a `QueueHandler` so disk I/O never blocks the
  caller (critical for the async Flet GUI — a coroutine that logs at
  INFO must NOT do a synchronous file rotation on the event loop).
- A background `QueueListener` thread fans out from the queue to the
  real handlers: `RotatingFileHandler` (with UTF-8 + delay), an
  optional console handler (dev only), a `RingBufferHandler` for the
  GUI (subscribe API), and an optional permanent-archive `FileHandler`.
- Crash hooks (sys / threading / asyncio + faulthandler) installed by
  `logging_crash.install_crash_handlers()` after the listener is up.
- Path redaction via `RedactingFormatter` masks the user home in
  user-visible handlers; the ring buffer keeps full paths so crash
  files have complete diagnostic context.

Locations
---------
Resolved via `platformdirs.user_log_dir("KnowledgeAgent",
appauthor=False)`:
- Windows: `%LOCALAPPDATA%\\KnowledgeAgent\\Logs\\`
- macOS:   `~/Library/Logs/KnowledgeAgent/`
- Linux:   `~/.local/state/research-literature-agent/log/`

Override via `KAGENT_LOG_DIR` env var (typical dev use: point at
`./logs/` inside the repo).

Files in that folder:
- `kagent.log` (+ `.1..N` rotated, gzipped) — main rolling log.
- `crashes/crash_<UTC>.log` — one file per uncaught exception, with
  full system context.
- `crashes/fault.log` — native crash log written by `faulthandler`.

Packaged-detection
------------------
We detect packaged-vs-dev via a 4-layer check (`sys.frozen` only fires
for PyInstaller; `flet build` uses Serious Python which doesn't set
it). To make the detection reliable in packaged builds, set
`KAGENT_PACKAGED=1` in your packaging config (`pyproject.toml` ->
`[tool.flet.app]` env_vars, or PyInstaller spec env).

asyncio hook wiring (one explicit line in the entry point)
----------------------------------------------------------
`init_logging()` returns the asyncio exception handler function. You
MUST install it on the running loop yourself, because asyncio binds
exception handlers to a specific loop instance — and Flet creates the
loop inside `ft.app(...)`, so no loop exists at the time
`init_logging()` runs.

Wire it like this in your Flet entry point::

    async def main(page: ft.Page) -> None:
        asyncio.get_running_loop().set_exception_handler(asyncio_handler)
        # ... rest of UI setup

    if __name__ == "__main__":
        asyncio_handler = init_logging()
        ft.app(target=main)

For non-Flet entry points (smoke scripts that use `asyncio.run`),
install it inside the top-level coroutine via the same pattern.

Not wiring it means async-callback crashes will be logged at WARNING
to the asyncio logger and disappear — no crash file, no traceback
in your main log. The other three hooks (sys / threading /
faulthandler) still fire, but they don't catch coroutine exceptions.

Hot-reload of settings is NOT supported. Settings changes via the GUI
require restart; the GUI surfaces a notice when settings change.
"""

from __future__ import annotations

import atexit
import contextlib
import gzip
import logging
import logging.config
import logging.handlers
import os
import queue
import re
import sys
from logging import LogRecord
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import platformdirs
import tomlkit
from pydantic import BaseModel, Field, field_validator

# Re-exported for backward compatibility — dictConfig dotted-path
# strings + existing test imports + GUI code can still reach these
# at their original `knowledge_agent.logging_setup.X` paths.
from knowledge_agent.logging_crash import (
    ALLOWLISTED_ENV_KEYS,
    FAULT_LOG_NAME,
    capture_system_info,
    cleanup_old_crashes,
    enable_faulthandler,
    install_crash_handlers,
    make_asyncio_exception_handler,
    make_sys_excepthook,
    make_threading_excepthook,
    write_crash_file,
)
from knowledge_agent.logging_ring_buffer import (
    RING_BUFFER_SIZE,
    RingBufferHandler,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "ALLOWLISTED_ENV_KEYS",
    # Constants other modules read
    "APP_NAME",
    "CRASH_SUBDIR",
    "ENV_LOG_CONSOLE",
    "ENV_LOG_DIR",
    "ENV_LOG_LEVEL",
    "ENV_PACKAGED",
    "FAULT_LOG_NAME",
    "LIBRARY_NOISE_CLAMP",
    "LOG_DATEFMT",
    "LOG_FILE_NAME",
    "LOG_FORMAT",
    "RING_BUFFER_SIZE",
    "LoggingSettings",
    "RedactingFormatter",
    "RingBufferHandler",
    # Re-exports from logging_crash (test imports still work)
    "capture_system_info",
    "cleanup_old_crashes",
    "enable_faulthandler",
    # Public API
    "init_logging",
    "install_crash_handlers",
    "make_asyncio_exception_handler",
    "make_sys_excepthook",
    "make_threading_excepthook",
    "write_crash_file",
]


# ---------------------------------------------------------------------------
# Module-level constants (single source of truth — not user-facing knobs).
# ---------------------------------------------------------------------------

APP_NAME = "KnowledgeAgent"
"""Pascal-case name used by platformdirs for the OS-standard folder."""

LOG_FILE_NAME = "kagent.log"
CRASH_SUBDIR = "crashes"
SETTINGS_FILE_NAME = "settings.toml"

ROTATING_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
ROTATING_BACKUP_COUNT = 10  # ~30-50 MB on disk with gzip

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"

ENV_LOG_DIR = "KAGENT_LOG_DIR"
ENV_LOG_LEVEL = "LOG_LEVEL"
ENV_LOG_CONSOLE = "KAGENT_LOG_CONSOLE"
ENV_PACKAGED = "KAGENT_PACKAGED"

LIBRARY_NOISE_CLAMP: dict[str, str] = {
    "httpx": "WARNING",
    "urllib3": "WARNING",
    "neo4j": "WARNING",
    "docling": "WARNING",
    "langchain": "WARNING",
    "langchain_core": "WARNING",
    "transformers": "ERROR",
    "huggingface_hub": "WARNING",
    "filelock": "WARNING",
    "PIL": "WARNING",
    "lancedb": "WARNING",
    "asyncio": "WARNING",
}
"""Loggers known to be chatty in the ML / Flet stack. Set at init."""

_HOME = str(Path.home())
_HOME_RE = re.compile(re.escape(_HOME), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public settings model.
#
# Reads from `settings.toml` in the user data dir + env-var overrides.
# Hardcoded defaults fall through if neither is set. The future GUI
# Settings panel will write to the same `settings.toml`; logging code
# reads, doesn't care which side wrote it.
# ---------------------------------------------------------------------------


class LoggingSettings(BaseModel):
    """User-facing logging preferences.

    Layered resolution at `init_logging()` time:
      1. Hardcoded class defaults (this model).
      2. Values from `settings.toml` in the user data dir (or
         `KAGENT_SETTINGS_FILE` if set).
      3. Env vars (`KAGENT_LOG_DIR`, `LOG_LEVEL`, `KAGENT_LOG_CONSOLE`)
         override anything from steps 1-2.

    The GUI Settings panel (future) writes to `settings.toml` only —
    env vars remain a developer escape hatch.
    """

    log_dir: Path | None = Field(
        default=None,
        description=(
            "Override for the rotating log directory. None means resolve "
            "via `platformdirs.user_log_dir(APP_NAME, appauthor=False)`. "
            "Set via `KAGENT_LOG_DIR` env var in dev (e.g. ./logs) so log "
            "files land inside the repo for fast tailing."
        ),
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description=(
            "Root logger level. Default INFO in packaged builds; dev runs "
            "typically set `LOG_LEVEL=DEBUG`. The ring buffer ALWAYS "
            "captures DEBUG regardless, so crash files have full context."
        ),
    )
    console: bool | None = Field(
        default=None,
        description=(
            "Force-enable / force-disable the console handler. None means "
            "'auto': enabled when not packaged, disabled when packaged. "
            "Override via `KAGENT_LOG_CONSOLE=1` (or =0)."
        ),
    )
    crash_retention_days: int = Field(
        default=30,
        ge=1,
        description=(
            "Delete crash files older than this many days at startup. "
            "Default 30 — long enough to investigate sporadic crashes, "
            "short enough to keep the folder tidy."
        ),
    )
    crash_max_files: int = Field(
        default=50,
        ge=1,
        description=(
            "Hard cap on crash files regardless of age — protects against "
            "a crash-loop filling the disk. Oldest files deleted first."
        ),
    )
    permanent_archive_path: Path | None = Field(
        default=None,
        description=(
            "Optional second log destination that NEVER rotates. Useful "
            "for an audit trail. The user is responsible for cleanup; the "
            "rotating logs continue to cap at ~50 MB at their standard "
            "location regardless of this setting."
        ),
    )

    @field_validator("log_dir", "permanent_archive_path", mode="before")
    @classmethod
    def _coerce_empty_to_none(cls, value: Any) -> Any:
        """Treat empty string from TOML / env var as None.

        TOML round-trips have a tendency to materialize `log_dir = ""`
        when the GUI clears a field; that should mean 'auto-resolve',
        not 'use the cwd'.
        """
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @classmethod
    def load(cls, settings_path: Path | None = None) -> LoggingSettings:
        """Resolve settings from TOML + env-var overrides.

        `settings_path` is the explicit TOML file to read. If None,
        resolves to `<user_data_dir>/settings.toml`. Missing file is
        not an error — defaults apply.
        """
        toml_values: dict[str, Any] = {}
        if settings_path is None:
            settings_path = (
                Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / SETTINGS_FILE_NAME
            )
        if settings_path.is_file():
            try:
                with settings_path.open("rb") as fh:
                    doc = tomlkit.load(fh)
                section = doc.get("logging", {})
                if section:
                    toml_values = dict(section)
            except (OSError, ValueError):
                # Bad TOML shouldn't break logging init — defaults apply.
                pass

        env_overrides: dict[str, Any] = {}
        if env_log_dir := os.environ.get(ENV_LOG_DIR):
            env_overrides["log_dir"] = env_log_dir
        if env_log_level := os.environ.get(ENV_LOG_LEVEL):
            env_overrides["log_level"] = env_log_level.upper()
        if (env_console := os.environ.get(ENV_LOG_CONSOLE)) is not None:
            env_overrides["console"] = env_console == "1"

        return cls(**{**toml_values, **env_overrides})


# ---------------------------------------------------------------------------
# Path-redacting formatter.
#
# Per-handler redaction: file / console / archive get masked paths;
# the ring buffer keeps full paths so crash files are diagnostic.
# ---------------------------------------------------------------------------


class RedactingFormatter(logging.Formatter):
    """Standard formatter that masks the user home dir to `~`.

    Applied to handlers whose output is human-facing in user-shareable
    contexts (file, console, archive). The ring buffer uses the plain
    formatter so crash files keep full paths for the developer.
    """

    def format(self, record: LogRecord) -> str:
        msg = super().format(record)
        return _HOME_RE.sub("~", msg)


# ---------------------------------------------------------------------------
# Packaged-vs-dev detection.
#
# `sys.frozen` only fires for PyInstaller / cx_Freeze. `flet build`
# uses Serious Python which embeds CPython inside a Flutter app
# without setting `sys.frozen`. We layer four signals so detection is
# robust across both packagers + future ones.
# ---------------------------------------------------------------------------


def _detect_packaged() -> bool:
    """True if we're running inside a packaged installer.

    Any of these signals counts as packaged:
      1. `sys.frozen` set (PyInstaller, cx_Freeze, py2exe).
      2. `KAGENT_PACKAGED=1` env var (the build process sets this in
         `pyproject.toml [tool.flet.app] env_vars`).
      3. `sys.stdout` or `sys.stderr` is None (windowed GUI process).
      4. `sys.executable` name doesn't look like a python interpreter
         (last-resort heuristic for Serious Python builds with stdio
         still wired through).
    """
    if getattr(sys, "frozen", False):
        return True
    if os.environ.get(ENV_PACKAGED) == "1":
        return True
    if sys.stdout is None or sys.stderr is None:
        return True
    exe_stem = Path(sys.executable).stem.lower()
    return bool(exe_stem and not exe_stem.startswith("python") and not exe_stem.startswith("py"))


# ---------------------------------------------------------------------------
# Log directory resolution + first-run dir creation.
# ---------------------------------------------------------------------------


def _resolve_log_dir(settings: LoggingSettings) -> Path:
    """Resolve the rotating log directory.

    Order: explicit `settings.log_dir` -> platformdirs OS standard.
    """
    if settings.log_dir is not None:
        return Path(settings.log_dir).expanduser().resolve()
    return Path(platformdirs.user_log_dir(APP_NAME, appauthor=False)).expanduser().resolve()


# ---------------------------------------------------------------------------
# Rotated-file gzip helper.
#
# RotatingFileHandler's `namer` + `rotator` pair lets us gzip backups:
# active log stays plain text (so live tailing works), but rolled
# files become `kagent.log.1.gz`, `.2.gz`, etc. ~10x effective history
# per MB on disk.
# ---------------------------------------------------------------------------


def _gzip_namer(default_name: str) -> str:
    return default_name + ".gz"


def _gzip_rotator(source: str, dest: str) -> None:
    try:
        with open(source, "rb") as sf, gzip.open(dest, "wb") as df:
            df.writelines(sf)
        os.remove(source)
    except OSError:
        # If gzip fails, fall back to plain move so we don't lose data.
        with contextlib.suppress(OSError):
            os.replace(source, dest.removesuffix(".gz"))


# ---------------------------------------------------------------------------
# dictConfig builder.
# ---------------------------------------------------------------------------

# Module-level queue used by the QueueHandler. Recreated per init so
# stale state from a previous test run doesn't bleed in.
_LOG_QUEUE: queue.Queue[Any] = queue.Queue(-1)
_LISTENER: logging.handlers.QueueListener | None = None
_INITIALIZED: bool = False


def _build_log_config(
    settings: LoggingSettings, log_dir: Path, console_enabled: bool
) -> dict[str, Any]:
    """Construct the `dictConfig` dict.

    Only the QueueHandler is attached to the root logger — the real
    handlers (file, console, ring, archive) are defined here so they
    get instantiated by dictConfig, but they're attached to the
    QueueListener by `init_logging()` (not to any logger directly).
    Look them up via `logging.getHandlerByName(...)` after dictConfig
    runs.
    """
    handlers: dict[str, dict[str, Any]] = {
        "queue": {
            "class": "logging.handlers.QueueHandler",
            "queue": _LOG_QUEUE,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_dir / LOG_FILE_NAME),
            "maxBytes": ROTATING_MAX_BYTES,
            "backupCount": ROTATING_BACKUP_COUNT,
            "encoding": "utf-8",
            "delay": True,
            "formatter": "redacting",
            "level": settings.log_level,
        },
        "ring": {
            "()": "knowledge_agent.logging_ring_buffer.RingBufferHandler",
            "maxlen": RING_BUFFER_SIZE,
            "level": logging.DEBUG,
            "formatter": "standard",
        },
    }
    if console_enabled:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "redacting",
            "level": settings.log_level,
        }
    if settings.permanent_archive_path is not None:
        archive_path = Path(settings.permanent_archive_path).expanduser().resolve()
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        handlers["archive"] = {
            "class": "logging.FileHandler",
            "filename": str(archive_path),
            "encoding": "utf-8",
            "delay": True,
            "formatter": "redacting",
            "level": settings.log_level,
        }

    loggers: dict[str, dict[str, Any]] = {
        name: {"level": level} for name, level in LIBRARY_NOISE_CLAMP.items()
    }

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": LOG_FORMAT,
                "datefmt": LOG_DATEFMT,
            },
            "redacting": {
                "()": "knowledge_agent.logging_setup.RedactingFormatter",
                "format": LOG_FORMAT,
                "datefmt": LOG_DATEFMT,
            },
        },
        "handlers": handlers,
        "loggers": loggers,
        "root": {
            "level": settings.log_level,
            "handlers": ["queue"],
        },
    }


# ---------------------------------------------------------------------------
# Public init entry point.
# ---------------------------------------------------------------------------


def init_logging(
    settings: LoggingSettings | None = None,
    settings_path: Path | None = None,
) -> Callable[[Any, dict[str, Any]], None]:
    """Initialise the application logging system.

    Wires up: dictConfig, QueueListener fan-out, file/console/ring
    handlers, library noise clamp, captureWarnings, faulthandler,
    sys.excepthook, threading.excepthook, and crash file cleanup.

    Call this ONCE at process start, from your entry point (Flet
    `main.py` or any smoke script) — BEFORE `ft.app(...)` or
    `asyncio.run(...)`.

    Idempotent: a second call returns the cached asyncio handler
    without rebuilding the config (no duplicate handlers, no second
    QueueListener thread).

    Args:
        settings: Pre-built settings. If None, resolves via
            `LoggingSettings.load(settings_path)`.
        settings_path: Path to `settings.toml`. If None, defaults to
            `<user_data_dir>/settings.toml`.

    Returns:
        The asyncio exception handler function. You MUST install it
        on the running loop yourself — asyncio binds exception
        handlers to a specific loop instance, and Flet creates the
        loop inside `ft.app(...)`, so no loop exists at the time
        this function runs.

        Wire it like this in the Flet entry point::

            async def main(page: ft.Page) -> None:
                asyncio.get_running_loop().set_exception_handler(asyncio_handler)
                # ... rest of UI setup

            if __name__ == "__main__":
                asyncio_handler = init_logging()
                ft.app(target=main)

        For non-Flet async entry points (smoke scripts using
        `asyncio.run`), install it inside the top-level coroutine via
        the same pattern. Sync smoke scripts can ignore the return
        value — the other three hooks (sys / threading / faulthandler)
        still fire.

        Not wiring it means async-callback crashes will be logged at
        WARNING to the asyncio logger and disappear — no crash file,
        no traceback in your main log.
    """
    global _LISTENER, _INITIALIZED, _LOG_QUEUE

    if settings is None:
        settings = LoggingSettings.load(settings_path)

    is_packaged = _detect_packaged()
    console_enabled = settings.console if settings.console is not None else not is_packaged

    log_dir = _resolve_log_dir(settings)
    log_dir.mkdir(parents=True, exist_ok=True)
    crash_dir = log_dir / CRASH_SUBDIR
    crash_dir.mkdir(parents=True, exist_ok=True)

    # Quiet the noisiest ML envs at the library level too. setdefault
    # so an explicit user override wins.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # If we're re-initialising, tear down the previous listener so we
    # don't accumulate threads. dictConfig itself is safe to re-call
    # with `disable_existing_loggers: False`.
    if _LISTENER is not None:
        with contextlib.suppress(Exception):
            _LISTENER.stop()
        _LISTENER = None
    # Reset the queue so stale records from a prior init don't replay.
    _LOG_QUEUE = queue.Queue(-1)

    config = _build_log_config(settings, log_dir, console_enabled)
    # Refresh the queue reference now that we reset it.
    config["handlers"]["queue"]["queue"] = _LOG_QUEUE
    logging.config.dictConfig(config)

    file_h = logging.getHandlerByName("file")
    ring_h = logging.getHandlerByName("ring")
    console_h = logging.getHandlerByName("console") if console_enabled else None
    archive_h = logging.getHandlerByName("archive")

    # Apply gzip rotation namer to the file handler.
    if isinstance(file_h, logging.handlers.RotatingFileHandler):
        file_h.namer = _gzip_namer
        file_h.rotator = _gzip_rotator

    real_handlers: list[logging.Handler] = [
        h for h in (file_h, console_h, ring_h, archive_h) if isinstance(h, logging.Handler)
    ]

    _LISTENER = logging.handlers.QueueListener(
        _LOG_QUEUE, *real_handlers, respect_handler_level=True
    )
    _LISTENER.start()
    if not _INITIALIZED:
        atexit.register(_shutdown_listener)

    logging.captureWarnings(True)

    ring_for_hooks = ring_h if isinstance(ring_h, RingBufferHandler) else None
    asyncio_handler = install_crash_handlers(
        crash_dir,
        ring_for_hooks,
        retention_days=settings.crash_retention_days,
        max_files=settings.crash_max_files,
    )

    if not is_packaged:
        # Dev-only debugging aids. Cheap to leave on; very useful when
        # an async leak or memory regression appears.
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        os.environ.setdefault("PYTHONASYNCIODEBUG", "1")

    _INITIALIZED = True
    logging.getLogger(__name__).info("Logs: %s", log_dir)
    return asyncio_handler


def _shutdown_listener() -> None:
    """`atexit` callback — stop the QueueListener cleanly so any
    queued records are flushed before the process dies."""
    global _LISTENER
    if _LISTENER is not None:
        with contextlib.suppress(Exception):
            _LISTENER.stop()
        _LISTENER = None
