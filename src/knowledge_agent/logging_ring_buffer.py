"""In-memory bounded ring buffer for log records, with pub/sub.

Single concern: the `RingBufferHandler` class. Lives alongside
`logging_setup.py` (which builds the dictConfig) and
`logging_crash.py` (which dumps the ring tail into crash files).

Use cases (both lean on the subscribe API):

  - **Crash file context**: the crash handlers in `logging_crash.py`
    call `get_snapshot()` to grab the last N records and embed them
    in the crash report.
  - **Flet "Recent Activity" panel** (future GUI): the panel
    `subscribe()`s a callback per emit; the consumer marshals to the
    UI thread via Flet's `page.update` / `run_thread_safe` (same
    pattern as the ResearchArticlesAgent bulk-op progress callback).

ALWAYS configured at DEBUG so crash files include verbose context
even when the file handler runs at INFO.

Re-exported from `logging_setup` for backward compatibility — tests
+ dictConfig strings still see it at `knowledge_agent.logging_setup.
RingBufferHandler`. New code should import directly from here.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from logging import LogRecord


RING_BUFFER_SIZE = 1000
"""Default ring capacity (records). Tuned to keep crash files
informative without holding excessive memory — 1000 records at the
project's typical message length is well under 1 MB."""


class RingBufferHandler(logging.Handler):
    """In-memory bounded ring of `LogRecord`s with pub/sub.

    Thread-safe:
      - `deque.append` is atomic in CPython.
      - `list(deque)` for snapshots is atomic.
      - Subscriber list mutations are lock-protected; iteration uses a
        copy so a callback can unsubscribe itself without races.

    Subscriber callbacks fire from `emit()` on whatever thread emitted
    the log. Callers are responsible for marshaling to the UI thread.
    A callback that raises does NOT crash the handler (its exception
    is swallowed silently — logging must not break the program).
    """

    def __init__(self, maxlen: int = RING_BUFFER_SIZE, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._records: deque[LogRecord] = deque(maxlen=maxlen)
        self._subscribers: list[Callable[[LogRecord], None]] = []
        self._lock = threading.Lock()

    def emit(self, record: LogRecord) -> None:
        self._records.append(record)
        with self._lock:
            subs = list(self._subscribers)
        for callback in subs:
            try:
                callback(record)
            except Exception:
                # A broken subscriber must not break logging.
                pass

    def get_snapshot(self) -> list[LogRecord]:
        """Return a copy of the current ring contents (oldest first)."""
        return list(self._records)

    def subscribe(
        self, callback: Callable[[LogRecord], None]
    ) -> Callable[[], None]:
        """Register a callback fired on every new record.

        Returns an `unsubscribe` zero-arg function. Calling it twice is
        a no-op.
        """
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe
