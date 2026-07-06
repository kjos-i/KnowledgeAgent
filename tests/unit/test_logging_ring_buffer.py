"""Unit tests for `logging_ring_buffer.RingBufferHandler` — the bounded ring
+ pub/sub. No global state, no I/O.
"""

from __future__ import annotations

import logging

from knowledge_agent.logging_ring_buffer import RingBufferHandler


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)


def test_emit_and_snapshot_oldest_first():
    h = RingBufferHandler(maxlen=10)
    h.emit(_record("a"))
    h.emit(_record("b"))
    assert [r.getMessage() for r in h.get_snapshot()] == ["a", "b"]


def test_ring_drops_oldest_past_maxlen():
    h = RingBufferHandler(maxlen=2)
    for m in ("a", "b", "c"):
        h.emit(_record(m))
    assert [r.getMessage() for r in h.get_snapshot()] == ["b", "c"]


def test_subscribe_receives_records_until_unsubscribed():
    h = RingBufferHandler()
    seen: list[str] = []
    unsub = h.subscribe(lambda r: seen.append(r.getMessage()))
    h.emit(_record("a"))
    unsub()
    h.emit(_record("b"))
    assert seen == ["a"]  # 'b' arrives after unsubscribe


def test_unsubscribe_twice_is_noop():
    h = RingBufferHandler()
    unsub = h.subscribe(lambda r: None)
    unsub()
    unsub()  # must not raise


def test_broken_subscriber_does_not_break_emit():
    h = RingBufferHandler()

    def boom(_r: logging.LogRecord) -> None:
        raise RuntimeError("subscriber blew up")

    seen: list[str] = []
    h.subscribe(boom)
    h.subscribe(lambda r: seen.append(r.getMessage()))
    h.emit(_record("a"))  # must not raise despite the broken subscriber
    assert seen == ["a"]  # the good subscriber still fired
    assert len(h.get_snapshot()) == 1  # record still stored
