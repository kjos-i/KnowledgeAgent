"""Tests for the Library → Ingest tab's action wiring (`IngestTab`).

Cover the pure result-formatters (no construction) plus the busy/loop
helpers. The picker + plan → execute paths are async and guarded off
without a running loop, so unit tests never touch the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

from knowledge_agent.gui.library.ingest import IngestTab


@dataclass(frozen=True)
class _FakeBulkResult:
    n_resolved: int
    n_no_work: int
    n_failed: int
    failures: tuple


# ---- result formatters (static) ----


def test_fmt_ingest_result_clean():
    r = SimpleNamespace(n_succeeded=3, n_failed=0, failures=())
    msg = IngestTab._fmt_ingest_result("Ingest folder", r)
    assert "3 succeeded" in msg
    assert "0 failed" in msg


def test_fmt_ingest_result_with_failure():
    r = SimpleNamespace(
        n_succeeded=1,
        n_failed=1,
        failures=(("bad.pdf", "boom"),),
    )
    msg = IngestTab._fmt_ingest_result("Re-ingest", r)
    assert "bad.pdf" in msg
    assert "boom" in msg


def test_fmt_sync_result():
    r = SimpleNamespace(
        n_new_ingested=2,
        n_edited_succeeded=1,
        n_orphans_deleted=3,
        n_new_failed=0,
        n_edited_failed=1,
        n_moved=4,
        # A real edited-failure is recorded in `failures` too (sync_execute
        # appends on every failure), so the list length is the truth.
        failures=(("EDITED bad.pdf", "boom"),),
    )
    msg = IngestTab._fmt_sync_result(r)
    assert "2 new" in msg
    assert "4 moved" in msg
    assert "1 re-ingested" in msg
    assert "3 removed" in msg
    assert "1 failed" in msg  # len(failures)


def test_fmt_sync_result_counts_moved_and_orphan_failures():
    """MOVED (path-patch) and ORPHAN (delete) failures live only in
    `failures`, not in n_new_failed / n_edited_failed. The summary counts
    the whole list so a failed move or delete never misreports '0 failed'."""
    r = SimpleNamespace(
        n_new_ingested=0,
        n_edited_succeeded=0,
        n_orphans_deleted=1,
        n_new_failed=0,
        n_edited_failed=0,
        n_moved=0,
        failures=(
            ("MOVED a.pdf", "boom"),
            ("ORPHAN b.pdf", "delete returned False"),
        ),
    )
    msg = IngestTab._fmt_sync_result(r)
    assert "2 failed" in msg  # both counted, though neither is new/edited


# ---- busy / loop helpers (need construction) ----


def test_set_busy_toggles_spinner_and_buttons(fake_app):
    tab = IngestTab(fake_app)
    tab._set_busy(True)
    assert tab.progress_ring.visible is True
    assert tab.ingest_folder_button.disabled is True
    assert tab.folder_browse_button.disabled is True
    tab._set_busy(False)
    assert tab.progress_ring.visible is False
    assert tab.ingest_folder_button.disabled is False


def test_loop_running_false_without_loop(fake_app):
    tab = IngestTab(fake_app)
    assert tab._loop_running() is False


def test_status_starts_as_empty_placeholder(fake_app):
    """Idle progress line shows the italic 'Empty' placeholder (#39)."""
    tab = IngestTab(fake_app)
    assert tab.status.value == "Empty"
    assert tab.status.italic is True


def test_write_status_clears_placeholder_styling(fake_app):
    """A real status message drops the italic/dim placeholder styling so it
    reads at full weight."""
    tab = IngestTab(fake_app)
    tab._write_status("Ingest folder: 3 / 3 files")
    assert tab.status.value == "Ingest folder: 3 / 3 files"
    assert tab.status.italic is False


def test_start_action_noop_while_busy(fake_app):
    """A second action can't start while one is in flight."""
    tab = IngestTab(fake_app)
    tab._busy = True
    # Should return immediately without touching the config editor.
    tab._start_action("Ingest folder")
    assert tab._bg_tasks == set()


async def test_run_action_empty_folder_clears_busy(fake_app, monkeypatch):
    """Ingest folder with an empty folder field must CLEAR busy, not leave the
    tab stuck spinning with every button disabled. Regression: this early
    return called `_set_status` without `_set_busy(False)` (all its siblings
    clear busy)."""
    tab = IngestTab(fake_app)
    tab.folder_field.value = "   "  # blank -> empty after strip
    fake_app.gui_config.corpus_config_path = "corpus.toml"  # not None
    monkeypatch.setattr(tab.config_editor, "get_ingest_args", lambda: ("Document", "Paper", False))
    monkeypatch.setattr(tab, "_missing_ingest_key", lambda config: None)
    # Skip the heavy backend import: stand-ins for (bulk_ops, load_corpus_config).
    monkeypatch.setattr(
        "asyncio.to_thread",
        AsyncMock(return_value=(SimpleNamespace(), lambda _p: SimpleNamespace())),
    )

    await tab._run_action("Ingest folder")

    assert tab._busy is False
    assert tab.progress_ring.visible is False
    assert tab.ingest_folder_button.disabled is False
    assert "Pick a folder" in tab.status.value


# ---- bulk-ops ----


def test_fmt_bulk_result_shows_ints_and_failures():
    r = _FakeBulkResult(
        n_resolved=2,
        n_no_work=1,
        n_failed=1,
        failures=(("d", "boom"),),
    )
    msg = IngestTab._fmt_bulk_result(r)
    assert "n_resolved=2" in msg
    assert "n_no_work=1" in msg
    assert "1 failures" in msg


def test_fmt_bulk_result_non_dataclass():
    assert IngestTab._fmt_bulk_result("x") == "done"


def test_bulk_op_noop_while_busy(fake_app):
    tab = IngestTab(fake_app)
    tab._busy = True
    tab._on_bulk_op_clicked("bulk_backfill_chunks")
    assert tab._bg_tasks == set()


def test_skip_manual_checkbox_relocated_here_default_on(fake_app):
    """Skip-manually-edited now lives on the Ingest tab (moved from the
    Documents table), default on."""
    tab = IngestTab(fake_app)
    assert tab.skip_manual_checkbox.value is True


# ---- cross-tab refresh callback ----


def test_on_ingest_complete_defaults_none(fake_app):
    tab = IngestTab(fake_app)
    assert tab.on_ingest_complete is None


def test_notify_ingest_complete_calls_callback(fake_app):
    tab = IngestTab(fake_app)
    called = []
    tab.on_ingest_complete = lambda: called.append(True)
    tab._notify_ingest_complete()
    assert called == [True]


def test_notify_ingest_complete_noop_when_unset(fake_app):
    tab = IngestTab(fake_app)
    tab._notify_ingest_complete()  # must not raise when no callback set
