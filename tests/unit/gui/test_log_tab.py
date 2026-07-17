"""Tests for the Log tab (tabs/log_tab) + the `log_file_path` accessor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui.tabs.log_tab import LogTab, _read_tail
from knowledge_agent.logging_setup import log_file_path

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    import pytest


def test_log_file_path_points_at_kagent_log():
    p = log_file_path()
    assert p.name == "kagent.log"
    assert p.is_absolute()


def test_read_tail_returns_last_n_and_total(tmp_path: Path):
    f = tmp_path / "x.log"
    f.write_text("\n".join(f"line{i}" for i in range(10)) + "\n", encoding="utf-8")
    text, total = _read_tail(f, 3)
    assert total == 10
    assert text.splitlines() == ["line7", "line8", "line9"]


def test_read_tail_missing_file(tmp_path: Path):
    text, total = _read_tail(tmp_path / "nope.log", 5)
    assert text == ""
    assert total == 0


def test_log_tab_builds_and_shows_path_and_tail(
    fake_app: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    logf = tmp_path / "kagent.log"
    logf.write_text("hello log\n", encoding="utf-8")
    monkeypatch.setattr("knowledge_agent.gui.tabs.log_tab.log_file_path", lambda: logf)

    tab = LogTab(fake_app)
    ctl = tab.build()

    assert isinstance(ctl, ft.Column)
    assert tab._path_text.value == str(logf)
    assert "hello log" in tab._body.value
    assert "1 of 1" in tab._status.value


def test_log_tab_empty_when_no_log(
    fake_app: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "knowledge_agent.gui.tabs.log_tab.log_file_path", lambda: tmp_path / "kagent.log"
    )
    tab = LogTab(fake_app)
    tab.build()
    assert "no log" in tab._body.value.lower()
    assert "No log entries yet." in tab._status.value
