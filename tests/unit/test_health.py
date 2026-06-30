"""Tests for `knowledge_agent.health` — the StatusReport orchestrator.

Per-check helpers are tested individually with mocked clients; the
orchestrator is tested for ordering, concurrent execution, and the
`all_ok` derivation. The CLI / GUI renderers are covered separately.

Each check is supposed to fail-soft (return ComponentStatus instead of
raising), so the assertions confirm that contract — a Neo4j raise
becomes `ok=False` with the exception repr in `detail`, not an
uncaught exception bubbling out of `system_status()`.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.health import (
    ComponentStatus,
    StatusReport,
    _check_lancedb,
    _check_neo4j,
    _check_provider_key,
    render_report_text,
    system_status,
)


# ---- ComponentStatus / StatusReport dataclasses ----


def test_component_status_is_frozen():
    """Renderers pass ComponentStatus around; in-place mutation must raise."""
    c = ComponentStatus("neo4j", True, "ok")
    with pytest.raises(Exception):
        c.ok = False  # type: ignore[misc]


def test_status_report_all_ok_true_when_all_components_ok():
    report = StatusReport(components=(
        ComponentStatus("a", True, ""),
        ComponentStatus("b", True, ""),
    ))
    assert report.all_ok is True


def test_status_report_all_ok_false_when_any_component_fail():
    report = StatusReport(components=(
        ComponentStatus("a", True, ""),
        ComponentStatus("b", False, "err"),
    ))
    assert report.all_ok is False


def test_status_report_all_ok_true_for_empty_components():
    """Vacuous: no components → no failures → all_ok=True. Documents
    the corner case so future renderers don't crash on an empty list."""
    assert StatusReport(components=()).all_ok is True


# ---- _check_neo4j ----


async def test_check_neo4j_returns_ok_when_round_trip_succeeds():
    fake_client = MagicMock()
    fake_client.read_query = AsyncMock(return_value=[{"ok": 1}])
    with patch(
        "knowledge_agent.kg.client.get_kg_client", return_value=fake_client,
    ):
        result = await _check_neo4j()
    assert result.name == "neo4j"
    assert result.ok is True


async def test_check_neo4j_returns_fail_when_unexpected_result():
    """A query that returns something OTHER than ok=1 is a real
    Neo4j-side bug — surface as fail with the result in detail."""
    fake_client = MagicMock()
    fake_client.read_query = AsyncMock(return_value=[{"ok": 999}])
    with patch(
        "knowledge_agent.kg.client.get_kg_client", return_value=fake_client,
    ):
        result = await _check_neo4j()
    assert result.ok is False
    assert "999" in result.detail


async def test_check_neo4j_catches_driver_exception_returns_fail():
    """Network errors propagate from the driver as exceptions; the check
    catches and reports the repr in detail."""
    fake_client = MagicMock()
    fake_client.read_query = AsyncMock(
        side_effect=RuntimeError("connection refused"),
    )
    with patch(
        "knowledge_agent.kg.client.get_kg_client", return_value=fake_client,
    ):
        result = await _check_neo4j()
    assert result.ok is False
    assert "connection refused" in result.detail


# ---- _check_lancedb ----


async def test_check_lancedb_returns_ok_with_table_count():
    fake_conn = MagicMock()
    fake_conn.table_names = AsyncMock(return_value=["chunks", "indexes"])
    fake_client = MagicMock()
    fake_client._ensure_conn = AsyncMock(return_value=fake_conn)
    with patch(
        "knowledge_agent.search.client.get_search_client",
        return_value=fake_client,
    ):
        result = await _check_lancedb()
    assert result.ok is True
    assert "2 table" in result.detail


async def test_check_lancedb_catches_open_failure():
    fake_client = MagicMock()
    fake_client._ensure_conn = AsyncMock(
        side_effect=FileNotFoundError("missing directory"),
    )
    with patch(
        "knowledge_agent.search.client.get_search_client",
        return_value=fake_client,
    ):
        result = await _check_lancedb()
    assert result.ok is False
    assert "missing directory" in result.detail


# ---- _check_provider_key ----


def test_provider_key_local_provider_returns_ok():
    """Ollama / HuggingFace: key_attr=None → always OK with local detail."""
    settings = MagicMock(spec=[])  # no attrs needed
    status = _check_provider_key("llm", "ollama", None, settings)
    assert status.ok is True
    assert "local" in status.detail


def test_provider_key_set_returns_ok():
    settings = MagicMock(anthropic_api_key="sk-ant-fake")
    status = _check_provider_key(
        "llm", "anthropic", "anthropic_api_key", settings,
    )
    assert status.ok is True
    assert "set" in status.detail


def test_provider_key_missing_returns_fail_with_env_var_name():
    """The detail message names the env var to set so the user can fix it."""
    settings = MagicMock(anthropic_api_key=None)
    status = _check_provider_key(
        "llm", "anthropic", "anthropic_api_key", settings,
    )
    assert status.ok is False
    assert "ANTHROPIC_API_KEY" in status.detail
    assert "MISSING" in status.detail


# ---- system_status orchestrator ----


async def test_system_status_returns_four_components_in_display_order():
    """Order is neo4j → lancedb → llm_key → embed_key. Locked because
    the renderers (CLI / GUI) rely on this for visual consistency."""
    fake_kg = MagicMock()
    fake_kg.read_query = AsyncMock(return_value=[{"ok": 1}])
    fake_conn = MagicMock()
    fake_conn.table_names = AsyncMock(return_value=[])
    fake_search = MagicMock()
    fake_search._ensure_conn = AsyncMock(return_value=fake_conn)
    fake_settings = MagicMock(
        anthropic_api_key="sk-ant",
        voyage_api_key="vy",
        llm_provider="anthropic",
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.health.get_settings", return_value=fake_settings),
        patch("knowledge_agent.kg.client.get_kg_client", return_value=fake_kg),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_search,
        ),
    ):
        report = await system_status()

    assert [c.name for c in report.components] == [
        "neo4j", "lancedb", "llm_key", "embed_key",
    ]
    assert report.all_ok is True


async def test_system_status_all_ok_false_when_neo4j_fails():
    """One component failing → report.all_ok=False; other components
    still run + report their status."""
    fake_kg = MagicMock()
    fake_kg.read_query = AsyncMock(side_effect=RuntimeError("conn"))
    fake_conn = MagicMock()
    fake_conn.table_names = AsyncMock(return_value=[])
    fake_search = MagicMock()
    fake_search._ensure_conn = AsyncMock(return_value=fake_conn)
    fake_settings = MagicMock(
        anthropic_api_key="sk-ant",
        voyage_api_key="vy",
        llm_provider="anthropic",
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.health.get_settings", return_value=fake_settings),
        patch("knowledge_agent.kg.client.get_kg_client", return_value=fake_kg),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_search,
        ),
    ):
        report = await system_status()

    assert report.all_ok is False
    # Other components still reported OK.
    assert report.components[1].ok is True  # lancedb
    assert report.components[2].ok is True  # llm_key


async def test_system_status_local_provider_skips_key_check():
    """Ollama LLM provider → llm_key reports OK with `local` detail
    regardless of which API keys are present."""
    fake_kg = MagicMock()
    fake_kg.read_query = AsyncMock(return_value=[{"ok": 1}])
    fake_conn = MagicMock()
    fake_conn.table_names = AsyncMock(return_value=[])
    fake_search = MagicMock()
    fake_search._ensure_conn = AsyncMock(return_value=fake_conn)
    fake_settings = MagicMock(
        anthropic_api_key=None,
        voyage_api_key="vy",
        llm_provider="ollama",        # <-- local
        embedding_provider="voyage",
    )
    with (
        patch("knowledge_agent.health.get_settings", return_value=fake_settings),
        patch("knowledge_agent.kg.client.get_kg_client", return_value=fake_kg),
        patch(
            "knowledge_agent.search.client.get_search_client",
            return_value=fake_search,
        ),
    ):
        report = await system_status()

    llm_key = next(c for c in report.components if c.name == "llm_key")
    assert llm_key.ok is True
    assert "local" in llm_key.detail


# ---- render_report_text ----


def test_render_report_text_one_line_per_component():
    report = StatusReport(components=(
        ComponentStatus("neo4j", True, "ok"),
        ComponentStatus("lancedb", False, "fail"),
    ))
    text = render_report_text(report)
    lines = text.split("\n")
    assert len(lines) == 2
    assert "OK" in lines[0]
    assert "FAIL" in lines[1]


def test_render_report_text_empty_components_returns_placeholder():
    text = render_report_text(StatusReport(components=()))
    assert "no components" in text.lower()
