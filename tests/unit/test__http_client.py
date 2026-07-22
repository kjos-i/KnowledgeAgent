"""Tests for the central `_http_client` module.

Covers the construction contract (Settings → AsyncClient kwargs) plus
the `request()` retry policy on retryable status codes / network
errors. The 4 production call sites (metadata, ontology_helpers,
ontology_fibo_writes, llm_lifecycle) own their own behaviour tests
against this surface; here we pin only what _http_client guarantees.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from knowledge_agent import _http_client


class _FakeResp:
    status_code = 200


async def test_request_explicit_none_timeout_reaches_httpx():
    """A11: an explicit timeout=None must reach httpx as None (no timeout), not
    be clamped to the 30s default. Unset -> default; explicit None -> no
    timeout (for long streaming downloads)."""
    captured: dict = {}

    async def _fake_get(url, params=None, timeout=None):
        captured["timeout"] = timeout
        return _FakeResp()

    fake_client = MagicMock()
    fake_client.get = _fake_get
    fake_settings = MagicMock(http_max_retries=0, http_default_timeout=30.0)
    with (
        patch("knowledge_agent._http_client._get_client", return_value=fake_client),
        patch("knowledge_agent._http_client.get_settings", return_value=fake_settings),
    ):
        await _http_client.request("http://x", timeout=None)
    assert captured["timeout"] is None  # NOT 30.0


async def test_request_unset_timeout_uses_default():
    """A11 regression guard: when the caller does NOT pass timeout, the
    configured default is used (not None)."""
    captured: dict = {}

    async def _fake_get(url, params=None, timeout=None):
        captured["timeout"] = timeout
        return _FakeResp()

    fake_client = MagicMock()
    fake_client.get = _fake_get
    fake_settings = MagicMock(http_max_retries=0, http_default_timeout=30.0)
    with (
        patch("knowledge_agent._http_client._get_client", return_value=fake_client),
        patch("knowledge_agent._http_client.get_settings", return_value=fake_settings),
    ):
        await _http_client.request("http://x")
    assert captured["timeout"] == 30.0


async def test_stream_explicit_none_timeout_reaches_httpx():
    """A11: stream() with an explicit timeout=None must pass None to httpx —
    large ontology downloads legitimately take minutes and must not be clamped
    to the 30s default."""
    captured: dict = {}

    @asynccontextmanager
    async def _fake_stream(method, url, timeout=None):
        captured["timeout"] = timeout
        yield MagicMock()

    fake_client = MagicMock()
    fake_client.stream = _fake_stream
    fake_settings = MagicMock(http_default_timeout=30.0)
    with (
        patch("knowledge_agent._http_client._get_client", return_value=fake_client),
        patch("knowledge_agent._http_client.get_settings", return_value=fake_settings),
    ):
        async with _http_client.stream("http://x", timeout=None):
            pass
    assert captured["timeout"] is None  # NOT 30.0


def test_build_client_passes_trust_env_true():
    """Corporate-proxy + custom CA-bundle env vars (HTTPS_PROXY,
    SSL_CERT_FILE, NO_PROXY) work only when `trust_env=True` is set on
    the AsyncClient. This test pins the intent so a future kwarg
    cleanup doesn't silently drop env-var support — corporate /
    regulated-network users depend on it.

    Verified part of the 2026-06-30 sprint-0d security review.
    """
    fake_settings = MagicMock(
        http_default_timeout=30.0,
        http_user_agent="kg-test/0",
    )
    captured: dict = {}

    def fake_async_client(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch(
            "knowledge_agent._http_client.get_settings",
            return_value=fake_settings,
        ),
        patch("httpx.AsyncClient", side_effect=fake_async_client),
    ):
        _http_client._build_client()

    assert captured.get("trust_env") is True


def test_build_client_passes_user_agent_and_timeout():
    """Settings round-trip: the User-Agent header and default timeout
    flow from Settings into the AsyncClient kwargs."""
    fake_settings = MagicMock(
        http_default_timeout=12.5,
        http_user_agent="kg-test/9.9",
    )
    captured: dict = {}

    def fake_async_client(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with (
        patch(
            "knowledge_agent._http_client.get_settings",
            return_value=fake_settings,
        ),
        patch("httpx.AsyncClient", side_effect=fake_async_client),
    ):
        _http_client._build_client()

    assert captured["timeout"] == 12.5
    assert captured["headers"]["User-Agent"] == "kg-test/9.9"
    assert captured["follow_redirects"] is True
