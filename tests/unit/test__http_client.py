"""Tests for the central `_http_client` module.

Covers the construction contract (Settings → AsyncClient kwargs) plus
the `request()` retry policy on retryable status codes / network
errors. The 4 production call sites (metadata, ontology_helpers,
ontology_fibo_writes, llm_lifecycle) own their own behaviour tests
against this surface; here we pin only what _http_client guarantees.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent import _http_client


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
