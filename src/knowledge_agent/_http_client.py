"""Central HTTP client for all outbound traffic — OpenAlex, GitHub Trees
API, Ollama daemon probe, ontology downloads.

ONE `httpx.AsyncClient` per process (cached). ONE place that owns
timeout policy, User-Agent header, and the retry/backoff loop. Replaces
4+ ad-hoc `httpx.get` / `httpx.stream` call sites that each carried
their own timeout (1s, 10s, 60s, None) with no shared config.

Two entry points:

  - `request(url, **kw)` — buffered async GET, retries on 429 / 5xx /
    network errors with exponential backoff (1s → 2s → 4s).
  - `stream(url, **kw)` — async context manager yielding the response
    open so the caller can `iter_raw`/`iter_bytes`. **NEVER retries**
    — partial-download replay is unsafe (truncated writes to disk).
    The bare ontology download caller is responsible for cleanup of
    half-written cache files on failure.

Configuration lives in `Settings`:
  - `http_default_timeout`  per-request timeout when not overridden
  - `http_max_retries`      retry budget for `request()`
  - `http_user_agent`       User-Agent header on every request

Streaming bodies INTENTIONALLY skip the retry loop because:
  1. The body has already started streaming when a mid-transfer error
     hits — replay would resend bytes the caller has already written
     to disk.
  2. Many ontology downloads are gigabytes — retry overhead is huge.
  3. Idempotency belongs at the caller's level (resume support, range
     requests, etc. — none of which we need today).

The client itself is opened lazily on first use and closed on
explicit `close()` only. Process lifetime is fine — no per-request
TLS handshake cost; httpx pools connections per host automatically.

## Corporate proxy + self-signed TLS (verified 2026-06-30, sprint 0d)

`httpx.AsyncClient()` defaults to `trust_env=True`, so every standard
proxy / TLS env var is honoured automatically without any code change
here:

  - `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` — proxy URLs per scheme.
  - `NO_PROXY` — comma-separated hostnames to bypass the proxy
    (typical for `localhost` + the Ollama daemon URL).
  - `SSL_CERT_FILE` — path to a custom CA bundle, for corporate MITM
    proxies that re-sign TLS. Common in regulated industries.
  - `SSL_CERT_DIR` — directory of hashed CA certificates (the
    OpenSSL-format dir layout).

Corporate-network users can set those in their shell / `.env` and the
ingest path, ontology downloads, OpenAlex DOI lookups, and Ollama
daemon probe all start respecting the proxy + CA bundle. We
deliberately do NOT add Settings knobs for these — the env-var route
is the platform-standard pattern that every other httpx-based app
(pip, huggingface_hub, requests) already follows. Documenting one
override mechanism in two places risks drift.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx

from knowledge_agent.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


_client: httpx.AsyncClient | None = None


def _build_client() -> httpx.AsyncClient:
    """Construct the singleton AsyncClient using current Settings.

    `trust_env=True` is set EXPLICITLY (not by default) so future
    refactors can't accidentally drop env-var support without the
    intent being visible. See the module docstring for the full list
    of env vars honoured (HTTPS_PROXY, NO_PROXY, SSL_CERT_FILE, etc.).
    """
    settings = get_settings()
    return httpx.AsyncClient(
        timeout=settings.http_default_timeout,
        headers={"User-Agent": settings.http_user_agent},
        follow_redirects=True,
        trust_env=True,
    )


def _get_client() -> httpx.AsyncClient:
    """Return the cached AsyncClient, opening on first use.

    NOT async — `httpx.AsyncClient()` construction itself is sync;
    only its requests are awaitable. Per-process, per-event-loop reuse
    is safe because httpx detects loop changes and reopens transports
    transparently.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


async def close() -> None:
    """Close the cached client. Call from process shutdown (atexit /
    Flet on_close) so connection pools drain cleanly.

    No-op when no client was opened. Safe to call multiple times — the
    second call hits the no-op branch.
    """
    global _client
    if _client is not None:
        c, _client = _client, None
        await c.aclose()


def _is_retryable_status(status_code: int) -> bool:
    """429 (rate limit) + any 5xx are worth a retry; other 4xx are caller errors."""
    return status_code == 429 or 500 <= status_code < 600


def _backoff_seconds(attempt: int) -> float:
    """Exponential 1s → 2s → 4s → 8s..."""
    return float(2**attempt)


# Sentinel distinguishing "caller passed nothing" (-> the configured
# http_default_timeout) from an explicit `timeout=None` (-> no timeout, for
# long streaming downloads). Typed Any so the `timeout: float | None` default
# stays type-clean; a real caller only ever passes a float or None.
_UNSET_TIMEOUT: Any = object()


async def request(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float | None = _UNSET_TIMEOUT,
    max_retries: int | None = None,
) -> httpx.Response:
    """Async GET with exponential-backoff retry on 429/5xx/network errors.

    Named `request` (not `get`) because `get` collides with `dict.get` /
    pydantic model `.get` everywhere in the codebase — the AST
    consistency guard can't distinguish them and emits false-positive
    "sync call to async 'get'" warnings on every dict access.

    Returns the response on success (any 2xx OR a non-retryable 4xx
    like 404 — those are MEANINGFUL responses the caller wants to see
    so they can `response.status_code` branch).

    Raises the LAST `httpx.HTTPError` after `max_retries` attempts
    of network failure, OR returns the LAST `httpx.Response` for
    `max_retries`-th retryable status code (so the caller can still
    log/inspect it).

    `timeout=None` is propagated to httpx unchanged (= no timeout —
    use only for streams).
    """
    settings = get_settings()
    resolved_timeout = settings.http_default_timeout if timeout is _UNSET_TIMEOUT else timeout
    retries = settings.http_max_retries if max_retries is None else max_retries
    client = _get_client()
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await client.get(
                url,
                params=params,
                timeout=resolved_timeout,
            )
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt < retries:
                wait = _backoff_seconds(attempt)
                logger.info(
                    "http get failed (network/timeout) attempt %d/%d for %s: "
                    "%s — retrying in %.1fs",
                    attempt + 1,
                    retries + 1,
                    url,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
                continue
            raise
        if _is_retryable_status(response.status_code) and attempt < retries:
            wait = _backoff_seconds(attempt)
            logger.info(
                "http get returned %d attempt %d/%d for %s — retrying in %.1fs",
                response.status_code,
                attempt + 1,
                retries + 1,
                url,
                wait,
            )
            await asyncio.sleep(wait)
            continue
        return response
    # Loop completes only when the LAST attempt was a retryable status — fall
    # through with that response.
    assert last_exc is None  # network failure paths raise above
    return response  # type: ignore[return-value]


@asynccontextmanager
async def stream(
    url: str,
    *,
    method: str = "GET",
    timeout: float | None = _UNSET_TIMEOUT,
) -> AsyncIterator[httpx.Response]:
    """Async streaming context manager. NEVER retries — see module docstring.

    Yields an open `httpx.Response`. The caller is expected to drain
    it via `response.aiter_raw()` / `aiter_bytes()`. The underlying
    connection stays open for the lifetime of the `async with`.

    `timeout=None` disables the per-request timeout — necessary for
    large ontology downloads that legitimately take minutes.
    """
    settings = get_settings()
    resolved_timeout = settings.http_default_timeout if timeout is _UNSET_TIMEOUT else timeout
    client = _get_client()
    async with client.stream(method, url, timeout=resolved_timeout) as response:
        yield response
