"""Provider-agnostic LLM factory.

Single dispatch point used by every LLM call site in the codebase
(`nodes._get_llm`, `entity_extractors/llm._get_llm`,
`ingestion/triples_extractor._get_llm`). All three previously imported
`ChatAnthropic` directly; they now go through `get_llm(...)` here
which dispatches on `settings.llm_provider`.

Pattern 1 from the design memo: LangChain's `init_chat_model` factory
returns a `BaseChatModel` regardless of provider, so downstream
`.with_structured_output(...)` and `.invoke(...)` calls work
identically across all four providers (anthropic / openai / google /
ollama). The factory's only job is wiring the right (model name,
api_key, base_url, temperature) into the LangChain call.

LAZY KEY VALIDATION — locked 2026-06-29: the active provider's API
key (or daemon URL for Ollama) is checked at first `get_llm()` call.
Mis-set provider raises a clear `ConfigError` naming the exact env
var to fix. Users who never touch a non-default provider never need
to set its key. NEVER validates at startup — settings.toml stays
loadable with only the default provider's key set.

NO AUTO-INSTALL — picking a provider in settings is settings-only.
If the provider's adapter package isn't installed, `init_chat_model`
raises a clear `ImportError` pointing at the pip extra. The GUI's
install dialog (`llm_lifecycle.py`) is the path that adds the
package; this factory never installs anything.

Caching: `@lru_cache` keyed by (provider, model, temperature) so
hot-path callers (per-chunk extraction) reuse the same LangChain
client. `maxsize=16` covers realistic combinations across the four
providers × six call sites with headroom for tests.

RATE LIMITING + RETRY (added 2026-06-29 in the async refactor):

- `InMemoryRateLimiter` is wired into `init_chat_model` per-provider
  when `settings.<provider>_requests_per_second` is set. The limiter
  is process-local — fine for a desktop single-user app; for a
  multi-user server deployment a Redis-backed limiter would be the
  correct upgrade. ONE bucket per (provider, rps) tuple — all
  cached LLM clients for the same provider share the same token
  bucket so concurrent calls coordinate correctly.

- `with_retry(runnable)` is a module-level helper applied at every
  LLM call site AFTER `.with_structured_output(...)`. It applies
  `stop_after_attempt=settings.llm_max_retries,
  wait_exponential_jitter=True` — covers residual 429s + transient
  network blips after the rate limiter has done its proactive job.
  Call-site usage:
    `await with_retry(llm.with_structured_output(Schema)).ainvoke(...)`.

  Rationale for call-site application (not factory-wrapped):
  `.with_retry()` returns `RunnableRetry`, which doesn't expose
  `.with_structured_output()`. The correct composition order is
  structured-output FIRST, retry SECOND. The helper keeps the
  config DRY — every caller reads from the same Settings field.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from knowledge_agent.config import LLM_PROVIDERS, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


# ========================================================================
# Temperature capability (Anthropic sampling quirk)
# ========================================================================
# Anthropic model IDs that REMOVED the sampling parameters (temperature /
# top_p / top_k). Sending `temperature` to any of these returns a hard
# HTTP 400 ("`temperature` is deprecated for this model"), so the factory
# omits it for them. Older Claude models (Opus 4.6, Sonnet 4.6, Sonnet 4.5,
# Haiku 4.5, …) still accept temperature and are NOT listed here.
# Single source of truth — extend this when a new sampling-free model ships.
_ANTHROPIC_NO_SAMPLING_MODELS: frozenset[str] = frozenset(
    {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
    }
)


def _supports_temperature(provider: str, model: str) -> bool:
    """Whether `temperature` may be sent to (provider, model).

    False only for the newest Anthropic models, which reject the sampling
    parameters with a 400. Every other provider/model accepts temperature.
    Match is case-insensitive on the bare model id (an accidental date
    suffix, if one is ever passed, still matches via prefix)."""
    if provider != "anthropic":
        return True
    m = model.strip().lower()
    return not any(m == name or m.startswith(f"{name}-") for name in _ANTHROPIC_NO_SAMPLING_MODELS)


def supports_temperature(provider: str, model: str) -> bool:
    """Public: whether (provider, model) accepts a `temperature` setting.

    Thin, stable wrapper over the internal check so callers (e.g. the GUI,
    which greys out its temperature sliders for sampling-free models) share
    this single source of truth instead of re-deriving the model set."""
    return _supports_temperature(provider, model)


# ========================================================================
# Provider config validation (lazy key check)
# ========================================================================
class ConfigError(RuntimeError):
    """Raised when the active provider is mis-configured.

    Carries a message naming the exact env var to set. Surfaced by the
    factory at first `get_llm()` call (lazy) rather than at startup.
    The orchestrator's existing typed-error catch (see
    `errors.ErrorDetail.from_exception`) handles this the same way
    it handles any other RuntimeError.
    """


def _validate_provider_config(provider: str) -> None:
    """Check the active provider has the credentials it needs.

    Raises `ConfigError` with a precise env-var name on miss. Does
    NOT check whether the provider's adapter package is installed —
    that's `init_chat_model`'s job and produces its own clear
    ImportError pointing at the pip extra.
    """
    settings = get_settings()
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Set it in .env or switch to a different llm_provider."
            )
    elif provider == "google":
        if not settings.google_api_key:
            raise ConfigError(
                "LLM_PROVIDER=google but GOOGLE_API_KEY is empty. "
                "Set it in .env or switch to a different llm_provider."
            )
    elif provider == "ollama":
        # Ollama needs no key, but the daemon must be reachable. We
        # don't ping it here — that's lifecycle's job at install plan
        # time, and a noisy network call on every cached factory miss
        # is the wrong place. `init_chat_model` will raise a clear
        # `ConnectionError` at first inference if the daemon is down.
        if not settings.ollama_base_url:
            raise ConfigError("LLM_PROVIDER=ollama but OLLAMA_BASE_URL is empty")
    else:
        # Unreachable given the Literal type on `settings.llm_provider`,
        # but the runtime guard keeps mypy happy and catches any future
        # provider added to the Literal but not wired here.
        raise ConfigError(f"unknown llm_provider: {provider!r}")


# ========================================================================
# Rate limiting (per-provider token buckets)
# ========================================================================
@lru_cache(maxsize=8)
def _get_rate_limiter(provider: str, requests_per_second: float) -> InMemoryRateLimiter:
    """Build and cache a per-provider token-bucket rate limiter.

    Cached so all `_build_llm(provider, ...)` cache misses share ONE
    bucket per provider — otherwise each new (model, temperature)
    combination would get its own bucket and the coordination
    guarantee evaporates.

    `check_every_n_seconds=0.1` is LangChain's default; finer grain
    wastes CPU, coarser grain delays callers when the bucket refills.
    `max_bucket_size` defaults to `requests_per_second` (also
    LangChain's default) which means burst-up-to-1-second-worth-of-
    requests then steady-state — sane for our use case where bursts
    come from the per-chunk fan-out.
    """
    # Lazy import — same rationale as `init_chat_model` below: the
    # module stays loadable without the langchain extras.
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(requests_per_second=requests_per_second)


def _rate_limiter_for(provider: str) -> InMemoryRateLimiter | None:
    """Resolve the rate limiter for the active provider, or None.

    Settings field is `<provider>_requests_per_second`; `None` (the
    default) means no rate limiter wired. Positive float means the
    cached bucket above. Negative or zero would have been rejected by
    the pydantic validator (`gt=0.0`), so we don't guard against it.
    """
    settings = get_settings()
    rps_map: dict[str, float | None] = {
        "anthropic": settings.anthropic_requests_per_second,
        "openai": settings.openai_requests_per_second,
        "google": settings.google_requests_per_second,
        "ollama": settings.ollama_requests_per_second,
    }
    rps = rps_map.get(provider)
    if rps is None:
        return None
    return _get_rate_limiter(provider, rps)


# ========================================================================
# Client construction (the cached builder)
# ========================================================================
@lru_cache(maxsize=16)
def _build_llm(provider: str, model: str, temperature: float) -> BaseChatModel:
    """Construct and cache the LangChain chat-model client.

    Cache key includes `provider` so a settings.llm_provider swap at
    runtime (rare — usually requires restart) doesn't return a stale
    client built against the old provider.
    """
    # Lazy import — keeps the module loadable without the langchain
    # extras installed (matters for tests + the lifecycle's
    # is_installed checks).
    from langchain.chat_models import init_chat_model

    settings = get_settings()
    kwargs: dict[str, object] = {
        "model": model,
        "model_provider": provider,
    }
    # `temperature` is REMOVED on the newest Anthropic models (Opus 4.8/4.7,
    # Sonnet 5, Fable 5) — sending it 400s. Only forward it where accepted;
    # GUI temperature knobs still apply to every model that supports them.
    if _supports_temperature(provider, model):
        kwargs["temperature"] = temperature
    else:
        logger.debug(
            "llm_factory: omitting temperature for %s/%s (model rejects sampling params)",
            provider,
            model,
        )
    # Provider-specific kwargs. `init_chat_model` accepts arbitrary
    # provider-specific keyword arguments and forwards them; the
    # mapping below mirrors each provider's LangChain integration
    # contract.
    if provider == "anthropic":
        kwargs["api_key"] = settings.anthropic_api_key
    elif provider == "openai":
        kwargs["api_key"] = settings.openai_api_key
    elif provider == "google":
        kwargs["api_key"] = settings.google_api_key
    elif provider == "ollama":
        kwargs["base_url"] = settings.ollama_base_url

    rate_limiter = _rate_limiter_for(provider)
    if rate_limiter is not None:
        kwargs["rate_limiter"] = rate_limiter

    return init_chat_model(**kwargs)


# ========================================================================
# Model-reference strings ("provider:model")
# ========================================================================
_KNOWN_PROVIDERS: frozenset[str] = frozenset(LLM_PROVIDERS)


def format_model_ref(provider: str, model: str) -> str:
    """Compose a (provider, model) selection into the stored reference string
    'provider:model' (e.g. 'anthropic:claude-sonnet-5', 'ollama:qwen2.5:7b').

    Counterpart to `parse_model_ref`. Part of the provider-per-model
    transition: a model choice carries its provider, so any picker can offer
    every installed provider's models rather than one global 'active provider'.
    """
    return f"{provider}:{model}"


def parse_model_ref(ref: str) -> tuple[str | None, str]:
    """Split a stored model reference into (provider, model).

    'provider:model' where `provider` is a known provider (`LLM_PROVIDERS`) →
    (provider, model). A bare model string — no provider prefix (legacy), or an
    Ollama tag whose left side isn't a provider like 'qwen2.5:7b' → (None, ref);
    the caller then falls back to the global provider. Splits on the FIRST ':'
    only, so multi-colon Ollama tags survive
    ('ollama:qwen2.5:7b' → ('ollama', 'qwen2.5:7b')).
    """
    head, sep, tail = ref.partition(":")
    if sep and head in _KNOWN_PROVIDERS:
        return head, tail
    return None, ref


def to_model_ref(ref: str, fallback_provider: str) -> str:
    """Normalize a stored model reference to a composite 'provider:model'.

    A bare / legacy value (no provider prefix) is wrapped with
    `fallback_provider` so it matches a picker option and dispatches
    explicitly; a value that already carries a provider is returned unchanged.
    The one place model-picker code turns a stored (possibly legacy) value into
    a display/dispatch ref.
    """
    provider, model = parse_model_ref(ref)
    return format_model_ref(provider or fallback_provider, model)


# ========================================================================
# Public entry points
# ========================================================================
def get_llm(model: str, temperature: float, *, provider: str | None = None) -> BaseChatModel:
    """Return the LLM client for the given (or active) provider.

    Args:
      model: Provider-specific model name (e.g. `claude-sonnet-4-6`,
        `gpt-4o`, `gemini-1.5-pro`, `qwen2.5:7b`).
      temperature: Sampling temperature, 0.0-1.0.
      provider: The LLM provider to dispatch through. When None (the default),
        falls back to the global `settings.llm_provider` — the legacy behavior,
        preserved while model choices migrate to carrying their own provider
        (see `parse_model_ref` / `format_model_ref`).

    Lazy key validation runs first; a mis-configured provider raises
    `ConfigError` with the exact env var to fix. If the provider's adapter
    package isn't installed, `init_chat_model` raises `ImportError` pointing
    at the pip extra.
    """
    settings = get_settings()
    resolved = provider or settings.llm_provider
    _validate_provider_config(resolved)
    return _build_llm(resolved, model, temperature)


def get_llm_ref(ref: str, temperature: float) -> BaseChatModel:
    """`get_llm` for a stored model reference.

    `ref` is a 'provider:model' string (new form) or a bare/legacy model name;
    this parses the provider out (`parse_model_ref`) and dispatches — a bare
    ref falls back to the global `settings.llm_provider`. This is the single
    seam every call site uses, so a stored selection's provider is always
    honored without each site re-implementing the parse.
    """
    provider, model = parse_model_ref(ref)
    return get_llm(model, temperature, provider=provider)


# ========================================================================
# Retry wrapper + cache management
# ========================================================================
def with_retry(runnable: Runnable) -> Runnable:
    """Wrap any LangChain `Runnable` with the standard retry policy.

    Applied at every LLM call site AFTER `.with_structured_output(...)`
    (LangChain composition order: structured-output first since
    `.with_retry()` returns a `RunnableRetry` which doesn't expose
    `.with_structured_output()`).

    Reads `settings.llm_max_retries` so a single Settings change
    propagates to every call site. Exponential backoff with jitter
    matches LangChain's recommended default for 429s + network
    blips.
    """
    settings = get_settings()
    return runnable.with_retry(
        stop_after_attempt=settings.llm_max_retries,
        wait_exponential_jitter=True,
    )


def clear_cache() -> None:
    """Drop the cached LangChain clients + rate limiters.

    Called by the lifecycle's switch-provider step so the next
    `get_llm()` re-resolves against the new provider. Also useful for
    tests that mutate `settings.llm_provider` or
    `settings.*_requests_per_second` between cases.
    """
    _build_llm.cache_clear()
    _get_rate_limiter.cache_clear()
