"""Provider-agnostic embedding factory.

Mirrors `llm_factory.py` but for the embedding side. `embed_texts()`
(sync) and `aembed_texts()` (async, added 2026-06-29 in the async
refactor) are the dispatch points used by `ingestion/embed.py` and
the agent's query-side embedding call.

Voyage (today's default, multimodal-capable) has a non-LangChain API
shape — `multimodal_embed(inputs=[[text]], ...)` — so it keeps its
native client path. The three add-on providers (OpenAI, Google,
HuggingFace) use LangChain's `init_embeddings` factory which returns
a `langchain_core.embeddings.Embeddings` instance exposing
`.embed_documents()` and `.embed_query()` (plus `.aembed_documents()`
/ `.aembed_query()` for the async path).

Lazy key validation + no-auto-install — same rule as `llm_factory.py`.
The active provider's key is checked at first call and raises
`ConfigError` (re-imported from llm_factory) with the exact env var
to fix. The factory NEVER pip-installs anything; missing adapter
package raises `ImportError` pointing at the pip extra, and the
GUI's `embedder_lifecycle.py` is the path that actually installs.

Caching: `@lru_cache(maxsize=8)` — fewer realistic combinations than
the LLM side since the embedder is a one-call-site singleton per
process. Distinct cache keys for the two input types (document vs.
query) on the Voyage path because Voyage's batch shape differs.

Dimension contract: the `embedding_dims` Settings field is the source
of truth read by LanceDB at table creation. Each provider's default
model has a fixed dimension (Voyage=1024, OpenAI=1536, Google=768,
HF varies); switching provider with a non-matching dimension breaks
the LanceDB schema and is guarded by
`embedder_lifecycle.switch_provider_plan` (NOT here — the factory
trusts the caller to have run the lifecycle's compatibility check).

ASYNC + RATE LIMITING (added 2026-06-29 in the async refactor):

- `aembed_texts()` is the async sibling of `embed_texts()`. For
  LangChain Embeddings (OpenAI/Google/HF) it dispatches to native
  `.aembed_documents()` / `.aembed_query()`. For Voyage's native
  client (no async API) it wraps the sync call in
  `asyncio.to_thread` so the event loop stays responsive while the
  Voyage HTTP request is in-flight.

- Unlike the LLM side, LangChain `Embeddings` clients do NOT accept
  a `rate_limiter` kwarg — there's no built-in proactive throttling
  for embedders. Bounded throughput on the embedder side is enforced
  by `settings.pipeline_max_concurrent_chunks` +
  `settings.bulk_ops_max_concurrent_docs` semaphores at the
  orchestrator level. The factory itself does not rate-limit; if
  embedder 429s become a problem in practice, drop the concurrency
  caps. The orchestrator's `.with_retry`-equivalent for embedders is
  handled differently (LangChain Embeddings exposes
  `.with_retry()` via `Runnable`); wiring is deferred to the
  orchestrators alongside `.with_retry` for the LLM call sites.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from knowledge_agent.config import get_settings
from knowledge_agent.llm_factory import ConfigError

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def _validate_embedder_config(provider: str) -> None:
    """Check the active embedder has the credentials it needs.

    Same shape as `llm_factory._validate_provider_config` but for
    embedders. OpenAI and Google share `*_api_key` Settings fields
    with the LLM side — the same key works for both endpoints.
    HuggingFace runs locally and needs no key.
    """
    settings = get_settings()
    if provider == "voyage":
        if not settings.voyage_api_key:
            raise ConfigError("EMBEDDING_PROVIDER=voyage but VOYAGE_API_KEY is empty")
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError(
                "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is empty. "
                "Set it in .env or switch to a different embedding_provider."
            )
    elif provider == "google":
        if not settings.google_api_key:
            raise ConfigError(
                "EMBEDDING_PROVIDER=google but GOOGLE_API_KEY is empty. "
                "Set it in .env or switch to a different embedding_provider."
            )
    elif provider == "huggingface":
        # No API key. Adapter requires `sentence-transformers` to be
        # installed — `_build_hf_embedder` will raise ImportError
        # pointing at the `embed-huggingface` extra if missing.
        pass
    else:
        raise ConfigError(f"unknown embedding_provider: {provider!r}")


@lru_cache(maxsize=1)
def _build_voyage_client():
    """Lazy Voyage client. Kept on its native API for multimodal support."""
    import voyageai

    return voyageai.Client(api_key=get_settings().voyage_api_key)


@lru_cache(maxsize=4)
def _build_langchain_embedder(provider: str, model: str) -> Embeddings:
    """Construct a LangChain Embeddings client (OpenAI / Google / HF).

    Each provider has its own adapter pulled in via its pip extra.
    Voyage is NOT routed here — it has a non-LangChain multimodal API
    and lives on `_build_voyage_client` above.
    """
    settings = get_settings()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model, api_key=settings.openai_api_key)
    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=settings.google_api_key)
    if provider == "huggingface":
        # langchain-huggingface ships HuggingFaceEmbeddings which wraps
        # sentence-transformers. The `embed-huggingface` extra pulls
        # sentence-transformers + torch directly; langchain-huggingface
        # is a thin wrapper — if it's not present we fall through to
        # raising a clear ImportError naming the extra.
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "huggingface embedding provider requires the "
                "`embed-huggingface` install extra. Install via the "
                "GUI's Embedder Provider settings, or manually with "
                "`pip install <this package>[embed-huggingface]`."
            ) from exc
        return HuggingFaceEmbeddings(model_name=model)
    raise ConfigError(f"unknown embedding provider: {provider!r}")


def _voyage_call(texts: list[str], input_type: str) -> list[list[float]]:
    """Native-client Voyage call, factored out so it can run in a thread.

    Used by `embed_texts` (wrapped in `asyncio.to_thread` since
    `voyageai.Client` has no async API).
    """
    settings = get_settings()
    client = _build_voyage_client()
    # Voyage's multimodal_embed expects each input as a list of
    # content items (text or PIL.Image); text-only path wraps each
    # text in a single-item list.
    inputs = [[text] for text in texts]
    result = client.multimodal_embed(
        inputs=inputs,
        model=settings.embedding_model,
        input_type=input_type,
    )
    return result.embeddings


def _voyage_multimodal_call(
    items: list[tuple[str, str | None]],
    input_type: str,
) -> list[list[float]]:
    """Native-client Voyage call for chunks that may carry an image.

    `items` is a list of `(text, image_ref)` pairs — one per chunk,
    aligned 1:1 with the output vectors. For each item:

      - Both `text` and `image_ref` present → input `[text, PIL.Image]`
        (caption-anchored multimodal — the shape RAA verified works
        against `voyage-multimodal-3`).
      - Only `text` present (image_ref is None or empty) → input
        `[text]` (text-only through the same multimodal API — same
        shape as the pre-multimodal `_voyage_call` above).
      - Only `image_ref` present (empty text — rare: PDF figure with
        no caption AND no OCR'd text) → input `[PIL.Image]`.

    Image bytes are loaded from disk via PIL. `image_ref` may be an
    absolute path (standalone image inputs; RAA's model) or a path
    inside the corpus folder (`<corpus folder>/figures/<doc_id>/<i>.png`
    — extracted PDF/Word/PPT pictures).
    """
    from PIL import Image

    settings = get_settings()
    client = _build_voyage_client()
    inputs: list[list[Any]] = []
    for text, image_ref in items:
        payload: list[Any] = []
        if text:
            payload.append(text)
        if image_ref:
            payload.append(Image.open(image_ref).convert("RGB"))
        if not payload:
            # Empty text AND no image — degenerate case (shouldn't
            # reach here in normal ingest). Fall back to empty text
            # so the row still gets a vector.
            payload.append("")
        inputs.append(payload)
    result = client.multimodal_embed(
        inputs=inputs,
        model=settings.embedding_model,
        input_type=input_type,
    )
    return result.embeddings


async def embed_chunks(
    chunks: list[Any],
    input_type: str = "document",
) -> list[list[float]]:
    """Embed chunks that may carry image_ref (multimodal-aware wrapper).

    Duck-typed input: each element must have `text` and (optionally)
    `image_ref` attributes/keys — matches `ParsedChunk` objects and
    LanceDB row dicts.

    Voyage path (multimodal-capable): text chunks embed as `[text]`;
    figure chunks (image_ref present + non-empty) embed as
    `[text, PIL.Image]` — one shared vector space via
    `multimodal_embed`. Empty text with image loads as `[PIL.Image]`.

    Non-Voyage providers (OpenAI / Google / HuggingFace): these
    embedders are text-only today. Chunks with image_ref are embedded
    from their text field alone — a warning logs once per call. To
    actually embed images with these providers, add multimodal support
    to the LangChain adapter path (out of scope for the current pass).

    Failures propagate; the orchestrator boundary catches. Empty
    input returns `[]` with no API call.
    """
    if not chunks:
        return []

    def _text_of(c: Any) -> str:
        v = getattr(c, "text", None)
        if v is None and isinstance(c, dict):
            v = c.get("text")
        return v or ""

    def _image_ref_of(c: Any) -> str | None:
        v = getattr(c, "image_ref", None)
        if v is None and isinstance(c, dict):
            v = c.get("image_ref")
        return v if v else None

    settings = get_settings()
    provider = settings.embedding_provider
    _validate_embedder_config(provider)

    if provider == "voyage":
        items = [(_text_of(c), _image_ref_of(c)) for c in chunks]
        return await asyncio.to_thread(
            _voyage_multimodal_call,
            items,
            input_type,
        )

    # Non-Voyage: text-only. Warn once if any chunk has image_ref.
    if any(_image_ref_of(c) for c in chunks):
        logger.warning(
            "embedder provider %r does not support multimodal input; "
            "figure chunks will be embedded from their text field only",
            provider,
        )
    texts = [_text_of(c) for c in chunks]
    model = _active_model_for(provider)
    embedder = _build_langchain_embedder(provider, model)
    if input_type == "query":
        return list(await asyncio.gather(*(embedder.aembed_query(t) for t in texts)))
    return await embedder.aembed_documents(texts)


async def embed_texts(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Embed `texts` into fixed-size vectors. Returns vectors aligned 1:1.

    `input_type` is "document" for indexing (chunks being stored) or
    "query" for the read side. Some providers use this hint to pick
    a different prompt internally (Voyage, OpenAI), others ignore it
    (HF). The factory passes it through where the provider supports
    it.

    Voyage path runs `_voyage_call` in a worker thread via
    `asyncio.to_thread` — voyageai.Client has no async API and HTTP
    I/O dominates the call cost, so a thread frees the event loop to
    progress other tasks (e.g. concurrent KG writes, LLM extraction)
    while Voyage's request is in flight.

    LangChain Embeddings path uses native `.aembed_documents()` /
    `.aembed_query()`. For query mode with multiple inputs, queries
    fan out concurrently via `asyncio.gather`.

    Failures propagate; the orchestrator boundary catches and records
    via `ErrorDetail.from_exception`. Empty input returns `[]` with
    no API call.
    """
    if not texts:
        return []

    settings = get_settings()
    provider = settings.embedding_provider
    _validate_embedder_config(provider)

    if provider == "voyage":
        return await asyncio.to_thread(_voyage_call, texts, input_type)

    model = _active_model_for(provider)
    embedder = _build_langchain_embedder(provider, model)
    if input_type == "query":
        # Fan out queries concurrently. For one-query callers (the
        # agent's read path) this collapses to a single await. For
        # multi-query callers the gather amortises round-trips.
        return list(await asyncio.gather(*(embedder.aembed_query(t) for t in texts)))
    return await embedder.aembed_documents(texts)


def _active_model_for(provider: str) -> str:
    """Pick the per-provider model name from Settings."""
    settings = get_settings()
    if provider == "openai":
        return settings.openai_embedding_model
    if provider == "google":
        return settings.google_embedding_model
    if provider == "huggingface":
        return settings.hf_embedding_model
    # voyage is handled inline above (uses settings.embedding_model
    # directly); shouldn't reach here.
    raise ConfigError(f"no per-provider model field for {provider!r}")


def clear_cache() -> None:
    """Drop the cached embedder clients.

    Called by the lifecycle's switch-provider step so the next
    `embed_texts()` re-resolves against the new provider. Also
    useful for tests that mutate `settings.embedding_provider`
    between cases.
    """
    _build_voyage_client.cache_clear()
    _build_langchain_embedder.cache_clear()
