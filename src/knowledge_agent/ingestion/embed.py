"""Embeddings for chunk text — provider-agnostic dispatch.

`embed_texts(texts, input_type)` (sync) and `aembed_texts(texts,
input_type)` (async) return vectors aligned 1:1 with the inputs. All
four embedding providers are supported (voyage / openai / google /
huggingface); dispatch happens inside `embedder_factory` based on the
active `settings.embedding_provider`. This module is a thin re-export
so existing imports (`from knowledge_agent.ingestion.embed import
embed_texts`) keep working unchanged, and the new async ingest path
imports `aembed_texts` from the same stable location.

API errors propagate — the caller (`pipeline.ingest_document`,
`pipeline.re_embed`, agent read-path nodes) is the orchestrator
boundary that catches and decides what to do (skip the doc, retry,
mark pending, surface in `IngestResult.embed_error`).

History: this module used to wrap `voyageai.Client` directly. The
2026-06-29 provider-swap refactor moved the dispatch logic to
`embedder_factory.py` so all four providers share one entry point;
this file now exists only as a stable import path for ingestion code
that doesn't need to know about providers. The async refactor on the
same day added `aembed_texts` alongside the sync entry point — same
contract, native await on the LangChain providers + `asyncio.to_thread`
wrap on Voyage's native (sync-only) client.
"""

from knowledge_agent.embedder_factory import (
    aembed_texts as aembed_texts,
    embed_texts as embed_texts,
)

__all__ = ["aembed_texts", "embed_texts"]
