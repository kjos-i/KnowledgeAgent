"""Embeddings for chunk text — provider-agnostic dispatch.

`embed_texts(texts, input_type)` returns vectors aligned 1:1 with the
inputs. All four embedding providers are supported (voyage / openai /
google / huggingface); dispatch happens inside `embedder_factory`
based on the active `settings.embedding_provider`. This module is a
thin re-export so existing imports (`from knowledge_agent.ingestion.
embed import embed_texts`) keep working unchanged.

`embed_texts` is async — call sites must `await` it. Voyage uses
`asyncio.to_thread` internally (its native client has no async API);
the LangChain providers use native `.aembed_documents()` /
`.aembed_query()`.

API errors propagate — the caller (`pipeline.ingest_document`,
`pipeline.re_embed`, agent read-path nodes) is the orchestrator
boundary that catches and decides what to do (skip the doc, retry,
mark pending, surface in `IngestResult.embed_error`).

History: this module used to wrap `voyageai.Client` directly. The
2026-06-29 provider-swap refactor moved the dispatch logic to
`embedder_factory.py` so all four providers share one entry point.
The 2026-06-29 async refactor made `embed_texts` async; the sync
version was deleted in Day 8 cleanup.
"""

from knowledge_agent.embedder_factory import (
    embed_chunks as embed_chunks,
)
from knowledge_agent.embedder_factory import (
    embed_texts as embed_texts,
)

__all__ = ["embed_chunks", "embed_texts"]
