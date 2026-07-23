# Architecture

One-page map of how KnowledgeAgent is organised, how data flows through it, and the major design decisions that shape the code. Read this once when picking the project back up; then trust the docstrings in each module for the rest.

For a high-level pitch see [README.md](README.md). For per-feature deep-dives, every module has a top-of-file docstring describing its contract.

---

## What it is

A desktop research-knowledge agent that ingests documents in many formats, stores them in two complementary backends, and lets the user query the result via a LangGraph agent that picks one of several retrieval topologies.

Two storage backends serve different query shapes:

- **[LanceDB](src/knowledge_agent/search/)** — embedded vector + BM25 hybrid search over chunked document text. Answers content questions ("what does the literature say about X?").
- **[Neo4j](src/knowledge_agent/kg/)** — knowledge graph spanning papers, authors, citations, venues, topics, entities, ontology canonicals, typed relations, and cross-document links. Answers structural and cross-document questions ("which authors cite each other?", "papers connected by shared concepts").

The agent ([nodes.py](src/knowledge_agent/nodes.py), [graph.py](src/knowledge_agent/graph.py)) orchestrates retrieval across both stores in one of five modes.

---

## Top-level layout

```
src/knowledge_agent/
├── config.py                 # Pydantic Settings — every runtime knob
├── nodes.py                  # 6 LangGraph node functions (agent loop)
├── graph.py                  # LangGraph wiring
├── state.py                  # AgentState dataclass (per-invocation state)
├── models.py                 # Public data shapes (Mention, KGHit, RetrievedChunk, ...)
├── errors.py                 # ErrorDetail — typed-error primitive
├── logging_setup.py          # Production-grade logging + crash handling
├── llm_factory.py            # Provider-agnostic LLM dispatch (4 providers)
├── embedder_factory.py       # Provider-agnostic embedding dispatch (4 providers)
├── llm_lifecycle.py          # GUI plan/execute for LLM provider install
├── embedder_lifecycle.py     # GUI plan/execute for embedder provider install
├── cli.py                    # Headless CLI — ka ingest / query / health / eval
├── health.py                 # Startup health checks (DB reachability, config)
├── artifacts.py              # Save / export answers + chats (md / txt / docx / json)
│
├── ingestion/                # Document ingest pipeline
│   ├── pipeline.py           # Per-doc orchestrator: parse → embed → write
│   ├── bulk_ops.py           # 14 multi-doc operations (delete, backfill, sync, ...)
│   ├── parse.py              # Format dispatcher
│   ├── parsers/              # docling_parser, code_parser, json_parser
│   ├── parser_lifecycle.py   # parsers-asr / parsers-code install ops
│   ├── triples_extractor.py  # LLM-based L8 typed-relation extraction
│   ├── embed.py              # Re-export of embedder_factory.embed_texts
│   └── ids.py                # doc_id / chunk_id derivation
│
├── kg/                       # Neo4j layer (KG writes + reads)
│   ├── client.py             # Neo4j driver wrapper, write/read facade
│   ├── schema.py             # Labels, properties, indexes
│   ├── schema_as_prompt.py   # Schema → LLM system prompt for Cypher generation
│   ├── cypher_safety.py      # Read-only validation, LIMIT wrapping
│   ├── openalex_writes.py    # L1-L4: papers, authors, citations, topics
│   ├── chunk_writes.py      # L5: per-chunk :Chunk nodes
│   ├── entity_writes.py      # L6: extracted entities
│   ├── ontology_*_writes.py  # L7: 18 ontology importers (one file per ontology)
│   ├── ontology_helpers.py   # Shared pronto + rdflib + Neo4j-write primitives
│   ├── ontology_xrefs.py     # L7 xref edges (MONDO ↔ MeSH, etc.)
│   ├── triples_writes.py     # L8: 15 typed entity-to-entity relations
│   ├── cross_doc_writes.py   # L9: :RELATED_TO via shared entities
│   ├── cross_doc_xrefs_writes.py  # L10: :RELATED_BY_XREF via shared canonicals
│   ├── ontology_lifecycle.py # GUI plan/execute for ontology import/delete
│   ├── ontology_provenance.py # OntologyProvenance dataclass
│   └── corpus_config.py      # Per-corpus toml schema
│
├── entity_extractors/        # NER adapters (L6)
│   ├── base.py               # Mention dataclass + adapter contract
│   ├── __init__.py           # Dispatch table: routes corpus.toml extractor to adapter
│   ├── llm.py                # Anthropic Haiku (or active LLM provider)
│   ├── gliner.py             # GLiNER multilingual zero-shot
│   ├── gliner_biomed.py      # GLiNER-BioMed biomedical zero-shot
│   ├── hunflair2.py          # HunFlair2 biomedical (Flair, 5-label)
│   └── extractor_lifecycle.py # GUI plan/execute for extractor install
│
├── search/                   # LanceDB layer
│   ├── client.py             # Connection lifecycle, write, hybrid/fts/vector search
│   └── schema.py             # chunks table schema (pinned at first creation)
│
├── evaluation/               # Evaluation harness — metrics, LLM-judge panel,
│                             #   SQLite ledger + JSON/CSV reports (ka eval / GUI)
│
└── gui/                      # Flet desktop app — Search / Library / Evaluation /
                              #   Settings / Log / Info tabs (entry point: ka-gui)
```

---

## Data flow: ingest path

When the user (or a bulk op) ingests one document:

```
┌────────────────────────────────────────────────────────────────────┐
│ source file (PDF / DOCX / HTML / MD / audio / video / ...)         │
└────────────────────────────────────────────────────────────────────┘
                                  │
              ingestion/parse.py + parsers/docling_parser.py
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│ list[ParsedChunk]  — text + section + page + content_type          │
└────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┴────────────────────┐
        │                                              │
        ▼                                              ▼
  embedder_factory.embed_texts                  ingest pipeline.py
  (Voyage / OpenAI / Google / HF)               coordinates everything
        │                                              │
        ▼                                              │
┌──────────────────────┐                               │
│  LanceDB chunks      │                               │
│  (vector + BM25)     │                               │
└──────────────────────┘                               │
                                                       │
                          ┌────────────────────────────┤
                          │                            │
                          ▼                            ▼
              KG L1-L4 (OpenAlex)             KG L5 (per-chunk nodes)
              papers, authors, citations,     :Chunk linked to :Document
              topics, venues
                          │                            │
                          ▼                            ▼
                                  KG L6 (NER)
                          per-chunk :MENTIONS to :Entity
                          (LLM or GLiNER or HunFlair2)
                                       │
                                       ▼
                                  KG L7 (ontology linking)
                          :CANONICAL_TO from :Entity to :OntologyTerm
                          when label/synonym matches
                                       │
                                       ▼
                                  KG L8 (triples)
                          15 typed :INHIBITS / :ACTIVATES / ... edges
                          between :Entity nodes (LLM-extracted)
                                       │
                                       ▼
                                  KG L9 (cross-document)
                          :RELATED_TO when two docs share ≥2 entities
                                       │
                                       ▼
                                  KG L10 (cross-doc via xrefs)
                          :RELATED_BY_XREF via shared canonicals
                          (joined through L7 ontology xref edges)
```

Each L7 ontology lives in its own write module ([kg/ontology_*_writes.py](src/knowledge_agent/kg/)). All 18 use shared helpers in [kg/ontology_helpers.py](src/knowledge_agent/kg/ontology_helpers.py).

Each layer is independently toggleable via [`corpus.toml`](corpus.toml.example).

---

## Data flow: query path

When the user asks a question:

```
┌─────────────────────────────────────┐
│  user question (natural language)   │
└─────────────────────────────────────┘
                  │
                  ▼
       nodes.py: mode_classifier_node
       (only when retrieval_mode="auto")
                  │
                  ▼
  ┌─────────────────────────────────────────────┐
  │  Picks one of 5 modes:                      │
  │   1. lancedb_only                           │
  │   2. neo4j_only                             │
  │   3. lancedb_then_neo4j                     │
  │   4. neo4j_then_lancedb                     │
  │   5. parallel_fused                         │
  └─────────────────────────────────────────────┘
                  │
       ┌──────────┴───────────┐
       │                      │
       ▼                      ▼
 query_builder_node     cypher_builder_node
 (rewrites for LanceDB) (writes safe read-only Cypher
                         from schema_as_prompt)
       │                      │
       ▼                      ▼
 lancedb_retriever_node   neo4j_retriever_node
 (hybrid/fts/vector       (CALL{...} RETURN * LIMIT N
  via search/client.py)    sandboxed via cypher_safety.py)
       │                      │
       │   list[RetrievedChunk]│ list[KGHit]
       │                      │
       └──────────┬───────────┘
                  ▼
       synthesizer_node
       (LLM produces final AgentAnswer with citations)
                  │
                  ▼
       AgentAnswer { answer, chunk_sources, kg_sources }
```

Per-invocation overrides (skip query_builder, skip synthesizer, change mode) live on [AgentState](src/knowledge_agent/state.py). Each node populates a typed error field on AgentState if its stage fails — see [errors.py](src/knowledge_agent/errors.py).

---

## The five lifecycle modules

A recurring pattern. Each lifecycle file owns a registry of installable things + a `plan_X()` / `execute_X()` pair surfaced to the GUI for install / uninstall / configure. The pattern was first established for ontologies, then replicated four more times:

| File | Manages | Registry size |
|---|---|---|
| [kg/ontology_lifecycle.py](src/knowledge_agent/kg/ontology_lifecycle.py) | L7 ontology import / link / delete | 18 ontologies |
| [entity_extractors/extractor_lifecycle.py](src/knowledge_agent/entity_extractors/extractor_lifecycle.py) | L6 NER extractor pip install | 4 extractors |
| [ingestion/parser_lifecycle.py](src/knowledge_agent/ingestion/parser_lifecycle.py) | parsers-asr / parsers-code extras | 2 extras |
| [llm_lifecycle.py](src/knowledge_agent/llm_lifecycle.py) | 4 LLM providers + Ollama model pulls | 4 providers + 4 models |
| [embedder_lifecycle.py](src/knowledge_agent/embedder_lifecycle.py) | 4 embedder providers + HF model downloads | 4 providers + 4 models |

Same contract for every one: `XPlan` dataclass with a `summary` property the GUI renders as a confirm dialog; `XResult` dataclass the GUI displays after execution.

---

## Provider model (LLM + embedder)

Both default providers (Anthropic LLM, Voyage embeddings) are **opt-in extras** as of 2026-06-29 — no provider is bundled in base deps. The first-launch wizard asks the user to pick + installs the relevant adapter. Settings + factory dispatch:

- [config.py](src/knowledge_agent/config.py) holds `llm_provider`, `embedding_provider`, and per-provider API keys (all `Optional`)
- [llm_factory.py](src/knowledge_agent/llm_factory.py) `get_llm(model, temp)` dispatches to LangChain's `init_chat_model` based on `settings.llm_provider`
- [embedder_factory.py](src/knowledge_agent/embedder_factory.py) `embed_texts(texts, input_type)` dispatches to Voyage's native client or LangChain's `OpenAIEmbeddings` / `GoogleGenerativeAIEmbeddings` / `HuggingFaceEmbeddings`
- Lazy key validation in both factories — the active provider's key is checked on first call, never at startup, so a user on pure-OpenAI never needs to set Anthropic/Voyage keys

---

## Settings + persistence

[config.py](src/knowledge_agent/config.py) is the single source of truth for runtime knobs. Loaded from `.env` (or `.env.test` via `load_test_env()`). Pydantic Settings validates types + ranges + cross-field invariants (e.g. `top_k ≤ num_candidates`).

Two `.env` files supported by design:
- `.env` — real corpus, real Neo4j instance
- `.env.test` — separate Neo4j instance + separate LanceDB folder for smokes (different password = a wrong-instance cross-connect fails authentication instead of silently corrupting real data)

Per-corpus settings (which layers are on, which extractor, which ontologies) live in each corpus folder's `corpus.toml` — schema in [corpus_config.py](src/knowledge_agent/corpus_config.py).

---

## Async surface

Shipped 2026-06-30. The entire I/O surface is `async def` end to end — no sync wrapper layer.

- **All `Neo4jClient` + `LanceClient` methods are `async def` on the native async drivers.** Neo4j uses `AsyncGraphDatabase` + `AsyncSession`; LanceDB uses `lancedb.connect_async` + `AsyncTable` / `AsyncQuery`. Every Cypher round-trip and every Lance read/write stays on the event loop — no `asyncio.to_thread` thread-pool dispatch in the hot path.
- **`pipeline.ingest_document` and `bulk_ops.*` are `async def`.** Per-chunk extraction in L6 and L8 fans out via `asyncio.gather` bounded by `asyncio.Semaphore(settings.pipeline_max_concurrent_chunks)` — this is the wall-clock win.
- **Embedder calls (`embed_texts`) are `async def`.** Voyage uses `asyncio.to_thread` over its sync SDK; OpenAI / Google / HuggingFace use LangChain's native `.aembed_documents()` / `.aembed_query()`.
- **LLM calls (`init_chat_model` runnables) use `.ainvoke()`** with an `InMemoryRateLimiter` wired into the factory and `.with_retry()` for `RunnableRetry` backoff.
- **Only sync escape hatch:** entry points (smoke scripts, the GUI's main loop) wrap `asyncio.run(main())` at the boundary. Inside the library everything is native async.
- **Pytest config:** `asyncio_mode = "auto"` (pytest-asyncio). Every `async def test_*` runs transparently — no `@pytest.mark.asyncio` decorators in the suite.

The performance shape: a single PDF with ~50 chunks at L6 goes from `O(50 * t_extract)` sequential to `O(50/8 * t_extract)` parallel — ~5-8× wall-clock improvement on extraction-bound workloads. KG writes still bottleneck at the Neo4j driver pool (`settings.neo4j_max_connection_pool_size`, default 100).

---

## Error handling contract

Locked 2026-06-26. See [errors.py](src/knowledge_agent/errors.py) for the dataclass.

- **Leaves raise.** Internal helpers (KG writes, HTTP fetches, parsers) propagate exceptions naturally. No swallow-and-return-False.
- **Orchestrators catch at the boundary** where a result dataclass (`IngestResult`, `BulkOpResult`, `AgentState`) is constructed. The catch handler populates `error: ErrorDetail | None` via `ErrorDetail.from_exception(exc)` and logs.
- **GUI reads the error field** off the result and shows the human `message`; `exception_type` is available for "Show details".

This matches how SQLAlchemy / Django / FastAPI all handle layered errors: exceptions for the mechanism, named result types at the API boundary.

---

## Testing tiers

Three tiers — see [tests/unit/](tests/unit/), [tests/integration/](tests/integration/), `tests/e2e/`.

- **Unit** — fast, pure-Python, no real services. Mocks for Neo4j / LanceDB / Voyage / LLMs. 2,400+ tests run in ~35 seconds.
- **Integration** — real services via the test instance (`@pytest.mark.integration`). Hits Neo4j Desktop test instance + LanceDB on-disk. Opt-in via `pytest -m integration`.
- **e2e** — Flet app launched end-to-end (`@pytest.mark.e2e`). Opt-in via `pytest -m e2e`.

Marker `@pytest.mark.slow` for tests that take noticeably long (real model inference, large ontology imports).

Smoke scripts in [scripts/](scripts/) are the human-supervised counterpart: install-and-use lifecycle exercises (10 smokes for provider installs, plus per-KG-layer smokes).

---

## Key architectural decisions (locked)

- **Layered KG with corpus-level toggles.** Each L1-L10 layer is independently on/off per corpus. Off-by-default for layers that cost LLM tokens (L6/L8).
- **Provider symmetry.** All 4 LLM providers + all 4 embedder providers are extras; no provider is privileged. First-launch wizard picks. Settings + factories enforce lazy validation.
- **Lifecycle pattern for all installable things.** Plan/execute dataclasses with `summary` property for the GUI. Five instances and counting.
- **Typed errors at orchestrator boundaries.** No silent failures; every result dataclass exposes its `*_error: ErrorDetail | None` fields.
- **Single source of truth for everything.** API keys → config; model names → config; schema dims → derived from active model; provenance → registry; cross-link surface → registry → plan dialog.
- **No auto-install on settings change.** Picking a provider in Settings is settings-only; the GUI surfaces a banner; the user clicks Install explicitly.
- **Blocking uninstall when active.** Cannot uninstall the active provider / cannot delete the active corpus's ontology — surfaces "switch first, then uninstall" instead.

---

## Deferred for future work

See `MEMORY.md` in `.claude/projects/` for the live deferred-items list. High-impact items:

- LangGraph SqliteSaver checkpointing + pause/cancel for in-flight ingest
- Structured progress events for the GUI (beyond log tailing)
- Backup/restore bulk ops (export_corpus / import_corpus)
- OpenTelemetry observability extra
- First-launch GUI wizard for provider picks
- Multi-user model decision (desktop install vs single-server multi-user)

---

## Pointers for onboarding

If you're picking this codebase up cold:

1. Read this file (15 minutes).
2. Read [config.py](src/knowledge_agent/config.py) top to bottom — every Settings field has a docstring explaining what it controls and why the default was chosen.
3. Trace one ingest end-to-end: open [ingestion/pipeline.py](src/knowledge_agent/ingestion/pipeline.py)'s `ingest_document`, follow the calls.
4. Trace one query end-to-end: open [graph.py](src/knowledge_agent/graph.py), follow the LangGraph nodes through [nodes.py](src/knowledge_agent/nodes.py).
5. For any layer you want to deep-dive, the per-module docstring is the canonical reference.

If you're returning to it after a break: the relevant memory files in `.claude/projects/c--Users-kjosi-ResearcherAgent/memory/` capture decisions in conversational form. MEMORY.md is the index.
