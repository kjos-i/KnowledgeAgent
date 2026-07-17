# Knowledge Agent

A desktop app for building a searchable **knowledge base** from your own documents
and asking questions of it in plain language. It combines **hybrid retrieval**
(vector + keyword search) with a **knowledge graph**, and answers with citations
back to the sources.

Knowledge Agent is **domain-agnostic** — it ships a broad tag taxonomy and adapts
to whatever documents you point it at, rather than being tied to one field.

## How it works

Two complementary stores work together:

- **LanceDB** — your documents are chunked and embedded for vector + BM25 hybrid
  search, so a question finds passages that are *semantically* similar even when
  the wording differs.
- **Neo4j** — a layered knowledge graph (documents, chunks, entities, ontology
  canonicals, typed relations, and cross-document links, plus optional OpenAlex
  bibliographic layers) answers questions that depend on *connections* across
  documents.

A **LangGraph** agent picks a retrieval strategy per question, retrieves from one
or both stores, and a language model synthesises a cited answer. The UI is a
**Flet** desktop app; there is also a headless CLI.

## Features

- **Build corpora** from many formats — PDF, DOCX, PPTX, XLSX, Markdown, HTML,
  LaTeX, CSV, JSON, images, audio/video, and source code.
- **Ask questions** and get cited, synthesised answers.
- **Tune retrieval** per query — result count, hybrid vs. vector-only, re-ranking,
  or a direct read-only Cypher query against the graph.
- **Layered knowledge graph** — chunks, entity extraction (LLM or GLiNER /
  HunFlair2), ontology linking across 18 ontologies, typed relations, and
  cross-document links — each layer independently toggleable per corpus.
- **Bring your own models** — Anthropic, OpenAI, Google, Voyage, Ollama (local),
  and Hugging Face — installed on demand; no provider is bundled.
- **Measure quality** — a built-in evaluation harness with registry-driven
  metrics, an optional LLM-as-judge panel, and JSON / CSV / SQLite reports.

## Install

Requires **Python 3.13+** and a running **Neo4j** instance (Neo4j Desktop or a
server). LanceDB is embedded — no setup.

    pip install -e ".[dev]"

Model providers and heavy extractors are **opt-in extras** — install only what a
corpus needs, e.g.:

    pip install -e ".[llm-anthropic,embed-voyage]"     # a provider pair
    pip install -e ".[entities-gliner,parsers-asr]"    # NER + audio/video

## Run

- **GUI:** `ka-gui`
- **CLI:** `ka ingest <folder>`, `ka query "..."`, `ka health`, `ka eval ...`

Configuration (database URIs, API keys, retrieval defaults) lives in `.env` or is
set through the GUI; per-corpus settings live in each corpus's `corpus.toml`.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — how the system is organised, the
  ingest/query data flows, and the design decisions.
- **[CLAUDE.md](CLAUDE.md)** — orientation for AI coding assistants working in the
  repository.
- In-app **Info** tabs cover the storage layout, the LanceDB + Neo4j schemas, and
  how to use each screen.

## Testing

    pytest                     # unit — fast, no real services
    pytest -m integration      # real Neo4j test instance + LanceDB (opt-in)
    pytest -m e2e              # launches the Flet app (opt-in)

## License & citation

Released under the **MIT License** — see [LICENSE](LICENSE). Free to use, modify,
and share, as long as the copyright notice is kept intact.

If you use Knowledge Agent in your work, a citation is appreciated:

> Kjos, I. (2026). *Knowledge Agent* (Version 0.1.0) [Computer software].
> ORCID: [0000-0002-9166-3074](https://orcid.org/0000-0002-9166-3074)
