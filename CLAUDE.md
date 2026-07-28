# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.
This is the fast-orientation map: what the project is, how to run and test it,
where each concern lives, and the conventions and gotchas that are not obvious
from any single file. **It is a living document — verify it against the code and
keep it current.**

## What this is

KnowledgeAgent is a **desktop research-knowledge agent**. It ingests documents
in many formats and stores them in two complementary backends:

- **LanceDB** — embedded vector + BM25 hybrid search over chunked document text
  (answers content questions).
- **Neo4j** — a layered knowledge graph (papers, authors, citations, venues,
  entities, ontology canonicals, typed relations, cross-document links) that
  answers structural / cross-document questions.

A **LangGraph** agent answers questions by retrieving across both stores; the UI
is a **Flet** desktop app. It is **domain-general** — it supports all content
domains, not just biomedical. Many of the bundled ontologies happen to be
biomedical, but that is a content choice, never assume the app is biomed-only.

- **Read `ARCHITECTURE.md` first.** It is the current, authoritative one-page map
  of subsystems, the ingest/query data flows, and the locked design decisions.
  This file does not duplicate it — it adds run/test instructions, a
  where-things-live index, and assistant-facing gotchas.
- Per-module top-of-file docstrings are the canonical per-feature reference.
- ⚠️ `README.md` is an outdated stub ("early scaffolding", "research articles",
  L1–L2 only). Trust `ARCHITECTURE.md` and the code until the README is
  refreshed.

## Running it

Requires **Python 3.13+**. Install editable with the dev extra:

    pip install -e ".[dev]"

LLM/embedder providers and heavy extractors are **opt-in extras** — no provider
is bundled. Install only what a corpus uses, e.g.:

    pip install -e ".[llm-anthropic,embed-voyage]"     # one provider pair
    pip install -e ".[entities-gliner,parsers-asr]"    # NER + audio/video

Extras are grouped in `pyproject.toml`: `llm-*`, `embed-*`, `entities-*`,
`parsers-*`. See ARCHITECTURE.md "Provider model" and the five `*_lifecycle.py`
modules for how installs are planned/executed from the GUI.

- **GUI:** `ka-gui`  (or `python -m knowledge_agent.gui`)
- **CLI:** `ka ingest <folder>`, `ka query "..."`, `ka health`, `ka eval ...`
  (or `python -m knowledge_agent`)

### External prerequisites (not pip-installable)

- **Neo4j must be running** (Neo4j Desktop or a server); the app connects over
  bolt using settings in `.env`. LanceDB is embedded — no setup.
- **Ollama** (optional) is a separate binary from https://ollama.com for local
  models; weights come via `ollama pull`.
- LLM/embedder API keys go in `.env` or the OS keyring (GUI Keys tab). Keys are
  validated **lazily** (first call), never at startup — a pure-OpenAI user never
  needs an Anthropic/Voyage key.

## Testing & linting

    pytest                     # unit only — fast, no real services
    pytest -m integration      # real Neo4j test instance + LanceDB (opt-in)
    pytest -m e2e              # launches the Flet app (opt-in)

- **Default `pytest` skips integration + e2e** (`pyproject.toml` addopts). A
  green default run therefore says nothing about integration-only modules —
  notably `kg/openalex_writes.py` and `kg/chunk_writes.py`, whose write logic is
  covered *only* by integration tests today.
- 🔒 **Tests must never touch a real instance.** The suite only uses `.env.test`
  (a separate Neo4j instance + separate `lancedb_test` folder) and `tmp_path`.
  Isolate `os.environ`; mock external writes. A wrong-instance cross-connect
  fails auth by design rather than corrupting real data.
- **Ruff runs on every commit** (pre-commit): `ruff check` + `ruff format`.
  Config in `pyproject.toml` (`line-length = 100`). Run `ruff check --fix` and
  `ruff format` before committing; do not add `# noqa` for codes that are not
  enabled (RUF100 will flag it).
- Async tests: `asyncio_mode = "auto"` — write `async def test_*`, no decorator.
- **Full verify + audit checklist:** `CHECKS.md` (repo root) groups every check
  (ruff, pre-commit, unit / integration / e2e, coverage, smoke scripts) and the
  manual audit types, with one line on what each looks for.

## Where things live (the "user asks about X → go here" index)

| Topic | Authoritative file(s) |
|---|---|
| Every runtime setting / knob | `config.py` (Pydantic Settings — SSOT) |
| Per-corpus layer toggles / extractor / ontologies | `corpus_config.py` + each corpus's `corpus.toml` |
| Agent loop / query topology (retrieval modes) | `graph.py` + `nodes.py`; per-run state in `state.py` |
| Public data shapes (Mention, KGHit, RetrievedChunk, AgentAnswer) | `models.py` |
| Ingest pipeline (parse → embed → write) | `ingestion/pipeline.py` |
| Multi-doc operations (delete, backfill, sync, …) | `ingestion/bulk_ops.py` |
| Parsing / format dispatch | `ingestion/parse.py` + `ingestion/parsers/` |
| doc_id / chunk_id; sync diff | `ingestion/ids.py`; `ingestion/sync_diff.py` (doc_id = SHA-256 content hash) |
| LanceDB layer + chunks table schema | `search/client.py`; `search/schema.py` (schema SSOT) |
| Neo4j driver / write+read facade | `kg/client.py` |
| Neo4j labels / edges / indexes | `kg/schema.py` |
| Neo4j schema → Cypher-gen prompt; Cypher safety | `kg/schema_as_prompt.py`; `kg/cypher_safety.py` (read-only + LIMIT wrap) |
| KG write layers L1–L10 | `kg/openalex_writes.py` (L1–4), `kg/chunk_writes.py` (L5), `kg/entity_writes.py` (L6), `kg/ontologies/*_writes.py` + `helpers.py` + `xrefs.py` (L7), `kg/triples_writes.py` (L8), `kg/cross_doc_writes.py` (L9), `kg/cross_doc_xrefs_writes.py` (L10) |
| NER extractors (L6) | `entity_extractors/` (base, dispatcher, llm, gliner, gliner_biomed, hunflair2) |
| LLM / embedder dispatch | `llm_factory.py` / `embedder_factory.py` (lazy key validation) |
| Install/uninstall of providers, extractors, parsers, ontologies | the five `*_lifecycle.py` (plan/execute pattern) |
| Typed errors | `errors.py` (`ErrorDetail`; leaves raise, orchestrators catch) |
| Logging + resolved log path | `logging_setup.py` (`log_file_path()`), `logging_crash.py`, `logging_ring_buffer.py` |
| Evaluation harness | `evaluation/` (`runner.py`, `engine.py`, `metrics.py`, `judge.py`, `registry.py`, `report.py`, `ledger.py`, `suite_gen.py`) |
| Save/export answers (md/txt/docx/json) | `artifacts.py` |
| GUI shell + tabs | `gui/app.py` (entry); tabs under `gui/tabs/`, `gui/library/`, `gui/evaluation/`, `gui/settings/`; result panel `gui/right_panel.py`; chat `gui/chat_panel.py` + `gui/chat_router.py` |
| In-app help / Info-tab content | `src/knowledge_agent/docs/*.md` (rendered by `gui/views/info_doc.py`) |

## Conventions & invariants

- **Single source of truth** — the repo's strongest rule. Never hardcode a value
  in two places: settings → `config.py`; data shapes → `models.py`; schema dims
  → derived from the active model; registries drive the plan dialogs.
- **Layered KG, per-corpus toggles.** L1–L10 are independently on/off per corpus;
  token-costly layers (L6/L8) default off.
- **Provider symmetry.** All 4 LLM + 4 embedder providers are extras; none is
  privileged. No auto-install on a settings change — the GUI surfaces a banner
  and the user installs explicitly. Cannot uninstall the active provider or
  delete the active corpus's ontology (switch first).
- **Typed errors at boundaries.** Leaves raise; orchestrators catch where a
  result dataclass is built and populate `*_error: ErrorDetail | None`. No
  swallow-and-return-False.
- **Async end-to-end.** The whole I/O surface is `async def` on native async
  drivers (Neo4j `AsyncGraphDatabase`, `lancedb.connect_async`). Only entry
  points wrap `asyncio.run`; don't add sync wrappers in the hot path.
- **Lifecycle pattern** for everything installable: an `XPlan`/`XResult` pair
  with a `summary` the GUI renders as a confirm dialog. Follow it for new
  installables.
- **Every backend setting is visible in the GUI** (editable or read-only) — never
  leave a setting code-only.

## Flet 0.85 GUI gotchas (these will bite)

- **Monospaced/code text renders unreliably** on Windows — ASCII column diagrams
  and trees drift. Use the native table renderer in `gui/_markdown.py`
  (`render_markdown`) for schemas/flows, not fenced code blocks.
- **`ft.Border.all(...)`** — capital `B` (not `ft.border.all`).
- **Module-level style objects** (e.g. the markdown stylesheet) are built at
  import; changing them needs an app **restart**, not a tab Refresh.
- `ft.Markdown` has **no table-border/column-width control** — that is why the
  custom Row-based table renderer exists.
- Defer view work: no fetches in a view's `_create_controls`, do them on
  first-show; keep heavy imports local. Reuse the constants in `gui/_styles.py`.

## Known benign issue (don't "fix" it)

- `pip check` reports one `huggingface-hub` / `click` mismatch. It is **cosmetic**
  — `click 8.3.3` is installed to satisfy deepeval; HF's newer-click need is only
  for its CLI, which we never call. Documented in `pyproject.toml`.

## Project status & direction (as of 2026-07-17)

Feature-complete across Search / Library / Evaluation / Log / Info, the
evaluation harness, and swappable providers; approaching a first version.
Planned remaining work, in order:

1. **Code review / audit** and an **app-wide info-icon help** buildout (help text
   spanning novice → technical), verifying and updating *this file* along the way.
2. **Packaging (last):** a desktop installer for non-technical users plus an
   install guide (Flet was chosen partly to enable desktop packaging). Neo4j and
   Ollama stay external dependencies the guide will walk users through.

Deferred / parked items live in `MEMORY.md` under
`.claude/projects/c--Users-kjosi-ResearcherAgent/memory/` (the returning-author
notes referenced by ARCHITECTURE.md).
