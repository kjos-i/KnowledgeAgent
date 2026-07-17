# Files & storage

Everything a corpus needs lives inside **one folder** — the corpus folder *is* the
project. It's self-contained and portable: copy the folder and you've copied the
whole knowledge base.

Jump to: [Inside the folder](#inside) · [Your documents](#documents) ·
[Vector store schema](#schema) · [Graph schema](#graph-schema) · [The idea](#idea) ·
[Other saved files](#saved) · [Installs downloads](#installs)

<a id="inside"></a>
## Inside a corpus folder

The corpus folder holds:

| Path | What it is |
|---|---|
| `corpus.toml` | config — the corpus's name, the embedder it was built with (vector dimension pinned at ingest), and other settings |
| `lancedb/` | the vector store (chunks + embeddings), rebuilt from your documents on ingest — [row schema below](#schema) |
| `figures/` | figure images pulled from documents, under `figures/<doc_id>/` |
| `.ka_session.json` | GUI session state (gitignored) |
| `<dataset>.json` | evaluation datasets you create |
| `eval_output/` | evaluation results — `eval_ledger.db` (run history), plus per-run `eval_report_<ts>_<dataset>.json` and `eval_summary_<ts>_<dataset>.csv` |

<a id="documents"></a>
## Your original documents

Your source documents are **referenced in place, not copied** into the corpus.
Ingest records each document's **content hash** (its identity) and its **path on
disk** — so the files can live anywhere and don't have to be in the corpus folder,
or even all in one folder.

Keeping them in a stable location does help, because **Sync** compares a folder on
disk against what's indexed and uses the content hash to classify each file:

- **moved** — same content, new path → Sync just updates the stored path, no
  re-ingest
- **edited** — same path, new content → Sync re-ingests only that file
- **new** — not yet indexed → ingested
- **gone** — indexed but no longer on disk → flagged as an orphan you can remove

Because identity is the content hash, Sync tolerates renames and moves *within the
folder you sync against*. What it can't do is see files you've scattered or deleted
— so a stable home keeps Sync and re-ingest easy. Putting the documents **inside
the corpus folder** is optional but has a bonus: the corpus becomes fully
self-contained, so you can hand the whole folder to someone else.

<a id="schema"></a>
## The vector store (LanceDB) schema

Ingest writes one **chunk** row per passage into the LanceDB `chunks` table:

| Column | Type | Req | Notes |
|---|---|---|---|
| `chunk_id` | string | ✓ | unique id of this chunk (joins to the graph) |
| `doc_id` | string | ✓ | the document's content hash — its identity |
| `chunk_index` | int32 | ✓ | position of the chunk within the document |
| `section` | string | | source section, when known |
| `page` | int32 | | source page, when known |
| `char_start` | int32 | | start offset in the source text |
| `char_end` | int32 | | end offset in the source text |
| `content_type` | string | ✓ | `text`, `figure`, or `table` |
| `image_ref` | string | | figure image path (figure chunks) |
| `text` | string | ✓ | the chunk text — BM25 / keyword search |
| `embedding` | vector | ✓ | float32; length = your embedder's dimension (vector search) |
| `main_label` | string | ✓ | the chunk's KG type |
| `sub_label` | string | | KG subtype (optional) |
| `doi` | string | | document DOI |
| `openalex_id` | string | | OpenAlex id |
| `title` | string | | document title |
| `year` | int32 | | publication year |
| `authors_display` | string | | authors, for display |
| `venue` | string | | publication venue |
| `source_url` | string | | source URL |
| `metadata_status` | string | ✓ | `enriched`, `pending`, `baseline`, or `manual` |
| `language` | string | | document language |
| `source_path` | string | | file path at last ingest — drives Sync's *moved* / *edited* detection |
| `ingested_at` | timestamp | ✓ | when the row was written |

**Req** = required (non-null). The `doi`…`language` columns are a denormalised
metadata cache, so a hit displays without a Neo4j round-trip. The embedder — and so
the vector dimension — is fixed per corpus, pinned at the first ingest.

<a id="graph-schema"></a>
## The knowledge graph (Neo4j) schema

The graph's shape depends on which **layers** a corpus enables (citations,
entities, ontologies, cross-document links), so the full schema is
config-dependent. The **core** labels below are the stable backbone; optional
layers add more. When you use Direct Cypher, the agent is handed the live schema
automatically — you don't have to memorise it.

**Node labels**

| Label | Key properties | What it is |
|---|---|---|
| `:Document` | `doc_id` (SHA-256 — joins to LanceDB) | a primary written work (paper, book, notes) |
| `:Artifact` | `doc_id` | supporting / derived content (dataset, code, figure, audio, video) |
| `:Chunk` | `chunk_id` (joins to LanceDB), `doc_id`, `chunk_index`, `section`, `page`, `content_type` | one chunk — identity + structure only; text + vector live in LanceDB |
| `:Entity` | `key` (lowercased match key), `entity_type`, `canonicalised` | an extracted entity mention |

Optional labels appear when their layer is on: `:Author` / `:Venue` / `:Topic` (the
OpenAlex citation graph), and `:OntologyTerm` with a subtype such as `:MeSHTerm` or
`:GOTerm` (18 ontologies — MeSH, GO, HPO, UBERON, MONDO, ChEBI, …), each carrying
`id`, `label`, `synonyms`, `definition`.

**Relationships**

| Pattern | What it means |
|---|---|
| `(:Chunk)-[:PART_OF]->(:Document)` | a chunk belongs to its parent (may be an `:Artifact`) |
| `(:Chunk)-[:MENTIONS]->(:Entity)` | the chunk's text named this entity |
| `(:Document)-[:CITES]->(:Document)` | citation (OpenAlex layer) |
| `(:Author)-[:AUTHORED]->(:Document)` | authorship — props `position`, `is_corresponding` |
| `(:Document)-[:PUBLISHED_IN]->(:Venue)` | publication venue |
| `(:Entity)-[:CANONICAL_TO]->(:OntologyTerm)` | entity linked to a canonical ontology term |
| `(:OntologyTerm)-[:<X>_IS_A]->(:OntologyTerm)` | ontology hierarchy — variable-depth with `*` |
| `(:Entity)-[<predicate>]->(:Entity)` | typed relation from text — 15 predicates (e.g. `:INHIBITS`, `:REGULATES`) |
| `(:Document)-[:RELATED_TO]-(:Document)` | cross-document link — shares ≥ 2 entities; undirected |

`doc_id` and `chunk_id` are the universal join keys back to the LanceDB store.

<a id="idea"></a>
## The idea

The corpus folder is meant to be **committed and shared** — with one exception:
`lancedb/` is *regenerable* from your documents, so it's the natural thing to
gitignore. Commit `corpus.toml` (and your source documents / gold sets); leave the
vector store out and rebuild it with a re-ingest. That keeps a corpus reproducible
without checking a large binary index into version control.

The Neo4j knowledge graph lives in your Neo4j database (not in this folder) — each
corpus can point at its own database. See *Getting Started* for connecting it.

<a id="saved"></a>
## Other saved files

- **Saved answers / chats** — written to the folders you set in **Settings → Save
  results** and **Save chat** (each has its own default folder and formats).
- **API keys** — stored in your operating system's keyring, never in the corpus
  folder.
- **Logs** — a rotating `kagent.log` in your OS log folder (on Windows,
  `%LOCALAPPDATA%\KnowledgeAgent\Logs\`; override with the `KAGENT_LOG_DIR`
  environment variable). The **Log** tab shows the live path and lets you read it
  in-app.

<a id="installs"></a>
## Installed models & downloads (Installs tab)

The **Installs** tab manages pieces that live *outside* any corpus folder and are
**shared across all corpora**:

- **Adapters** (LLM providers, embedders, parsers, entity extractors) — pip
  packages installed into your Python environment.
- **Model weights** — the Hugging Face embedder, entity-extractor weights, and
  Docling's Whisper model land in the shared **Hugging Face Hub cache**.
- **Ollama models** — pulled into Ollama's own model store.
- **Ontology source files** — downloaded to the **ontology downloads directory**
  you set in Installs (shared across corpora; blank = the backend default).
