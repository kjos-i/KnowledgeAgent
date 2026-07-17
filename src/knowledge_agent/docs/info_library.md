# Library

The Library is where you build and manage **corpora** — each corpus is one
knowledge base. (For what's inside a corpus folder, see *Info → Files & storage*.)

Jump to: [Create New](#create) · [Ingest](#ingest) · [File types](#file-types) ·
[Metadata](#metadata) · [Sync](#sync)

<a id="create"></a>
## Create New

Make a new corpus. You choose a folder that *becomes* the corpus — its
`corpus.toml`, `lancedb/`, and `figures/` are written directly inside it — and the
**embedder** it will use. The embedder is fixed per corpus (the vector dimension is
pinned at the first ingest), so choose it before you ingest.

<a id="ingest"></a>
## Ingest

Add documents to the selected corpus. Choose a folder or files and start; each
document is chunked, embedded into LanceDB, and its entities and relationships are
extracted into Neo4j. On this sub-tab you can also:

- edit the corpus config (chunking, extractors, ontologies) before ingesting,
- watch progress and **Cancel** a folder run (it stops at the next document
  boundary),
- run bulk operations — including **Sync** (below) — to keep the index in step
  with your files.

<a id="file-types"></a>
## What file types can I ingest?

Ingest accepts anything the app has a parser for — a broad range:

- **Documents** — PDF, Word (`docx`), PowerPoint (`pptx`), Excel (`xlsx`).
- **Text & markup** — plain text, Markdown, HTML, AsciiDoc, LaTeX (`tex`),
  Quarto / R Markdown (`qmd`, `rmd`), and XML / JATS (`nxml`) for scientific
  articles.
- **Data** — CSV, TSV, and JSON / JSONL.
- **Images** — PNG, JPEG, TIFF, BMP, WebP (their text is extracted).
- **Audio & video** — MP3, WAV, M4A, MP4 and more are **transcribed** to text (a
  Whisper model downloads on first use), plus `vtt` caption files.
- **Code** — Python, JavaScript / TypeScript, Java, Go, Rust, C / C++.

Pick a file the app can't parse and it's skipped with a clear message — the exact
list is enforced at ingest time.

<a id="metadata"></a>
## Metadata

The selected corpus's info card plus a table of its indexed documents. Open a
document to view or edit its metadata. (Corpus *selection* is the global dropdown
at the top-right, so this sub-tab shows the active corpus — it isn't a picker.)

<a id="sync"></a>
## Sync — keeping the index in step

A document's identity is the **hash of its contents**, so the app can compare a
folder on disk against what's indexed and sort each file into one of five buckets:

| Bucket | Meaning | Action |
|---|---|---|
| **New** | on disk, not indexed | ingest it |
| **Unchanged** | same content, same path | nothing to do |
| **Moved** | same content, new path | just update the stored path — no re-ingest |
| **Edited** | same path, new content | delete the old, ingest the new |
| **Orphan** | indexed, no matching file on disk | offered for removal (with confirmation) |

Because identity is the content hash, Sync tolerates renames and moves within the
folder you sync against. See *Info → Files & storage* for where your original
documents live.
