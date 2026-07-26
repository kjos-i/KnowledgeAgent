# Library

The Library is where you build and manage **corpora**. Each corpus is one knowledge
base. (For what's inside a corpus folder, see *Info → Files & storage*.)

Jump to: [Create New](#create) · [Ingest](#ingest) · [File types](#file-types) ·
[Metadata](#metadata) · [Sync](#sync)

<a id="create"></a>
## Create New

Register a new corpus. This is a storage-only step: you give it a name, the Neo4j
connection, and a folder, then pick how that folder is used:

- **Create a new folder named after the corpus** (the default): the folder you pick
  is a *parent*, and a subfolder named after the corpus is made inside it. That
  subfolder is the corpus home and holds `corpus.toml`, `lancedb/`, and `figures/`.
- **Use the selected folder as the corpus home**: the folder you pick *is* the
  corpus, and those same files are written directly inside it (useful for adopting a
  folder you already made).

You do **not** choose the embedder here. A new corpus starts with a default
embedder; you set or change it in the **Ingest** tab's config editor before your
first ingest. That matters because the embedder is fixed per corpus: the vector
dimension is pinned when the first document is ingested. Switching later to an
embedder with a *different* dimension means re-ingesting the corpus (a same-size
model swap can be refreshed with **Re-embed** instead).

<a id="ingest"></a>
## Ingest

Add documents to the selected corpus. Choose a folder or files and start. Every
document is chunked, embedded into LanceDB, and recorded as chunk nodes in Neo4j.
Extracting **entities** and **relationships** into Neo4j is optional: those layers
are off by default (they cost extra time and LLM calls) and you turn them on in the
corpus config. On this sub-tab you can also:

- edit the corpus config (chunking, extractors, ontologies, embedder, and which
  knowledge-graph layers to run) before ingesting,
- watch progress and **Cancel** a folder run (it stops at the next document
  boundary, so the file being processed finishes first),
- run bulk operations, including **Sync** (below), to keep the index in step with
  your files.

<a id="file-types"></a>
## What file types can I ingest?

Ingest accepts anything the app has a parser for, a broad range:

- **Documents:** PDF, Word (`docx`), PowerPoint (`pptx`), Excel (`xlsx`).
- **Text & markup:** plain text, Markdown, HTML, AsciiDoc, LaTeX (`tex`),
  Quarto / R Markdown (`qmd`, `rmd`), and XML / JATS (`nxml`) for scientific
  articles.
- **Data:** CSV, TSV, and JSON / JSONL.
- **Images:** PNG, JPEG, TIFF, BMP, WebP (their text is extracted by OCR).
- **Audio & video:** MP3, WAV, M4A, MP4 and more are **transcribed** to text, plus
  `vtt` caption files. Transcription needs the audio/video add-on installed (it
  pulls a Whisper model on first use and relies on ffmpeg). Video is transcribed
  from its audio track only; on-screen visuals are not captured.
- **Code:** Python, JavaScript / TypeScript, Java, Go, Rust, C / C++ (needs the
  code add-on installed).

If you pick a **single file** the app can't parse, the ingest stops with a clear
message naming the unsupported type. In a **folder** run, unsupported files are
skipped silently (they never enter the plan). Either way, the exact supported list
is enforced at ingest time.

<a id="metadata"></a>
## Metadata

The selected corpus's info card, plus a browsable list of its indexed documents
(one card each). Open a document to view or edit its metadata. Corpus *selection* is
the global dropdown at the top-right, so this sub-tab always shows the active
corpus. It is not a picker.

<a id="sync"></a>
## Sync: keeping the index in step

A document's identity is the **SHA-256 hash of its contents**, so the app can
compare a folder on disk against what's indexed and sort each file into one of five
buckets:

| Bucket | Meaning | Action |
|---|---|---|
| **New** | on disk, not indexed | ingest it |
| **Unchanged** | same content, same path | nothing to do |
| **Moved** | same content, new path | update the stored path only (no re-ingest) |
| **Edited** | same path, new content | delete the old, ingest the new |
| **Orphan** | indexed, no matching file on disk | offered for removal (with confirmation) |

Because identity is the content hash, Sync tolerates renames and moves within the
folder you sync against. See *Info → Files & storage* for where your original
documents live.
