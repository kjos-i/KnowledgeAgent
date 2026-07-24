"""Central registry of (i) help-icon text, keyed by a stable ID.

Each entry is an `InfoText` (a title plus up to three tier strings:
standard, beginner, technical). Call sites render one with
`info(app, "<key>")` instead of hand-writing the prose inline — that
keeps the GUI layout code readable and gathers every help string in one
place to review and tone-check as a body.

Keys are namespaced by location, `"<area>.<control>"` (e.g.
`"diagnostics.system_health"`, `"library.corpus.embedder"`), so the key
tells you where the icon lives. A `KeyError` from `info(app, key)` means
a typo'd or unregistered key — it fails loudly in dev rather than
rendering blank.

The widget, the three tiers (standard = blue `(i)`, beginner = green
cap, technical = orange wrench), and their Settings → App toggles live in
`info_icon.py`. This module is data only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui._widgets.info_icon import InfoText, info_point

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


INFO: dict[str, InfoText] = {
    "diagnostics.system_health": InfoText(
        title="System health",
        standard=(
            "Checks the app can reach everything it needs to answer a query: "
            "the Neo4j knowledge graph, the LanceDB store, and the API keys "
            "for your active providers. These only pass once a corpus exists "
            "— a corpus supplies the database connection. If none is created "
            "yet, make one in the Library tab and press Re-run."
        ),
        beginner=(
            "A quick self-test of the app's plumbing. Until you've created a "
            "corpus (your own searchable collection of documents) the app has "
            "no database to connect to, so instead of the check results it shows "
            "a reminder to make one. Create a corpus in the Library tab, press "
            "Re-run, and each working part lights up green."
        ),
        technical=(
            "Runs `health.system_status()`: a Neo4j `RETURN 1` round-trip, a "
            "LanceDB `list_tables()`, and a presence check on the active LLM + "
            "embedder API keys (it only tests a key is set, not that it works; "
            "local providers like Ollama auto-pass). The report is gated on "
            "`get_settings()`, which requires NEO4J_PASSWORD — put in the "
            "environment only by an active corpus that has a saved password. "
            "Without one, `get_settings()` raises and the panel shows the "
            "matching 'not set — create a corpus in Library' prompt instead of "
            "the four chips."
        ),
    ),
    "library.create_dataset": InfoText(
        title="Create a new corpus",
        standard=(
            "Registers a new corpus (a searchable collection with its own Neo4j "
            "graph and LanceDB store). Set up a database in Neo4j Desktop first, "
            "then fill in a unique name, its Neo4j connection (the URI + user "
            "defaults fit a standard local Neo4j Desktop; the password is the one "
            "you set for that database), and a folder to keep the corpus's files "
            "in. Create tests the connection, writes the corpus, and makes it the "
            "active one — then configure layers and add documents in the Ingest tab."
        ),
        beginner=(
            "A corpus is your own searchable collection of documents. Each one "
            "needs a place to store its files and a Neo4j database (the graph half "
            "of the search), so before this form open Neo4j Desktop and start a "
            "database. That gives you three things to copy in here: the connection "
            "address (URI), the user, and the password. The defaults usually fit a "
            "fresh local Neo4j Desktop, so often you only need to type the password. "
            "Give the corpus a short name, pick a folder for it, and press Create — "
            "the app checks it can reach Neo4j, then switches to your new (still "
            "empty) corpus. You add documents afterward in the Ingest tab; nothing "
            "is searched yet."
        ),
        technical=(
            "Create runs a real `RETURN 1` against the URI with the entered "
            "credentials — a wrong password or unreachable host aborts creation (no "
            "corpus.toml is written). On success it writes a seed `corpus.toml` "
            "(chunks layer on, all other L1-L10 layers off, xrefs none, all "
            "sub-labels allowed, no ontologies/extractor — plus the default embedder "
            "+ chunker settings), saves the password to the OS keyring as "
            "`neo4j-<name>` (never on disk), and registers the corpus as active. The "
            "corpus home holds `lancedb/`, `corpus.toml`, and `figures/`; create-mode "
            "makes it a `<name>` subfolder under the folder you pick, adopt-mode uses "
            "the folder itself. Neo4j's own data stays in Neo4j Desktop, outside the "
            "folder. Layers, ontologies, extractor and thresholds are set in the "
            "Ingest tab, not here."
        ),
    ),
    "library.corpus_folder": InfoText(
        title="Why one folder per corpus",
        standard=(
            "A corpus keeps its files together in one folder — its config "
            "(corpus.toml) and vector store (lancedb/), plus figures/ for "
            "multimodal ingests — so a whole corpus can be moved or backed up as "
            "a unit. (The Neo4j graph and your API keys live outside it: in Neo4j "
            "Desktop and the OS keyring.) Keeping your source documents in this "
            "folder too is optional — the app reads them wherever they are, in "
            "place, and never copies them — but doing so makes the corpus fully "
            "self-contained and makes Sync and Re-ingest reliable. Only corpus.toml "
            "(plus lancedb/ once you ingest) is truly required; the documents are "
            "up to you."
        ),
        beginner=(
            "The idea: everything a corpus needs sits in one folder, so it's easy "
            "to find, move, or back up. The app puts its own files there — a small "
            "settings file (corpus.toml) and the search index (lancedb/), plus a "
            "figures folder if your documents contain pictures. Two things stay "
            "outside: the Neo4j database (it lives in Neo4j Desktop) and your API "
            "keys (kept in your system's secure keychain). You can also keep your "
            "actual documents in this folder — that's optional, the app can read "
            "them from anywhere — but if you do, two things work more smoothly: "
            "Sync (noticing when you add, change, or remove documents) and Re-ingest "
            "(re-reading a document), because both need to find your original files. "
            "The part you can't skip is the corpus's own settings and index files — "
            "those ARE the corpus."
        ),
        technical=(
            "The folder holds the app-managed artifacts: `corpus.toml` (the config "
            "`load_corpus_config` requires — no corpus without it) and `lancedb/` "
            "(the vector store, created on first ingest), plus `figures/<doc_id>/` "
            "beside lancedb when `extract_figures` is on. (Eval runs add "
            "`eval_output/` + dataset JSON; the GUI adds `.ka_session.json` — all "
            "situational.) The Neo4j graph stays in Neo4j Desktop (only URI/user are "
            "stored; the password is in the OS keyring), so the folder isn't "
            "literally everything. Source files are never copied — ingest records "
            "each file's path and reads it in place from any folder you point at. "
            "Keeping them in a stable, corpus-owned location matters for Sync (it "
            "hashes the folder's files and compares stored paths -> "
            "NEW/MOVED/EDITED/ORPHAN; a moved-away or missing path shows as ORPHAN) "
            "and per-document Re-ingest (which re-reads the stored path and fails if "
            "it's gone). Re-embed and the backfills read chunk text from LanceDB, so "
            "they don't need the source file. Adopt-mode is the payoff: point 'Use "
            "the selected folder as the corpus home' at a folder that already holds "
            "your documents and the app writes lancedb/ + corpus.toml + figures/ "
            "alongside — a fully self-contained, portable corpus."
        ),
    ),
    "ingest.folder_actions": InfoText(
        title="Ingest folder, Re-ingest & Sync",
        standard=(
            "These three buttons all act on the folder you picked above, and each ends "
            "in a summary you confirm before anything runs. They differ in what they do "
            "to documents already in the corpus.\n\n"
            "Ingest folder adds only files whose content isn't already in the corpus; "
            "anything already there is skipped, so it's safe to run repeatedly. It only "
            "ever adds — it won't remove documents you deleted from the folder, and a "
            "file you edited in place is added as a second copy beside the old one.\n\n"
            "Re-ingest reprocesses every file in the folder from scratch, replacing the "
            "data of documents already present (no duplicates of unchanged files). Use "
            "it after changing the corpus's settings, to rebuild existing documents "
            "under them, or to fix a document that parsed badly. Like Ingest folder, it "
            "won't delete removed files or clean up edited-file copies.\n\n"
            "Sync makes the corpus match the folder exactly: it adds new files, updates "
            "moved files, replaces edited files, and DELETES any corpus document with no "
            "matching file in the folder. It treats the folder as the complete set for "
            "this corpus, so a partial or wrong folder makes it propose deleting "
            "everything else — deletion can't be undone, though the confirmation lists "
            "every document it will remove.\n\n"
            "Rule of thumb: Ingest folder to add, Re-ingest to rebuild, Sync to mirror "
            "a folder exactly."
        ),
        beginner=(
            "A corpus is your own searchable collection of documents. These three "
            "buttons all work on the folder you picked above and all show a summary to "
            "confirm first — they differ in how they treat documents you already "
            "added.\n\n"
            "Ingest folder adds documents that aren't in your corpus yet and skips the "
            "rest, so running it again is safe and never duplicates unchanged files. It "
            "only adds: it won't remove a document because you deleted its file, and if "
            "you change a file after adding it, you'll end up with two copies.\n\n"
            "Re-ingest rebuilds documents from your files. It goes over every file in "
            "the folder again from scratch and replaces what's stored rather than "
            "duplicating it. Reach for it when you changed a corpus setting and want "
            "documents you already added rebuilt with it, or to give a document that "
            "came out wrong a clean run.\n\n"
            "Sync makes your corpus match the folder exactly — think of the folder as "
            "the master copy. It adds new files, follows moved ones, replaces edited "
            "ones, and removes from the corpus any document whose file is no longer in "
            "the folder. That removal is permanent, so if you pick a folder missing "
            "some of your documents, Sync will offer to delete them (it shows the full "
            "list first). When unsure, use Ingest folder — it only adds.\n\n"
            "Short version: Ingest folder to add new documents, Re-ingest to rebuild "
            "existing ones, Sync to make the corpus mirror a folder."
        ),
        technical=(
            "All three act on the picked folder, share the plan→confirm dialog, a "
            "cooperative Cancel (stops at a document boundary), and a missing-key "
            "pre-flight (a cloud provider's key is required; local Ollama / HuggingFace "
            "need none). Each first saves the config editor to `corpus.toml` and runs "
            "`reconcile_*_to_config` — see the (i) on Ingestion settings. `doc_id` is a "
            "SHA-256 of the file's bytes (`ingestion/ids.py`).\n\n"
            "Ingest folder → `bulk_ops.add_plan` / `add_execute`: walks + hashes the "
            "folder, skips any file whose `doc_id` already has chunks in LanceDB "
            "(`get_chunks_by_doc_id`), ingests the rest via `pipeline.ingest_document`. "
            "Dedup is by content hash, not path; it never enumerates the index, never "
            "deletes, never re-paths. An edited-in-place file (new `doc_id`) is added "
            "while the old `doc_id` remains — a duplicate.\n\n"
            "Re-ingest → `bulk_ops.ingest_folder_plan` / `ingest_folder_execute`: runs "
            "the full pipeline on every file regardless of whether it's indexed; "
            "`ingest_document` deletes each doc across LanceDB + all Neo4j layers, then "
            "rewrites it (replace, not duplicate). No orphan handling. The Overwrite "
            "checkbox is `preserve_existing_labels` (default preserve): off keeps a "
            "doc's stored `(main, sub)` labels, on forces the picked labels — labels "
            "only, content and graph are rebuilt either way.\n\n"
            "Sync → `bulk_ops.sync_plan` / `sync_execute` over `sync_diff.classify`: "
            "hashes the folder AND enumerates the whole corpus index "
            "(`list_indexed_docs`), classifying NEW / UNCHANGED / MOVED (patch "
            "`source_path`) / EDITED (delete old + ingest new, labels carried) / ORPHAN. "
            "ORPHAN scope is the whole corpus, so a partial folder orphans everything "
            "else; each is deleted via `pipeline.delete_doc` (irreversible) and the "
            "confirm lists them (`orphan_display_names`). Path matching is "
            "case-sensitive."
        ),
    ),
    "ingest.single_file": InfoText(
        title="Ingest single file",
        standard=(
            "Ingest single file adds one document — the file shown in the File field "
            "above (pick it with that Browse button) — into the active corpus. It's "
            "the same ingest as the folder buttons, just for one file you choose "
            "directly, wherever it lives on disk. It asks 'Ingest <name>?' before "
            "running.\n\n"
            "When to use it: to add or refresh a single document without scanning a "
            "whole folder — one new file, or a re-run of one document that parsed "
            "badly. Ingesting the exact same file again just replaces its data (no "
            "duplicate).\n\n"
            "When not to use it: for many files at once (use Ingest folder); to remove "
            "documents or mirror a folder (use Sync); to rebuild everything after a "
            "settings change (use Re-ingest). And note it doesn't clean up edits — if "
            "you ingest a file you've changed since first adding it, the old version "
            "stays behind as a separate copy, so use Sync to truly replace an edited "
            "file."
        ),
        beginner=(
            "A corpus is your own searchable collection of documents. This button adds "
            "one document to it — the file in the File field just above (click its "
            "Browse to choose one). It's the single-file version of Ingest folder: same "
            "result, but for exactly one file you point at, wherever it is on your "
            "computer. It shows 'Ingest <name>?' and waits for you to confirm.\n\n"
            "Use it when you just want to add one document, or re-do one that came out "
            "wrong — no need to gather files into a folder first. Adding the very same "
            "file again simply refreshes it; you won't get two copies.\n\n"
            "Skip it when you have lots of files to add (Ingest folder does the whole "
            "folder at once), or when you want your corpus to match a folder — adding "
            "new files, removing deleted ones — which is Sync's job. One thing to "
            "watch: if you changed a file after first adding it and then ingest it "
            "here, the older version isn't removed and you'd end up with both. To "
            "replace an edited document cleanly, use Sync."
        ),
        technical=(
            "The button → `_start_action('Ingest single file')` → `_plan_single_file` "
            "→ `_execute_single_file`, which calls `pipeline.ingest_document(path, "
            "config, main, sub, preserve_existing_labels=not overwrite)` on the single "
            "path in the File field (its own picker, separate from the folder picker "
            "the other three share). No folder walk/hash: the confirm is a plain "
            "'Ingest <name>?' rather than a scanned plan summary, and the run isn't "
            "cancellable (the Cancel button is armed only for folder executes).\n\n"
            "It shares the rest of the ingest contract: `_start_action` saves the "
            "config editor first, the missing-key pre-flight runs, and "
            "`ingest_document` runs the corpus-wide `reconcile_*_to_config` — so "
            "ingesting one file with a layer just disabled still wipes that layer "
            "across the corpus. `doc_id` is the content hash: ingesting an identical "
            "file re-runs delete-then-write on the same `doc_id` (replace, no "
            "duplicate); an edited file (new `doc_id`) is written new while the old "
            "`doc_id` lingers — only Sync reconciles that. The path must satisfy "
            "`Path.is_file()` and have a supported extension (else `ingest_document` "
            "raises, surfaced as 'Ingest failed')."
        ),
    ),
    "ingest.progress": InfoText(
        title="Progress",
        standard=(
            "The Progress line is the live status of the current job; when idle it "
            "reads 'Empty'. Every ingest action opens with a spinner and '<action>: "
            "preparing…' while the ingestion engine loads (slow only on the first run "
            "of a session).\n\n"
            "A folder action (Ingest folder / Re-ingest / Sync) then shows '<action>: "
            "scanning <folder>…' while it hashes every file to build the plan. After "
            "you confirm, a progress bar replaces the spinner and the line counts "
            "documents — '<action>: 3 / 20 files' — ending in a summary like 'Ingest "
            "folder done: 18 succeeded, 2 failed' or 'Sync done: 5 new, 1 re-ingested, "
            "2 removed, 0 failed'.\n\n"
            "Ingest single file skips the scan and just shows 'Ingesting <name>…' then "
            "'Ingested <name>.' (no bar). Other messages: 'nothing to do' when the "
            "folder holds nothing to change; '<action> failed: …' on error; and if you "
            "press Cancel during a folder run, it finishes the current document then "
            "stops, prefixing the summary with 'Cancelled'."
        ),
        beginner=(
            "This line tells you what the app is doing right now. Before you start "
            "anything it just says 'Empty'.\n\n"
            "When you run one of the ingest buttons a small spinner appears and the "
            "line describes each step in words — first it gets ready (and, for a "
            "folder, reads through your files), then a bar fills up and the line counts "
            "documents as it goes, like '3 / 20 files', so you can see how far along it "
            "is. When it finishes, the line shows a short summary of what happened — how "
            "many documents were added, and how many failed if any.\n\n"
            "Some messages you might see: 'Ingesting <name>…' then 'Ingested <name>.' "
            "for a single file; 'nothing to do' if there was nothing new to process; "
            "or, if you pressed Cancel, it finishes the document it's on and then stops. "
            "While the spinner is turning it's still working — let it finish."
        ),
        technical=(
            "Three controls share this section — an indeterminate spinner "
            "(`progress_ring`), the status `Text`, and a `Cancel` button — with a "
            "determinate `progress_bar` below.\n\n"
            "Every action opens with '<action>: preparing…' (spinner) during the "
            "off-thread backend import. Folder actions (`_run_action` → "
            "`_execute_action`) then show '<action>: scanning <folder>…' during "
            "`_walk_and_hash`; an empty plan short-circuits to '<action>: nothing to do "
            "in <folder>.'. After confirm: '<action>: working…', then `_begin_progress` "
            "hides the spinner and shows the bar, and `_on_ingest_progress` sets it to "
            "done/total while writing '<action>: {done} / {total} files' per document "
            "(for Sync, per item across the new/moved/edited/orphan buckets). The final "
            "line is `_fmt_ingest_result` ('<action> done: N succeeded, M failed' [+ "
            "'First failure: <name> — <err>']) or `_fmt_sync_result` ('Sync done: X "
            "new, Y re-ingested, Z removed, W failed'); a raised error shows '<action> "
            "failed: <exc>'.\n\n"
            "The bar appears only for folder executes. Ingest single file runs with the "
            "spinner only ('Ingesting <name>…' → 'Ingested <name>.'), as do the Bulk "
            "operations ('<op>: working…' → '<op> done: …' / 'failed: …') — all on this "
            "same line. Cancel is armed only during a folder execute: pressing it writes "
            "'Cancelling — will stop after the current file…' and prefixes the result "
            "with 'Cancelled —' (a cooperative stop at the next document boundary)."
        ),
    ),
    "ingest.settings_reconcile": InfoText(
        title="How these settings are saved and applied",
        standard=(
            "The panel on the right holds this corpus's ingestion settings — its "
            "document labels, which knowledge-graph layers are on, the entity "
            "extractor, ontologies, and thresholds. Editing here doesn't take effect on "
            "its own: a dot marks unsaved changes, and Discard reverts them.\n\n"
            "The changes are saved and applied the next time you ingest. Whenever you "
            "click Ingest folder, Re-ingest, or Sync, the app first writes these "
            "settings to the corpus, then brings the corpus in line with them before "
            "ingesting — and that means any layer you turn off (or narrow) has its "
            "existing data removed across the whole corpus, not just for the files "
            "you're ingesting now. Turning a layer on is the opposite: new documents get "
            "it immediately, but documents already in the corpus gain it only when you "
            "Re-ingest them or run that layer's backfill under Bulk operations."
        ),
        beginner=(
            "These settings on the right control how documents are processed for this "
            "corpus — what type they're labelled, how much of the knowledge graph is "
            "built, and which tools are used. Changing one here doesn't do anything by "
            "itself: a small dot shows you have unsaved changes, and Discard undoes "
            "them.\n\n"
            "Your changes take effect the next time you add documents. When you click "
            "Ingest folder, Re-ingest, or Sync, the app saves these settings first and "
            "then brings the whole corpus in line with them before ingesting. The part "
            "to be aware of: if you switch a graph layer off, the data that layer had "
            "built is removed for every document in the corpus — not only the ones "
            "you're adding right now. Switching a layer on works the other way: "
            "documents you add from now on include it, but documents already in the "
            "corpus only get it after you Re-ingest them (or run that layer's backfill "
            "in the Bulk operations section)."
        ),
        technical=(
            "The right pane is `CorpusConfigEditor`; edits live in memory with a dirty "
            "indicator until saved. Every ingest action calls "
            "`try_save_and_get_error()` first, persisting the draft to the corpus's "
            "`corpus.toml`, then `pipeline.ingest_document` runs "
            "`reconcile_ontologies/entities/triples/cross_doc/cross_doc_xrefs_to_config` "
            "(`kg/reconcile.py`) before writing the doc.\n\n"
            "Reconcile is destructive in one direction: it wipes corpus-wide any layer "
            "or selection the new config makes more restrictive — a disabled layer, a "
            "narrowed `entity_types` set — and is fail-hard (a Cypher error propagates "
            "rather than half-reconciling). It does not populate newly-enabled layers; "
            "those reach existing docs only via Re-ingest or the matching "
            "`bulk_backfill_*`. Because reconcile runs at the top of each "
            "`ingest_document` call, it fires the moment you ingest at least one "
            "document — so a folder action scoped to a few new files can still trigger a "
            "whole-corpus layer wipe."
        ),
    ),
}


def info(app: GuiApp, key: str) -> ft.Control:
    """Render the (i) help-icon row for a registry key.

    Looks up `INFO[key]` and delegates to `info_point`. A `KeyError` means
    the key isn't registered (typo / missing entry) — surfaced loudly so it
    is caught in dev, never silently blank.
    """
    return info_point(app, INFO[key])
