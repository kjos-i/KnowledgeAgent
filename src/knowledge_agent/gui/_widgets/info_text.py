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
    "ingest.ingest_folder": InfoText(
        title="Ingest folder",
        standard=(
            "Ingest folder adds new documents from the selected folder into the "
            "active corpus. It scans every supported file in the folder and its "
            "sub-folders and ingests only the ones whose content isn't already in "
            "the corpus; anything already there is skipped. That makes it safe to "
            "run repeatedly — you won't get duplicates of unchanged files.\n\n"
            "It only ever adds. It doesn't remove documents you've deleted from the "
            "folder, and it doesn't detect a file you've edited in place — an edited "
            "file counts as new content, so it's added as a second copy beside the "
            "old one. To update edited files or remove deleted ones, use Sync. "
            "You'll review a summary and confirm before anything runs."
        ),
        beginner=(
            "A corpus is your own searchable collection of documents — the app reads "
            "your files, splits them into passages, and stores them so you can search "
            "and ask questions. Ingest folder is how you put documents in.\n\n"
            "Point it at a folder and it goes through every supported file there "
            "(sub-folders included) and adds the ones not already in your corpus. Run "
            "it again later and files already added are skipped, so you'll never add "
            "the same document twice — running it more than once is safe.\n\n"
            "The thing to know: this button only adds. Delete a file from the folder "
            "and it still stays in your corpus. Change a file after adding it and this "
            "button treats the new version as a brand-new document, leaving you with "
            "two copies. When you want the corpus to match the folder exactly — add "
            "new, update changed, remove deleted — use Sync instead. Either way, "
            "you'll see what's about to happen and confirm first."
        ),
        technical=(
            "Maps to `bulk_ops.add_plan` / `add_execute` — a purely additive sync. "
            "`_walk_and_hash` walks the folder recursively and computes each supported "
            "file's `doc_id` (`compute_doc_id` in `ingestion/ids.py` — a SHA-256 of the "
            "file's bytes); any file whose `doc_id` already has chunks in LanceDB "
            "(`get_chunks_by_doc_id`) is skipped, and the rest run through "
            "`pipeline.ingest_document`. Dedup is by content hash, not path — a moved or "
            "renamed copy of an indexed file is still recognised and skipped, and its "
            "stored path is not updated.\n\n"
            "It never enumerates the whole index and never deletes. Moved files aren't "
            "re-pathed, removed files are left in place, and an edited-in-place file "
            "(new bytes → new `doc_id`) is ingested as a new document while the old "
            "`doc_id` remains — a duplicate. Only Sync reconciles those.\n\n"
            "Shared with all three buttons (see the (i) on Ingestion settings): the "
            "click first saves the config editor to `corpus.toml` and runs "
            "`reconcile_*_to_config`, so disabling a layer there wipes it corpus-wide on "
            "the next ingested file. A missing key for a cloud provider the corpus uses "
            "aborts with a dialog first; local providers (Ollama / HuggingFace) need "
            "none. Cancel stops after the current file, keeping what's done."
        ),
    ),
    "ingest.reingest": InfoText(
        title="Re-ingest",
        standard=(
            "Re-ingest reprocesses every supported file in the selected folder from "
            "scratch, whether or not it's already in the corpus. For files already "
            "present it deletes their current data and writes it again — it replaces, "
            "so unchanged files aren't duplicated.\n\n"
            "Reach for it when you've changed the corpus's settings (turned on a layer, "
            "changed the extractor, adjusted chunking or embedding) and want documents "
            "you already have rebuilt under the new settings — or when a document parsed "
            "badly and you want a clean run. Like Ingest folder it only touches files in "
            "the folder: it won't delete documents you've removed, and a file edited in "
            "place is written as a new copy while the old one stays. It keeps each "
            "document's type label unless you tick Overwrite. You'll confirm a summary "
            "first."
        ),
        beginner=(
            "Re-ingest rebuilds documents from your files. Where Ingest folder skips "
            "anything already added, Re-ingest processes every supported file in the "
            "folder again from scratch — replacing what's stored for that document "
            "rather than duplicating it.\n\n"
            "Why do that? Usually because you changed a setting for the corpus (on the "
            "right of this tab) and want documents you already added rebuilt with it. "
            "It's also how you fix a document that didn't come out right the first time "
            "— Re-ingest gives it a clean run.\n\n"
            "It still works only on files in the folder: it won't delete documents you "
            "removed, and an edited file is added as a new copy while the old one stays. "
            "To make the corpus match the folder exactly, use Sync. You'll see a summary "
            "and confirm before it runs."
        ),
        technical=(
            "Maps to `bulk_ops.ingest_folder_plan` / `ingest_folder_execute` — a forced "
            "re-run. It walks + hashes every supported file and calls "
            "`pipeline.ingest_document` on all of them regardless of whether the "
            "`doc_id` is already indexed. `ingest_document` deletes each doc across "
            "LanceDB (`delete_chunks_by_doc_id`) and every Neo4j layer, then rewrites "
            "it, so present docs are replaced, not duplicated.\n\n"
            "No orphan handling: files removed from the folder aren't deleted, and an "
            "edited-in-place file (new `doc_id`) is written new while the old content's "
            "`doc_id` lingers — same straggler behaviour as Ingest folder. Only Sync "
            "cleans those up.\n\n"
            "The Overwrite checkbox is `preserve_existing_labels` (default: preserve "
            "on). Off, a re-ingested doc keeps its stored `(main, sub)` labels; tick "
            "Overwrite to force the labels chosen in the settings panel. It affects "
            "labels only — content, chunks, embeddings and graph data are rebuilt "
            "either way. For a settings change that doesn't need re-parsing, the Bulk "
            "operations below are cheaper: `bulk_re_embed` re-embeds existing chunks, "
            "and the per-layer `bulk_backfill_*` rebuild one layer without re-reading "
            "source. Same save + `reconcile_*_to_config`, missing-key pre-flight, and "
            "Cancel behaviour as the others."
        ),
    ),
    "ingest.sync": InfoText(
        title="Sync",
        standard=(
            "Sync makes the corpus match the selected folder exactly. In one pass it "
            "compares every document in the corpus against the files in the folder and: "
            "adds new files, updates the stored location of moved files, replaces edited "
            "files, and DELETES from the corpus any document with no matching file in "
            "the folder.\n\n"
            "That last part is the one to watch. Sync treats the folder as the complete, "
            "authoritative set for this corpus — so if you point it at a partial folder, "
            "or the wrong one, every corpus document not found there is treated as "
            "removed and deleted. Deletion clears the document from search and from the "
            "graph and can't be undone. The confirmation lists every document Sync will "
            "delete, so read it before approving, and make sure the folder holds "
            "everything you want to keep."
        ),
        beginner=(
            "Sync makes your corpus match a folder exactly — think of the folder as the "
            "master copy and the corpus as a mirror of it.\n\n"
            "In one step it adds files that are new, notices files you've moved and "
            "updates where they're stored, replaces files you've edited with the new "
            "version, and removes from the corpus any document whose file is no longer "
            "in the folder.\n\n"
            "That last part is powerful and permanent. Sync assumes the folder you pick "
            "holds everything the corpus should have. Pick a folder that's missing some "
            "of your documents — or the wrong folder — and Sync treats those as deleted "
            "and removes them; removed documents are gone from search and can't be "
            "recovered. Before it does anything, Sync shows a summary and the full list "
            "of documents it will delete — read it, then confirm. When in doubt, use "
            "Ingest folder — it only adds, never deletes."
        ),
        technical=(
            "Maps to `bulk_ops.sync_plan` / `sync_execute` — a bidirectional reconcile "
            "over the 5-bucket diff in `ingestion/sync_diff.py` (`classify`). It walks + "
            "hashes the folder and enumerates the entire corpus index "
            "(`list_indexed_docs` in `search/client.py`), then classifies into NEW, "
            "UNCHANGED, MOVED (same hash, new path → patch `source_path` only), EDITED "
            "(same path, new hash → delete old + ingest new, carrying old labels forward "
            "via `get_focal_labels_by_doc_id`), and ORPHAN.\n\n"
            "ORPHAN = any indexed doc matched by no disk file, by hash or path — scoped "
            "to the whole corpus, not the folder's subtree. So a partial or wrong folder "
            "makes every other corpus doc an orphan. Each orphan is deleted via "
            "`pipeline.delete_doc` (LanceDB chunks + all KG layers), irreversibly. The "
            "confirm dialog lists every orphan (`SyncPlan.orphan_display_names` — title, "
            "else stored path, else short doc_id) so you see exactly what will be removed "
            "before approving.\n\n"
            "Same save + `reconcile_*_to_config` and missing-key pre-flight as the "
            "others; Cancel stops at a bucket boundary (applied items stay). Path "
            "matching is case-sensitive (`as_posix`, no case-fold), so on Windows a "
            "case-only path change can misclassify UNCHANGED as EDITED/ORPHAN."
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
