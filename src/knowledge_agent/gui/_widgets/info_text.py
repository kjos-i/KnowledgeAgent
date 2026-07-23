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
}


def info(app: GuiApp, key: str) -> ft.Control:
    """Render the (i) help-icon row for a registry key.

    Looks up `INFO[key]` and delegates to `info_point`. A `KeyError` means
    the key isn't registered (typo / missing entry) — surfaced loudly so it
    is caught in dev, never silently blank.
    """
    return info_point(app, INFO[key])
