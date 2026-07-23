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
}


def info(app: GuiApp, key: str) -> ft.Control:
    """Render the (i) help-icon row for a registry key.

    Looks up `INFO[key]` and delegates to `info_point`. A `KeyError` means
    the key isn't registered (typo / missing entry) — surfaced loudly so it
    is caught in dev, never silently blank.
    """
    return info_point(app, INFO[key])
