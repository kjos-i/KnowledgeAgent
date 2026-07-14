"""Settings — the app-level Settings tab and the Keys tab.

`Settings` (app-level config) and `Keys` (API credentials) are each their
own top-level tab, mounted directly by `app.py`. They were split out of a
former single `SettingsView` "Keys / App" sub-tab shell (2026-07-14).

Retrieval and LLM moved to the Search tab's sub-tabs (per-query knobs);
Embedding moved to the per-corpus Ingest config editor (the embedder is
per-corpus now).
"""

from knowledge_agent.gui.settings.app_tab import AppTab
from knowledge_agent.gui.settings.keys_tab import KeysTab

__all__ = ["AppTab", "KeysTab"]
