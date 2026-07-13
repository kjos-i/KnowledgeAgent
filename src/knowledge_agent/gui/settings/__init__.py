"""Settings — sub-tabs for the top-level Settings tab.

`SettingsView` is the coordinator that composes three sub-tabs:

  - Keys
  - Embedding
  - App

Retrieval and LLM moved to the Search tab's sub-tabs (per-query knobs);
`app.py` mounts `SettingsView(app).build()` as a top-level tab.
"""

from knowledge_agent.gui.settings.settings_view import SettingsView

__all__ = ["SettingsView"]
