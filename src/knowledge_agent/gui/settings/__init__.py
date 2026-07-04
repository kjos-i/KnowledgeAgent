"""Settings panel — sub-tabs for the right panel's Settings mode.

`SettingsView` is the top-level coordinator that composes five sub-tabs:

  - Keys
  - Retrieval
  - LLM
  - Embedding
  - App

Right panel's MODE_SETTINGS dispatches to `SettingsView(app).build()`.
"""

from knowledge_agent.gui.settings.settings_view import SettingsView

__all__ = ["SettingsView"]
