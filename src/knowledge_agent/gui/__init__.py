"""Desktop GUI for the knowledge agent (Flet).

Package layout (slice 1, 2026-06-30):

  - `__main__.py`        — entry; calls `app.main()`.
  - `app.py`             — `GuiApp` coordinator (session state + handlers + build()).
  - `_styles.py`         — shared visual constants (colors, placeholder text).
  - `chat_panel.py`      — left column (chat output + input + Send/Stop/Save/Clear).
  - `display_panel.py`   — right column (NavigationRail view-switcher + view area
                           + inline action row Save Answer / Open Result / paste).
  - `chat_router.py`     — small LLM deciding chat-vs-retrieve before invoking the
                           agent graph.
  - `artifacts.py`       — save_answer / save_chat to disk.
  - `config_store.py`    — `GuiConfig` persistence (JSON + OS keyring for keys).
  - `views/_frame.py`    — `view_header`, `empty_state`, `view_with_header` helpers.
  - `views/latest_view.py` — renders the agent's structured `AgentAnswer`.
  - `views/file_view.py`   — renders an opened `.md` answer file.

Sister projects (ResearchArticlesAgent) use the same shell — chat / display panels
+ mode-switching views — so the orchestration patterns transplant cleanly.
"""
