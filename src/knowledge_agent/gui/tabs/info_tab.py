"""Info top-level tab — the help / reference surface.

Promoted to a top-level tab (it was a Search right-panel placeholder mode)
so it's one click away while working, per the user's request.

Placeholder for now — the full contents (what each tab does, where every
setting lives, the getting-started workflow, search / Cypher tips, the
file-layout map, and the ledger-column reference) land later; see
`notes.md` item 23. Kept as a real tab so that content has a home to grow
into.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from knowledge_agent.gui.views._frame import empty_state, view_with_header

if TYPE_CHECKING:
    import flet as ft

    from knowledge_agent.gui.app import GuiApp


class InfoTab:
    """Static Info tab (placeholder until the how-to guide is written)."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        return view_with_header(
            "Information",
            empty_state(
                "Help / how-to guide — coming soon. This panel will describe "
                "what each tab does, where every setting lives, the full "
                "workflow, and search tips (including Cypher)."
            ),
        )
