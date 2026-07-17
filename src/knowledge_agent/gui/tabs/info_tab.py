"""Info top-level tab — the app-wide help surface.

A secondary sub-tab strip. Most tabs are backed by a shipped `.md` in
`knowledge_agent/docs/` and rendered through the shared `views/info_doc.InfoDoc`
(the same widget the Metrics Guide and the per-area Info sub-tabs use); the final
**Dependencies** tab is generated live from the installed package metadata. This
tab holds only the *general* help; per-area help lives as an Info sub-tab inside
Search / Library / Evaluation.

The docs are first-drafts meant to be edited in place — `InfoDoc` re-reads the
file on each build, so edits show on the next launch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._markdown import themed_markdown
from knowledge_agent.gui.tabs._dependencies import dependencies_markdown
from knowledge_agent.gui.views.info_doc import InfoDoc, docs_dir

if TYPE_CHECKING:
    from knowledge_agent.gui.app import GuiApp

# Sub-tab label -> its doc filename (under knowledge_agent/docs/). One .md per
# sub-tab, matching the item-23 sections: what the app does, where things are,
# the getting-started workflow, and the files/storage map.
_SECTIONS: tuple[tuple[str, str], ...] = (
    ("About", "info_about.md"),
    ("Where things are", "info_layout.md"),
    ("Getting Started", "info_getting_started.md"),
    ("Files & storage", "info_storage.md"),
)

# The generated tab appended after the shipped-doc sections.
_DEPENDENCIES_LABEL = "Dependencies"


class InfoTab:
    """App-wide Info tab — About / Where things are / Getting Started."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app

    def build(self) -> ft.Control:
        labels = [label for label, _ in _SECTIONS] + [_DEPENDENCIES_LABEL]
        bodies: list[ft.Control] = []
        for label, filename in _SECTIONS:
            doc = InfoDoc(
                self.app,
                docs_dir() / filename,
                missing_hint=f"'{label}' help is not written yet ({filename}).",
            )
            bodies.append(ft.Container(content=doc.build(), padding=8, expand=True))
        bodies.append(ft.Container(content=self._dependencies_body(), padding=8, expand=True))
        # Same secondary-tab shape as the Search right panel / Library / Evaluation
        # so the sub-strip reads one level below the primary tab bar.
        sub_bar = ft.TabBar(
            tabs=[ft.Tab(label=label) for label in labels],
            secondary=True,
        )
        sub_bodies = ft.TabBarView(controls=bodies, expand=True)
        return ft.Tabs(
            length=len(labels),
            selected_index=0,
            content=ft.Column(controls=[sub_bar, sub_bodies], expand=True, spacing=0),
            expand=True,
        )

    def _dependencies_body(self) -> ft.Control:
        """The generated Dependencies tab — declared packages + installed versions,
        read live from the package metadata (not a shipped doc)."""
        return ft.Column(
            controls=[themed_markdown(dependencies_markdown())],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
