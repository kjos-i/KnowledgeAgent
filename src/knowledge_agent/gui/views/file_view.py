"""File view — renders an opened `.md` answer file loaded via the Search
tab's `Open Result` button or paste-path field.

Stateless: takes the filename + contents at construct time and
renders the contents as Markdown. Same shape as `LatestView` so the
right panel's mode switch is uniform.
"""

import flet as ft

from knowledge_agent.gui.views._frame import view_with_header


class FileView:
    """Renders an opened saved answer file in the Search right panel."""

    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.content = content

    def build(self) -> ft.Control:
        body = ft.Column(
            controls=[
                ft.Markdown(
                    self.content,
                    selectable=True,
                    extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                )
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return view_with_header(f"File: {self.name}", body)
