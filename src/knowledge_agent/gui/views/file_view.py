"""File view — renders an opened `.md` / `.txt` answer file loaded via the
Search tab's `Open Result` button or paste-path field.

Stateless: takes the filename + contents at construct time. `.md` renders as
Markdown; `.txt` renders as plain monospace text (no Markdown parsing) so a
saved plain-text answer shows exactly its characters. Same shape as
`LatestView` so the right panel's mode switch is uniform.
"""

import flet as ft

from knowledge_agent.gui.views._frame import view_with_header


class FileView:
    """Renders an opened saved answer file in the Search right panel."""

    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.content = content

    def build(self) -> ft.Control:
        # A .txt renders as-is (monospace, no Markdown parsing) so it shows
        # exactly its characters; everything else renders as Markdown.
        if self.name.lower().endswith(".txt"):
            content_control: ft.Control = ft.Text(
                self.content,
                selectable=True,
                font_family="monospace",
            )
        else:
            content_control = ft.Markdown(
                self.content,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            )
        body = ft.Column(
            controls=[content_control],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return view_with_header(f"File: {self.name}", body)
