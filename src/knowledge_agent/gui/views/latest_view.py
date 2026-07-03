"""Latest result view — shows the synthesized `AgentAnswer` + sources,
or a centered empty-state when no answer has been produced yet.

Rendered as the default mode of the Search tab's right panel. The
view itself is stateless — it takes the answer + query at construct
time and builds a Flet Column with:

  1. Markdown block: answer text + text-chunk sources + KG sources
     (via `render_answer_markdown`).
  2. Figure gallery (multimodal): if any `chunk_sources` entry has
     `content_type='figure'` AND a valid `image_ref` on disk, a Row
     of thumbnails is rendered under the markdown. Clicking a
     thumbnail opens a modal AlertDialog with the full-size image and
     full metadata (doc_id, chunk_id, image_ref).

Updates happen by switching the right panel's mode (re-build the view).
"""
from __future__ import annotations

from pathlib import Path

import flet as ft

from knowledge_agent.gui.artifacts import render_answer_markdown
from knowledge_agent.gui.views._frame import empty_state, view_with_header
from knowledge_agent.models import AgentAnswer, ChunkSource

# Thumbnail size for the figure gallery. Small enough not to dominate
# the right-panel layout; click-to-expand shows the full-size image.
_THUMB_WIDTH = 160
_THUMB_HEIGHT = 120

# Max size for the click-to-expand modal image. Flet scales down to
# fit if the source is larger; smaller sources render at native size.
_MODAL_MAX_WIDTH = 900
_MODAL_MAX_HEIGHT = 700


class LatestView:
    """The 'latest answer + sources' view for the Search tab right panel."""

    def __init__(
        self,
        answer: AgentAnswer | None,
        query: str,
        page: ft.Page | None = None,
    ) -> None:
        self.answer = answer
        self.query = query
        # `page` is needed to show the click-to-expand dialog for
        # figure thumbnails. None disables the click behavior (the
        # thumbnails still render, just not interactive).
        self.page = page

    def build(self) -> ft.Control:
        if self.answer is None:
            body: ft.Control = empty_state(
                "Search result will show here once you ask a question."
            )
            return view_with_header("Latest Result", body)

        md = render_answer_markdown(self.answer, self.query)
        controls: list[ft.Control] = [
            ft.Markdown(
                md,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            )
        ]

        # Extract figure sources and render a thumbnail gallery. Any
        # ChunkSource with content_type='figure' and an existing file
        # at image_ref qualifies. Silently skips figures whose PNGs
        # are missing (moved / deleted since ingest); text-based
        # citation still appears via the markdown block above.
        figure_sources = [
            (i, cs) for i, cs in enumerate(self.answer.chunk_sources, start=1)
            if cs.content_type == "figure"
            and cs.image_ref
            and Path(cs.image_ref).is_file()
        ]
        if figure_sources:
            controls.append(ft.Divider())
            controls.append(
                ft.Text(
                    f"Figures cited ({len(figure_sources)})",
                    weight=ft.FontWeight.BOLD, size=14,
                )
            )
            controls.append(
                ft.Row(
                    controls=[
                        self._build_thumbnail(index, cs)
                        for index, cs in figure_sources
                    ],
                    wrap=True,
                    spacing=8,
                    run_spacing=8,
                )
            )

        body = ft.Column(
            controls=controls,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        return view_with_header("Latest Result", body)

    def _build_thumbnail(
        self, source_index: int, cs: ChunkSource,
    ) -> ft.Control:
        """One clickable thumbnail with a caption strip below.

        The Image is wrapped in an ft.Container so on_click is
        available (Flet Images themselves have no on_click). Clicking
        opens `_show_full_image_dialog`.
        """
        assert cs.image_ref is not None  # narrowed by caller

        caption = f"[{source_index}] {cs.doc_id[:12]}…"
        if cs.page:
            caption = f"[{source_index}] page {cs.page} · {cs.doc_id[:12]}…"

        thumb = ft.Image(
            src=cs.image_ref,
            width=_THUMB_WIDTH,
            height=_THUMB_HEIGHT,
            fit=ft.ImageFit.CONTAIN,
        )
        return ft.Container(
            content=ft.Column(
                controls=[
                    thumb,
                    ft.Text(
                        caption, size=10, color=ft.Colors.GREY_400,
                        no_wrap=False,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                spacing=2,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=4,
            border=ft.border.all(1, ft.Colors.GREY_700),
            border_radius=4,
            on_click=(
                (lambda e, s=cs: self._show_full_image_dialog(s))
                if self.page is not None
                else None
            ),
            ink=self.page is not None,
            tooltip=(
                f"Click to expand\n"
                f"chunk_id: {cs.chunk_id}\n"
                f"image_ref: {cs.image_ref}"
            ),
        )

    def _show_full_image_dialog(self, cs: ChunkSource) -> None:
        """Modal showing the full-size image + full citation metadata."""
        if self.page is None or cs.image_ref is None:
            return

        def _close(_ev):
            self.page.pop_dialog()

        # Metadata rows shown alongside the image. Compact — full path
        # is selectable so the user can copy it.
        metadata_rows: list[ft.Control] = [
            ft.Text(f"doc_id: {cs.doc_id}", size=11, selectable=True),
            ft.Text(f"chunk_id: {cs.chunk_id}", size=11, selectable=True),
            ft.Text(
                f"image_ref: {cs.image_ref}",
                size=11, selectable=True,
            ),
        ]
        if cs.page:
            metadata_rows.append(
                ft.Text(f"page: {cs.page}", size=11)
            )
        if cs.quote:
            metadata_rows.append(
                ft.Text(
                    f"caption: {cs.quote}",
                    size=11, selectable=True, italic=True,
                )
            )

        dialog = ft.AlertDialog(
            modal=True,
            content=ft.Container(
                width=_MODAL_MAX_WIDTH,
                content=ft.Column(
                    scroll=ft.ScrollMode.AUTO,
                    tight=True,
                    spacing=8,
                    controls=[
                        ft.Image(
                            src=cs.image_ref,
                            width=_MODAL_MAX_WIDTH,
                            height=_MODAL_MAX_HEIGHT,
                            fit=ft.ImageFit.CONTAIN,
                        ),
                        ft.Divider(),
                        *metadata_rows,
                    ],
                ),
            ),
            actions=[ft.TextButton("Close", on_click=_close)],
        )
        self.page.show_dialog(dialog)
