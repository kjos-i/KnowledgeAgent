"""Latest result view — shows the synthesized `AgentAnswer` + sources,
or a centered empty-state when no answer has been produced yet.

Rendered as the default mode of the Search tab's right panel. The
view itself is stateless — it takes the answer + query at construct
time and builds a Flet Markdown control. Updates happen by switching
the right panel's mode (re-build the view).
"""
import flet as ft

from knowledge_agent.gui.artifacts import render_answer_markdown
from knowledge_agent.gui.views._frame import empty_state, view_with_header
from knowledge_agent.models import AgentAnswer


class LatestView:
    """The 'latest answer + sources' view for the Search tab right panel."""

    def __init__(self, answer: AgentAnswer | None, query: str) -> None:
        self.answer = answer
        self.query = query

    def build(self) -> ft.Control:
        if self.answer is None:
            body: ft.Control = empty_state(
                "Search result will show here once you ask a question."
            )
        else:
            md = render_answer_markdown(self.answer, self.query)
            body = ft.Column(
                controls=[
                    ft.Markdown(
                        md,
                        selectable=True,
                        extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                    )
                ],
                scroll=ft.ScrollMode.AUTO,
                expand=True,
                horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
            )
        return view_with_header("Latest Result", body)
