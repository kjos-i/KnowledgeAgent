"""Evaluation → Dataset sub-tab — browse a gold dataset (read-only, slice 1).

Master-detail: pick a dataset, scroll the full list of cases on the left, and
see EVERY field of the selected case on the right — grouped exactly like the
`EvalCase` schema so the form doubles as visual documentation of what a gold
case contains.

Slice 1 is READ-ONLY (loads via `load_cases`, no backend writes). Editing,
Add/Edit/Delete, the `origin` provenance field, the Search-tab capture, and
LLM generation land in later slices (they touch the backend — save helper +
schema field + generator).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import FRAME_BORDER_COLOR
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.evaluation.models import EvalCase
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_LIST_W = 300


class DatasetTab:
    """Read-only gold-dataset browser (slice 1 of the authoring UI)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.dataset_dropdown: ft.Dropdown | None = None
        self.case_list: ft.Column | None = None
        self.detail: ft.Column | None = None
        self.status: ft.Text | None = None
        self._cases: list[EvalCase] = []
        self._selected: int | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.config import DEFAULT_DATASET_PATH

        datasets_dir = DEFAULT_DATASET_PATH.parent
        options = [
            ft.DropdownOption(key=str(p), text=p.name) for p in sorted(datasets_dir.glob("*.json"))
        ]
        self.dataset_dropdown = ft.Dropdown(
            label="Dataset",
            editable=True,
            options=options,
            value=str(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH.exists() else None,
            width=_LIST_W,
        )
        load_button = ft.TextButton("Load", icon=ft.Icons.FOLDER_OPEN, on_click=self._on_load)
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_500)

        self.case_list = ft.Column(
            controls=[ft.Text("Load a dataset to browse its cases.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=2,
        )
        self.detail = ft.Column(
            controls=[ft.Text("Select a case to see all its fields.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=6,
        )

        left = ft.Container(
            width=_LIST_W,
            content=ft.Column(
                [
                    ft.Row([self.dataset_dropdown, load_button], spacing=4, wrap=True),
                    self.status,
                    ft.Divider(height=1),
                    self.case_list,
                ],
                expand=True,
                spacing=6,
            ),
        )
        right = ft.Container(
            expand=True,
            border=ft.Border.all(1, FRAME_BORDER_COLOR),
            border_radius=6,
            padding=12,
            content=self.detail,
        )
        body = ft.Row([left, right], expand=True, spacing=12)
        return ft.Column([view_header("Dataset"), body], expand=True, spacing=8)

    # ---- data / load ------------------------------------------------------

    def _on_load(self, _e: ft.Event) -> None:
        if not (self.dataset_dropdown and self.dataset_dropdown.value):
            self._set_status("Pick a dataset first.")
            return
        self._load(Path(self.dataset_dropdown.value))

    def _load(self, path: Path) -> None:
        from knowledge_agent.evaluation.models import load_cases

        try:
            self._cases = load_cases(path)
        except Exception as exc:  # broad: surface any parse/validation error in-line
            self._set_status(f"could not load: {exc}")
            return
        self._selected = None
        self._set_status(f"{len(self._cases)} case(s)")
        self._render_list()
        if self.detail is not None:
            self.detail.controls = [ft.Text("Select a case to see all its fields.", italic=True)]
        self.app.page.update()

    def _render_list(self) -> None:
        if self.case_list is None:
            return
        if not self._cases:
            self.case_list.controls = [ft.Text("Dataset has no cases.", italic=True)]
            return
        rows: list[ft.Control] = []
        for i, case in enumerate(self._cases):
            selected = i == self._selected
            rows.append(
                ft.Container(
                    on_click=lambda _e, idx=i: self._select(idx),
                    padding=8,
                    border_radius=4,
                    bgcolor=ft.Colors.BLUE_GREY_900 if selected else None,
                    content=ft.Column(
                        [
                            ft.Text(case.id, weight=ft.FontWeight.BOLD, size=12),
                            ft.Text(
                                case.question,
                                size=11,
                                color=ft.Colors.GREY_400,
                                max_lines=2,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=1,
                    ),
                )
            )
        self.case_list.controls = rows

    def _select(self, idx: int) -> None:
        self._selected = idx
        self._render_list()
        self._render_detail(self._cases[idx])
        self.app.page.update()

    # ---- detail (full-field, grouped like the schema) ---------------------

    def _render_detail(self, case: EvalCase) -> None:
        if self.detail is None:
            return
        r = case.retrieval
        self.detail.controls = [
            *self._group(
                "Identity",
                [
                    ("id", case.id),
                    ("question", case.question),
                    ("category", case.category or "—"),
                    ("notes", case.notes or "—"),
                ],
            ),
            *self._group(
                "Retrieval settings",
                [
                    ("retrieval_mode", r.retrieval_mode),
                    ("lancedb_search_mode", r.lancedb_search_mode),
                    ("top_k", str(r.top_k)),
                    ("skip_query_builder", _yn(r.skip_query_builder)),
                    ("direct_retrieval", _yn(r.direct_retrieval)),
                ],
            ),
            *self._group(
                "Retrieval / chunk gold",
                [
                    ("expected_sources", _lst(case.expected_sources)),
                    ("expected_chunks", _lst(case.expected_chunks)),
                ],
            ),
            *self._group(
                "Keyword checks",
                [
                    ("required_keywords", _lst(case.required_keywords)),
                    ("disallowed_keywords", _lst(case.disallowed_keywords)),
                ],
            ),
            *self._group(
                "Judge gold",
                [("expected_answer_points", _lst(case.expected_answer_points))],
            ),
            *self._group(
                "KG gold",
                [
                    ("expected_entities", _lst(case.expected_entities)),
                    ("expected_mode", case.expected_mode or "—"),
                    ("user_cypher", case.user_cypher or "—"),
                ],
            ),
        ]

    def _group(self, title: str, fields: list[tuple[str, str]]) -> list[ft.Control]:
        rows = [
            ft.Row(
                [
                    ft.Text(f"{label}", size=12, color=ft.Colors.GREY_400, width=170),
                    ft.Text(value, size=12, selectable=True, expand=True),
                ],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
            for label, value in fields
        ]
        return [ft.Text(title, weight=ft.FontWeight.BOLD, size=13), *rows, ft.Divider(height=1)]

    # ---- helpers ----------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        if self.status is not None:
            self.status.value = msg
            self.app.page.update()


def _yn(value: bool) -> str:
    return "yes" if value else "no"


def _lst(values: list[str]) -> str:
    return "\n".join(f"• {v}" for v in values) if values else "—"
