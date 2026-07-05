"""Evaluation → Dataset sub-tab — browse + edit a gold dataset (slice 3).

Master-detail: pick a dataset, scroll the full list of cases on the left, and
edit EVERY field of the selected case on the right in a form grouped like the
`EvalCase` schema (so it doubles as documentation of what a case contains).
New / Save / Delete a case, and edit the dataset `name` + `status` header;
persistence goes through the backend `save_dataset` helper (object format).

Dropdown options are derived from the model's own `Literal`s (via
`typing.get_args`) so the choices can never drift from the schema. List-valued
fields (expected_sources, keywords, …) are edited as one-item-per-line text.

"Generate (LLM)" drafts candidate cases from the active corpus (one per
doc) via the backend `generator` — flagged `origin=llm` for human review.

Deliberately out of scope here (later slices): the doc-picker that resolves
`expected_sources` to real ingested `doc_id`s, and a Search-tab capture
button.
"""

from __future__ import annotations

import typing
from pathlib import Path
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.gui._styles import FRAME_BORDER_COLOR
from knowledge_agent.gui.views._frame import view_header

if TYPE_CHECKING:
    from knowledge_agent.evaluation.models import EvalCase, EvalDataset
    from knowledge_agent.gui.app import GuiApp
    from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

_LIST_W = 320
_NONE = ""  # the expected_mode "(none)" sentinel (maps to None on read)


class DatasetTab:
    """Browse + edit a gold dataset (slice 3 of the authoring UI)."""

    def __init__(self, app: GuiApp, coordinator: EvaluationView) -> None:
        self.app = app
        self.coordinator = coordinator
        self.dataset_dropdown: ft.Dropdown | None = None
        self.name_field: ft.TextField | None = None
        self.status_dropdown: ft.Dropdown | None = None
        self.case_list: ft.Column | None = None
        self.form: ft.Column | None = None
        self.status: ft.Text | None = None
        self.gen_count: ft.TextField | None = None
        self.gen_button: ft.Control | None = None
        # form field widgets (created in build)
        self.f: dict[str, ft.Control] = {}
        # state
        self._dataset: EvalDataset | None = None
        self._cases: list[EvalCase] = []
        self._selected: int | None = None
        self._path: Path | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        from knowledge_agent.evaluation.config import DEFAULT_DATASET_PATH
        from knowledge_agent.evaluation.models import (
            CaseOrigin,
            DatasetStatus,
            LanceDbSearchMode,
            RetrievalMode,
        )

        origins = list(typing.get_args(CaseOrigin))
        statuses = list(typing.get_args(DatasetStatus))
        modes = list(typing.get_args(RetrievalMode))
        search_modes = list(typing.get_args(LanceDbSearchMode))

        datasets_dir = DEFAULT_DATASET_PATH.parent
        options = [
            ft.DropdownOption(key=str(p), text=p.name) for p in sorted(datasets_dir.glob("*.json"))
        ]
        self.dataset_dropdown = ft.Dropdown(
            label="Dataset (pick or type a path)",
            editable=True,
            options=options,
            value=str(DEFAULT_DATASET_PATH) if DEFAULT_DATASET_PATH.exists() else None,
            width=_LIST_W,
        )
        load_button = ft.TextButton("Load", icon=ft.Icons.FOLDER_OPEN, on_click=self._on_load)
        self.name_field = ft.TextField(label="Dataset name", width=_LIST_W, dense=True)
        self.status_dropdown = ft.Dropdown(
            label="Status",
            options=[ft.DropdownOption(key=s, text=s) for s in statuses],
            value="draft",
            width=_LIST_W,
        )
        new_button = ft.TextButton("New case", icon=ft.Icons.ADD, on_click=self._on_new)
        capture_button = ft.TextButton(
            "From last search", icon=ft.Icons.SEARCH, on_click=self._on_capture_from_search
        )
        self.gen_count = ft.TextField(label="count", value="5", width=80, dense=True)
        self.gen_button = ft.TextButton(
            "Generate (LLM)", icon=ft.Icons.AUTO_AWESOME, on_click=self._on_generate_llm
        )
        self.status = ft.Text("", size=11, color=ft.Colors.GREY_500)

        self.case_list = ft.Column(
            controls=[ft.Text("Load a dataset to browse + edit its cases.", italic=True)],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=2,
        )

        # ---- form field widgets ----
        self.f = {
            "id": ft.TextField(label="id", dense=True),
            "question": ft.TextField(label="question", multiline=True, min_lines=2, max_lines=4),
            "origin": ft.Dropdown(
                label="origin",
                options=[ft.DropdownOption(key=o, text=o) for o in origins],
                value="manual",
            ),
            "category": ft.TextField(label="category", dense=True),
            "notes": ft.TextField(label="notes", multiline=True, min_lines=1, max_lines=3),
            "retrieval_mode": ft.Dropdown(
                label="retrieval_mode",
                options=[ft.DropdownOption(key=m, text=m) for m in modes],
                value="lancedb_only",
            ),
            "lancedb_search_mode": ft.Dropdown(
                label="lancedb_search_mode",
                options=[ft.DropdownOption(key=m, text=m) for m in search_modes],
                value="hybrid",
            ),
            "top_k": ft.TextField(label="top_k", value="5", width=100, dense=True),
            "skip_query_builder": ft.Checkbox(label="skip_query_builder", value=False),
            "direct_retrieval": ft.Checkbox(label="direct_retrieval", value=False),
            "expected_sources": _list_field("expected_sources (one doc_id per line)"),
            "expected_chunks": _list_field("expected_chunks (one substring per line)"),
            "required_keywords": _list_field("required_keywords (one per line)"),
            "disallowed_keywords": _list_field("disallowed_keywords (one per line)"),
            "expected_answer_points": _list_field("expected_answer_points (one per line)"),
            "expected_entities": _list_field("expected_entities (one per line)"),
            "expected_mode": ft.Dropdown(
                label="expected_mode",
                options=[
                    ft.DropdownOption(key=_NONE, text="(none)"),
                    *(ft.DropdownOption(key=m, text=m) for m in modes),
                ],
                value=_NONE,
            ),
            "user_cypher": ft.TextField(
                label="user_cypher", multiline=True, min_lines=1, max_lines=4
            ),
        }

        save_button = ft.Button("Save case", icon=ft.Icons.SAVE, on_click=self._on_save_case)
        delete_button = ft.TextButton(
            "Delete case", icon=ft.Icons.DELETE_OUTLINE, on_click=self._on_delete
        )

        self.form = ft.Column(
            controls=[
                ft.Row([save_button, delete_button], spacing=8),
                *self._group("Identity", ["id", "question", "origin", "category", "notes"]),
                *self._group(
                    "Retrieval settings",
                    [
                        "retrieval_mode",
                        "lancedb_search_mode",
                        "top_k",
                        "skip_query_builder",
                        "direct_retrieval",
                    ],
                ),
                *self._group("Retrieval / chunk gold", ["expected_sources", "expected_chunks"]),
                *self._group("Keyword checks", ["required_keywords", "disallowed_keywords"]),
                *self._group("Judge gold", ["expected_answer_points"]),
                *self._group("KG gold", ["expected_entities", "expected_mode", "user_cypher"]),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
            spacing=6,
        )

        left = ft.Container(
            width=_LIST_W,
            content=ft.Column(
                [
                    ft.Row([self.dataset_dropdown, load_button], spacing=4, wrap=True),
                    self.name_field,
                    self.status_dropdown,
                    self.status,
                    ft.Row([new_button, capture_button], wrap=True),
                    ft.Row(
                        [self.gen_button, self.gen_count],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
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
            content=self.form,
        )
        body = ft.Row([left, right], expand=True, spacing=12)
        return ft.Column([view_header("Dataset"), body], expand=True, spacing=8)

    def _group(self, title: str, keys: list[str]) -> list[ft.Control]:
        return [
            ft.Text(title, weight=ft.FontWeight.BOLD, size=13),
            *(self.f[k] for k in keys),
            ft.Divider(height=1),
        ]

    # ---- load -------------------------------------------------------------

    def _on_load(self, _e: ft.Event) -> None:
        if not (self.dataset_dropdown and self.dataset_dropdown.value):
            self._set_status("Pick or type a dataset path first.")
            return
        self._load(Path(self.dataset_dropdown.value))

    def _load(self, path: Path) -> None:
        from knowledge_agent.evaluation.models import load_dataset

        try:
            ds = load_dataset(path)
        except Exception as exc:  # broad: surface any parse/validation error in-line
            self._set_status(f"could not load: {exc}")
            return
        self._dataset = ds
        self._cases = ds.cases
        self._path = path
        self._selected = None
        if self.name_field is not None:
            self.name_field.value = ds.name
        if self.status_dropdown is not None:
            self.status_dropdown.value = ds.status
        self._clear_form()
        self._render_list()
        self._set_status(f"{len(self._cases)} case(s)")
        self.app.page.update()

    # ---- list -------------------------------------------------------------

    def _render_list(self) -> None:
        if self.case_list is None:
            return
        if not self._cases:
            self.case_list.controls = [ft.Text("No cases yet — New case to add one.", italic=True)]
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
                            ft.Row(
                                [
                                    ft.Text(
                                        case.id, weight=ft.FontWeight.BOLD, size=12, expand=True
                                    ),
                                    ft.Text(case.origin, size=10, color=ft.Colors.GREY_500),
                                ]
                            ),
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
        self._fill_form(self._cases[idx])
        self._render_list()
        self.app.page.update()

    # ---- form <-> case ----------------------------------------------------

    def _fill_form(self, case: EvalCase) -> None:
        f = self.f
        f["id"].value = case.id
        f["question"].value = case.question
        f["origin"].value = case.origin
        f["category"].value = case.category
        f["notes"].value = case.notes
        f["retrieval_mode"].value = case.retrieval.retrieval_mode
        f["lancedb_search_mode"].value = case.retrieval.lancedb_search_mode
        f["top_k"].value = str(case.retrieval.top_k)
        f["skip_query_builder"].value = case.retrieval.skip_query_builder
        f["direct_retrieval"].value = case.retrieval.direct_retrieval
        f["expected_sources"].value = _text(case.expected_sources)
        f["expected_chunks"].value = _text(case.expected_chunks)
        f["required_keywords"].value = _text(case.required_keywords)
        f["disallowed_keywords"].value = _text(case.disallowed_keywords)
        f["expected_answer_points"].value = _text(case.expected_answer_points)
        f["expected_entities"].value = _text(case.expected_entities)
        f["expected_mode"].value = case.expected_mode or _NONE
        f["user_cypher"].value = case.user_cypher or ""

    def _clear_form(self) -> None:
        f = self.f
        for key in ("id", "question", "category", "notes", "user_cypher"):
            f[key].value = ""
        for key in (
            "expected_sources",
            "expected_chunks",
            "required_keywords",
            "disallowed_keywords",
            "expected_answer_points",
            "expected_entities",
        ):
            f[key].value = ""
        f["origin"].value = "manual"
        f["retrieval_mode"].value = "lancedb_only"
        f["lancedb_search_mode"].value = "hybrid"
        f["top_k"].value = "5"
        f["skip_query_builder"].value = False
        f["direct_retrieval"].value = False
        f["expected_mode"].value = _NONE

    def _read_form(self) -> EvalCase:
        """Build an `EvalCase` from the form (raises ValidationError/ValueError
        on bad input — the caller surfaces it)."""
        from knowledge_agent.evaluation.models import EvalCase

        f = self.f
        return EvalCase(
            id=(f["id"].value or "").strip(),
            question=(f["question"].value or "").strip(),
            origin=f["origin"].value or "manual",
            category=(f["category"].value or "").strip(),
            notes=(f["notes"].value or "").strip(),
            expected_sources=_lines(f["expected_sources"].value),
            expected_chunks=_lines(f["expected_chunks"].value),
            required_keywords=_lines(f["required_keywords"].value),
            disallowed_keywords=_lines(f["disallowed_keywords"].value),
            expected_answer_points=_lines(f["expected_answer_points"].value),
            expected_entities=_lines(f["expected_entities"].value),
            expected_mode=f["expected_mode"].value or None,
            user_cypher=(f["user_cypher"].value or "").strip() or None,
            retrieval={
                "retrieval_mode": f["retrieval_mode"].value,
                "lancedb_search_mode": f["lancedb_search_mode"].value,
                "top_k": int((f["top_k"].value or "5").strip()),
                "skip_query_builder": bool(f["skip_query_builder"].value),
                "direct_retrieval": bool(f["direct_retrieval"].value),
            },
        )

    # ---- CRUD -------------------------------------------------------------

    def _current_path(self) -> Path | None:
        if self._path is not None:
            return self._path
        v = self.dataset_dropdown.value if self.dataset_dropdown else None
        return Path(v) if v else None

    def _on_new(self, _e: ft.Event) -> None:
        self._selected = None
        self._clear_form()
        self._render_list()
        self._set_status("new case — fill the form and Save")
        self.app.page.update()

    def _on_capture_from_search(self, _e: ft.Event) -> None:
        """Pre-fill a NEW case from the last Search result — the question +
        the retrieved doc_ids (deduped, order-preserved) as expected_sources,
        `origin=search`. The user reviews (keywords / answer points) + Saves.

        This is the authoring-side of the 'capture from search' flow: no
        cross-view navigation — the user is already here to curate the set."""
        answer = getattr(self.app, "last_answer", None)
        query = getattr(self.app, "last_query", None)
        if answer is None or not query:
            self._set_status("No search result to capture — run a query in Search first.")
            return
        seen: set[str] = set()
        sources: list[str] = []
        for cs in getattr(answer, "chunk_sources", None) or []:
            doc_id = getattr(cs, "doc_id", None)
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                sources.append(doc_id)
        self._selected = None
        self._clear_form()
        self.f["question"].value = query
        self.f["expected_sources"].value = _text(sources)
        self.f["origin"].value = "search"
        self._render_list()
        self._set_status(f"captured from search ({len(sources)} source(s)) — review + Save")
        self.app.page.update()

    async def _on_generate_llm(self, _e: ft.Event) -> None:
        """Draft `count` LLM candidate cases from the ACTIVE corpus and append
        them (flagged `origin=llm`) to the loaded dataset for review.

        `generator.generate_from_corpus` samples one passage per corpus doc
        and drafts a question + answer points + keywords via the active
        provider's model. The cases are saved to the dataset path; the user
        then reviews / edits / deletes each in the form. Async — the LLM calls
        run off the UI thread and the button disables while it works."""
        from knowledge_agent.evaluation.generator import generate_from_corpus
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        path = self._current_path()
        if path is None:
            self._set_status("Pick or type a dataset path first.")
            return
        try:
            n = int((self.gen_count.value or "5").strip()) if self.gen_count else 5
        except ValueError:
            self._set_status("Count must be a whole number.")
            return
        if n <= 0:
            self._set_status("Count must be at least 1.")
            return

        if self.gen_button is not None:
            self.gen_button.disabled = True
        self._set_status(f"generating {n} candidate case(s) from the active corpus…")
        try:
            cases = await generate_from_corpus(n)
        except Exception as exc:  # broad: provider / corpus / LLM errors → status
            if self.gen_button is not None:
                self.gen_button.disabled = False
            self._set_status(f"generation failed: {exc}")
            return
        if self.gen_button is not None:
            self.gen_button.disabled = False

        if not cases:
            self._set_status("no candidates generated (empty corpus or all passages too short)")
            return

        if self._dataset is None:
            self._dataset = EvalDataset()
            if self.name_field is not None:
                self.name_field.value = self._dataset.name
            if self.status_dropdown is not None:
                self.status_dropdown.value = self._dataset.status
        self._dataset.cases.extend(cases)
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status
            self._set_status(f"generated {len(cases)} but save failed: {exc}")
            return
        self._cases = self._dataset.cases
        self._path = path
        self._selected = None
        self._render_list()
        self._set_status(
            f"generated {len(cases)} LLM candidate(s) — review each (origin=llm), keep/edit/delete"
        )

    def _on_save_case(self, _e: ft.Event) -> None:
        from knowledge_agent.evaluation.models import EvalDataset, save_dataset

        path = self._current_path()
        if path is None:
            self._set_status("Pick or type a dataset path first.")
            return
        try:
            case = self._read_form()
        except Exception as exc:  # broad: validation / int-parse errors → status line
            self._set_status(f"invalid case: {exc}")
            return
        if self._dataset is None:
            self._dataset = EvalDataset()
        # header edits ride along with any save
        if self.name_field is not None:
            self._dataset.name = (self.name_field.value or "").strip()
        if self.status_dropdown is not None:
            self._dataset.status = self.status_dropdown.value or "draft"

        if self._selected is not None and 0 <= self._selected < len(self._dataset.cases):
            self._dataset.cases[self._selected] = case
            verb = "updated"
        else:
            self._dataset.cases.append(case)
            self._selected = len(self._dataset.cases) - 1
            verb = "added"
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"save failed: {exc}")
            return
        self._cases = self._dataset.cases
        self._path = path
        self._render_list()
        self._set_status(f"{verb} — saved {len(self._cases)} case(s) to {path.name}")
        self.app.page.update()

    def _on_delete(self, _e: ft.Event) -> None:
        from knowledge_agent.evaluation.models import save_dataset

        if self._dataset is None or self._selected is None:
            self._set_status("Select a case to delete.")
            return
        path = self._current_path()
        if path is None:
            self._set_status("No dataset path to save to.")
            return
        del self._dataset.cases[self._selected]
        try:
            save_dataset(self._dataset, path)
        except Exception as exc:  # broad: I/O failure → status line
            self._set_status(f"save failed: {exc}")
            return
        self._selected = None
        self._cases = self._dataset.cases
        self._clear_form()
        self._render_list()
        self._set_status(f"deleted — {len(self._cases)} case(s) remain")
        self.app.page.update()

    # ---- helpers ----------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        if self.status is not None:
            self.status.value = msg
            self.app.page.update()


def _list_field(hint: str) -> ft.TextField:
    return ft.TextField(label=hint, multiline=True, min_lines=2, max_lines=5)


def _lines(text: str | None) -> list[str]:
    return [line.strip() for line in (text or "").splitlines() if line.strip()]


def _text(values: list[str]) -> str:
    return "\n".join(values)
