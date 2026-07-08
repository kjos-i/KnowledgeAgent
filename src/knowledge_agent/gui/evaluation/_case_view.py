"""Shared case-card renderer for the Evaluation tabs.

Both the Run tab (read-only — "see exactly what will run") and the Dataset
tab (edit + delete) render a dataset's cases as the SAME card list, so they
look identical and stay in sync. Single source: change a card here and both
tabs move together.

Read-only when no `on_edit` / `on_delete` callbacks are given; pass either
(Dataset tab) to get a clickable card + Edit / Delete buttons. `detailed=True`
(Run tab) expands every non-empty gold field on the card, since there's no
edit form there to inspect the case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import flet as ft

if TYPE_CHECKING:
    from collections.abc import Callable

_ELLIPSIS = ft.TextOverflow.ELLIPSIS

# (attribute, caption) for the gold fields shown in `detailed` mode, in order.
# List-valued fields are joined; empties are skipped so the card only shows
# what the case actually asserts.
_DETAIL_FIELDS: tuple[tuple[str, str], ...] = (
    ("category", "category"),
    ("notes", "notes"),
    ("expected_sources", "expected_sources"),
    ("expected_chunks", "expected_chunks"),
    ("required_keywords", "required_keywords"),
    ("disallowed_keywords", "disallowed_keywords"),
    ("expected_answer_points", "expected_answer_points"),
    ("expected_entities", "expected_entities"),
    ("expected_mode", "expected_mode"),
    ("user_cypher", "user_cypher"),
)


def _retrieval_summary(case: Any) -> str:
    """One-line read of the case's retrieval settings."""
    r = getattr(case, "retrieval", None)
    if r is None:
        return ""
    parts = [
        str(getattr(r, "retrieval_mode", "")),
        str(getattr(r, "lancedb_search_mode", "")),
        f"top_k={getattr(r, 'top_k', '')}",
    ]
    if getattr(r, "direct_retrieval", False):
        parts.append("direct")
    if getattr(r, "skip_query_builder", False):
        parts.append("skip-qb")
    return " · ".join(p for p in parts if p and p != "None")


def _field_value(case: Any, attr: str) -> str:
    """Display string for a gold field — list → comma-joined, else str."""
    v = getattr(case, attr, None)
    if v is None or v == "":
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def _kv_line(caption: str, value: str) -> ft.Control:
    """A `caption: value` detail line (caption column + wrapping value)."""
    return ft.Row(
        controls=[
            ft.Text(f"{caption}:", size=12, color=ft.Colors.GREY_500, width=160),
            ft.Text(value, size=12, color=ft.Colors.GREY_300, expand=True, selectable=True),
        ],
        vertical_alignment=ft.CrossAxisAlignment.START,
    )


def case_card(
    case: Any,
    idx: int,
    *,
    selected: int | None = None,
    detailed: bool = False,
    on_edit: Callable[[int], None] | None = None,
    on_delete: Callable[[int], None] | None = None,
) -> ft.Control:
    """One case rendered as a card. `detailed` expands every non-empty gold
    field; editable (on_edit/on_delete) adds a click-to-edit body + buttons."""
    editable = on_edit is not None or on_delete is not None
    question = ft.Text(
        case.question,
        size=12,
        color=ft.Colors.GREY_400,
        selectable=detailed,
        # Detailed cards show the whole question; compact cards clip it.
        max_lines=None if detailed else 2,
        overflow=None if detailed else _ELLIPSIS,
    )
    body: list[ft.Control] = [
        ft.Row(
            controls=[
                ft.Text(case.id, weight=ft.FontWeight.BOLD, size=12, expand=True),
                ft.Text(getattr(case, "origin", ""), size=12, color=ft.Colors.GREY_500),
            ],
        ),
        question,
        ft.Text(_retrieval_summary(case), size=12, color=ft.Colors.GREY_600),
    ]
    if detailed:
        for attr, caption in _DETAIL_FIELDS:
            value = _field_value(case, attr)
            if value:
                body.append(_kv_line(caption, value))
    if editable:
        buttons: list[ft.Control] = []
        if on_edit is not None:
            buttons.append(
                ft.TextButton(
                    "Edit",
                    icon=ft.Icons.EDIT_OUTLINED,
                    on_click=lambda _e, i=idx: on_edit(i),
                )
            )
        if on_delete is not None:
            buttons.append(
                ft.TextButton(
                    "Delete",
                    icon=ft.Icons.DELETE_OUTLINE,
                    on_click=lambda _e, i=idx: on_delete(i),
                )
            )
        body.append(ft.Row(buttons, spacing=4))
    return ft.Container(
        padding=8,
        border_radius=4,
        border=ft.Border.all(1, ft.Colors.GREY_800),
        bgcolor=ft.Colors.BLUE_GREY_900 if selected == idx else None,
        # Click the card body to edit it (Dataset tab); read-only cards don't
        # react to clicks.
        on_click=(lambda _e, i=idx: on_edit(i)) if on_edit is not None else None,
        content=ft.Column(body, spacing=2),
    )


def render_case_cards(
    cases: list[Any],
    *,
    selected: int | None = None,
    detailed: bool = False,
    on_edit: Callable[[int], None] | None = None,
    on_delete: Callable[[int], None] | None = None,
    empty_hint: str = "No cases.",
) -> list[ft.Control]:
    """Render a list of cases as cards (or a single empty-state hint)."""
    if not cases:
        return [ft.Text(empty_hint, italic=True, color=ft.Colors.GREY_500)]
    return [
        case_card(
            c,
            i,
            selected=selected,
            detailed=detailed,
            on_edit=on_edit,
            on_delete=on_delete,
        )
        for i, c in enumerate(cases)
    ]
