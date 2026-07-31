"""Unit tests for the shared Evaluation case-card renderer (`_case_view`)."""

from types import SimpleNamespace

import flet as ft

from knowledge_agent.evaluation.models import EvalCase
from knowledge_agent.gui.evaluation._case_view import case_card, render_case_cards


def _case(cid: str = "c1") -> SimpleNamespace:
    return SimpleNamespace(
        id=cid,
        question="How do ESCRTs work?",
        origin="manual",
        retrieval=SimpleNamespace(
            retrieval_mode="lancedb_only",
            top_k=5,
            direct_retrieval=False,
            skip_query_builder=False,
        ),
    )


def _buttons(card: ft.Control) -> list[str]:
    """Collect the text of any TextButtons anywhere in a card."""
    found: list[str] = []

    def walk(ctl: object) -> None:
        if isinstance(ctl, ft.TextButton):
            found.append(ctl.content)  # TextButton label lives in `content`
        for child in getattr(ctl, "controls", None) or []:
            walk(child)
        inner = getattr(ctl, "content", None)
        if inner is not None and not isinstance(inner, str):
            walk(inner)

    walk(card)
    return found


def test_render_empty_shows_hint() -> None:
    out = render_case_cards([], empty_hint="Nothing here.")
    assert len(out) == 1
    assert isinstance(out[0], ft.Text)
    assert out[0].value == "Nothing here."


def test_readonly_card_has_no_buttons_or_click() -> None:
    """Run-tab preview: read-only cards have no Edit/Delete and aren't clickable."""
    card = case_card(_case(), 0)
    assert _buttons(card) == []
    assert card.on_click is None


def test_editable_card_has_edit_and_delete() -> None:
    """Dataset tab: passing callbacks adds Edit + Delete, but the body is NOT
    clickable — selecting/editing happens via the Edit button only."""
    edited: list[int] = []
    deleted: list[int] = []
    card = case_card(
        _case(),
        2,
        on_edit=edited.append,
        on_delete=deleted.append,
    )
    assert set(_buttons(card)) == {"Edit", "Delete"}
    assert card.on_click is None


def test_cancel_button_only_on_selected_card() -> None:
    """Cancel shows only on the selected card (the one being edited)."""
    selected_card = case_card(
        _case(),
        1,
        selected=1,
        on_edit=lambda _i: None,
        on_delete=lambda _i: None,
        on_cancel=lambda _i: None,
    )
    assert set(_buttons(selected_card)) == {"Edit", "Cancel", "Delete"}
    other_card = case_card(
        _case(),
        0,
        selected=1,
        on_edit=lambda _i: None,
        on_delete=lambda _i: None,
        on_cancel=lambda _i: None,
    )
    assert "Cancel" not in _buttons(other_card)


def _all_text(card: ft.Control) -> str:
    """Concatenate every Text value anywhere in a card."""
    out: list[str] = []

    def walk(ctl: object) -> None:
        v = getattr(ctl, "value", None)
        if isinstance(v, str):
            out.append(v)
        for child in getattr(ctl, "controls", None) or []:
            walk(child)
        inner = getattr(ctl, "content", None)
        if inner is not None and not isinstance(inner, str):
            walk(inner)

    walk(card)
    return " ".join(out)


def test_detailed_card_shows_nonempty_gold_fields() -> None:
    """detailed=True surfaces every non-empty gold field (Run-tab preview)."""
    case = _case()
    case.category = "recall"
    case.expected_sources = ["doc1", "doc2"]
    text = _all_text(case_card(case, 0, detailed=True))
    assert "category" in text and "recall" in text
    assert "expected_sources" in text and "doc1" in text and "doc2" in text


def test_compact_card_omits_gold_fields() -> None:
    """Default (compact) cards don't expand the gold fields."""
    case = _case()
    case.category = "recall"
    assert "recall" not in _all_text(case_card(case, 0))


def test_render_case_cards_selected_highlights() -> None:
    cards = render_case_cards([_case("a"), _case("b")], selected=1, on_edit=lambda _i: None)
    # The selected card (index 1) is tinted; the other is not.
    assert cards[1].bgcolor is not None
    assert cards[0].bgcolor is None


def test_card_warns_on_unrunnable_case() -> None:
    """A real EvalCase with a required retrieval knob left blank shows a
    'not runnable' warning on its card; a fully-pinned one does not."""
    invalid = EvalCase(id="bad", question="Q?")  # defaults: auto + None knobs (not runnable)
    assert "not runnable" in _all_text(case_card(invalid, 0))

    valid = EvalCase(
        id="good",
        question="Q?",
        retrieval={
            "retrieval_mode": "lancedb_only",
            "lancedb_search_mode": "hybrid",
            "top_k": 5,
            "num_candidates": 40,
            "rrf_rank_constant": 60,
        },
    )
    assert "not runnable" not in _all_text(case_card(valid, 0))
