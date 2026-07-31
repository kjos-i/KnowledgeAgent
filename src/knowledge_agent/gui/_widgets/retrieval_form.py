"""Shared retrieval-form logic for BOTH the Settings Retrieval tab (global
config) and the Eval dataset form (per-case), so the two GUI forms can never
drift on which knob a given mode uses.

The mode-to-leg rule itself (which knobs each mode reads) is NOT defined here.
It lives once in `knowledge_agent.retrieval_rules`, imported by both this GUI
gray-out and the eval harness's `evaluation.models.required_knobs`, so the two
can never disagree. This module imports only that shared leaf plus flet.

`apply_gray_out` turns that rule into disabled/faded controls. A control the
current mode can't use is BOTH disabled (non-interactive) AND faded, because
Flet doesn't recolor a disabled Radio, so a block-level opacity fade is what
makes it read as grayed. A faded/disabled knob is never fed to the search
either: the leg that would read it simply doesn't run for that mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.config import retrieval_defaults
from knowledge_agent.retrieval_rules import LANCE_MODES, active_knobs, mmr_applicable

if TYPE_CHECKING:
    from collections.abc import Callable

FADED_OPACITY = 0.4  # opacity of a block the current mode can't use

# The fixed baseline retrieval defaults, sourced directly from the config.py
# Settings field defaults (via `retrieval_defaults`) so the eval Dataset form
# and the Settings retrieval tab can never drift from the backend. These are the
# fixed field defaults, NOT the user's live settings, so an authored case is
# reproducible regardless of anyone's config. Full knob set (categorical +
# numeric), keyed by the per-case knob names.
DEFAULT_VALUES: dict[str, object] = retrieval_defaults()


@dataclass
class RetrievalControls:
    """The gray-out-able controls of a retrieval form. Leaf controls (fields,
    checkbox, slider) rely on Flet's built-in `disabled`, which visibly grays
    them. The ONE exception is the radio block: Flet neither cascades a
    RadioGroup's disabled to its radios nor recolors a disabled radio, so we
    wrap it (`lancedb_mode_box`) and fade its opacity. Any field left None is
    skipped, so a host only passes the controls it has."""

    lancedb_mode_box: ft.Container | None = None
    lancedb_radios: list[ft.Radio] = field(default_factory=list)
    # The within-store search-mode control — a RadioGroup (Settings) OR a
    # Dropdown (eval form). Either way it just gets `disabled`; a Dropdown grays
    # via the built-in, radios need the box's opacity fade above.
    lancedb_mode_control: ft.Control | None = None
    num_candidates_field: ft.Control | None = None
    rrf_constant_field: ft.Control | None = None
    use_mmr_checkbox: ft.Checkbox | None = None
    mmr_lambda_control: ft.Control | None = None
    kg_max_rows_field: ft.Control | None = None


def mmr_help_text(*, lance_on: bool, is_fts: bool, use_mmr: bool) -> str:
    """The one-line MMR help string for the current state (caller assigns it
    wherever it shows one)."""
    if not lance_on:
        return "Disabled. This mode doesn't run the LanceDB leg."
    if is_fts:
        return "Disabled. FTS has no vectors to diversify."
    if not use_mmr:
        return "MMR is off. Enable it to diversify the result pool."
    return "lambda 1.0 is pure relevance; lambda 0.0 is pure diversity."


def apply_gray_out(
    c: RetrievalControls,
    *,
    retrieval_mode: str,
    lancedb_search_mode: str,
    use_mmr: bool,
) -> None:
    """Disable + fade every control the mode/toggles can't use.

    Which tuning knobs a mode uses comes from the shared
    `retrieval_rules.active_knobs` (the ONE source this gray-out and the eval
    harness's `required_knobs` both read), so the two can never drift. The radio
    block and the MMR checkbox aren't tuning knobs, so they use the leg
    predicates (`LANCE_MODES` membership, `mmr_applicable`) from the same leaf.

    A disabled control is never fed to the search either: the leg that would
    read it simply doesn't run for that mode. top_k / skip_query_builder /
    direct_retrieval are always active (every mode reads them), so they aren't
    handled here.
    """
    lance_on = retrieval_mode in LANCE_MODES
    knobs = active_knobs(retrieval_mode, lancedb_search_mode, use_mmr)
    mmr_ok = mmr_applicable(retrieval_mode, lancedb_search_mode)

    # ---- LanceDB search-mode radios ----
    # Flet doesn't cascade a RadioGroup's disabled to its radios, and doesn't
    # recolor a disabled radio, so set each radio AND fade the whole block.
    for radio in c.lancedb_radios:
        radio.disabled = not lance_on
    if c.lancedb_mode_control is not None:
        c.lancedb_mode_control.disabled = not lance_on
    if c.lancedb_mode_box is not None:
        c.lancedb_mode_box.opacity = 1.0 if lance_on else FADED_OPACITY

    # ---- LanceDB numeric knobs (from the shared active-knobs rule) ----
    if c.num_candidates_field is not None:
        c.num_candidates_field.disabled = "num_candidates" not in knobs
    if c.rrf_constant_field is not None:
        c.rrf_constant_field.disabled = "rrf_rank_constant" not in knobs

    # ---- MMR (built-in disabled grays these fine) ----
    if c.use_mmr_checkbox is not None:
        c.use_mmr_checkbox.disabled = not mmr_ok
        # Never show a checked box the mode can't honour.
        c.use_mmr_checkbox.value = use_mmr and mmr_ok
    if c.mmr_lambda_control is not None:
        c.mmr_lambda_control.disabled = "mmr_lambda" not in knobs

    # ---- Neo4j leg ----
    if c.kg_max_rows_field is not None:
        c.kg_max_rows_field.disabled = "kg_max_rows" not in knobs


# ---------------------------------------------------------------------------
# Shared control builders — so Settings → Retrieval and the eval Dataset form
# render the IDENTICAL widgets (not two look-alikes). Each host wires its own
# event handlers (Settings persists on blur / change-end; the eval form
# refreshes its preview), but the widgets themselves come from one place.
# ---------------------------------------------------------------------------

LANCEDB_MODE_LABELS: list[tuple[str, str]] = [
    ("hybrid", "Hybrid (BM25 + vector, RRF-fused)"),
    ("fts", "FTS (BM25 only)"),
    ("vector", "Vector (kNN cosine only)"),
]


def build_search_mode_radios(
    value: str, on_change: Callable[[ft.Event], None]
) -> tuple[list[ft.Radio], ft.RadioGroup, ft.Container]:
    """The within-store search-mode control the way Settings builds it: three
    radios (all options visible at once — the intuitive part) wrapped in a
    Container so the gray-out can fade the whole block (Flet won't recolor a
    disabled radio). Returns the radios (for per-radio disable), the group, and
    the box (for the opacity fade)."""
    radios = [ft.Radio(value=k, label=lbl.split(" (")[0]) for k, lbl in LANCEDB_MODE_LABELS]
    group = ft.RadioGroup(
        value=value, on_change=on_change, content=ft.Row(controls=radios, spacing=16)
    )
    box = ft.Container(
        content=ft.Column(
            spacing=6,
            controls=[
                ft.Text("LanceDB search mode (within-store):", size=12, color=ft.Colors.GREY_400),
                group,
            ],
        )
    )
    return radios, group, box


def build_mmr_slider(
    value: float,
    *,
    on_change: Callable[[ft.Event], None] | None = None,
    on_change_end: Callable[[ft.Event], None] | None = None,
) -> tuple[ft.Slider, ft.Text]:
    """The MMR-λ slider + its numeric value display, the way Settings builds it.
    The slider keeps the value text in sync on every slide then forwards to the
    host's `on_change`; `on_change_end` fires when the user lifts off (Settings
    persists there, to avoid a save per pixel)."""
    value_text = ft.Text(f"{float(value):.2f}", size=12, color=ft.Colors.WHITE, width=42)

    def _slide(e: ft.Event) -> None:
        value_text.value = f"{float(e.control.value):.2f}"
        if on_change is not None:
            on_change(e)

    slider = ft.Slider(
        value=value, min=0.0, max=1.0, divisions=20, on_change=_slide, on_change_end=on_change_end
    )
    return slider, value_text


# ---------------------------------------------------------------------------
# Input-mode radio — the "how is my text interpreted as a query" axis, SHARED
# by Settings → Retrieval and the eval Dataset form so the two can't drift.
# Only the three modes that map to real backend graph knobs live here:
#   refined       -> the query-builder rewrites the text (skip_query_builder=False)
#   direct_query  -> the text is the search query verbatim (skip_query_builder=True)
#   direct_cypher -> the text is raw Cypher (user_cypher set)
# The store (retrieval_mode) is an INDEPENDENT knob, NOT bundled here — except
# direct_cypher, which can only run on the graph (`store_forced_by_mode`).
# "Conversational" (the GUI chat router) is a chat-only PRE-step on a different
# axis; the Settings tab prepends it itself. It is deliberately NOT here — the
# eval harness invokes the backend graph directly and has no router.
# ---------------------------------------------------------------------------

INPUT_MODE_LABELS: list[tuple[str, str]] = [
    ("refined", "Refined query — the query builder rewrites your text before searching"),
    (
        "direct_query",
        "Direct query — search your exact text, unrewritten (skips the query builder)",
    ),
    ("direct_cypher", "Direct Cypher — run your text as raw Cypher on the knowledge graph"),
]


def build_input_mode_radios() -> list[ft.Radio]:
    """The three shared query-mode radios (refined / direct_query / direct_cypher).

    A host wraps them in its OWN RadioGroup — the Settings tab prepends its
    chat-only 'Conversational' radio, the eval form uses just these three — so
    both render the SAME options + labels from one source. Fresh instances per
    call (a control can't be shared across the tree)."""
    return [ft.Radio(value=key, label=label) for key, label in INPUT_MODE_LABELS]


def query_mode_to_knobs(mode: str, *, cypher_text: str | None = None) -> dict[str, object]:
    """Map a query mode to the two graph knobs it controls: `skip_query_builder`
    (direct_query) and `user_cypher` (direct_cypher). The store (`retrieval_mode`)
    is independent and NOT set here — see `store_forced_by_mode` for the one
    exception. Single source for both the chat dispatch and the eval-case save,
    so they can't disagree on what a mode means."""
    return {
        "skip_query_builder": mode == "direct_query",
        "user_cypher": ((cypher_text or "").strip() or None) if mode == "direct_cypher" else None,
    }


def knobs_to_query_mode(*, skip_query_builder: bool, user_cypher: str | None) -> str:
    """Inverse of `query_mode_to_knobs`: derive the radio value from a stored
    case's knobs (Cypher wins, then skip-builder, else refined). Used when the
    eval form loads an existing case back into the radio."""
    if user_cypher:
        return "direct_cypher"
    if skip_query_builder:
        return "direct_query"
    return "refined"


def store_forced_by_mode(mode: str) -> str | None:
    """Direct Cypher can only execute on the graph, so it pins the store to
    `neo4j_only`; every other query mode leaves the store INDEPENDENT (the
    user's own retrieval_mode). Returns the forced retrieval_mode, or None when
    the mode doesn't constrain the store."""
    return "neo4j_only" if mode == "direct_cypher" else None
