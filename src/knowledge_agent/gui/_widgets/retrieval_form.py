"""Shared retrieval-form logic for BOTH Settings → Retrieval (global config)
and the Eval dataset form (per-case), so the two can never drift on which knob
a given mode actually uses.

GUI-only module — no backend import. The eval harness enforces the same
mode → leg rule independently (`evaluation.models.required_knobs`); the two
copies are kept in sync by a per-layer test, not a cross-layer import, per the
backend/GUI separation rule.

`apply_gray_out` is the single source for the rule. A control the current mode
can't use is BOTH disabled (non-interactive) AND faded — Flet doesn't recolor a
disabled Radio, so a block-level opacity fade is what actually makes it read as
grayed. A faded/disabled knob is never fed to the search either, because the
leg that would read it simply doesn't run for that mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import flet as ft

if TYPE_CHECKING:
    from collections.abc import Callable

# Which retrieval legs each agent-level mode runs. `auto` is classified at run
# time, so it's treated as running BOTH legs (nothing grayed) to stay safe.
LANCE_MODES: frozenset[str] = frozenset(
    {"auto", "lancedb_only", "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused"}
)
NEO4J_MODES: frozenset[str] = frozenset(
    {"auto", "neo4j_only", "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused"}
)

FADED_OPACITY = 0.4  # opacity of a block the current mode can't use

# Standard starting values for a retrieval config — the defaults the eval
# Dataset form seeds a NEW case with (a fixed baseline, NOT the user's live
# global settings, so a case is reproducible regardless of anyone's config).
# Chosen to match the config.py Settings field defaults (the standard baseline).
DEFAULT_VALUES: dict[str, int | float] = {
    "top_k": 5,
    "num_candidates": 100,
    "rrf_rank_constant": 60,
    "mmr_lambda": 0.6,
    "kg_max_rows": 50,
}


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
        return "Disabled — this mode doesn't run the LanceDB leg."
    if is_fts:
        return "Disabled — FTS has no vectors to diversify."
    if not use_mmr:
        return "MMR off — enable to diversify the result pool."
    return "λ = 1.0 → pure relevance; λ = 0.0 → pure diversity."


def apply_gray_out(
    c: RetrievalControls,
    *,
    retrieval_mode: str,
    lancedb_search_mode: str,
    use_mmr: bool,
) -> None:
    """Disable + fade every control the mode/toggles can't use.

    - LanceDB leg (lancedb_search_mode, num_candidates, rrf, MMR) runs for every
      mode EXCEPT neo4j_only.
    - rrf fuses only in hybrid; MMR applies only in hybrid/vector (FTS has no
      vectors); mmr_lambda only when MMR is on.
    - Neo4j leg (kg_max_rows) runs for every mode EXCEPT lancedb_only.

    top_k / skip_query_builder / direct_retrieval are always active (every mode
    reads them), so they aren't handled here.
    """
    lance_on = retrieval_mode in LANCE_MODES
    neo4j_on = retrieval_mode in NEO4J_MODES
    is_fts = lancedb_search_mode == "fts"
    is_hybrid = lancedb_search_mode == "hybrid"
    mmr_available = lance_on and not is_fts

    # ---- LanceDB search-mode radios ----
    # Flet doesn't cascade a RadioGroup's disabled to its radios, and doesn't
    # recolor a disabled radio, so set each radio AND fade the whole block.
    for radio in c.lancedb_radios:
        radio.disabled = not lance_on
    if c.lancedb_mode_control is not None:
        c.lancedb_mode_control.disabled = not lance_on
    if c.lancedb_mode_box is not None:
        c.lancedb_mode_box.opacity = 1.0 if lance_on else FADED_OPACITY

    # ---- LanceDB numeric knobs ----
    if c.num_candidates_field is not None:
        c.num_candidates_field.disabled = not lance_on
    if c.rrf_constant_field is not None:
        c.rrf_constant_field.disabled = not (lance_on and is_hybrid)

    # ---- MMR (built-in disabled grays these fine) ----
    if c.use_mmr_checkbox is not None:
        c.use_mmr_checkbox.disabled = not mmr_available
        # Never show a checked box the mode can't honour.
        c.use_mmr_checkbox.value = use_mmr and mmr_available
    if c.mmr_lambda_control is not None:
        c.mmr_lambda_control.disabled = not (mmr_available and use_mmr)

    # ---- Neo4j leg ----
    if c.kg_max_rows_field is not None:
        c.kg_max_rows_field.disabled = not neo4j_on


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
