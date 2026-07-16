"""Shared recipe editor — the `EvalRecipe` controls, mounted by BOTH eval tabs.

A dataset's *recipe* is its canonical "how to run me": which metric groups
run, the judge panel, and the three gate thresholds. This widget is the single
implementation of those controls:

  * **Create test cases** tab — mounts it fully editable; the recipe rides
    along with the dataset on save.
  * **Run evaluation** tab — mounts the SAME widget loaded from the dataset's
    saved recipe. It stays editable until the dataset is frozen; freezing
    (`set_enabled(False)`) locks the whole recipe and **Unfreeze** reopens it.

SSOT: the metric-group checks + judge panel used to live inline in `run_tab`;
they moved here so the dataset tab reuses the exact same controls instead of a
parallel copy. `to_recipe()` / `load()` are the two conversion points.

Whether a run measures answer QUALITY (judge on, gates strict) or PROFILES
retrieval knobs (judge off, gates at 0 so you read the metric curves) is
expressed directly here via the metric groups + thresholds — it is a RUN
choice, not a dataset "kind". Knob sweeps are modelled as a SUITE of files that
share the same facts but pin different retrieval knobs (see
`compute_facts_hash`). The gate thresholds DO gate — they feed `EvalConfig` at
run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flet as ft

from knowledge_agent.evaluation.config import DEFAULT_ENABLED_GROUPS
from knowledge_agent.gui._styles import labeled_field, sub_section_header
from knowledge_agent.gui._widgets.info_icon import info_icon
from knowledge_agent.gui.settings.llm_tab import model_options

if TYPE_CHECKING:
    from knowledge_agent.evaluation.models import EvalRecipe
    from knowledge_agent.gui.app import GuiApp

# Ordered metric-group toggles (ALL_TOGGLE_GROUPS is an unordered frozenset).
# judge sits last and defaults OFF: it calls paid LLM judges; the rest are
# deterministic + free.
_GROUPS: tuple[str, ...] = ("source", "chunk", "kg", "judge")

# Fixed caption column so the three threshold inputs line up down a shared
# edge (widest label is "required_keyword_threshold").
_THRESHOLD_LABEL_WIDTH = 210
_THRESHOLDS: tuple[tuple[str, float], ...] = (
    ("judge_threshold", 0.5),
    ("metadata_match_threshold", 0.8),
    ("required_keyword_threshold", 0.5),
)


def _fmt(value: float) -> str:
    """A threshold's field text — trims trailing zeros (0.5, 0.8, 0)."""
    return f"{value:g}"


def _unit_float(raw: str | None, default: float) -> float:
    """Parse a 0–1 threshold field, clamped to [0, 1]; blank/bad → default."""
    s = (raw or "").strip()
    if not s:
        return default
    try:
        return max(0.0, min(1.0, float(s)))
    except ValueError:
        return default


class RecipeForm:
    """The `EvalRecipe` controls: metric groups + judge panel + gate
    thresholds. Editable everywhere; `set_enabled(False)` freezes it for the
    Run tab's read-only display."""

    def __init__(self, app: GuiApp) -> None:
        self.app = app
        # controls (built in build())
        self.group_checks: dict[str, ft.Checkbox] = {}
        self.threshold_fields: dict[str, ft.TextField] = {}
        # judge panel (moved verbatim from run_tab — SSOT)
        self.judge_dropdown: ft.Dropdown | None = None
        self.judge_models: list[str] = []  # names Added into the list below
        self.add_judge_button: ft.TextButton | None = None
        self.judge_panel: ft.Column | None = None
        self.judge_fallback_hint: ft.Text | None = None
        self.judge_section: ft.Container | None = None
        # Outer container whose `disabled` freezes/unfreezes the whole widget.
        self._wrapper: ft.Container | None = None

    # ---- build ------------------------------------------------------------

    def build(self) -> ft.Control:
        self.group_checks = {
            g: ft.Checkbox(
                label=g,
                value=g in DEFAULT_ENABLED_GROUPS,
                on_change=self._on_group_change,
            )
            for g in _GROUPS
        }
        self.threshold_fields = {
            name: ft.TextField(value=_fmt(default), width=90, dense=True)
            for name, default in _THRESHOLDS
        }

        # ---- judge panel (identical to the old run_tab block) ----
        judge_on = bool(self.group_checks["judge"].value)
        self.judge_dropdown = ft.Dropdown(editable=True, options=self._judge_options())
        self.add_judge_button = ft.TextButton(
            "Add judge model", icon=ft.Icons.ADD, on_click=self._on_add_judge_clicked
        )
        self.judge_panel = ft.Column(controls=[], spacing=4)  # added-judge list
        self.judge_fallback_hint = ft.Text(
            "No judges added — one default judge from your active provider is used.",
            size=12,
            color=ft.Colors.GREY_600,
            italic=True,
        )
        self.judge_section = ft.Container(
            disabled=not judge_on,  # greys out (stays visible) when judge is off
            content=ft.Column(
                [
                    sub_section_header(
                        "Judge panel",
                        trailing=info_icon(
                            self.app,
                            title="Judge panel",
                            text=(
                                "LLM judges — they call your provider and cost "
                                "money, which is why the judge metric is OFF by "
                                "default. Each model you add scores every case; the "
                                "panel's mean is the judge score. Add none and one "
                                "default judge from your active provider is used."
                            ),
                        ),
                    ),
                    ft.Row(
                        [
                            ft.Container(
                                content=labeled_field("Judge model", self.judge_dropdown),
                                expand=True,
                            ),
                            self.add_judge_button,
                        ],
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    self.judge_panel,
                    self.judge_fallback_hint,
                ],
                spacing=6,
            ),
        )
        self._refresh_judge_list()

        content = ft.Column(
            [
                sub_section_header("Metric groups"),
                ft.Row([self.group_checks[g] for g in _GROUPS], wrap=True),
                self.judge_section,
                sub_section_header(
                    "Gate thresholds",
                    trailing=info_icon(
                        self.app,
                        title="Gate thresholds",
                        text=(
                            "The pass/fail cutoffs a case is judged against: the "
                            "judge score, the metadata (source) match rate, and the "
                            "required-keyword hit rate each need to clear its "
                            "threshold. Set a threshold to 0 to stop it gating (a "
                            "knob-profiling run does this to read metric curves "
                            "without pass/fail)."
                        ),
                    ),
                ),
                *(
                    labeled_field(
                        name, self.threshold_fields[name], label_width=_THRESHOLD_LABEL_WIDTH
                    )
                    for name, _ in _THRESHOLDS
                ),
            ],
            spacing=8,
        )
        self._wrapper = ft.Container(content=content)
        return self._wrapper

    # ---- recipe <-> controls ---------------------------------------------

    def load(self, recipe: EvalRecipe | None) -> None:
        """Populate the controls from `recipe` (None → a fresh default recipe).
        Pure — no page.update; the caller flushes."""
        from knowledge_agent.evaluation.models import EvalRecipe

        r = recipe or EvalRecipe()
        for g, cb in self.group_checks.items():
            cb.value = g in r.enabled_groups
        self.judge_models = list(r.judge_models)
        for name, default in _THRESHOLDS:
            self.threshold_fields[name].value = _fmt(getattr(r, name, default))
        self._refresh_judge_list()
        self._sync_judge_grey()

    def to_recipe(self) -> EvalRecipe:
        """Build an `EvalRecipe` from the current controls."""
        from knowledge_agent.evaluation.models import EvalRecipe

        groups = [g for g in _GROUPS if self.group_checks[g].value]
        return EvalRecipe(
            enabled_groups=groups,
            judge_models=list(self.judge_models),
            judge_threshold=_unit_float(self.threshold_fields["judge_threshold"].value, 0.5),
            metadata_match_threshold=_unit_float(
                self.threshold_fields["metadata_match_threshold"].value, 0.8
            ),
            required_keyword_threshold=_unit_float(
                self.threshold_fields["required_keyword_threshold"].value, 0.5
            ),
        )

    def set_enabled(self, enabled: bool) -> None:
        """Freeze (False) / unfreeze (True) every control at once. The Run tab
        shows the saved recipe frozen and unfreezes for a one-off deviation.
        Under the freeze the judge section keeps its own judge-on/off grey-out,
        so unfreezing reveals it in the right state. Pure — caller flushes."""
        if self._wrapper is not None:
            self._wrapper.disabled = not enabled

    def any_group_selected(self) -> bool:
        """True when at least one metric group is ticked (a run needs one)."""
        return any(cb.value for cb in self.group_checks.values())

    # ---- metric groups ----------------------------------------------------

    def _on_group_change(self, _e: ft.Event | None = None) -> None:
        # Toggling the judge group greys / reveals the judge panel below it.
        self._sync_judge_grey()
        self.app.page.update()

    # ---- judge panel (moved from run_tab) ---------------------------------

    def _judge_options(self) -> list[ft.DropdownOption]:
        # Every installed provider's models — a judge PANEL spans providers, and
        # each judge is stored as a 'provider:model' ref the backend dispatches
        # per-judge (so one panel can mix Claude + GPT + Gemini).
        return model_options()

    def _sync_judge_grey(self) -> None:
        """Grey the whole judge box out (still visible) when judge is off —
        `disabled` cascades to the picker, the list, and the Add button."""
        if self.judge_section is not None:
            self.judge_section.disabled = not bool(self.group_checks["judge"].value)

    def _refresh_judge_list(self) -> None:
        """Rebuild the added-judge list rows + toggle the empty fallback note."""
        if self.judge_panel is not None:
            self.judge_panel.controls = [self._judge_list_row(n) for n in self.judge_models]
        if self.judge_fallback_hint is not None:
            self.judge_fallback_hint.visible = not self.judge_models

    def _judge_list_row(self, name: str) -> ft.Control:
        """One added-judge list item: bullet + model name + a fixed 'temp 0'
        badge + remove button. Judges always run deterministically at
        temperature 0; the badge just surfaces that."""
        return ft.Row(
            [
                ft.Icon(ft.Icons.CIRCLE, size=6, color=ft.Colors.GREY_500),
                ft.Text(name, size=12, color=ft.Colors.GREY_200, expand=True),
                ft.Text("temp 0", size=12, color=ft.Colors.GREY_500, italic=True),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_size=16,
                    tooltip="Remove this judge",
                    on_click=lambda _e, n=name: self._remove_judge(n),
                ),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=6,
        )

    def _on_add_judge_clicked(self, _e: ft.Event) -> None:
        """Move the picked model into the judge list (dedup), then clear the
        picker for the next add."""
        if self.judge_dropdown is None:
            return
        name = (self.judge_dropdown.value or "").strip()
        if name and name not in self.judge_models:
            self.judge_models.append(name)
            self.judge_dropdown.value = None
            self._refresh_judge_list()
        self.app.page.update()

    def _remove_judge(self, name: str) -> None:
        if name in self.judge_models:
            self.judge_models.remove(name)
            self._refresh_judge_list()
            self.app.page.update()
