"""Tests for the shared RecipeForm widget (the EvalRecipe controls: metric
groups + judge panel + gate thresholds).

Built statically — no page render. Covers the load()/to_recipe() round-trip,
freeze, and the judge panel (which moved here from run_tab).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from knowledge_agent.evaluation.models import EvalRecipe
from knowledge_agent.gui.evaluation._recipe_form import RecipeForm


def _form(fake_app: MagicMock) -> RecipeForm:
    fake_app.gui_config.llm_provider = "anthropic"
    rf = RecipeForm(fake_app)
    rf.build()
    return rf


def test_defaults_match_harness(fake_app: MagicMock):
    """A fresh form == the harness defaults: source/chunk/kg on, judge off,
    default thresholds."""
    r = _form(fake_app).to_recipe()
    assert set(r.enabled_groups) == {"source", "chunk", "kg"}
    assert (r.judge_threshold, r.metadata_match_threshold, r.required_keyword_threshold) == (
        0.5,
        0.8,
        0.5,
    )


def test_load_round_trip(fake_app: MagicMock):
    """load(recipe) populates every control; to_recipe() reads them back."""
    rf = _form(fake_app)
    rf.load(
        EvalRecipe(
            enabled_groups=["source", "judge"],
            judge_models=["m1", "m2"],
            judge_threshold=0.6,
            metadata_match_threshold=0.7,
            required_keyword_threshold=0.4,
        )
    )
    assert rf.group_checks["source"].value is True
    assert rf.group_checks["chunk"].value is False
    assert rf.group_checks["judge"].value is True
    assert rf.judge_models == ["m1", "m2"]

    out = rf.to_recipe()
    assert set(out.enabled_groups) == {"source", "judge"}
    assert out.judge_models == ["m1", "m2"]
    assert (out.judge_threshold, out.metadata_match_threshold, out.required_keyword_threshold) == (
        0.6,
        0.7,
        0.4,
    )


def test_load_none_gives_defaults(fake_app: MagicMock):
    """load(None) resets a dirtied form to the harness defaults."""
    rf = _form(fake_app)
    rf.group_checks["judge"].value = True  # dirty it
    rf.threshold_fields["judge_threshold"].value = "0.9"
    rf.load(None)
    r = rf.to_recipe()
    assert set(r.enabled_groups) == {"source", "chunk", "kg"}
    assert r.judge_threshold == 0.5


def test_judge_section_greys_until_judge_on(fake_app: MagicMock):
    """The judge box stays visible but greys out (disabled) until judge is
    ticked — the group-change handler syncs it."""
    rf = _form(fake_app)
    assert rf.judge_section.disabled is True  # judge off by default
    rf.group_checks["judge"].value = True
    rf._on_group_change(MagicMock())
    assert rf.judge_section.disabled is False


def test_add_and_remove_judges(fake_app: MagicMock):
    """Pick a model → Add appends it (dedup); X removes it; picker clears."""
    rf = _form(fake_app)
    assert rf.judge_models == []
    rf.judge_dropdown.value = "model-a"
    rf._on_add_judge_clicked(MagicMock())
    assert rf.judge_models == ["model-a"]
    assert rf.judge_dropdown.value is None  # cleared for the next add
    assert len(rf.judge_panel.controls) == 1
    rf.judge_dropdown.value = "model-a"  # dedup
    rf._on_add_judge_clicked(MagicMock())
    assert rf.judge_models == ["model-a"]
    rf._remove_judge("model-a")
    assert rf.judge_models == []


def test_judge_fallback_hint_toggles(fake_app: MagicMock):
    """The 'no judges → default fallback' note shows only when the list empty."""
    rf = _form(fake_app)
    assert rf.judge_fallback_hint.visible is True
    rf.judge_dropdown.value = "model-a"
    rf._on_add_judge_clicked(MagicMock())
    assert rf.judge_fallback_hint.visible is False


def test_set_enabled_freezes_and_unfreezes(fake_app: MagicMock):
    rf = _form(fake_app)
    rf.set_enabled(False)
    assert rf._wrapper.disabled is True
    rf.set_enabled(True)
    assert rf._wrapper.disabled is False


def test_any_group_selected(fake_app: MagicMock):
    rf = _form(fake_app)
    assert rf.any_group_selected() is True
    for cb in rf.group_checks.values():
        cb.value = False
    assert rf.any_group_selected() is False


def test_thresholds_clamp_and_default(fake_app: MagicMock):
    """A blank / out-of-range / non-numeric threshold falls back or clamps —
    to_recipe() never raises on a half-typed field."""
    rf = _form(fake_app)
    rf.threshold_fields["judge_threshold"].value = ""  # blank → default
    rf.threshold_fields["metadata_match_threshold"].value = "5"  # >1 → clamp
    rf.threshold_fields["required_keyword_threshold"].value = "x"  # bad → default
    r = rf.to_recipe()
    assert r.judge_threshold == 0.5
    assert r.metadata_match_threshold == 1.0
    assert r.required_keyword_threshold == 0.5
