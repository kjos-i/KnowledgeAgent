"""Tests for the shared RecipeForm widget (the EvalRecipe controls: profile +
metric groups + judge panel + gate thresholds).

Built statically — no page render. Covers the profile presets, the hand-edit →
Custom rule, the load()/to_recipe() round-trip, freeze, and the judge panel
(which moved here from run_tab).
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
    """A fresh form == the harness defaults: source/chunk/kg on, judge off, no
    profile (Custom), default thresholds."""
    r = _form(fake_app).to_recipe()
    assert set(r.enabled_groups) == {"source", "chunk", "kg"}
    assert r.dataset_kind is None
    assert (r.judge_threshold, r.metadata_match_threshold, r.required_keyword_threshold) == (
        0.5,
        0.8,
        0.5,
    )


def test_fact_profile_preset(fake_app: MagicMock):
    """Fact = judge ON, gates at the harness defaults, tagged fact."""
    rf = _form(fake_app)
    rf.profile_group.value = "fact"
    rf._on_profile_change(MagicMock())
    r = rf.to_recipe()
    assert r.dataset_kind == "fact"
    assert set(r.enabled_groups) == {"source", "chunk", "kg", "judge"}
    assert (r.judge_threshold, r.metadata_match_threshold, r.required_keyword_threshold) == (
        0.5,
        0.8,
        0.5,
    )


def test_knob_profile_preset(fake_app: MagicMock):
    """Knob = judge OFF, every gate at 0 (read the curves), tagged knob."""
    rf = _form(fake_app)
    rf.profile_group.value = "knob"
    rf._on_profile_change(MagicMock())
    r = rf.to_recipe()
    assert r.dataset_kind == "knob"
    assert "judge" not in r.enabled_groups
    assert (r.judge_threshold, r.metadata_match_threshold, r.required_keyword_threshold) == (
        0.0,
        0.0,
        0.0,
    )


def test_group_edit_clears_profile_to_custom(fake_app: MagicMock):
    """Toggling a metric group by hand drops the profile back to Custom."""
    rf = _form(fake_app)
    rf.profile_group.value = "fact"
    rf._on_profile_change(MagicMock())
    assert rf.profile_group.value == "fact"
    rf.group_checks["chunk"].value = False
    rf._on_group_change(MagicMock())
    assert rf.profile_group.value is None
    assert rf.to_recipe().dataset_kind is None


def test_threshold_edit_clears_profile_to_custom(fake_app: MagicMock):
    """Editing a threshold by hand also drops the profile to Custom."""
    rf = _form(fake_app)
    rf.profile_group.value = "knob"
    rf._on_profile_change(MagicMock())
    rf.threshold_fields["judge_threshold"].value = "0.7"
    rf._on_threshold_change(MagicMock())
    assert rf.profile_group.value is None


def test_load_round_trip(fake_app: MagicMock):
    """load(recipe) populates every control; to_recipe() reads them back."""
    rf = _form(fake_app)
    rf.load(
        EvalRecipe(
            dataset_kind="fact",
            enabled_groups=["source", "judge"],
            judge_models=["m1", "m2"],
            judge_threshold=0.6,
            metadata_match_threshold=0.7,
            required_keyword_threshold=0.4,
        )
    )
    assert rf.profile_group.value == "fact"
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
    rf.profile_group.value = "knob"
    rf._on_profile_change(MagicMock())
    rf.load(None)
    r = rf.to_recipe()
    assert r.dataset_kind is None
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


def test_adding_judge_keeps_profile(fake_app: MagicMock):
    """The judge PANEL is independent of the profile — adding a judge does NOT
    drop Fact/Knob to Custom (only groups + thresholds do)."""
    rf = _form(fake_app)
    rf.profile_group.value = "fact"
    rf._on_profile_change(MagicMock())
    rf.judge_dropdown.value = "model-a"
    rf._on_add_judge_clicked(MagicMock())
    assert rf.profile_group.value == "fact"


def test_set_enabled_freezes_and_unfreezes(fake_app: MagicMock):
    rf = _form(fake_app)
    rf.set_enabled(False)
    assert rf._wrapper.disabled is True
    rf.set_enabled(True)
    assert rf._wrapper.disabled is False


def test_case_type_readonly_stays_locked_when_unfrozen(fake_app: MagicMock):
    """The Run tab mounts the form with case_type_readonly=True: the Case Type
    radio is disabled independent of the wrapper freeze — it stays locked even
    after set_enabled(True) unfreezes the rest. It still shows the loaded value
    and writes it back unchanged. A default mount leaves the radio editable."""
    fake_app.gui_config.llm_provider = "anthropic"
    ro = RecipeForm(fake_app, case_type_readonly=True)
    ro.build()
    assert ro.profile_group.disabled is True  # Case Type read-only...
    assert ro._wrapper.disabled is False  # ...the rest editable
    ro.set_enabled(True)  # unfreeze the rest
    assert ro.profile_group.disabled is True  # STILL locked (owned by Create test cases)
    ro.load(EvalRecipe(dataset_kind="knob"))
    assert ro.profile_group.value == "knob"  # displays the loaded value
    assert ro.to_recipe().dataset_kind == "knob"  # writes it back unchanged

    assert not _form(fake_app).profile_group.disabled  # default mount → editable


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
