"""Tests for the Evaluation Dataset sub-tab (browse + edit, slice 3).

Loads a tiny gold dataset from tmp_path and exercises: list + select → form
populate, New/Save (creates a file), Edit (persists), Delete (persists), and
invalid-input handling. Writes go through the real `save_dataset`, so the
round-trips reload from disk to verify.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from knowledge_agent.evaluation.models import EvalCase, load_dataset
from knowledge_agent.gui.evaluation.dataset_tab import DatasetTab


def _dataset_file(tmp_path) -> object:
    cases = [
        {
            "id": "c1",
            "question": "What drives membrane scission?",
            "required_keywords": ["ESCRT-III", "scission"],
            "retrieval": {
                "retrieval_mode": "neo4j_only",
                "direct_retrieval": True,
                "kg_max_rows": 50,
            },
            "user_cypher": "MATCH (e:Entity) RETURN e.key LIMIT 5",
            "origin": "llm",
        },
        {"id": "c2", "question": "second question"},
    ]
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(cases), encoding="utf-8")
    return p


def _tab(fake_app, path=None) -> DatasetTab:
    tab = DatasetTab(fake_app, coordinator=MagicMock())
    tab.build()
    # build() auto-loads the packaged default dataset (50a); reset to a clean
    # slate so each test controls its own dataset context and never writes the
    # shipped default. Pass `path` to target a specific (tmp) file.
    tab._dataset = None
    tab._cases = []
    tab._path = None
    tab._selected = None
    tab.dataset_field.value = str(path) if path else ""
    return tab


def test_dataset_tab_builds(fake_app):
    assert DatasetTab(fake_app, coordinator=MagicMock()).build() is not None


def test_gen_temp_slider_greyed_for_sampling_free_model(fake_app):
    """Picking a sampling-free case-generation model (e.g. Opus 4.8) greys
    out the generation temperature slider; a temp-taking model (Haiku 4.5)
    re-enables it."""
    fake_app.gui_config.llm_provider = "anthropic"
    tab = _tab(fake_app)
    tab.gen_model_dropdown.value = "claude-opus-4-8"
    tab._on_gen_model_changed()
    assert tab.gen_temp_slider.disabled is True
    assert "temperature" in (tab.gen_temp_slider.tooltip or "")

    tab.gen_model_dropdown.value = "claude-haiku-4-5"
    tab._on_gen_model_changed()
    assert tab.gen_temp_slider.disabled is False
    assert tab.gen_temp_slider.tooltip is None


def test_load_and_select_populates_form(fake_app, tmp_path):
    tab = _tab(fake_app)
    tab._load(_dataset_file(tmp_path))
    assert len(tab._cases) == 2
    tab._select(0)
    assert tab.f["id"].value == "c1"
    assert tab.f["retrieval_mode"].value == "neo4j_only"
    assert tab.f["direct_retrieval"].value is True
    assert tab.f["user_cypher"].value == "MATCH (e:Entity) RETURN e.key LIMIT 5"
    assert "ESCRT-III" in tab.f["required_keywords"].value  # one-per-line text
    assert tab.f["origin"].value == "llm"


def test_load_bad_dataset_surfaces_error(fake_app, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("42", encoding="utf-8")  # JSON scalar → neither array nor object
    tab = _tab(fake_app)
    tab._load(bad)
    assert "could not load" in tab.status.value
    assert tab._cases == []


def test_new_case_saves_to_file(fake_app, tmp_path):
    p = tmp_path / "new.json"
    tab = _tab(fake_app, p)
    tab._on_new(MagicMock())  # knobs pre-fill with the standard defaults
    tab.f["id"].value = "brandnew"
    tab.f["question"].value = "a new question?"
    tab.f["required_keywords"].value = "alpha\nbeta"
    tab.status_group.value = "final"
    tab._on_save_case(MagicMock())

    assert p.exists()
    ds = load_dataset(p)
    assert ds.status == "final"  # status header rode along (name field dropped)
    assert [c.id for c in ds.cases] == ["brandnew"]
    assert ds.cases[0].required_keywords == ["alpha", "beta"]  # lines → list


def test_edit_selected_case_persists(fake_app, tmp_path):
    p = _dataset_file(tmp_path)
    tab = _tab(fake_app)
    tab._load(p)
    tab._select(0)
    tab.f["question"].value = "edited question?"
    tab._on_save_case(MagicMock())

    reloaded = load_dataset(p)
    assert reloaded.cases[0].question == "edited question?"
    assert len(reloaded.cases) == 2  # edited in place, not appended


def test_delete_case_persists(fake_app, tmp_path):
    p = _dataset_file(tmp_path)
    tab = _tab(fake_app)
    tab._load(p)
    # Each case card's Delete button calls _delete_case with its index.
    tab._delete_case(0)

    assert [c.id for c in load_dataset(p).cases] == ["c2"]


def test_save_invalid_case_surfaces_error(fake_app, tmp_path):
    p = tmp_path / "x.json"
    tab = _tab(fake_app, p)
    tab._on_new(MagicMock())
    tab.f["id"].value = ""  # id is required → ValidationError
    tab.f["question"].value = "q?"
    tab._on_save_case(MagicMock())

    assert "invalid case" in tab.status.value
    assert not p.exists()  # nothing written on invalid input


def test_capture_from_search_prefills_new_case(fake_app):
    fake_app.last_query = "why did the valve fail?"
    fake_app.last_answer = SimpleNamespace(
        chunk_sources=[
            SimpleNamespace(doc_id="doc_a"),
            SimpleNamespace(doc_id="doc_b"),
            SimpleNamespace(doc_id="doc_a"),  # dup → collapsed
        ]
    )
    tab = _tab(fake_app)
    tab._on_capture_from_search(MagicMock())
    assert tab._selected is None  # new-case mode
    assert tab.f["question"].value == "why did the valve fail?"
    assert tab.f["origin"].value == "search"
    assert tab.f["expected_sources"].value == "doc_a\ndoc_b"  # deduped, order-preserved


def test_capture_from_search_no_result(fake_app):
    fake_app.last_answer = None
    fake_app.last_query = None
    tab = _tab(fake_app)
    tab._on_capture_from_search(MagicMock())
    assert "No search result" in tab.status.value


async def test_generate_multiple_appends_candidates_and_saves(fake_app, tmp_path):
    """Generate multiple drafts candidates via the backend, appends them
    (origin=llm) to the dataset, and saves to the path."""
    p = tmp_path / "gen.json"
    tab = _tab(fake_app, p)
    tab.gen_model_dropdown.value = "some-model"  # chosen generation model
    fake_cases = [
        EvalCase(id="gen-00-a", question="A?", origin="llm", expected_sources=["d1"]),
        EvalCase(id="gen-01-b", question="B?", origin="llm", expected_sources=["d2"]),
    ]
    with patch(
        "knowledge_agent.evaluation.generator.generate_from_corpus",
        new_callable=AsyncMock,
        return_value=fake_cases,
    ) as gen_mock:
        await tab._run_generate_multiple(2, p)

    gen_mock.assert_awaited_once_with(
        2, model="some-model", temperature=0.3
    )  # picker threads through
    ds = load_dataset(p)
    assert [c.id for c in ds.cases] == ["gen-00-a", "gen-01-b"]
    assert all(c.origin == "llm" for c in ds.cases)
    assert "generated 2" in tab.status.value


def test_generate_multiple_requires_dataset_path(fake_app):
    tab = _tab(fake_app)  # no path → cleared dataset field
    tab.gen_model_dropdown.value = "some-model"  # past the required-model guard
    tab._on_generate_multiple(MagicMock())
    assert "choose a dataset" in tab.status.value.lower()


def test_generate_multiple_confirms_before_running(fake_app, tmp_path):
    """Generate multiple asks first — it opens a confirm dialog rather than
    generating immediately (nothing written until you confirm)."""
    p = tmp_path / "gen.json"
    tab = _tab(fake_app, p)
    tab.gen_model_dropdown.value = "some-model"  # past the required-model guard
    tab.gen_count.value = "3"
    tab._on_generate_multiple(MagicMock())
    fake_app.page.show_dialog.assert_called_once()
    assert not p.exists()  # nothing generated or written yet


async def test_generate_one_fills_form_without_saving(fake_app, tmp_path):
    """Generate one drafts a single candidate into the form for review — it
    fills the form (origin=llm) and writes nothing until Add case."""
    p = tmp_path / "one.json"
    tab = _tab(fake_app, p)
    tab.gen_model_dropdown.value = "some-model"  # past the required-model guard
    candidate = EvalCase(id="draft-1", question="Q?", origin="llm", required_keywords=["k"])
    with patch(
        "knowledge_agent.evaluation.generator.generate_from_corpus",
        new_callable=AsyncMock,
        return_value=[candidate],
    ):
        await tab._on_generate_one(MagicMock())
    assert tab.f["id"].value == "draft-1"
    assert tab.f["origin"].value == "llm"
    assert tab._selected is None  # new case → commits via Add
    assert not p.exists()  # nothing written until Add case


def test_build_opens_empty(fake_app):
    """No baked-in default any more: the tab opens with an empty dataset field
    and nothing loaded — the user Browses / New dataset in the corpus folder."""
    tab = DatasetTab(fake_app, coordinator=MagicMock())
    tab.build()
    assert tab.dataset_field.value == ""
    assert tab._dataset is None
    assert tab._path is None


def test_commit_buttons_enable_by_mode(fake_app, tmp_path):
    """Two commit buttons: Add is live (Update grey) for a new case; loading an
    existing case from the list flips it — Update live, Add grey."""
    tab = _tab(fake_app)
    tab._on_new(MagicMock())
    assert tab.add_button.disabled is False
    assert tab.update_button.disabled is True
    tab._load(_dataset_file(tmp_path))
    tab._select(0)
    assert tab.add_button.disabled is True
    assert tab.update_button.disabled is False


def test_cancel_edit_clears_form_and_deselects(fake_app, tmp_path):
    """The card's Cancel button blanks the form and deselects — back to new
    mode (Add live, Update grey)."""
    tab = _tab(fake_app)
    tab._load(_dataset_file(tmp_path))
    tab._select(0)
    assert tab._selected == 0
    assert tab.f["id"].value == "c1"
    tab._on_cancel_edit()
    assert tab._selected is None
    assert tab.f["id"].value == ""
    assert tab.add_button.disabled is False
    assert tab.update_button.disabled is True


def test_new_dataset_creates_empty_file(fake_app, tmp_path):
    """New dataset writes an empty but valid JSON immediately (Create = create)
    and makes it the current dataset."""
    p = tmp_path / "fresh.json"
    tab = _tab(fake_app)
    tab._start_new_dataset(p)
    assert p.exists()
    assert load_dataset(p).cases == []
    assert tab._path == p
    assert tab.dataset_field.value == str(p)


def test_gray_out_reflects_mode(fake_app):
    """The retrieval knobs the chosen mode can't use are grayed out (disabled),
    same rule as Settings → Retrieval: neo4j_only disables the LanceDB block;
    lancedb_only disables kg_max_rows."""
    tab = _tab(fake_app)
    tab.f["retrieval_mode"].value = "neo4j_only"
    tab._sync_retrieval_gray_out()
    # Same shared radios as Settings: each radio disabled + the block faded.
    assert all(r.disabled for r in tab._lancedb_radios)
    assert tab._lancedb_mode_box.opacity == 0.4
    assert tab.f["num_candidates"].disabled is True
    assert tab.f["rrf_rank_constant"].disabled is True
    assert tab.f["kg_max_rows"].disabled is False  # the Neo4j leg runs

    tab.f["retrieval_mode"].value = "lancedb_only"
    tab.f["lancedb_search_mode"].value = "hybrid"
    tab._sync_retrieval_gray_out()
    assert not any(r.disabled for r in tab._lancedb_radios)
    assert tab._lancedb_mode_box.opacity == 1.0
    assert tab.f["num_candidates"].disabled is False
    assert tab.f["rrf_rank_constant"].disabled is False
    assert tab.f["kg_max_rows"].disabled is True  # no Neo4j leg


def test_input_mode_maps_to_case_fields(fake_app, tmp_path):
    """The Input-mode radio maps to the case's skip_query_builder / user_cypher:
    Direct query → skip_query_builder; Direct Cypher → user_cypher; Refined →
    neither (the intuitive radio replacing the scattered checkbox/field)."""
    p = tmp_path / "im.json"
    tab = _tab(fake_app, p)
    tab._on_new(MagicMock())
    tab.f["id"].value = "dq"
    tab.f["question"].value = "q?"
    tab.f["input_mode"].value = "direct_query"
    tab._on_save_case(MagicMock())
    c = load_dataset(p).cases[-1]
    assert c.retrieval.skip_query_builder is True
    assert c.user_cypher is None

    tab._on_new(MagicMock())
    tab.f["id"].value = "dc"
    tab.f["question"].value = "q?"
    tab.f["input_mode"].value = "direct_cypher"
    tab.f["user_cypher"].value = "MATCH (n) RETURN n LIMIT 5"
    tab._on_save_case(MagicMock())
    c = load_dataset(p).cases[-1]
    assert c.user_cypher == "MATCH (n) RETURN n LIMIT 5"
    assert c.retrieval.skip_query_builder is False


def test_input_mode_derived_on_load(fake_app, tmp_path):
    """Loading a case sets the radio: user_cypher → Direct Cypher, else a plain
    case → Refined query."""
    tab = _tab(fake_app)
    tab._load(_dataset_file(tmp_path))
    tab._select(0)  # c1 carries user_cypher
    assert tab.f["input_mode"].value == "direct_cypher"
    tab._select(1)  # c2 is a plain case
    assert tab.f["input_mode"].value == "refined"


def test_recipe_saved_with_dataset(fake_app, tmp_path):
    """The recipe authored on the tab persists with the dataset on a case save
    — it rides along like the status header (`_stamp_header`)."""
    p = tmp_path / "r.json"
    tab = _tab(fake_app, p)
    tab._on_new(MagicMock())
    tab.f["id"].value = "c1"
    tab.f["question"].value = "q?"
    # Pick the Knob profile on the recipe form (judge off, gates at 0).
    tab.recipe_form.profile_group.value = "knob"
    tab.recipe_form._on_profile_change(MagicMock())
    tab._on_save_case(MagicMock())

    ds = load_dataset(p)
    assert ds.recipe is not None
    assert ds.recipe.dataset_kind == "knob"
    assert "judge" not in ds.recipe.enabled_groups


def test_recipe_loaded_from_dataset(fake_app, tmp_path):
    """Loading a dataset populates the recipe form from its saved recipe."""
    from knowledge_agent.evaluation.models import EvalDataset, EvalRecipe, save_dataset

    p = tmp_path / "r.json"
    save_dataset(
        EvalDataset(
            recipe=EvalRecipe(dataset_kind="fact", judge_threshold=0.9),
            cases=[EvalCase(id="c", question="q?")],
        ),
        p,
    )
    tab = _tab(fake_app)
    tab._load(p)
    assert tab.recipe_form.profile_group.value == "fact"
    assert tab.recipe_form.threshold_fields["judge_threshold"].value == "0.9"


def _frozen_file(tmp_path):
    """A final + frozen one-case dataset with a recipe; return its path."""
    from knowledge_agent.evaluation.models import EvalDataset, EvalRecipe, save_dataset

    p = tmp_path / "frozen.json"
    save_dataset(
        EvalDataset(
            status="final",
            frozen=True,
            recipe=EvalRecipe(dataset_kind="fact"),
            cases=[EvalCase(id="c", question="q?")],
        ),
        p,
    )
    return p


def test_frozen_dataset_recipe_readonly(fake_app, tmp_path):
    """Loading a frozen dataset makes the recipe read-only + shows Unfreeze."""
    tab = _tab(fake_app)
    tab._load(_frozen_file(tmp_path))
    assert tab.recipe_form._wrapper.disabled is True  # recipe read-only
    assert tab.unfreeze_button.visible is True
    assert tab.frozen_indicator.visible is True


def test_dataset_unfreeze_persists(fake_app, tmp_path):
    """Unfreeze clears the frozen flag on disk and re-enables the recipe."""
    from knowledge_agent.evaluation.models import load_dataset

    p = _frozen_file(tmp_path)
    tab = _tab(fake_app)
    tab._load(p)
    tab._do_unfreeze()
    assert load_dataset(p).frozen is False
    assert tab.recipe_form._wrapper.disabled is False  # editable again


def test_status_to_draft_clears_frozen(fake_app, tmp_path):
    """Setting status below final clears the frozen lock in-session (invariant),
    re-enabling the recipe."""
    tab = _tab(fake_app)
    tab._load(_frozen_file(tmp_path))
    assert tab._dataset.frozen is True
    tab.status_group.value = "draft"
    tab._on_status_change(MagicMock())
    assert tab._dataset.frozen is False
    assert tab.recipe_form._wrapper.disabled is False


def test_new_knobs_round_trip(fake_app, tmp_path):
    """The new per-case knobs are read from the form and persisted on the case."""
    p = tmp_path / "k.json"
    tab = _tab(fake_app, p)
    tab._on_new(MagicMock())
    tab.f["id"].value = "k1"
    tab.f["question"].value = "q?"
    tab.f["num_candidates"].value = "80"
    tab.f["rrf_rank_constant"].value = "45"
    tab.f["use_mmr"].value = True
    tab.f["mmr_lambda"].value = 0.7  # mmr_lambda is a slider → float
    tab._on_save_case(MagicMock())

    rs = load_dataset(p).cases[0].retrieval
    assert rs.num_candidates == 80
    assert rs.rrf_rank_constant == 45
    assert rs.use_mmr is True
    assert rs.mmr_lambda == 0.7
