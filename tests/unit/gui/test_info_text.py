"""Guard the (i) help-icon text registry stays well-formed.

Cheap structural checks so a typo'd key or an empty entry fails in CI
rather than rendering a blank / broken icon in the GUI.
"""

from knowledge_agent.gui._widgets.info_icon import InfoText
from knowledge_agent.gui._widgets.info_text import INFO


def test_registry_not_empty():
    assert INFO, "INFO registry should have entries"


def test_every_entry_well_formed():
    for key, spec in INFO.items():
        assert isinstance(spec, InfoText), f"{key}: not an InfoText"
        assert spec.title and spec.title.strip(), f"{key}: empty title"
        # at least one tier carries text — an entry with none renders nothing
        assert any((spec.standard, spec.beginner, spec.technical)), f"{key}: no tier text"


def test_keys_namespaced_and_lowercase():
    for key in INFO:
        assert key == key.lower(), f"{key}: keys should be lowercase"
        assert "." in key, f"{key}: keys should be namespaced '<area>.<control>'"


def test_documents_table_content_icons_present():
    """The two 'Table contents' icons under the Documents filter (cards +
    actions, and the Edit dialog) are registered with all three tiers filled."""
    for key in ("select.documents_cards", "select.documents_edit"):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"


def test_installs_section_icons_present():
    """The Installs tab overview + 5 section icons are registered with all
    three tiers filled (they render from the registry via section_header key=)."""
    for key in (
        "installs.overview",
        "installs.llm_providers",
        "installs.embedding_providers",
        "installs.parsers",
        "installs.entity_extractors",
        "installs.ontologies",
    ):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"


def test_keys_overview_icon_present():
    """The Keys tab overview icon is registered with all three tiers filled."""
    spec = INFO["keys.overview"]
    assert spec.standard and spec.beginner and spec.technical, "keys.overview: a tier is empty"


def test_settings_section_icons_present():
    """The Settings tab (AppTab) section headers + system-health board render
    from the registry, all under the settings.* prefix with three tiers filled
    (system_health was renamed from diagnostics.system_health, not duplicated)."""
    for key in (
        "settings.save_results",
        "settings.save_chat",
        "settings.app_behaviour",
        "settings.diagnostics",
        "settings.database_connection",
        "settings.system_health",
    ):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"
    assert "diagnostics.system_health" not in INFO, "old key should be renamed, not kept"


def test_llms_and_retrieval_section_icons_present():
    """The LLMs + Retrieval sub-tabs render their section headers from the
    registry (migrated out of local module constants); every key has all three
    tiers filled."""
    for key in (
        "llms.installed_providers",
        "llms.models",
        "llms.rate_limits",
        "retrieval.mode",
        "retrieval.result_size",
        "retrieval.rrf",
        "retrieval.mmr",
        "retrieval.knowledge_graph",
        "retrieval.input_mode",
    ):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"


def test_log_overview_icon_present():
    """The Log tab's single overview icon (after the title) is registered with
    all three tiers filled."""
    spec = INFO["log.overview"]
    assert spec.standard and spec.beginner and spec.technical, "log.overview: a tier is empty"


def test_eval_run_section_icons_present():
    """The Evaluation Run sub-tab icons (2 column titles + 3 sections + 4
    recipe/run sub-sections; the recipe ones are shared with Create test
    cases) are registered with all three tiers filled."""
    for key in (
        "eval_run.overview",
        "eval_run.test_cases",
        "eval_run.evaluation_cases",
        "eval_run.run_options",
        "eval_run.run",
        "eval_run.metric_groups",
        "eval_run.judge_panel",
        "eval_run.gate_thresholds",
        "eval_run.tracing",
    ):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"


def test_eval_dataset_section_icons_present():
    """The Evaluation Create-test-cases sub-tab icons (2 column titles + 4
    section headers + 3 per-case-form sub-sections + the suite panel) are
    registered with all three tiers filled."""
    for key in (
        "eval_dataset.editor_overview",
        "eval_dataset.cases_overview",
        "eval_dataset.evaluation_cases",
        "eval_dataset.progress",
        "eval_dataset.add_cases",
        "eval_dataset.dataset_cases",
        "eval_dataset.form_information",
        "eval_dataset.form_facts",
        "eval_dataset.form_retrieval",
        "eval_dataset.suite",
    ):
        spec = INFO[key]
        assert spec.standard and spec.beginner and spec.technical, f"{key}: a tier is empty"


def test_eval_dashboard_overview_icon_present():
    """The shared Evaluation dashboard left-column icon (one 3-tier icon at the
    top of the rail, right of Refresh) is registered with all three tiers filled."""
    spec = INFO["eval_dashboard.overview"]
    assert spec.standard and spec.beginner and spec.technical, (
        "eval_dashboard.overview: a tier is empty"
    )
