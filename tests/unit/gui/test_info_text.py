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
