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
