"""SSOT guard for the pip distribution name.

The lifecycle code once defaulted `distribution_name` to the pre-rename
`research-literature-agent`, so every extractor / parser install ran
`pip install research-literature-agent[...]` and failed (release-blocker).
`knowledge_agent.DISTRIBUTION_NAME` is now the single source; these tests
pin it to pyproject's `[project].name` and to the four install-execute
defaults so a future rename can't drift back.
"""

import inspect
import tomllib
from pathlib import Path

from knowledge_agent import DISTRIBUTION_NAME
from knowledge_agent.embedder_lifecycle import install_embedder_provider_execute
from knowledge_agent.entity_extractors.extractor_lifecycle import install_extractor_execute
from knowledge_agent.ingestion.parser_lifecycle import install_parser_extra_execute
from knowledge_agent.llm_lifecycle import install_llm_provider_execute


def test_distribution_name_is_knowledge_agent():
    assert DISTRIBUTION_NAME == "knowledge-agent"


def test_distribution_name_matches_pyproject_project_name():
    """SSOT: the constant must equal pyproject's [project].name, so a rename
    there is caught here rather than silently breaking every install."""
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["name"] == DISTRIBUTION_NAME


def test_install_execute_defaults_use_distribution_name():
    """Every install-execute's `distribution_name` default must be the shared
    constant, not the stale pre-rename literal (the release-blocker)."""
    for fn in (
        install_extractor_execute,
        install_parser_extra_execute,
        install_embedder_provider_execute,
        install_llm_provider_execute,
    ):
        default = inspect.signature(fn).parameters["distribution_name"].default
        assert default == DISTRIBUTION_NAME, f"{fn.__name__} default is {default!r}"
        assert default != "research-literature-agent"
