"""KnowledgeAgent — LangGraph agent for research literature with LanceDB + Neo4j."""

__version__ = "0.1.0"

DISTRIBUTION_NAME = "knowledge-agent"
"""The pip distribution name — matches `[project].name` in pyproject.toml.

Single source for every `pip install <DISTRIBUTION_NAME>[extra]` install
target and `importlib.metadata` lookup, so a future rename touches ONE line.
Deliberately NOT the same constant as the config/keyring app identity
(`config.APP_NAME`), which stays independent so a dist rename never
relocates a user's stored settings / keyring entries."""

__all__ = ["DISTRIBUTION_NAME", "__version__"]
