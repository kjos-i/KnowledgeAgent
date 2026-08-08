"""Unit-tier shared fixtures + pytest hooks.

Scope: tests under `tests/unit/` only — fast, pure-Python, no real
DB / HTTP / Voyage / Anthropic calls. Anything in here that boots a
real service belongs in `tests/integration/conftest.py` instead.

Subdirectories (kg/, ingestion/, entity_extractors/, search/, gui/)
mirror `src/knowledge_agent/`. Per-subdirectory conftest
files can be added later if a subdir grows fixtures unique to its
modules; until then, this file is the single conftest serving them
all.
"""

import pytest


@pytest.fixture(autouse=True)
def _block_real_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-block the developer's real `.env` for every unit test.

    `Settings` reads `.env` (`env_file=.env`) for any field absent from the
    environment, so a unit test building `Settings()` (or calling
    `get_settings()`) would silently pick up the developer's real API keys and
    DB creds from `.env`. Flip the same `disable_env_file()` switch the GUI uses
    so `get_settings()` consults ONLY `os.environ`: the opt-in provider API keys
    resolve to `None`, and the four required-no-default connection fields are
    supplied as fakes here (`NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` for the
    graph connection, `LANCEDB_PATH`). Tests that construct `Settings` explicitly
    via `_minimal_required()` are unaffected. Tests must never reach `.env`; a
    value the fakes / `.env.test` don't provide fails loudly instead of falling
    back to real credentials.
    """
    from knowledge_agent import config

    monkeypatch.setattr(config, "_env_file_disabled", True)
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "fake-neo4j-password")  # pragma: allowlist secret
    monkeypatch.setenv("LANCEDB_PATH", "./lancedb")
    config.get_settings.cache_clear()
