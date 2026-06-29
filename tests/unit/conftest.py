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
