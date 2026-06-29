"""Top-level pytest configuration + shared fixtures.

Fixtures defined here are available to every test in the suite —
unit/, integration/, and e2e/. Use this for genuinely cross-cutting
helpers (e.g. a `tmp_corpus_dir` fixture that builds a fake corpus
folder, useful across tiers).

Per-tier specifics live in the nested `conftest.py` files:
  - tests/unit/conftest.py        — unit-only (no real services)
  - tests/integration/conftest.py — real Neo4j / LanceDB / Voyage
  - tests/e2e/conftest.py         — full app launch fixtures

Keep this file lean: anything specific to one tier belongs there,
not here, so unit-test imports don't pay for integration-only
fixture wiring.
"""
