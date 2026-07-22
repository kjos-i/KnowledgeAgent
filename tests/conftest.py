"""Top-level pytest configuration + shared fixtures.

Fixtures defined here are available to every test in the suite —
unit/, integration/, and e2e/. Use this for genuinely cross-cutting
helpers (e.g. a `tmp_corpus_dir` fixture that builds a fake corpus
folder, useful across tiers). Keep it lean: anything specific to one
tier belongs in that tier's nested conftest, not here.

===========================================================================
Test-suite conventions (THE single source — kept here so it's easy to
rediscover a week later).
===========================================================================

Three tiers, mirroring `src/knowledge_agent/` subdir-for-subdir:

  - tests/unit/         — fast, pure-Python, NO real services. Mocks +
                          `tmp_path` + isolated `os.environ`. Runs by
                          DEFAULT (`pytest`).
  - tests/integration/  — hit the real test-instance Neo4j + LanceDB (+
                          Voyage/Anthropic when needed). Every test marked
                          `@pytest.mark.integration`; SKIPPED by default,
                          opt in via `pytest -m integration`.
  - tests/e2e/          — full backend user-journeys (ingest → query →
                          answer). Marked `@pytest.mark.e2e`; opt in via
                          `pytest -m e2e`. No headless-Flet driver exists
                          (2026), so e2e drives the same BACKEND the GUI
                          calls, not the Flet UI (see tests/e2e/conftest).

  `addopts` in pyproject.toml carries `-m "not integration and not e2e"`
  so the default run is unit-only; `--strict-markers` makes the `markers`
  list in pyproject the single source of truth.

THE SAFETY INVARIANT — tests/smokes only ever touch the TEST instance,
never real data:

  - Any tier that touches a real service calls `config.load_test_env()`
    FIRST, which loads `.env.test` (a separate Neo4j Desktop instance +
    a `lancedb_test` path). Smoke scripts call it at the very TOP, before
    any import that could trigger `get_settings()`.
  - Isolation is enforced by PASSWORD, not by convention: the test
    instance's password differs from the real one, so a wrong-instance
    cross-connect fails AUTH rather than corrupting real data.
  - Belt-and-suspenders: all four DB settings (`neo4j_uri/user/password`,
    `lancedb_path`) are required-no-default, so a missing/forgotten value
    raises at Settings construction instead of falling back to something
    that points at the real corpus.
  - The GUI never reads `.env` at all (`disable_env_file()` + keyring →
    env bridge). `.env` = the CLI/dev path only.

WIPE conventions:

  - integration: `clean_kg` (`MATCH (n) DETACH DELETE n`) and `clean_lance`
    (drop chunks table) wipe the test stores BEFORE each test that mutates
    them — no after-wipe (next test's before-wipe covers it).
  - smokes (persistent-write): clear-at-start → write → PAUSE for manual
    inspection → optional cleanup-at-end (Enter cleans, Ctrl+C keeps).
    See scripts/smoke_kg_l1_l5.py.
  - install smokes: a different lifecycle (plan → confirm → install →
    verify → pause → offer-uninstall); see scripts/_install_smoke_lib.py.

COVERAGE philosophy (what "covered" means here):

  - Backend logic is covered by unit (mocked) + integration (real). Modules
    with no test file NAMED after them (e.g. reconcile, metadata_resolution,
    the ontology loaders) are exercised through their callers'/`*_writes`
    tests; reconcile also has a dedicated test_reconcile.py for its
    destructive paths.
  - Pure-layout GUI shells (tabs/*, views/*, _styles, _frame,
    resizable_split) are exercised by the view/tab build-tests
    (test_tabs.py, test_views.py, …) — no per-widget unit test.
  - There is NO line-coverage gate (pytest-cov not wired); coverage is by
    module-mirroring + integration, not a measured %.
"""
