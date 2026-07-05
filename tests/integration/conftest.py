"""Integration-tier shared fixtures + pytest hooks.

Scope: tests under `tests/integration/` — hit real Neo4j (via the
test-instance credentials in `.env.test`), real LanceDB (at
`LANCEDB_PATH` from `.env.test`), real Voyage/Anthropic when needed.
Per [[test-instance-setup]] the test instance is isolated by
password from the real corpus so a cross-connect fails auth rather
than corrupting real data.

Every test in this tier MUST be decorated with `@pytest.mark.integration`
— the default `pytest` run skips this tier (`-m "not integration and
not e2e"` in pyproject.toml). Opt in via `pytest -m integration`.

Fixtures shipped here:

  - `kg_client` — module-scoped `Neo4jClient` pointing at the test
    instance. Built once per file, closed at teardown. Calls
    `load_test_env()` before the first construction so the smoke
    instance is targeted, not the real one.

  - `clean_kg` — function-scoped: wipes the test instance via
    `MATCH (n) DETACH DELETE n` before each test, so tests don't
    contaminate one another. Use this whenever a test mutates the
    graph state.

  - `ensure_constraints` — session-scoped: runs once at the start
    of the integration run to install schema constraints.

Run integration tests:
    pytest -m integration                    # this tier only
    pytest -m integration tests/integration/kg/  # just the kg subset
    pytest                                   # default: skips integration

When this conftest grows, also create per-subdirectory conftest
files (kg/, ingestion/, search/) for subdir-specific fixtures —
keep this file for cross-subdir helpers only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from knowledge_agent.config import load_test_env

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def _test_env_loaded() -> None:
    """Switch the process to the smoke-test Neo4j instance ONCE per
    pytest session, before any other RLA import that might trigger
    `get_settings()` against the real `.env`.

    Per [[test-instance-setup]] the test instance has a different
    password so a wrong-instance state fails auth rather than
    corrupting real data.
    """
    load_test_env()


@pytest.fixture(scope="module")
def kg_client(_test_env_loaded: None) -> Iterator[Any]:
    """Module-scoped `Neo4jClient` pointing at the test instance.

    The driver is the connection pool (one per app), so building it
    per file keeps wire overhead low without leaking state across
    test files. Each test still gets the same client instance for the
    duration of its module.

    `close()` is async post-refactor; we drive it via asyncio.run on
    teardown since the fixture itself is sync (module-scoped fixtures
    can't be async generators that yield across many tests).
    """
    import asyncio

    from knowledge_agent.kg.client import Neo4jClient

    client = Neo4jClient()
    try:
        yield client
    finally:
        asyncio.run(client.close())


@pytest.fixture(scope="session")
def ensure_constraints(_test_env_loaded: None) -> None:
    """Install schema constraints once at the start of the integration
    run. Constraints are idempotent (`IF NOT EXISTS`) so re-running
    is a no-op."""
    import asyncio

    from knowledge_agent.kg.client import Neo4jClient

    async def _run() -> None:
        client = Neo4jClient()
        try:
            await client.ensure_constraints()
        finally:
            await client.close()

    asyncio.run(_run())


@pytest.fixture
def clean_kg(kg_client: Any) -> Iterator[None]:
    """Wipe the test KG before each test that uses this fixture.

    Pattern: `MATCH (n) DETACH DELETE n`. The test instance is for
    test artefacts only; per [[test-instance-setup]] it's isolated
    by password from the real corpus, so a wipe here cannot reach
    real data.

    Use this fixture whenever a test mutates the graph. Don't use
    it for read-only inspections of pre-existing state (there should
    not BE pre-existing state at this tier, but be explicit).
    """
    with kg_client.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    yield


# ---------------------------------------------------------------------------
# LanceDB fixtures — mirror the KG side but for the LanceDB
# `LANCEDB_PATH` from `.env.test`.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lance_client(_test_env_loaded: None) -> Iterator[Any]:
    """Module-scoped `LanceClient` pointing at the LanceDB path from
    `.env.test`. The connection is a handle to an on-disk directory
    (no process, no port); one per file keeps it cheap."""
    import asyncio

    from knowledge_agent.search.client import LanceClient

    client = LanceClient()
    try:
        yield client
    finally:
        asyncio.run(client.close())


@pytest.fixture
def clean_lance(lance_client: Any) -> Iterator[None]:
    """Drop the chunks table before each test that uses this fixture.

    `drop_chunks_table()` is idempotent — no-op when the table
    doesn't exist yet. After teardown the next `ensure_schema()` call
    recreates the table from the current `chunks_schema()` — so each
    test starts from an empty, schema-fresh table.
    """
    import asyncio

    asyncio.run(lance_client.drop_chunks_table())
    yield


# ---------------------------------------------------------------------------
# Sample-document fixtures — pick whichever supported file is in
# `test_documents/` so the integration suite doesn't break when the
# corpus rotates. Tests that need a specific format (PDF vs DOCX vs
# JATS XML) take the matching fixture.
# ---------------------------------------------------------------------------


TEST_DOCS = Path(__file__).resolve().parents[2] / "test_documents"
"""Path to the test_documents/ directory at project root, shared
across all integration tests. Single source of truth — the fixtures
below resolve real files relative to this.

`parents[2]` because this conftest lives at
`tests/integration/conftest.py`: parents[0]=integration/,
parents[1]=tests/, parents[2]=project root."""


def _pick_first(extension: str) -> Path | None:
    """First file with the given extension (sorted for determinism)."""
    matches = sorted(TEST_DOCS.glob(f"*{extension}"))
    return matches[0] if matches else None


def _pick_first_of(*extensions: str) -> Path | None:
    """First file matching any of the given extensions (sorted, dedup).

    For format families that span multiple suffixes (image = png/jpg/...,
    audio = m4a/mp3/..., video = mp4/mov/...) — the parse layer treats
    them as one category, so the fixture surfaces whichever is in the
    folder first.
    """
    matches: list[Path] = []
    for ext in extensions:
        matches.extend(TEST_DOCS.glob(f"*{ext}"))
    return sorted(set(matches))[0] if matches else None


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    """First `.pdf` file in `test_documents/` (sorted for determinism).

    Lets tests swap PDFs freely — drop a new file in, replace the old
    one, no test code touched. Skips the test (rather than crashing
    with a confusing FileNotFoundError deep in docling) if the folder
    contains no PDF.
    """
    pdf = _pick_first(".pdf")
    if pdf is None:
        pytest.skip(f"no PDF fixtures in {TEST_DOCS}")
    return pdf


@pytest.fixture(scope="session")
def sample_jats_xml() -> Path:
    """First `.xml` file in `test_documents/` (assumed JATS-shaped).

    Same flexibility as `sample_pdf`. Skips when absent."""
    xml = _pick_first(".xml")
    if xml is None:
        pytest.skip(f"no XML fixtures in {TEST_DOCS}")
    return xml


@pytest.fixture(scope="session")
def sample_docx() -> Path:
    """First `.docx` file in `test_documents/`. Skips when absent."""
    docx = _pick_first(".docx")
    if docx is None:
        pytest.skip(f"no DOCX fixtures in {TEST_DOCS}")
    return docx


# ---- Office formats (pptx, xlsx) — docling native support, no extra needed.


@pytest.fixture(scope="session")
def sample_pptx() -> Path:
    """First `.pptx` file in `test_documents/`."""
    p = _pick_first(".pptx")
    if p is None:
        pytest.skip(f"no PPTX fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_xlsx() -> Path:
    """First `.xlsx` file in `test_documents/`."""
    p = _pick_first(".xlsx")
    if p is None:
        pytest.skip(f"no XLSX fixtures in {TEST_DOCS}")
    return p


# ---- Markup / structured text (html, md, tex, adoc, csv, vtt).
# All docling-native in the base install.


@pytest.fixture(scope="session")
def sample_html() -> Path:
    """First `.html` file in `test_documents/`."""
    p = _pick_first(".html")
    if p is None:
        pytest.skip(f"no HTML fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_md() -> Path:
    """First `.md` Markdown file in `test_documents/`."""
    p = _pick_first(".md")
    if p is None:
        pytest.skip(f"no Markdown fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_tex() -> Path:
    """First `.tex` LaTeX file in `test_documents/`."""
    p = _pick_first(".tex")
    if p is None:
        pytest.skip(f"no LaTeX fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_adoc() -> Path:
    """First `.adoc` AsciiDoc file in `test_documents/`."""
    p = _pick_first(".adoc")
    if p is None:
        pytest.skip(f"no AsciiDoc fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """First `.csv` file in `test_documents/`."""
    p = _pick_first(".csv")
    if p is None:
        pytest.skip(f"no CSV fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_vtt() -> Path:
    """First `.vtt` WebVTT subtitle file in `test_documents/`."""
    p = _pick_first(".vtt")
    if p is None:
        pytest.skip(f"no VTT fixtures in {TEST_DOCS}")
    return p


# ---- Image / audio / video — multi-extension families. Each fixture
# surfaces whichever family member is in the folder first.


@pytest.fixture(scope="session")
def sample_image() -> Path:
    """First image file in `test_documents/` (png / jpg / jpeg /
    tiff / webp). Docling's IMAGE pipeline handles all of these
    uniformly so a single fixture covers the category. gif is NOT a
    docling image format, so it's intentionally excluded."""
    p = _pick_first_of(".png", ".jpg", ".jpeg", ".tiff", ".webp")
    if p is None:
        pytest.skip(f"no image fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_audio() -> Path:
    """First audio file (m4a / mp3 / wav / flac / ogg). Requires the
    `parsers-asr` extra at parse time; the fixture itself just picks
    a file regardless. Tests that parse it must guard on the import."""
    p = _pick_first_of(".m4a", ".mp3", ".wav", ".flac", ".ogg")
    if p is None:
        pytest.skip(f"no audio fixtures in {TEST_DOCS}")
    return p


@pytest.fixture(scope="session")
def sample_video() -> Path:
    """First video file (mp4 / mov / avi / mkv / webm). Same caveat
    as `sample_audio` — parse path needs the `parsers-asr` extra
    (audio track gets transcribed; visual frames are NOT OCRed today,
    see [[deferred-video-frame-extraction]])."""
    p = _pick_first_of(".mp4", ".mov", ".avi", ".mkv", ".webm")
    if p is None:
        pytest.skip(f"no video fixtures in {TEST_DOCS}")
    return p
