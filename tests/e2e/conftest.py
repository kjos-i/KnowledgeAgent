"""E2E-tier shared fixtures + pytest hooks.

Scope: tests under `tests/e2e/` — full user journeys with the Flet
app launched as a real process. These tests are heavy (seconds-to-
minutes per case); run them selectively, not on every commit.

Standard fixture conventions to add as the GUI lands:
  - `app_process`     — session-scoped: launches the Flet entry point,
                        yields a driver handle, terminates on teardown
  - `clean_state`     — wipes both stores so each journey starts fresh
  - `sample_corpus`   — checked-in tiny corpus under tests/data/e2e/

The honest state of art (2026): no mature headless-Flet-driver
exists. Practical patterns for the GUI e2e slot:
  - boot Flet in a subprocess + drive it via OS-level click
    automation (pyautogui, etc.) — works but flaky
  - refactor Flet handlers to call backend functions that are
    themselves testable — covers most bugs without a Flet driver
  - keep the heaviest journey checks under `scripts/` for manual run

Run e2e selectively:
    pytest tests/e2e/
    pytest -m e2e
"""

from __future__ import annotations

import pytest

from knowledge_agent.config import load_test_env


@pytest.fixture(scope="session", autouse=True)
def _e2e_env() -> None:
    """Point the whole e2e session at the TEST instance (`.env.test`) BEFORE
    any test triggers `get_settings()` — the same isolation guarantee the
    integration tier relies on (password-isolated; see the root conftest's
    SAFETY INVARIANT). Autouse so no e2e test can forget it."""
    load_test_env()
