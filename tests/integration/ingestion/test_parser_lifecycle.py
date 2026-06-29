"""Integration tests for `ingestion/parser_lifecycle` — plan/execute
for parser-extra install/uninstall.

What is and is NOT tested here:

  - **PLAN** functions: tested directly against real registry state.
    They produce dataclasses summarising what install/uninstall would
    do. No mutation, no pip subprocess.
  - **EXECUTE** functions: NOT run against real pip — installing
    arbitrary packages into the test process would mutate the test
    environment in surprising ways AND a successful test would mean
    the next test runs against a different package set than the
    previous one. Unit tests cover the execute() functions with pip
    mocked; this file pins the plan() side against the real registry
    + the real Python interpreter `pip` subprocess plumbing for
    `pip --version` (a safe read-only probe).

Manual interactive counterpart: none — parser_lifecycle is exercised
by the GUI's "Install parsers-asr" / "Install parsers-code" buttons.

Skipped by default; opt in via `pytest -m integration`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from knowledge_agent.ingestion.parser_lifecycle import (
    InstallParserExtraPlan,
    UninstallParserExtraPlan,
    install_parser_extra_plan,
    uninstall_parser_extra_plan,
)

pytestmark = pytest.mark.integration


def test_install_parser_extra_plan_returns_dataclass_for_asr() -> None:
    """plan() builds a valid InstallParserExtraPlan against the real
    registry for the `parsers-asr` extra."""
    plan = install_parser_extra_plan("asr")
    assert isinstance(plan, InstallParserExtraPlan)
    assert plan.extra_name == "asr"
    # Plan carries a non-empty summary the dialog renders.
    assert plan.summary


def test_install_parser_extra_plan_returns_dataclass_for_code() -> None:
    """Same contract for the `parsers-code` extra."""
    plan = install_parser_extra_plan("code")
    assert isinstance(plan, InstallParserExtraPlan)
    assert plan.extra_name == "code"
    assert plan.summary


def test_install_parser_extra_plan_rejects_unknown_extra() -> None:
    """Unknown extras raise — catches typos at plan time before the
    user sees an unhelpful pip error."""
    with pytest.raises(Exception):
        install_parser_extra_plan("not-a-real-extra")


def test_uninstall_parser_extra_plan_for_known_extras() -> None:
    """Uninstall plan() also builds against the real registry."""
    for extra in ("asr", "code"):
        plan = uninstall_parser_extra_plan(extra)
        assert isinstance(plan, UninstallParserExtraPlan)
        assert plan.extra_name == extra
        assert plan.summary


def test_pip_subprocess_plumbing_works() -> None:
    """The execute() functions shell out to `pip install` / `pip
    uninstall`. We don't actually install anything in tests, but we
    DO verify the subprocess plumbing the executes rely on is alive
    by running a safe read-only equivalent: `pip --version`."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0
    assert "pip" in result.stdout.lower()
