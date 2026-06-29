"""Tests for `knowledge_agent.errors` — ErrorDetail.

`ErrorDetail` is the typed-error foundation. Orchestrators that
catch exceptions at result-construction boundaries build an
ErrorDetail via `from_exception` (or directly for synthetic
failures) and place it on the matching `error: ErrorDetail | None`
field of the user-facing result dataclass.
"""

from __future__ import annotations

import pytest

from knowledge_agent.errors import ErrorDetail


def test_error_detail_minimal() -> None:
    detail = ErrorDetail(message="something broke")
    assert detail.message == "something broke"
    assert detail.exception_type is None


def test_error_detail_with_exception_type() -> None:
    detail = ErrorDetail(
        message="connection refused",
        exception_type="neo4j.exceptions.ServiceUnavailable",
    )
    assert detail.exception_type == "neo4j.exceptions.ServiceUnavailable"


def test_error_detail_is_frozen() -> None:
    detail = ErrorDetail(message="x")
    with pytest.raises(Exception):  # FrozenInstanceError
        detail.message = "y"  # type: ignore[misc]


def test_error_detail_from_exception() -> None:
    try:
        raise ValueError("bad input")
    except ValueError as exc:
        detail = ErrorDetail.from_exception(exc)
    assert detail.message == "bad input"
    assert detail.exception_type == "builtins.ValueError"


def test_error_detail_from_exception_with_override_message() -> None:
    try:
        raise ConnectionRefusedError("technical detail")
    except ConnectionRefusedError as exc:
        detail = ErrorDetail.from_exception(
            exc, override_message="Couldn't reach Neo4j — is it running?"
        )
    assert detail.message == "Couldn't reach Neo4j — is it running?"
    # exception_type still preserved for the "Show details" panel.
    assert detail.exception_type == "builtins.ConnectionRefusedError"


def test_error_detail_from_exception_dotted_qualname() -> None:
    """Custom exception classes should produce module.qualname strings
    so the GUI's 'Show details' panel can render them unambiguously."""

    class _MyError(RuntimeError):
        pass

    try:
        raise _MyError("synthetic")
    except _MyError as exc:
        detail = ErrorDetail.from_exception(exc)
    # Qualname includes the test-function path so the assertion uses
    # endswith to stay robust to where the test lives.
    assert detail.exception_type is not None
    assert detail.exception_type.endswith("_MyError")
