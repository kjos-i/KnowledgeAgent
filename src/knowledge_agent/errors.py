"""Typed-error primitives for the application.

One type:

  - `ErrorDetail` — what every user-facing result dataclass carries
    when an operation failed. A human-readable `message` (GUI-
    displayable) plus the original `exception_type` (e.g.
    `neo4j.exceptions.ServiceUnavailable`) for power-user
    troubleshooting.

Error-handling pattern (Python community standard):

  - **Leaves raise.** Internal helpers (KG writes, HTTP fetches,
    parsers) propagate exceptions naturally. No `try/except + log +
    return False` swallowing.

  - **Orchestrators catch at the boundary** where the user-facing
    result dataclass (`IngestResult`, `BulkOp*Result`, `AgentState`)
    is constructed. The catch handler populates the matching
    `error: ErrorDetail | None` field on the result via
    `ErrorDetail.from_exception(exc)`, and logs the failure.

  - **GUI reads the error field** off the result and shows the
    `message`; the `exception_type` is available for a "Show
    details" panel.

This matches the way SQLAlchemy / Django / FastAPI / HTTPX all
handle layered errors: exceptions for the mechanism, named result
types at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorDetail:
    """Structured error info carried by failed result dataclasses.

    `message` is the GUI-facing summary — shown to the user when a
    result reports a stage failure.

    `exception_type` is the dotted module path of the original
    exception (e.g. `neo4j.exceptions.ServiceUnavailable`), surfaced
    for the "Show details" panel + crash files. None when the
    failure was synthetic (no underlying exception).
    """

    message: str
    exception_type: str | None = None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        override_message: str | None = None,
    ) -> ErrorDetail:
        """Build an ErrorDetail from a caught exception.

        Uses `str(exc)` as the message unless `override_message` is
        provided — useful when the raw exception string is too
        cryptic and the caller wants to wrap it in a user-friendly
        context (e.g., "Couldn't connect to Neo4j: <inner>")."""
        message = override_message if override_message is not None else str(exc)
        exception_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        return cls(message=message, exception_type=exception_type)
