"""System status / readiness checks — one report consumed by both UIs.

`system_status() → StatusReport` is the single source of truth for "is
the app ready to serve requests?". Two renderers:

  - CLI `ka health` (`cli._cmd_health`) — prints the report to stdout.
  - GUI Settings → Diagnostics panel (sprint 1) — renders chips + retry
    button against the same dataclass.

Checks today:

  - Neo4j: `RETURN 1` round-trip via `kg_client.read_query`.
  - LanceDB: open the connection + list tables (any call that proves
    the directory is accessible).
  - Active LLM provider key: per `settings.llm_provider`, check the
    matching `*_api_key` is non-empty. Local providers (`ollama`) skip
    the check and report OK with `local` detail.
  - Active embedder provider key: same shape for
    `settings.embedding_provider`. Local providers (`huggingface`)
    skip.

Each check fail-soft: if the underlying call raises (Neo4j down,
LanceDB directory missing), the check catches and reports `ok=False`
with the exception repr in `detail`. The orchestrator (`system_status`)
never raises: if `Settings` can't even be built (e.g. no active corpus,
so NEO4J_PASSWORD isn't bridged), it comes back as `report.config_error`
with no components; otherwise failures surface as `ok=False` components.

Component checks run concurrently via `asyncio.gather` — the slowest
check (Neo4j network round-trip) doesn't gate the others. Wall-time
is the slowest single check, not the sum.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from knowledge_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Public dataclasses — what the renderers consume.
# =============================================================================


@dataclass(frozen=True)
class ComponentStatus:
    """One check's result. Frozen so renderers can pass it around safely.

    `name` is the stable identifier the GUI uses for chip rendering +
    test assertions. `detail` is a free-form human-readable string the
    GUI shows on hover / the CLI prints inline; never None for failures
    (carries the exception repr or "MISSING").
    """

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class StatusReport:
    """The full system snapshot.

    `components` is the ordered tuple of per-check results in display
    order (Neo4j first, then LanceDB, then LLM key, then embedder key).
    `all_ok` is a derived convenience for "should the green chip light
    up at the top of the Diagnostics panel".

    `config_error` is set when `Settings` couldn't be built at all (e.g.
    no active corpus, so NEO4J_PASSWORD isn't bridged into the env). It's
    a one-line actionable hint; when set, `components` is empty and the
    checks did not run — renderers show the hint instead of chips.
    """

    components: tuple[ComponentStatus, ...]
    config_error: str | None = None

    @property
    def all_ok(self) -> bool:
        return self.config_error is None and all(c.ok for c in self.components)


# =============================================================================
# Per-check helpers — each NEVER raises; returns a ComponentStatus.
# =============================================================================


async def _check_neo4j() -> ComponentStatus:
    """`RETURN 1` round-trip. Verifies driver, auth, network, server."""
    try:
        # Lazy import: health.py shouldn't drag kg_client into modules
        # that only touch settings / config.
        from knowledge_agent.kg.client import get_kg_client

        client = get_kg_client()
        rows = await client.read_query("RETURN 1 AS ok")
        if rows and rows[0].get("ok") == 1:
            return ComponentStatus("neo4j", True, "RETURN 1 round-trip OK")
        return ComponentStatus(
            "neo4j",
            False,
            f"unexpected result from RETURN 1: {rows!r}",
        )
    except Exception as exc:
        return ComponentStatus("neo4j", False, repr(exc))


async def _check_lancedb() -> ComponentStatus:
    """Open the connection + list tables — proves the directory is OK."""
    try:
        from knowledge_agent.search.client import get_search_client

        client = get_search_client()
        conn = await client._ensure_conn()
        tables = (await conn.list_tables()).tables
        return ComponentStatus(
            "lancedb",
            True,
            f"connected ({len(tables)} table(s))",
        )
    except Exception as exc:
        return ComponentStatus("lancedb", False, repr(exc))


# Provider → (settings attribute holding the key, "local" flag).
# Local providers (ollama / huggingface) skip the key check and always
# report OK — they have no API key concept. Misconfigured local setups
# fail elsewhere (factory raises on missing daemon / missing model).
_LLM_PROVIDER_KEY_ATTR: dict[str, str | None] = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "google": "google_api_key",
    "ollama": None,  # local
}

_EMBED_PROVIDER_KEY_ATTR: dict[str, str | None] = {
    "voyage": "voyage_api_key",
    "openai": "openai_api_key",
    "google": "google_api_key",
    "huggingface": None,  # local
}


def _check_provider_key(
    role: str,
    provider: str,
    key_attr: str | None,
    settings: Settings,
) -> ComponentStatus:
    """One provider key check. `role` is `"llm"` or `"embed"` — drives
    the component name + the user-facing detail string.

    `key_attr=None` marks a local provider (Ollama / HuggingFace);
    those skip the check and report OK with a `local` detail.
    """
    name = f"{role}_key"
    if key_attr is None:
        return ComponentStatus(
            name,
            True,
            f"{provider}: local (no key required)",
        )
    key = getattr(settings, key_attr, None)
    if key:
        return ComponentStatus(name, True, f"{provider}: set")
    return ComponentStatus(
        name,
        False,
        f"{provider}: MISSING (set {key_attr.upper()} in .env)",
    )


# =============================================================================
# Orchestrator.
# =============================================================================


# Human-readable guidance per known required-no-default field, so a config
# failure renders "missing X — do Y" instead of a raw pydantic error. The
# Neo4j params + LanceDB path are per-corpus, hence "create a corpus".
_FIELD_GUIDANCE: dict[str, str] = {
    "neo4j_password": "Neo4j password not set; create a corpus in Library",
    "neo4j_uri": "Neo4j URI not set; create a corpus in Library",
    "neo4j_user": "Neo4j user not set; create a corpus in Library",
    "lancedb_path": "LanceDB path not set; create a corpus in Library",
}


def config_error_message(exc: Exception) -> str:
    """Translate a `get_settings()` failure into a one-line actionable hint.

    Pydantic missing-field errors map to `_FIELD_GUIDANCE` (the most
    actionable thing right now); other errors fall back to the class name +
    message. The caller logs the raw `repr(exc)` so debug info isn't lost.
    """
    if isinstance(exc, ValidationError):
        missing = [
            str(err["loc"][-1])
            for err in exc.errors()
            if err.get("type") == "missing" and err.get("loc")
        ]
        for field in missing:
            if field in _FIELD_GUIDANCE:
                return _FIELD_GUIDANCE[field]
        if missing:
            return f"missing required setting(s): {', '.join(missing)}"
    return f"{type(exc).__name__}: {exc}"


async def system_status() -> StatusReport:
    """Run all checks concurrently; return a snapshot. NEVER raises.

    If `Settings` can't be built — e.g. no active corpus, so NEO4J_PASSWORD
    isn't bridged into the env — this returns a report with `config_error`
    set (a one-line hint) and no components, rather than raising. Otherwise
    every component's failure surfaces in its `ComponentStatus.ok = False`.
    Callers branch on `report.config_error` first (show the hint), then
    `report.all_ok` / `report.components`.
    """
    try:
        settings = get_settings()
    except Exception as exc:
        logger.warning("system_status: could not build Settings: %r", exc)
        return StatusReport(components=(), config_error=config_error_message(exc))

    llm_provider = settings.llm_provider
    embed_provider = settings.embedding_provider
    llm_key_attr = _LLM_PROVIDER_KEY_ATTR.get(llm_provider)
    embed_key_attr = _EMBED_PROVIDER_KEY_ATTR.get(embed_provider)

    # Run the two I/O checks concurrently — the slowest gates the report,
    # not the sum. Provider-key checks are synchronous (no I/O) so they
    # run inline after the gather.
    neo4j_status, lancedb_status = await asyncio.gather(
        _check_neo4j(),
        _check_lancedb(),
    )
    llm_status = _check_provider_key(
        "llm",
        llm_provider,
        llm_key_attr,
        settings,
    )
    embed_status = _check_provider_key(
        "embed",
        embed_provider,
        embed_key_attr,
        settings,
    )

    return StatusReport(
        components=(neo4j_status, lancedb_status, llm_status, embed_status),
    )


def render_report_text(report: StatusReport) -> str:
    """Render the report as the CLI's text output. The GUI does NOT use
    this — it builds its own Flet widgets from the same dataclass.

    Format: one line per component, `<name>: <OK|FAIL> <detail>`, aligned
    so the OK/FAIL column lines up across rows.
    """
    if report.config_error:
        return report.config_error
    if not report.components:
        return "(no components in report)"
    name_width = max(len(c.name) for c in report.components)
    lines: list[str] = []
    for c in report.components:
        marker = "OK  " if c.ok else "FAIL"
        lines.append(f"{c.name:<{name_width}}  {marker}  {c.detail}")
    return "\n".join(lines)
