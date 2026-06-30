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
with the exception repr in `detail`. The orchestrator
(`system_status`) never raises — the only failure mode is "report
says one or more components are not OK".

Component checks run concurrently via `asyncio.gather` — the slowest
check (Neo4j network round-trip) doesn't gate the others. Wall-time
is the slowest single check, not the sum.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

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
    """

    components: tuple[ComponentStatus, ...]

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.components)


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
            "neo4j", False,
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
        tables = await conn.table_names()
        return ComponentStatus(
            "lancedb", True, f"connected ({len(tables)} table(s))",
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
            name, True, f"{provider}: local (no key required)",
        )
    key = getattr(settings, key_attr, None)
    if key:
        return ComponentStatus(name, True, f"{provider}: set")
    return ComponentStatus(
        name, False,
        f"{provider}: MISSING (set {key_attr.upper()} in .env)",
    )


# =============================================================================
# Orchestrator.
# =============================================================================


async def system_status() -> StatusReport:
    """Run all checks concurrently; return a snapshot.

    NEVER raises — every component's failure surfaces in its
    `ComponentStatus.ok = False`. Callers branch on
    `report.all_ok` (full green) or iterate `report.components` (per-
    component rendering).
    """
    settings = get_settings()

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
        "llm", llm_provider, llm_key_attr, settings,
    )
    embed_status = _check_provider_key(
        "embed", embed_provider, embed_key_attr, settings,
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
    if not report.components:
        return "(no components in report)"
    name_width = max(len(c.name) for c in report.components)
    lines: list[str] = []
    for c in report.components:
        marker = "OK  " if c.ok else "FAIL"
        lines.append(f"{c.name:<{name_width}}  {marker}  {c.detail}")
    return "\n".join(lines)
