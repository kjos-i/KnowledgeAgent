"""GUI-facing LLM provider lifecycle ops (install / uninstall / model pull).

Same plan/execute UI contract as `ingestion/parser_lifecycle.py` and
`entity_extractors/extractor_lifecycle.py`. Manages the three
non-default LLM providers (OpenAI / Google / Ollama); Anthropic is
the bundled default and has no install/uninstall ops.

LOCKED RULE 2026-06-29 — NO AUTO-INSTALL.
Picking a provider in Settings is settings-only. The GUI surfaces a
banner ("OpenAI is your active LLM provider but is not installed.
[Install] [Switch back]") and `llm_factory.get_llm` raises a clear
ImportError (pointing at the pip extra) if code invokes an
uninstalled provider. This module is the SOLE path that calls pip
install; the factory never installs anything.

LOCKED RULE 2026-06-29 — BLOCKING UNINSTALL.
Uninstalling a provider that is the current `settings.llm_provider`
is blocked with a clear "switch active provider first" message.
Symmetric to the parser_lifecycle uninstall-while-active guard.

Ollama is the partial-install case:

  1. Adapter (`langchain-ollama`, ~1-2 MB) — pip-installable, this
     module handles.
  2. Daemon (Go binary, ~150-300 MB) — manual. The plan summary
     surfaces https://ollama.com with platform-specific install
     instructions when the daemon isn't detected on PATH.
  3. Model weights (2-40 GB) — `ollama pull <model>`, separate
     `pull_ollama_model_*` ops. Each model in the curated menu has
     its own provenance.

Curated menus locked 2026-06-29 (4 entries each, see memory
[[llm-embedder-provider-swap]]):

  - Anthropic models: Haiku (cheap) + Sonnet (smart) + Opus (top).
  - OpenAI models: GPT-4o-mini + GPT-4o + GPT-4o-2024-11-20.
  - Google models: Gemini 1.5 Flash + 1.5 Pro + 2.0 Flash.
  - Ollama models: llama3.2:3b / qwen2.5:7b / qwen2.5:14b /
    llama3.3:70b (laptop / mid / high / workstation tiers).

NOTE: subprocess pip calls assume a writeable Python environment
(venv or per-user install). In a frozen / immutable distribution,
the install op would need to surface "this distribution is read-only;
re-install the app with the extra included" - deferred until we
actually ship a frozen distribution.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from dataclasses import dataclass
from typing import Any

from knowledge_agent.config import get_settings

logger = logging.getLogger(__name__)


# ---- provenance dataclasses (security disclosure for install dialog) ----


@dataclass(frozen=True)
class LLMProviderProvenance:
    """Where an LLM provider's code + service come from.

    One per provider (anthropic / openai / google / ollama). Models
    within a provider share this; per-model facts (size, license,
    hardware tier) live on `OllamaModelProvenance` for the local
    case.

    All fields are simple types (str / bool) so the GUI can render
    them in a flat info panel without per-field formatting logic.
    """

    provider_name: str
    """Canonical provider id matching `settings.llm_provider`."""

    display_name: str
    """User-facing name (e.g. "Anthropic Claude", "Ollama (local)")."""

    publisher: str
    """Who runs the service (cloud) or maintains the runtime (local)."""

    license: str
    """Of the SERVICE itself, not specific models. For cloud APIs
    this is the provider's API terms; for Ollama it's the runtime
    license (MIT). Per-model licenses live on `OllamaModelProvenance`
    or the docstring of each model entry."""

    source_url: str
    """Direct URL the user can click for the provider's homepage /
    docs."""

    pip_extras: str | None
    """Name of the optional-dependency group in `pyproject.toml`.
    `None` for Anthropic (bundled). Surfaced in the install dialog
    so the user can replicate the install via CLI if needed."""

    requires_api_key: bool
    """True for cloud providers. The install plan blocks (and the
    GUI surfaces a "set your API key first" banner) when this is
    True but the relevant `*_api_key` Setting is empty."""

    requires_daemon: bool
    """True for Ollama only — the daemon must be running on PATH /
    at `ollama_base_url`. The install plan surfaces the manual
    install URL when this is True and the daemon isn't detected."""

    daemon_install_url: str | None
    """For `requires_daemon=True`: the URL pointing the user at the
    manual installer. None for cloud providers."""


@dataclass(frozen=True)
class OllamaModelProvenance:
    """Per-model facts for the Ollama curated menu.

    Surfaced in the pull-model dialog so users see the disk cost +
    hardware tier BEFORE running `ollama pull <name>`. Unlike cloud
    LLM models (where one provider hosts many models on the same
    service), each Ollama model is a distinct local download with
    its own size and licence.
    """

    model_id: str
    """The exact tag passed to `ollama pull` (e.g. "qwen2.5:7b")."""

    display_name: str
    """User-facing name (e.g. "Qwen 2.5 7B")."""

    download_size_gb: float
    """Approximate disk cost in GB. Surfaced prominently — these
    range from 2 GB (laptop) to 40 GB (workstation)."""

    hardware_tier: str
    """Short human label: "Laptop CPU" / "Mid GPU" / "High-end GPU" /
    "Workstation"."""

    min_ram_gb: int
    """Minimum RAM in GB. The install dialog warns when the system
    has less than this (future check, today informational only)."""

    publisher: str
    """Who released the model (Meta / Alibaba / etc)."""

    license: str
    """Model weight license — varies (Apache-2.0, Llama 3.3
    Community License, etc). Important because Ollama itself is MIT
    but the model weights are not."""


# ---- per-provider provenance constants ----


_ANTHROPIC_PROVENANCE = LLMProviderProvenance(
    provider_name="anthropic",
    display_name="Anthropic Claude",
    publisher="Anthropic PBC",
    license="Proprietary cloud API (Anthropic Usage Policy)",
    source_url="https://www.anthropic.com/api",
    pip_extras="llm-anthropic",
    requires_api_key=True,
    requires_daemon=False,
    daemon_install_url=None,
)


_OPENAI_PROVENANCE = LLMProviderProvenance(
    provider_name="openai",
    display_name="OpenAI GPT",
    publisher="OpenAI, Inc.",
    license="Proprietary cloud API (OpenAI Usage Policies)",
    source_url="https://platform.openai.com/docs",
    pip_extras="llm-openai",
    requires_api_key=True,
    requires_daemon=False,
    daemon_install_url=None,
)


_GOOGLE_PROVENANCE = LLMProviderProvenance(
    provider_name="google",
    display_name="Google Gemini",
    publisher="Google LLC",
    license="Proprietary cloud API (Google AI Studio terms)",
    source_url="https://ai.google.dev/docs",
    pip_extras="llm-google",
    requires_api_key=True,
    requires_daemon=False,
    daemon_install_url=None,
)


_OLLAMA_PROVENANCE = LLMProviderProvenance(
    provider_name="ollama",
    display_name="Ollama (local)",
    publisher="Ollama Project (Jeffrey Morgan et al.)",
    license="MIT (runtime); per-model licences vary",
    source_url="https://ollama.com",
    pip_extras="llm-ollama",
    requires_api_key=False,
    requires_daemon=True,
    daemon_install_url="https://ollama.com/download",
)


# ---- Ollama curated model menu (LOCKED 4 entries 2026-06-29) ----


OLLAMA_MODELS: dict[str, OllamaModelProvenance] = {
    "llama3.2:3b": OllamaModelProvenance(
        model_id="llama3.2:3b",
        display_name="Llama 3.2 3B",
        download_size_gb=2.0,
        hardware_tier="Laptop CPU",
        min_ram_gb=8,
        publisher="Meta Platforms, Inc.",
        license="Llama 3.2 Community License",
    ),
    "qwen2.5:7b": OllamaModelProvenance(
        model_id="qwen2.5:7b",
        display_name="Qwen 2.5 7B",
        download_size_gb=4.7,
        hardware_tier="Mid GPU",
        min_ram_gb=16,
        publisher="Alibaba Cloud (Qwen team)",
        license="Apache-2.0",
    ),
    "qwen2.5:14b": OllamaModelProvenance(
        model_id="qwen2.5:14b",
        display_name="Qwen 2.5 14B",
        download_size_gb=9.0,
        hardware_tier="High-end GPU",
        min_ram_gb=24,
        publisher="Alibaba Cloud (Qwen team)",
        license="Apache-2.0",
    ),
    "llama3.3:70b": OllamaModelProvenance(
        model_id="llama3.3:70b",
        display_name="Llama 3.3 70B (Q4 quantised)",
        download_size_gb=40.0,
        hardware_tier="Workstation",
        min_ram_gb=64,
        publisher="Meta Platforms, Inc.",
        license="Llama 3.3 Community License",
    ),
}


# ---- per-provider library detection ----


def _anthropic_is_installed() -> bool:
    """True iff `langchain-anthropic` is importable.

    Pre-2026-06-29 this returned True unconditionally because the
    package shipped in base deps. Post-2026-06-29 Anthropic is a
    provider like any other — installed only if the user picks it in
    the first-launch wizard.
    """
    try:
        import langchain_anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _openai_is_installed() -> bool:
    """True iff `langchain-openai` is importable."""
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True


def _google_is_installed() -> bool:
    """True iff `langchain-google-genai` is importable."""
    try:
        import langchain_google_genai  # noqa: F401
    except ImportError:
        return False
    return True


def _ollama_adapter_is_installed() -> bool:
    """True iff `langchain-ollama` is importable.

    Distinct from the daemon check (`_ollama_daemon_is_reachable`):
    a user can have the adapter pip-installed without the daemon
    running, and vice versa. The install dialog surfaces both
    statuses.
    """
    try:
        import langchain_ollama  # noqa: F401
    except ImportError:
        return False
    return True


async def _ollama_daemon_is_reachable() -> bool:
    """True iff the Ollama daemon binary is on PATH OR responding at
    `ollama_base_url`.

    Uses `shutil.which` first (fast, no network); falls back to an
    async HTTP GET on /api/tags (1s timeout) through the central
    `_http_client` (so its retry policy / User-Agent are uniform with
    every other outbound HTTP call). Order: PATH check is cheap +
    handles the common case where the daemon is installed but not
    started; HTTP check handles remote daemons + custom base URLs.

    `max_retries=0` so a missing daemon fails fast — this is a
    liveness probe, not an API call worth retrying.
    """
    if shutil.which("ollama") is not None:
        return True
    settings = get_settings()
    try:
        from knowledge_agent import _http_client

        resp = await _http_client.request(
            f"{settings.ollama_base_url}/api/tags",
            timeout=1.0,
            max_retries=0,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ---- registry ----


LLM_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "anthropic": {
        "display_name": "Anthropic Claude",
        "bundled": False,
        "is_installed_fn": _anthropic_is_installed,
        "library_packages": ("langchain-anthropic",),
        "provenance": _ANTHROPIC_PROVENANCE,
        # Default model names for each call site. The lifecycle's
        # switch-provider step copies these into the matching
        # Settings fields. Tier mapping: classifier/builder=cheap,
        # cypher/synthesizer/extractor=smart.
        "default_models": {
            "mode_classifier": "claude-haiku-4-5-20251001",
            "query_builder": "claude-haiku-4-5-20251001",
            "cypher_builder": "claude-sonnet-4-6",
            "synthesizer": "claude-sonnet-4-6",
            "entity_extractor": "claude-haiku-4-5-20251001",
            "triples_extractor": "claude-haiku-4-5-20251001",
        },
    },
    "openai": {
        "display_name": "OpenAI GPT",
        "bundled": False,
        "is_installed_fn": _openai_is_installed,
        "library_packages": ("langchain-openai",),
        "provenance": _OPENAI_PROVENANCE,
        "default_models": {
            "mode_classifier": "gpt-4o-mini",
            "query_builder": "gpt-4o-mini",
            "cypher_builder": "gpt-4o",
            "synthesizer": "gpt-4o",
            "entity_extractor": "gpt-4o-mini",
            "triples_extractor": "gpt-4o-mini",
        },
    },
    "google": {
        "display_name": "Google Gemini",
        "bundled": False,
        "is_installed_fn": _google_is_installed,
        "library_packages": ("langchain-google-genai",),
        "provenance": _GOOGLE_PROVENANCE,
        "default_models": {
            "mode_classifier": "gemini-1.5-flash",
            "query_builder": "gemini-1.5-flash",
            "cypher_builder": "gemini-1.5-pro",
            "synthesizer": "gemini-1.5-pro",
            "entity_extractor": "gemini-1.5-flash",
            "triples_extractor": "gemini-1.5-flash",
        },
    },
    "ollama": {
        "display_name": "Ollama (local)",
        "bundled": False,
        "is_installed_fn": _ollama_adapter_is_installed,
        "library_packages": ("langchain-ollama",),
        "provenance": _OLLAMA_PROVENANCE,
        # Every call site uses the same Ollama model — local models
        # don't have a cheap/smart tier split the way cloud providers
        # do. Users override via Settings if they want to mix sizes.
        # Default to the mid-GPU tier (most laptops with a GPU); the
        # lifecycle's pull-model step lets users pick a different one.
        "default_models": {
            "mode_classifier": "qwen2.5:7b",
            "query_builder": "qwen2.5:7b",
            "cypher_builder": "qwen2.5:7b",
            "synthesizer": "qwen2.5:7b",
            "entity_extractor": "qwen2.5:7b",
            "triples_extractor": "qwen2.5:7b",
        },
    },
}


# ---- subprocess wrapper (mockable) ----


async def _run_pip(args: list[str], timeout: float = 600.0) -> tuple[bool, str]:
    """Run `python -m pip <args>` async; return (success, combined output).

    Same shape as `parser_lifecycle._run_pip` and
    `extractor_lifecycle._run_pip` — kept duplicated rather than
    extracted to a shared helper to match the existing codebase
    pattern (each lifecycle module is self-contained).

    Uses `asyncio.create_subprocess_exec` so the caller's event loop
    (Flet GUI / CLI / eval harness) stays responsive while pip runs.
    On timeout the child is killed with `proc.kill()` + reaped via
    `proc.wait()` so no zombie remains.
    """
    cmd = [sys.executable, "-m", "pip", *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"pip command timed out after {timeout}s"
        output = (stdout.decode("utf-8", errors="replace") if stdout else "") + (
            stderr.decode("utf-8", errors="replace") if stderr else ""
        )
        return proc.returncode == 0, output
    except Exception as exc:
        return False, f"pip subprocess failed: {exc!r}"


async def _run_ollama(args: list[str], timeout: float = 1800.0) -> tuple[bool, str]:
    """Run `ollama <args>` async; return (success, combined output).

    Used for `ollama pull <model>` — large models take time, so the
    default timeout is 30 minutes. The daemon must be running; the
    pull will fail fast with a clear error if it isn't.

    Uses `asyncio.create_subprocess_exec` so the caller's event loop
    (Flet GUI / CLI / eval harness) stays responsive across the multi-
    minute pull. On timeout the child is killed + reaped.
    `FileNotFoundError` from create_subprocess_exec surfaces when the
    `ollama` binary isn't on PATH — translated to the same friendly
    message as before.
    """
    cmd = ["ollama", *args]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, (
            "`ollama` not found on PATH. Install the daemon from https://ollama.com/download first."
        )
    except Exception as exc:
        return False, f"ollama subprocess failed: {exc!r}"
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return False, f"ollama command timed out after {timeout}s"
    output = (stdout.decode("utf-8", errors="replace") if stdout else "") + (
        stderr.decode("utf-8", errors="replace") if stderr else ""
    )
    return proc.returncode == 0, output


# ---- shared helpers ----


def _registry_entry(provider: str) -> dict[str, Any]:
    """Look up a provider in the registry, raise ValueError on miss."""
    entry = LLM_PROVIDER_REGISTRY.get(provider)
    if entry is None:
        raise ValueError(
            f"Unknown LLM provider {provider!r}. Known: {sorted(LLM_PROVIDER_REGISTRY)}."
        )
    return entry


def _ollama_model_provenance(model_id: str) -> OllamaModelProvenance:
    """Look up a curated Ollama model entry, raise on miss."""
    prov = OLLAMA_MODELS.get(model_id)
    if prov is None:
        raise ValueError(
            f"Unknown Ollama model {model_id!r}. Curated menu: {sorted(OLLAMA_MODELS)}."
        )
    return prov


# ---- install_llm_provider ----


@dataclass(frozen=True)
class InstallLLMProviderPlan:
    """Plan for installing an LLM provider's adapter.

    For Ollama, additionally reports `daemon_reachable` so the GUI
    can show the two install steps separately (adapter via pip,
    daemon via manual download).
    """

    provider_name: str
    display_name: str
    pip_extras: str | None
    already_installed: bool
    bundled: bool
    provenance: LLMProviderProvenance
    daemon_reachable: bool | None
    """For Ollama: whether the daemon binary is on PATH or reachable
    at ollama_base_url. None for cloud providers (no daemon)."""

    @property
    def summary(self) -> str:
        if self.bundled:
            return f"{self.display_name} ships by default - nothing to install."
        if self.already_installed:
            base = f"{self.display_name} adapter is already installed."
            if self.provenance.requires_daemon and self.daemon_reachable is False:
                base += (
                    f" Ollama DAEMON is not reachable — install it from "
                    f"{self.provenance.daemon_install_url} and start it."
                )
            return base
        daemon_clause = ""
        if self.provenance.requires_daemon and self.daemon_reachable is False:
            daemon_clause = (
                f" NOTE: also requires the Ollama daemon — install "
                f"manually from {self.provenance.daemon_install_url}."
            )
        return (
            f"Install {self.display_name}: runs "
            f"`pip install <this package>[{self.pip_extras}]`. "
            f"Published by {self.provenance.publisher} "
            f"({self.provenance.license}). "
            f"Source: {self.provenance.source_url}. "
            f"A restart is required after install.{daemon_clause}"
        )


@dataclass(frozen=True)
class InstallLLMProviderResult:
    """Outcome of `install_llm_provider_execute`."""

    provider_name: str
    did_install: bool
    install_ok: bool
    restart_required: bool
    pip_output: str


async def install_llm_provider_plan(provider: str) -> InstallLLMProviderPlan:
    """Build an install plan for the given LLM provider."""
    entry = _registry_entry(provider)
    prov: LLMProviderProvenance = entry["provenance"]
    daemon_reachable: bool | None = None
    if prov.requires_daemon:
        daemon_reachable = await _ollama_daemon_is_reachable()
    return InstallLLMProviderPlan(
        provider_name=provider,
        display_name=entry["display_name"],
        pip_extras=prov.pip_extras,
        already_installed=bool(entry["is_installed_fn"]()),
        bundled=entry["bundled"],
        provenance=prov,
        daemon_reachable=daemon_reachable,
    )


async def install_llm_provider_execute(
    plan: InstallLLMProviderPlan,
    *,
    distribution_name: str = "knowledge-agent",
) -> InstallLLMProviderResult:
    """Run `pip install <dist>[<pip_extras>]` if not already installed.

    The Ollama daemon is NOT installed by this op (no reliable
    cross-platform path); the plan summary surfaces the manual URL.
    """
    if plan.bundled or plan.already_installed:
        return InstallLLMProviderResult(
            provider_name=plan.provider_name,
            did_install=False,
            install_ok=True,
            restart_required=False,
            pip_output="",
        )
    target = f"{distribution_name}[{plan.pip_extras}]"
    ok, output = await _run_pip(["install", target])
    return InstallLLMProviderResult(
        provider_name=plan.provider_name,
        did_install=True,
        install_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- uninstall_llm_provider ----


@dataclass(frozen=True)
class UninstallLLMProviderPlan:
    """Plan for uninstalling an LLM provider's adapter.

    Blocking: when `is_active=True` the plan summary refuses the
    uninstall and tells the user to switch active provider first.
    """

    provider_name: str
    display_name: str
    packages_to_remove: tuple[str, ...]
    installed: bool
    bundled: bool
    is_active: bool

    @property
    def summary(self) -> str:
        if self.bundled:
            return f"{self.display_name} is the bundled default — uninstall is not supported."
        if self.is_active:
            return (
                f"{self.display_name} is your ACTIVE LLM provider. "
                f"Switch to a different provider in Settings first, "
                f"then uninstall."
            )
        if not self.installed:
            return f"{self.display_name} is not installed."
        return (
            f"Uninstall {self.display_name}: runs "
            f"`pip uninstall {' '.join(self.packages_to_remove)}`. "
            f"A restart is required after uninstall."
        )


@dataclass(frozen=True)
class UninstallLLMProviderResult:
    """Outcome of `uninstall_llm_provider_execute`."""

    provider_name: str
    did_uninstall: bool
    uninstall_ok: bool
    restart_required: bool
    pip_output: str


def uninstall_llm_provider_plan(provider: str) -> UninstallLLMProviderPlan:
    """Build an uninstall plan."""
    entry = _registry_entry(provider)
    settings = get_settings()
    return UninstallLLMProviderPlan(
        provider_name=provider,
        display_name=entry["display_name"],
        packages_to_remove=tuple(entry["library_packages"]),
        installed=bool(entry["is_installed_fn"]()),
        bundled=entry["bundled"],
        is_active=(settings.llm_provider == provider),
    )


async def uninstall_llm_provider_execute(
    plan: UninstallLLMProviderPlan,
) -> UninstallLLMProviderResult:
    """Pip-uninstall the provider's adapter package.

    No-ops when bundled / not installed / active. The active-provider
    block is the policy guard from
    [[llm-embedder-provider-swap]] — symmetric with the
    parser_lifecycle and ontology_lifecycle delete-while-active
    guards.
    """
    if plan.bundled or not plan.installed or plan.is_active:
        return UninstallLLMProviderResult(
            provider_name=plan.provider_name,
            did_uninstall=False,
            uninstall_ok=True,
            restart_required=False,
            pip_output="",
        )
    ok, output = await _run_pip(["uninstall", "-y", *plan.packages_to_remove])
    return UninstallLLMProviderResult(
        provider_name=plan.provider_name,
        did_uninstall=True,
        uninstall_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- pull_ollama_model ----


@dataclass(frozen=True)
class PullOllamaModelPlan:
    """Plan for `ollama pull <model>` of a curated menu entry.

    `daemon_reachable` gates execution — pull fails clearly if the
    daemon isn't running. Surfaced separately so the GUI can guide
    the user through "start the daemon first".
    """

    model_id: str
    provenance: OllamaModelProvenance
    daemon_reachable: bool

    @property
    def summary(self) -> str:
        if not self.daemon_reachable:
            return (
                f"Cannot pull {self.provenance.display_name} — Ollama "
                f"daemon is not reachable. Start it (or install from "
                f"https://ollama.com/download) first."
            )
        return (
            f"Pull {self.provenance.display_name}: runs "
            f"`ollama pull {self.model_id}`. Downloads "
            f"~{self.provenance.download_size_gb:.1f} GB from "
            f"{self.provenance.publisher} ({self.provenance.license}). "
            f"Hardware tier: {self.provenance.hardware_tier} "
            f"(min {self.provenance.min_ram_gb} GB RAM)."
        )


@dataclass(frozen=True)
class PullOllamaModelResult:
    """Outcome of `pull_ollama_model_execute`."""

    model_id: str
    did_pull: bool
    pull_ok: bool
    ollama_output: str


async def pull_ollama_model_plan(model_id: str) -> PullOllamaModelPlan:
    """Build a plan for pulling one curated Ollama model.

    `model_id` must be a key in `OLLAMA_MODELS` — outside-of-menu
    models are not supported by the GUI (curated menu is the
    interface; advanced users can run `ollama pull` manually if
    they want a different model).
    """
    prov = _ollama_model_provenance(model_id)
    return PullOllamaModelPlan(
        model_id=model_id,
        provenance=prov,
        daemon_reachable=await _ollama_daemon_is_reachable(),
    )


async def pull_ollama_model_execute(
    plan: PullOllamaModelPlan,
) -> PullOllamaModelResult:
    """Run `ollama pull <model_id>` against the local daemon."""
    if not plan.daemon_reachable:
        return PullOllamaModelResult(
            model_id=plan.model_id,
            did_pull=False,
            pull_ok=False,
            ollama_output=(
                "Ollama daemon not reachable. Install from "
                "https://ollama.com/download and start it first."
            ),
        )
    ok, output = await _run_ollama(["pull", plan.model_id])
    return PullOllamaModelResult(
        model_id=plan.model_id,
        did_pull=True,
        pull_ok=ok,
        ollama_output=output,
    )
