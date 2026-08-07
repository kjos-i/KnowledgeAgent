"""GUI-facing embedder provider lifecycle ops.

Mirror of `llm_lifecycle.py` for the embedding side. Manages the
three non-default embedder providers (OpenAI / Google / HuggingFace);
Voyage is the bundled default and has no install/uninstall ops.

LOCKED RULE 2026-06-29 — NO AUTO-INSTALL.
Same as the LLM side. Picking a provider in Settings is settings-
only; the GUI surfaces a banner and `embedder_factory.embed_texts`
raises ImportError pointing at the pip extra if code invokes an
uninstalled provider.

LOCKED RULE 2026-06-29 — BLOCKING UNINSTALL when active.
Same as the LLM side.

LOCKED RULE 2026-06-29 — DIMENSION-CHANGE GUARD.
LanceDB pins the embedding dimension at table creation. Switching
to a provider whose default dimension differs from the current
corpus's embedding column AND the table has rows = DESTRUCTIVE.
`switch_embedder_plan` reports `dim_mismatch=True`+`existing_rows=N`
so the GUI can surface "this switch requires wiping and re-
embedding the corpus" with a confirm dialog. Approved switches go
through `drop_chunks_table` then full re-ingest.

HuggingFace is the 2-step case:

  1. Install libs (`sentence-transformers` + `torch`, ~2-3 GB). The
     `embed-huggingface` pip extra. Heaviest install in the menu —
     the install dialog surfaces the disk cost prominently.
  2. Download model (`download_hf_model_*` ops). Each curated menu
     entry has its own provenance with pinned commit SHA, size, and
     safetensors flag.

Curated HF menu locked 2026-06-29 (4 entries):

  - BAAI/bge-m3              — multilingual default (1024-dim, 2.3 GB)
  - mixedbread-ai/mxbai-...  — English top-tier (1024-dim, 1.3 GB)
  - BAAI/bge-small-en-v1.5   — compact English (384-dim, 130 MB)
  - sentence-transformers/   — classic / fastest (384-dim, 90 MB)
    all-MiniLM-L6-v2

Dimension reference (read by the lifecycle's switch step):
  - voyage (default)  → 1024
  - openai            → 1536 (text-embedding-3-*)
  - google            → 768  (text-embedding-004)
  - huggingface       → varies per model (see HF_EMBEDDING_MODELS)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any

from knowledge_agent import DISTRIBUTION_NAME
from knowledge_agent.config import (
    EMBEDDING_DIM_DEFAULTS,
    EMBEDDING_MODEL_DEFAULTS,
    get_settings,
)

logger = logging.getLogger(__name__)


# ---- provenance dataclasses (security disclosure for install dialog) ----


@dataclass(frozen=True)
class EmbedderProviderProvenance:
    """Where an embedder provider's code + service come from.

    One per provider (voyage / openai / google / huggingface). Same
    shape as `llm_lifecycle.LLMProviderProvenance` but trimmed —
    embedder providers don't need a daemon (HF is local but loads
    in-process, no separate runtime).

    `default_dim` is the active model's output dimension at default
    settings. The lifecycle's switch-provider step compares this
    against the current LanceDB table's vector column to detect
    destructive switches.
    """

    provider_name: str
    """Canonical provider id matching `settings.embedding_provider`."""

    display_name: str
    """User-facing name (e.g. "Voyage AI", "HuggingFace (local)")."""

    publisher: str
    """Who runs the service (cloud) or maintains the library
    (local)."""

    license: str
    """Of the service / library. Per-model licences for HF live on
    `HFEmbedderProvenance`."""

    source_url: str
    """Provider homepage / docs URL."""

    pip_extras: str | None
    """Optional-dependency group in `pyproject.toml`. `None` for
    Voyage (bundled)."""

    requires_api_key: bool
    """True for cloud providers (voyage / openai / google). False
    for HF (local model file)."""

    default_dim: int
    """Output dimension of the provider's default model. Used by
    the dimension-change guard to detect destructive switches."""


@dataclass(frozen=True)
class HFEmbedderProvenance:
    """Per-model facts for the HuggingFace curated menu.

    Same shape as `extractor_lifecycle.ModelProvenance` — pinned
    commit SHA + safetensors flag + size + license. Surfaced in the
    model-download dialog so users see what they're about to pull.
    """

    model_id: str
    """HF repo id (e.g. "BAAI/bge-m3")."""

    display_name: str
    """User-facing name (e.g. "BGE-m3 (multilingual)")."""

    source_url: str
    """HF model card URL."""

    download_size_mb: int
    """Approximate model file size in MB. Distinct from the 2-3 GB
    library install (sentence-transformers + torch); the library is
    one-time, model files are per-model."""

    dimensions: int
    """Output vector dimension. Compared against the LanceDB table
    schema by `switch_embedder_plan`."""

    language: str
    """Coverage: "Multilingual (100+)" / "English" / etc."""

    publisher: str
    """Model publisher (e.g. "Beijing Academy of AI")."""

    license: str
    """Model weights license (Apache-2.0 / MIT for the curated menu —
    no commercial-restricted models on the list)."""

    safetensors: bool
    """True iff the model loads from `.safetensors` only (no pickle
    execution path). All four curated entries ship safetensors."""

    pinned_revision: str
    """Specific commit SHA used by the adapter. NEVER `main` — that
    auto-updates with no notice. Bump only after re-verification.
    First adapter load uses `snapshot_download(revision=...)` to
    pin."""


# ---- per-provider provenance constants ----


_VOYAGE_PROVENANCE = EmbedderProviderProvenance(
    provider_name="voyage",
    display_name="Voyage AI",
    publisher="Voyage AI (Stanford spinout)",
    license="Proprietary cloud API (Voyage Terms)",
    source_url="https://www.voyageai.com/embedding-models",
    pip_extras="embed-voyage",
    requires_api_key=True,
    default_dim=EMBEDDING_DIM_DEFAULTS["voyage"],
)


_OPENAI_EMBED_PROVENANCE = EmbedderProviderProvenance(
    provider_name="openai",
    display_name="OpenAI Embeddings",
    publisher="OpenAI, Inc.",
    license="Proprietary cloud API (OpenAI Usage Policies)",
    source_url="https://platform.openai.com/docs/guides/embeddings",
    pip_extras="embed-openai",
    requires_api_key=True,
    # text-embedding-3-small default; 3-large is also 1536 by default
    # but supports shortening via dimensions param.
    default_dim=EMBEDDING_DIM_DEFAULTS["openai"],
)


_GOOGLE_EMBED_PROVENANCE = EmbedderProviderProvenance(
    provider_name="google",
    display_name="Google Gemini Embeddings",
    publisher="Google LLC",
    license="Proprietary cloud API (Google AI Studio terms)",
    source_url="https://ai.google.dev/gemini-api/docs/embeddings",
    pip_extras="embed-google",
    requires_api_key=True,
    default_dim=EMBEDDING_DIM_DEFAULTS["google"],  # text-embedding-004
)


_HF_EMBED_PROVENANCE = EmbedderProviderProvenance(
    provider_name="huggingface",
    display_name="HuggingFace (local)",
    publisher="HuggingFace + open-source community",
    license="Library: Apache-2.0; per-model licences vary",
    source_url="https://huggingface.co/models?library=sentence-transformers",
    pip_extras="embed-huggingface",
    requires_api_key=False,
    # Default model = BGE-m3 (1024-dim multilingual). User can pick
    # any entry from the curated menu via the GUI; the switch step
    # validates dimension against the LanceDB schema.
    default_dim=EMBEDDING_DIM_DEFAULTS["huggingface"],
)


# ---- HF curated model menu (LOCKED 4 entries 2026-06-29) ----


HF_EMBEDDING_MODELS: dict[str, HFEmbedderProvenance] = {
    "BAAI/bge-m3": HFEmbedderProvenance(
        model_id="BAAI/bge-m3",
        display_name="BGE-m3 (multilingual)",
        source_url="https://huggingface.co/BAAI/bge-m3",
        download_size_mb=2300,
        dimensions=1024,
        language="Multilingual (100+)",
        publisher="Beijing Academy of Artificial Intelligence",
        license="MIT",
        safetensors=True,
        # Pinned 2026-06-29 — bump only after re-verifying the HF
        # model card. Verify on bump: safetensors still present,
        # licence unchanged, dimensions still 1024.
        pinned_revision="5617a9f61b028005a4858fdac845db406aefb181",
    ),
    "mixedbread-ai/mxbai-embed-large-v1": HFEmbedderProvenance(
        model_id="mixedbread-ai/mxbai-embed-large-v1",
        display_name="mxbai-embed-large (English)",
        source_url="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
        download_size_mb=1300,
        dimensions=1024,
        language="English",
        publisher="Mixedbread AI",
        license="Apache-2.0",
        safetensors=True,
        pinned_revision="db9d1fe0f31addb4978201b2bf3e577f3f8900d2",
    ),
    "BAAI/bge-small-en-v1.5": HFEmbedderProvenance(
        model_id="BAAI/bge-small-en-v1.5",
        display_name="BGE-small-en (compact)",
        source_url="https://huggingface.co/BAAI/bge-small-en-v1.5",
        download_size_mb=130,
        dimensions=384,
        language="English",
        publisher="Beijing Academy of Artificial Intelligence",
        license="MIT",
        safetensors=True,
        pinned_revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    ),
    "sentence-transformers/all-MiniLM-L6-v2": HFEmbedderProvenance(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        display_name="all-MiniLM-L6 (classic / fastest)",
        source_url=("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"),
        download_size_mb=90,
        dimensions=384,
        language="English",
        publisher="UKP Lab / sentence-transformers",
        license="Apache-2.0",
        safetensors=True,
        pinned_revision="c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    ),
}


# ---- per-provider library detection ----


def _voyage_is_installed() -> bool:
    """True iff `voyageai` is importable.

    Pre-2026-06-29 this returned True unconditionally because the
    package shipped in base deps. Post-2026-06-29 Voyage is a provider
    like any other — installed only if the user picks it in the
    first-launch wizard.
    """
    try:
        import voyageai  # noqa: F401
    except ImportError:
        return False
    return True


def _openai_embed_is_installed() -> bool:
    """True iff `langchain-openai` is importable."""
    try:
        import langchain_openai  # noqa: F401
    except ImportError:
        return False
    return True


def _google_embed_is_installed() -> bool:
    """True iff `langchain-google-genai` is importable."""
    try:
        import langchain_google_genai  # noqa: F401
    except ImportError:
        return False
    return True


def _hf_embed_libs_installed() -> bool:
    """True iff `langchain_huggingface`, `sentence-transformers` AND `torch`
    are all importable.

    All three are pulled in by the `embed-huggingface` extra. We check all
    three (not just sentence-transformers + torch) because the factory loads
    the embedder through `langchain_huggingface.HuggingFaceEmbeddings` — if
    that adapter is missing the row must NOT show green, or the user installs
    "successfully" and then hits an ImportError on first embed. torch is also
    verified independently because it sometimes lingers from other extras
    (ASR / extractors).
    """
    try:
        import langchain_huggingface  # noqa: F401
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


# ---- registry ----


EMBEDDER_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "voyage": {
        "display_name": "Voyage AI",
        "bundled": False,
        "is_installed_fn": _voyage_is_installed,
        "library_packages": ("voyageai",),
        "provenance": _VOYAGE_PROVENANCE,
        "default_model": EMBEDDING_MODEL_DEFAULTS["voyage"],
    },
    "openai": {
        "display_name": "OpenAI Embeddings",
        "bundled": False,
        "is_installed_fn": _openai_embed_is_installed,
        "library_packages": ("langchain-openai",),
        "provenance": _OPENAI_EMBED_PROVENANCE,
        "default_model": EMBEDDING_MODEL_DEFAULTS["openai"],
    },
    "google": {
        "display_name": "Google Gemini Embeddings",
        "bundled": False,
        "is_installed_fn": _google_embed_is_installed,
        "library_packages": ("langchain-google-genai",),
        "provenance": _GOOGLE_EMBED_PROVENANCE,
        "default_model": EMBEDDING_MODEL_DEFAULTS["google"],
    },
    "huggingface": {
        "display_name": "HuggingFace (local)",
        "bundled": False,
        "is_installed_fn": _hf_embed_libs_installed,
        # torch is deliberately excluded from the pip-uninstall set: it is a
        # shared BASE dependency (docling's document parsing, lancedb, and
        # transformers all require it), so removing it when the user uninstalls
        # the HF embedder would break the app. Only the HF-exclusive packages
        # are removed. (Install detection stays via _hf_embed_libs_installed.)
        "library_packages": ("langchain-huggingface", "sentence-transformers"),
        "provenance": _HF_EMBED_PROVENANCE,
        # HF provider's "default model" is the menu entry initially
        # picked; settings.hf_embedding_model holds the active choice.
        "default_model": EMBEDDING_MODEL_DEFAULTS["huggingface"],
    },
}


# ---- subprocess wrapper (mockable) ----


async def _run_pip(args: list[str], timeout: float = 1200.0) -> tuple[bool, str]:
    """Run `python -m pip <args>` async; return (success, combined output).

    Timeout default is 20 minutes — the HF extra pulls torch (large
    compile chain on first install).

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


# ---- shared helpers ----


def _registry_entry(provider: str) -> dict[str, Any]:
    """Look up an embedder provider in the registry."""
    entry = EMBEDDER_PROVIDER_REGISTRY.get(provider)
    if entry is None:
        raise ValueError(
            f"Unknown embedder provider {provider!r}. Known: {sorted(EMBEDDER_PROVIDER_REGISTRY)}."
        )
    return entry


def _hf_model_provenance(model_id: str) -> HFEmbedderProvenance:
    """Look up a curated HF model entry, raise on miss."""
    prov = HF_EMBEDDING_MODELS.get(model_id)
    if prov is None:
        raise ValueError(
            f"Unknown HF embedding model {model_id!r}. Curated menu: {sorted(HF_EMBEDDING_MODELS)}."
        )
    return prov


# ---- install_embedder_provider ----


@dataclass(frozen=True)
class InstallEmbedderProviderPlan:
    """Plan for installing an embedder provider's adapter (or libs)."""

    provider_name: str
    display_name: str
    pip_extras: str | None
    already_installed: bool
    bundled: bool
    provenance: EmbedderProviderProvenance

    @property
    def summary(self) -> str:
        if self.bundled:
            return f"{self.display_name} ships by default - nothing to install."
        if self.already_installed:
            return f"{self.display_name} is already installed."
        size_note = ""
        if self.provider_name == "huggingface":
            # Surface the ~2-3 GB cost prominently — heaviest install
            # on the menu.
            size_note = (
                " WARNING: pulls sentence-transformers + torch "
                "(~2-3 GB depending on CUDA wheel). Model file is a "
                "separate download (90 MB - 2.3 GB) via the curated "
                "menu after libs are installed."
            )
        return (
            f"Install {self.display_name}: runs "
            f"`pip install <this package>[{self.pip_extras}]`. "
            f"Published by {self.provenance.publisher} "
            f"({self.provenance.license}). "
            f"Source: {self.provenance.source_url}. "
            f"Default model dimension: {self.provenance.default_dim}. "
            f"A restart is required after install.{size_note}"
        )


@dataclass(frozen=True)
class InstallEmbedderProviderResult:
    """Outcome of `install_embedder_provider_execute`."""

    provider_name: str
    did_install: bool
    install_ok: bool
    restart_required: bool
    pip_output: str


def install_embedder_provider_plan(
    provider: str,
) -> InstallEmbedderProviderPlan:
    """Build an install plan for the given embedder provider."""
    entry = _registry_entry(provider)
    prov: EmbedderProviderProvenance = entry["provenance"]
    return InstallEmbedderProviderPlan(
        provider_name=provider,
        display_name=entry["display_name"],
        pip_extras=prov.pip_extras,
        already_installed=bool(entry["is_installed_fn"]()),
        bundled=entry["bundled"],
        provenance=prov,
    )


async def install_embedder_provider_execute(
    plan: InstallEmbedderProviderPlan,
    *,
    distribution_name: str = DISTRIBUTION_NAME,
) -> InstallEmbedderProviderResult:
    """Run `pip install <dist>[<pip_extras>]` if not already installed."""
    if plan.bundled or plan.already_installed:
        return InstallEmbedderProviderResult(
            provider_name=plan.provider_name,
            did_install=False,
            install_ok=True,
            restart_required=False,
            pip_output="",
        )
    target = f"{distribution_name}[{plan.pip_extras}]"
    ok, output = await _run_pip(["install", target])
    return InstallEmbedderProviderResult(
        provider_name=plan.provider_name,
        did_install=True,
        install_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- uninstall_embedder_provider ----


@dataclass(frozen=True)
class UninstallEmbedderProviderPlan:
    """Plan for uninstalling an embedder provider's adapter.

    Blocking when `is_active=True` — same rule as
    `llm_lifecycle.UninstallLLMProviderPlan`.
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
                f"{self.display_name} is your ACTIVE embedding "
                f"provider. Switch to a different provider in "
                f"Settings first, then uninstall."
            )
        if not self.installed:
            return f"{self.display_name} is not installed."
        return (
            f"Uninstall {self.display_name}: runs "
            f"`pip uninstall {' '.join(self.packages_to_remove)}`. "
            f"A restart is required after uninstall."
        )


@dataclass(frozen=True)
class UninstallEmbedderProviderResult:
    """Outcome of `uninstall_embedder_provider_execute`."""

    provider_name: str
    did_uninstall: bool
    uninstall_ok: bool
    restart_required: bool
    pip_output: str


def uninstall_embedder_provider_plan(
    provider: str,
) -> UninstallEmbedderProviderPlan:
    """Build an uninstall plan."""
    entry = _registry_entry(provider)
    settings = get_settings()
    return UninstallEmbedderProviderPlan(
        provider_name=provider,
        display_name=entry["display_name"],
        packages_to_remove=tuple(entry["library_packages"]),
        installed=bool(entry["is_installed_fn"]()),
        bundled=entry["bundled"],
        is_active=(settings.embedding_provider == provider),
    )


async def uninstall_embedder_provider_execute(
    plan: UninstallEmbedderProviderPlan,
) -> UninstallEmbedderProviderResult:
    """Pip-uninstall the provider's adapter package."""
    if plan.bundled or not plan.installed or plan.is_active:
        return UninstallEmbedderProviderResult(
            provider_name=plan.provider_name,
            did_uninstall=False,
            uninstall_ok=True,
            restart_required=False,
            pip_output="",
        )
    ok, output = await _run_pip(["uninstall", "-y", *plan.packages_to_remove])
    return UninstallEmbedderProviderResult(
        provider_name=plan.provider_name,
        did_uninstall=True,
        uninstall_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- download_hf_model ----


@dataclass(frozen=True)
class DownloadHFModelPlan:
    """Plan for downloading one curated HF embedding model."""

    model_id: str
    provenance: HFEmbedderProvenance
    libs_installed: bool
    """sentence-transformers + torch must be installed first. When
    False the GUI should chain Install-libs → Download-model rather
    than letting the user try to skip ahead."""

    @property
    def summary(self) -> str:
        if not self.libs_installed:
            return (
                f"Cannot download {self.provenance.display_name} — "
                f"sentence-transformers + torch are not installed. "
                f"Install the HuggingFace libs first "
                f"(~2-3 GB), then return to this dialog."
            )
        # Future-proof: today all 4 curated HF entries ship safetensors,
        # but the warning helper will surface a prominent line if a
        # future menu entry ever has safetensors=False.
        from knowledge_agent._provenance import security_warning_text

        return (
            f"Download {self.provenance.display_name}: pulls "
            f"~{self.provenance.download_size_mb} MB from "
            f"{self.provenance.publisher} "
            f"({self.provenance.license}). "
            f"Dimensions: {self.provenance.dimensions}. "
            f"Language: {self.provenance.language}. "
            f"Source: {self.provenance.source_url} "
            f"(pinned to {self.provenance.pinned_revision[:12]}). "
            f"safetensors={self.provenance.safetensors}."
            f"{security_warning_text(safetensors=self.provenance.safetensors)}"
        )


@dataclass(frozen=True)
class DownloadHFModelResult:
    """Outcome of `download_hf_model_execute`."""

    model_id: str
    did_download: bool
    download_ok: bool
    error_output: str


def download_hf_model_plan(model_id: str) -> DownloadHFModelPlan:
    """Build a plan for downloading one curated HF embedding model."""
    prov = _hf_model_provenance(model_id)
    return DownloadHFModelPlan(
        model_id=model_id,
        provenance=prov,
        libs_installed=_hf_embed_libs_installed(),
    )


def _hf_model_revision(model_id: str) -> str:
    """Pinned revision for a curated embedding model; 'main' for an off-menu id.

    Curated models pin an exact SHA (provenance-verified). An off-menu id has no
    pin and resolves to 'main' — the download revision, encoded in the folder
    name so a path load is the pinned copy.
    """
    prov = HF_EMBEDDING_MODELS.get(model_id)
    return prov.pinned_revision if prov is not None else "main"


def _hf_model_dir(model_id: str):
    """Flat local folder holding this embedding model's weights.

    Single source of the on-disk layout is `Settings.model_dir` — one flat copy
    per pinned revision, NO HuggingFace cache/symlinks. Returns None when
    settings are unreachable (e.g. an unconfigured GUI).
    """
    try:
        return get_settings().model_dir(model_id, _hf_model_revision(model_id))
    except Exception:
        return None


def download_hf_model_execute(plan: DownloadHFModelPlan) -> DownloadHFModelResult:
    """Pre-download the model via `huggingface_hub.snapshot_download`.

    Downloads flat into `Settings.model_dir` (one copy per pinned revision, no
    HuggingFace cache/symlinks) pinned to the menu entry's `pinned_revision`, so
    the folder matches what was provenance-verified. `sentence-transformers`
    then loads straight from that folder path, no network access.
    """
    if not plan.libs_installed:
        return DownloadHFModelResult(
            model_id=plan.model_id,
            did_download=False,
            download_ok=False,
            error_output=(
                "sentence-transformers + torch not installed. "
                "Install the embed-huggingface extra first."
            ),
        )
    try:
        from huggingface_hub import snapshot_download

        # Download flat into our own per-revision folder (no HF cache, no
        # symlinks) — one copy, works on Windows without admin. `_hf_model_dir`
        # is the single resolver the factory load path shares.
        target = _hf_model_dir(plan.model_id)
        if target is None:
            return DownloadHFModelResult(
                model_id=plan.model_id,
                did_download=False,
                download_ok=False,
                error_output="could not resolve the models folder (settings unavailable).",
            )
        snapshot_download(
            repo_id=plan.model_id,
            revision=plan.provenance.pinned_revision,
            local_dir=str(target),
        )
        return DownloadHFModelResult(
            model_id=plan.model_id,
            did_download=True,
            download_ok=True,
            error_output="",
        )
    except Exception as exc:
        logger.exception("HF model download failed for %s", plan.model_id)
        return DownloadHFModelResult(
            model_id=plan.model_id,
            did_download=True,
            download_ok=False,
            error_output=f"{type(exc).__name__}: {exc}",
        )


# ---- switch_embedder_provider (with dimension-change guard) ----


@dataclass(frozen=True)
class SwitchEmbedderPlan:
    """Plan for switching the active embedding provider.

    `dim_mismatch=True` means the corpus already has rows at a
    different vector dimension than the new provider's default —
    the switch is DESTRUCTIVE (requires `drop_chunks_table` +
    re-ingest). The GUI surfaces this as a hard confirm dialog
    listing the number of rows that would be lost.

    `existing_rows` is the count of chunks the user would be wiping;
    surfaces in the confirm dialog so the cost is visible.
    """

    from_provider: str
    to_provider: str
    from_dim: int | None
    """Current corpus's pinned dimension, or None when there's no
    existing data."""
    to_dim: int
    """New provider's default dimension."""
    existing_rows: int
    """Chunks count in the current corpus."""
    dim_mismatch: bool
    """True iff `from_dim is not None and from_dim != to_dim`."""

    @property
    def summary(self) -> str:
        if self.from_provider == self.to_provider:
            return f"Already using {self.to_provider} — no change."
        if not self.dim_mismatch:
            if self.existing_rows == 0:
                return (
                    f"Switch from {self.from_provider} to "
                    f"{self.to_provider}. No data yet — no re-ingest "
                    f"needed."
                )
            return (
                f"Switch from {self.from_provider} to "
                f"{self.to_provider}. Both use "
                f"{self.to_dim}-dim embeddings; existing "
                f"{self.existing_rows} chunks stay valid but their "
                f"embeddings come from a different model — consider "
                f"re-embedding for retrieval-quality consistency."
            )
        return (
            f"DESTRUCTIVE switch: {self.from_provider} "
            f"({self.from_dim}-dim) → {self.to_provider} "
            f"({self.to_dim}-dim). The LanceDB chunks table pins the "
            f"vector dimension at creation; the {self.existing_rows} "
            f"existing chunks will be DROPPED and must be re-ingested "
            f"from source under the new provider. Confirm before "
            f"running."
        )


async def switch_embedder_plan(to_provider: str) -> SwitchEmbedderPlan:
    """Build a plan for switching to `to_provider`.

    Reads the current corpus's vector dim + row count from the LanceDB
    chunks table (if one exists) and compares against the new provider's
    default dim. The plan surfaces both the destructive flag and the row
    count so the GUI can warn appropriately.

    Async because the LanceDB read goes through the native-async
    `LanceClient` (`_ensure_conn` — there is no sync connection path
    since the 2026-06 async refactor). The GUI calls this from its async
    embedding-model handler.
    """
    settings = get_settings()
    from_provider = settings.embedding_provider
    to_entry = _registry_entry(to_provider)
    to_dim: int = to_entry["provenance"].default_dim

    from_dim: int | None = None
    existing_rows: int = 0
    try:
        from knowledge_agent.search.client import get_search_client
        from knowledge_agent.search.schema import CHUNKS_TABLE

        client = get_search_client()
        conn = await client._ensure_conn()
        if CHUNKS_TABLE in (await conn.list_tables()).tables:
            table = await conn.open_table(CHUNKS_TABLE)
            existing_rows = int(await table.count_rows())
            if existing_rows > 0:
                from_dim = (await table.schema()).field("embedding").type.list_size
    except Exception as exc:
        # If LanceDB isn't reachable we cannot tell — treat as "no
        # dim known", surface this as a safe-default plan that does
        # NOT flag destructive (the GUI can still warn via a separate
        # banner about LanceDB being down).
        logger.warning(
            "switch_embedder_plan: could not read LanceDB schema (%r); assuming no corpus yet",
            exc,
        )

    dim_mismatch = from_dim is not None and from_dim != to_dim
    return SwitchEmbedderPlan(
        from_provider=from_provider,
        to_provider=to_provider,
        from_dim=from_dim,
        to_dim=to_dim,
        existing_rows=existing_rows,
        dim_mismatch=dim_mismatch,
    )
