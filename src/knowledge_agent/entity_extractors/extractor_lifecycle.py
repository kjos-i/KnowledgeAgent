"""GUI-facing extractor lifecycle ops (Python-environment admin).

Same plan/execute UI contract as `kg/ontology_lifecycle.py`, but
operates on the user's Python environment (pip-installed packages +
model cache files), not on the KG. Lives in `entity_extractors/`
because the per-extractor lifecycle details (pip extras name, cache
location, install check) belong with each adapter.

Three ops:

  - `install_extractor` — `pip install rla[entities-<name>]` via
    subprocess against the same Python interpreter the GUI is running
    under. Prompts the user to restart the app (Python can't reliably
    pick up new packages mid-process).

  - `delete_extractor_cache` — free model weights without uninstalling
    the package. For SciSpaCy that means uninstalling the model wheel
    (`en_ner_bc5cdr_md`) while keeping the `scispacy` library so a
    future re-download is one button click away.

  - `uninstall_extractor` — full removal: cache + package. Symmetric
    with `install_extractor`.

`EXTRACTOR_REGISTRY` declares per-adapter lifecycle metadata. The LLM
adapter is bundled (no install/uninstall ops apply); other adapters
have a `pip_extras` key naming the optional-dependency group.

Each registry entry MAY also carry a `provenance: ModelProvenance`
describing where the model weights come from + their security posture.
The GUI's install dialog surfaces these fields (HF URL, publisher,
license, safetensors availability, trust_remote_code requirement,
pinned commit SHA) so the user can make an informed-consent decision
before downloading anything. Per `backend-no-ui-prompts`, this module
only PROVIDES the data; the dialog rendering lives in the GUI.

NOTE: subprocess pip calls assume a writeable Python environment
(venv or per-user install). In a frozen / immutable distribution, the
install op would need to surface "this distribution is read-only;
re-install the app with the extra included" - deferred until we
actually ship a frozen distribution.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---- model provenance (security disclosure for the install dialog) ----


@dataclass(frozen=True)
class ModelProvenance:
    """Where the model weights come from + their security posture.

    Populated by adapters whose install downloads model weights
    (HuggingFace adapters mostly). The install dialog reads these
    fields to show the user what they're about to download before
    pip/HF runs.

    All fields are intentionally simple types (str / int / bool) so
    the GUI can render them in a flat info panel without per-field
    formatting logic.
    """

    model_name: str
    """Canonical model identifier. For HF: the repo ID like
    `urchade/gliner_multi-v2.1`. For pip-only adapters: the pypi
    package name."""

    source_url: str
    """Direct URL the user can click to inspect the model themselves.
    For HF models this is the model card page."""

    download_size_mb: int
    """Approximate disk size after install, in MB. Surfaced so the
    user knows what they're committing to."""

    publisher: str
    """Who maintains the model. Free-form, but should be specific
    enough to verify provenance ("Urchade Zaratiana (Sorbonne)" rather
    than just "academic")."""

    license: str
    """License of the model weights (e.g. `Apache-2.0`, `MIT`).
    Important for downstream redistribution decisions."""

    safetensors: bool
    """True iff the model loads from `.safetensors` files (no pickle
    execution path). The dialog flags False prominently — pickle
    loading is a known attack vector."""

    trust_remote_code: bool
    """True iff loading the model REQUIRES executing custom Python
    code from the HF repo. The dialog flags True prominently — this
    is the highest-severity security concern the user should see. The
    adapter's source-file header MUST document which commit's code
    was audited when this is True."""

    pinned_revision: str
    """The specific commit SHA (or version tag for pip-only) the
    adapter loads. NEVER `main` — that auto-updates with no notice.
    The adapter code uses this revision when calling
    `from_pretrained`. Bump only after re-verification."""

    domain_tags: tuple[str, ...] = ()
    """The 20-tag taxonomy slots this extractor's default labels
    serve. Drives the install dialog's "this extractor covers ..."
    line AND the cross-link helper's "extractor → ontology
    candidates" routing. Default `()` for back-compat with
    transitional callers; in practice every shipped extractor
    populates it. Empty when the extractor is genuinely
    domain-agnostic (open-vocabulary LLM)."""


# ---- per-extractor registry ----


def _gliner_is_installed() -> bool:
    """True iff the gliner library is importable. The model weights
    themselves auto-download on first inference call via HF, so
    "installed" here means "the library is available" — same shape
    as the LLM adapter (which only needs langchain_anthropic to be
    available)."""
    try:
        import gliner  # noqa: F401
    except ImportError:
        return False
    return True


def _gliner_biomed_is_installed() -> bool:
    """True iff the gliner library is importable. Same check as
    `_gliner_is_installed` because GLiNER-BioMed reuses the same
    library — the two adapters differ only by which HF checkpoint
    they pin. Installing one extras installs the library for both
    (and pip de-duplicates the dep)."""
    try:
        import gliner  # noqa: F401
    except ImportError:
        return False
    return True


def _hunflair2_is_installed() -> bool:
    """True iff the flair library is importable. Model weights
    auto-download on first inference, so "installed" here means
    "library available" — same shape as the gliner adapters."""
    try:
        import flair  # noqa: F401
    except ImportError:
        return False
    return True


# GLiNER model provenance — surfaced by the install dialog before any
# download. Pinned 2026-06-23: model.safetensors only (~1.1 GB),
# Apache-2.0, no trust_remote_code. Mirrors the constants in
# `entity_extractors/gliner.py` (MODEL_NAME + MODEL_REVISION) — keep
# the two in sync when bumping. Re-verify all fields on the HF model
# page before changing pinned_revision.
_GLINER_PROVENANCE = ModelProvenance(
    model_name="urchade/gliner_multi-v2.1",
    source_url="https://huggingface.co/urchade/gliner_multi-v2.1",
    download_size_mb=1100,
    publisher="Urchade Zaratiana (Sorbonne) + Knowledgator",
    license="Apache-2.0",
    safetensors=True,
    trust_remote_code=False,
    pinned_revision="443d26d654e0324125a96bebd8e796c14ff2efe6",
    # General-purpose zero-shot NER — default labels (PERSON,
    # ORGANIZATION, LOCATION, EVENT, DATE, MISC) target general
    # corpora. Override `entity_types` in corpus.toml for any
    # specialised domain.
    domain_tags=("general",),
)


# GLiNER-BioMed model provenance. Pinned 2026-06-23 to the bi-large
# variant — bi-encoder paradigm, best on BC5CDR + CHIA + NCBI Disease,
# ~1.1 GB. Mirrors `entity_extractors/gliner_biomed.py` constants.
# IMPORTANT: this checkpoint has NO .safetensors file — loading uses
# pickle deserialisation (pytorch_model.bin). The install dialog
# surfaces safetensors=False prominently so the user can make an
# informed choice. Provenance (academic publisher + peer-reviewed
# Bioinformatics 2025 paper + Apache-2.0 + pinned SHA) mitigates but
# does not eliminate the pickle exec risk.
_GLINER_BIOMED_PROVENANCE = ModelProvenance(
    model_name="Ihor/gliner-biomed-bi-large-v1.0",
    source_url="https://huggingface.co/Ihor/gliner-biomed-bi-large-v1.0",
    download_size_mb=1100,
    publisher="Ihor Stepanov + DS4DH (University of Geneva)",
    license="Apache-2.0",
    safetensors=False,
    trust_remote_code=False,
    pinned_revision="75fb10d6d5500c5e98c285493578116540ec47d3",
    # Biomedical zero-shot — default labels (DISEASE, CHEMICAL,
    # GENE, PROTEIN, SPECIES, CELL_LINE, CELL_TYPE, ANATOMY) span
    # the standard biomedical NER vocabulary.
    domain_tags=(
        "medicine", "biology", "chemistry", "proteins", "cell biology",
    ),
)


# HunFlair2 provenance. Pinned 2026-06-23 to the unified hunflair2-ner
# model (~1.24 GB). Same pickle-only situation as GLiNER-BioMed —
# install dialog flags the risk; provenance (established academic
# group, MIT-licensed Flair library, pinned SHA) mitigates.
_HUNFLAIR2_PROVENANCE = ModelProvenance(
    model_name="hunflair/hunflair2-ner",
    source_url="https://huggingface.co/hunflair/hunflair2-ner",
    download_size_mb=1240,
    publisher="HU Berlin Bioinformatics (Flair NLP team)",
    license="MIT (inherited from Flair, Zalando SE 2018)",
    safetensors=False,
    trust_remote_code=False,
    pinned_revision="3af2b8972f7af2910ce8d9ae724da09b3d7a166c",
    # Fixed biomedical 5-label set (DISEASE, CHEMICAL, GENE,
    # SPECIES, CELL_LINE). Narrower than GLiNER-BioMed — no PROTEIN
    # / CELL_TYPE / ANATOMY — but the unified Flair tagger gives
    # tight per-label scores.
    domain_tags=("medicine", "biology", "chemistry"),
)


# Default labels each NER adapter emits — referenced from the
# registry below. Imported as constants rather than at registry-build
# time so the install dialog can describe each adapter's output
# vocabulary BEFORE the adapter's heavy dependencies (PyTorch / Flair)
# are pip-installed. Each per-adapter module's `DEFAULT_LABELS` constant
# is itself dependency-free (plain tuple), so this re-import is safe
# without paying the model-load cost.
#
# For LLM the value is `None` (open vocabulary — runtime-chosen);
# HunFlair2's value mirrors the project-side normalised labels emitted
# by its adapter (Flair internally tags title-case; the adapter
# uppercases at the Mention boundary).
from knowledge_agent.entity_extractors.gliner import (  # noqa: E402
    DEFAULT_LABELS as _GLINER_EMITTED_LABELS,
)
from knowledge_agent.entity_extractors.gliner_biomed import (  # noqa: E402
    DEFAULT_LABELS as _GLINER_BIOMED_EMITTED_LABELS,
)

# HunFlair2 emits a fixed 5-label set; the adapter's
# `_FLAIR_TO_OUR_LABELS` mapping defines them, but we hard-code here
# to avoid loading the adapter (which would try to import Flair).
_HUNFLAIR2_EMITTED_LABELS: tuple[str, ...] = (
    "DISEASE", "CHEMICAL", "GENE", "SPECIES", "CELL_LINE",
)


EXTRACTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "llm": {
        "display_name": "LLM (Claude Haiku)",
        # Bundled with the base package - install/uninstall are no-ops.
        "bundled": True,
        "pip_extras": None,
        "model_packages": (),
        "is_installed_fn": lambda: True,
        # Bundled: nothing downloads on install, no provenance to surface.
        "provenance": None,
        # Open-vocabulary: the LLM picks labels at runtime from the
        # chunk text. `None` is the explicit "no static label set"
        # marker the cross-link helpers check for.
        "emitted_labels": None,
        # Domain-agnostic by design — covers anything the user asks
        # for via `entity_types` in corpus.toml.
        "domain_tags": (),
    },
    "gliner": {
        "display_name": "GLiNER (zero-shot, multilingual)",
        "bundled": False,
        "pip_extras": "entities-gliner",
        # Model weights auto-download from HF on first inference
        # (HF cache, not pip-installed). No separate model_packages
        # to uninstall via pip — delete_cache for gliner would mean
        # clearing the HF cache directory, which is a future
        # follow-up (currently model_packages stays empty so
        # delete_cache is a no-op for gliner).
        "model_packages": (),
        "is_installed_fn": _gliner_is_installed,
        # Provenance surfaces in the install dialog before download.
        "provenance": _GLINER_PROVENANCE,
        "emitted_labels": _GLINER_EMITTED_LABELS,
        "domain_tags": _GLINER_PROVENANCE.domain_tags,
    },
    "gliner_biomed": {
        "display_name": "GLiNER-BioMed (biomedical zero-shot)",
        "bundled": False,
        "pip_extras": "entities-gliner-biomed",
        # Same library as general gliner — pip de-duplicates if both
        # extras are installed. HF cache holds the per-model weights.
        "model_packages": (),
        "is_installed_fn": _gliner_biomed_is_installed,
        # Provenance flags safetensors=False — pickle exec risk
        # surfaced in install dialog.
        "provenance": _GLINER_BIOMED_PROVENANCE,
        "emitted_labels": _GLINER_BIOMED_EMITTED_LABELS,
        "domain_tags": _GLINER_BIOMED_PROVENANCE.domain_tags,
    },
    "hunflair2": {
        "display_name": "HunFlair2 (biomedical NER, all-or-nothing)",
        "bundled": False,
        "pip_extras": "entities-hunflair2",
        # Flair model weights live in the HF cache; no pip
        # model_packages to uninstall. Delete-cache for hunflair2
        # would clear the HF cache for the pinned model — future
        # follow-up; currently a no-op.
        "model_packages": (),
        "is_installed_fn": _hunflair2_is_installed,
        # Provenance flags safetensors=False — pickle exec risk
        # surfaced in install dialog.
        "provenance": _HUNFLAIR2_PROVENANCE,
        "emitted_labels": _HUNFLAIR2_EMITTED_LABELS,
        "domain_tags": _HUNFLAIR2_PROVENANCE.domain_tags,
    },
}


# ---- subprocess wrapper (mockable) ----


async def _run_pip(
    args: list[str], timeout: float = 600.0
) -> tuple[bool, str]:
    """Run `python -m pip <args>` async; return (success, combined output).

    `sys.executable` ensures the pip call targets the SAME interpreter
    the app is running under, so the new package is visible after a
    restart. Timeout default is generous (10 minutes) because heavy
    extractor extras can pull in PyTorch + CUDA wheels.

    Uses `asyncio.create_subprocess_exec` so the caller's event loop
    (Flet GUI / CLI / eval harness) stays responsive while pip runs.
    On timeout the child is killed with `proc.kill()` + reaped via
    `proc.wait()` so no zombie remains.
    """
    cmd = [sys.executable, "-m", "pip"] + args
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return False, f"pip command timed out after {timeout}s"
        output = (
            (stdout.decode("utf-8", errors="replace") if stdout else "")
            + (stderr.decode("utf-8", errors="replace") if stderr else "")
        )
        return proc.returncode == 0, output
    except Exception as exc:
        return False, f"pip subprocess failed: {exc!r}"


# ---- install_extractor ----


@dataclass(frozen=True)
class InstallExtractorPlan:
    """Plan for installing an extractor.

    `bundled=True` means the extractor ships with the base package
    (LLM) - the plan's summary says "already available" and execute
    is a no-op. For non-bundled extractors, `pip_extras` carries the
    optional-dependency group name; execute shells out to pip.

    `provenance` carries model-download security disclosure (HF URL,
    publisher, license, safetensors flag, trust_remote_code flag,
    pinned revision SHA, download size) for adapters whose install
    pulls model weights. `None` means no model weights download as
    part of install (bundled adapters, pip-only adapters that don't
    pull model files until first inference). The GUI's install dialog
    renders these fields when present so the user can decide before
    download. See `backend-no-ui-prompts` — this dataclass is pure
    data; the actual confirmation dialog lives in the GUI.

    `domain_tags` and `emitted_labels` come from the registry. They
    drive the install dialog's "this extractor covers ..." +
    "emits labels: ..." lines so the user can decide which extractor
    matches their corpus shape before commiting to the download.
    Open-vocabulary extractors (LLM) carry `emitted_labels=None`; the
    summary surfaces "open vocabulary (LLM-chosen at runtime)" in that
    case.
    """

    extractor_name: str
    display_name: str
    bundled: bool
    pip_extras: str | None
    already_installed: bool
    provenance: ModelProvenance | None = None
    domain_tags: tuple[str, ...] = ()
    emitted_labels: tuple[str, ...] | None = None
    canonicalization_candidates: (
        tuple[tuple[str, tuple[str, ...]], ...] | None
    ) = None
    """Pre-computed cross-link surface for the coverage clause.

    `tuple[(label, ontology_names), ...]` — one entry per emitted
    label, with the ontology names whose `covers_labels` include it.
    Labels with no matching ontology have an empty `ontology_names`
    tuple.

    `None` is the open-vocabulary marker (LLM `emitted_labels=None`):
    the summary surfaces a "matches any ontology lexically" line
    instead of a per-label mapping.

    Filled by `install_extractor_plan` via
    `kg.ontology_lifecycle.get_canonicalization_candidates`."""

    @property
    def summary(self) -> str:
        if self.bundled:
            return (
                f"{self.display_name} ships by default - nothing to "
                f"install.{self._coverage_clause()}"
            )
        if self.already_installed:
            return (
                f"{self.display_name} is already installed."
                f"{self._coverage_clause()}"
            )
        if self.provenance is None:
            return (
                f"Install {self.display_name}? Will run "
                f"`pip install <this package>[{self.pip_extras}]`. "
                f"A restart is required after install."
                f"{self._coverage_clause()}"
            )
        # Surface provenance facts in the summary so even text-only
        # callers (CLI, smoke tests) see what's about to download.
        # Security-elevated flags (no-safetensors, trust_remote_code)
        # get a prominent appended warning via `_provenance`.
        from knowledge_agent._provenance import security_warning_text

        return (
            f"Install {self.display_name}? Will run "
            f"`pip install <this package>[{self.pip_extras}]`. "
            f"Downloads {self.provenance.model_name} "
            f"(~{self.provenance.download_size_mb} MB) from "
            f"{self.provenance.publisher} under {self.provenance.license}. "
            f"Source: {self.provenance.source_url} "
            f"(pinned to {self.provenance.pinned_revision[:12]}). "
            f"safetensors={self.provenance.safetensors}, "
            f"trust_remote_code={self.provenance.trust_remote_code}. "
            f"A restart is required after install."
            f"{security_warning_text(safetensors=self.provenance.safetensors, trust_remote_code=self.provenance.trust_remote_code)}"
            f"{self._coverage_clause()}"
        )

    def _coverage_clause(self) -> str:
        """One-line append describing the adapter's domain coverage,
        emitted-label vocabulary, and cross-link to ontologies. Empty
        when nothing's set (back-compat for callers that construct
        the plan directly without registry-sourced fields)."""
        parts: list[str] = []
        if self.domain_tags:
            parts.append(
                f"Domain tags: {', '.join(self.domain_tags)}."
            )
        else:
            # Empty tuple is meaningful for LLM ("domain-agnostic"); the
            # absence of the field altogether is what we want silent.
            if self.emitted_labels is None and self.bundled:
                parts.append(
                    "Domain tags: (domain-agnostic — open vocabulary)."
                )
        if self.emitted_labels is None:
            parts.append(
                "Emitted labels: open vocabulary "
                "(LLM-chosen at runtime; constrain via "
                "`entity_types` in corpus.toml)."
            )
        elif self.emitted_labels:
            parts.append(
                f"Emitted labels: {', '.join(self.emitted_labels)}."
            )

        # Cross-link surface: per-label ontology candidates.
        if self.canonicalization_candidates is None:
            # Open-vocabulary marker — only meaningful when emitted is
            # also None (LLM). Skip if the plan was built without
            # registry sourcing (back-compat default).
            if self.emitted_labels is None and self.bundled:
                parts.append(
                    "Canonicalisation: any ontology lexically — "
                    "runtime label choices match against whatever "
                    "ontologies are imported."
                )
        elif self.canonicalization_candidates:
            target_parts: list[str] = []
            for label, ont_names in self.canonicalization_candidates:
                if ont_names:
                    target_parts.append(
                        f"{label} → {', '.join(ont_names)}"
                    )
                else:
                    target_parts.append(
                        f"{label} → (no shipped ontology covers this)"
                    )
            if target_parts:
                parts.append(
                    "Canonicalisation targets: "
                    + "; ".join(target_parts) + "."
                )

        if not parts:
            return ""
        return " " + " ".join(parts)


@dataclass(frozen=True)
class InstallExtractorResult:
    """Outcome of `install_extractor_execute`.

    `did_install` is False when the extractor was bundled OR already
    installed (op is a no-op). `install_ok` is False when pip ran and
    failed; `pip_output` carries the combined stdout/stderr for the
    GUI to surface ("see why it failed").

    `restart_required` mirrors `did_install` - if pip actually ran,
    the user must restart for the new package to be importable.
    """

    extractor_name: str
    did_install: bool
    install_ok: bool
    restart_required: bool
    pip_output: str


def _registry_entry(extractor_name: str) -> dict[str, Any]:
    """Look up an extractor in the registry, raise ValueError on miss."""
    entry = EXTRACTOR_REGISTRY.get(extractor_name)
    if entry is None:
        raise ValueError(
            f"Unknown extractor {extractor_name!r}. "
            f"Known extractors: {sorted(EXTRACTOR_REGISTRY)}."
        )
    return entry


def install_extractor_plan(extractor_name: str) -> InstallExtractorPlan:
    """Build a plan for installing the named extractor.

    Calls the per-adapter `is_installed_fn` to detect current state.
    Threads the optional `provenance` ModelProvenance from the registry
    so the GUI dialog can show security disclosure (HF URL, publisher,
    license, safetensors, trust_remote_code, pinned revision). Raises
    `ValueError` for an unknown extractor.
    """
    entry = _registry_entry(extractor_name)
    already_installed = bool(entry["is_installed_fn"]())
    # Pre-compute the cross-link surface at plan-build time. Lazy
    # import avoids the import-graph cycle: extractor_lifecycle is a
    # leaf module that kg/ontology_lifecycle imports from, so we
    # can't have a reverse top-level import here without cycling.
    from knowledge_agent.kg.ontology_lifecycle import (
        get_canonicalization_candidates,
    )
    candidates_dict = get_canonicalization_candidates(extractor_name)
    # Open-vocabulary marker: emitted_labels=None -> candidates={}.
    # Preserve the None sentinel so the summary surfaces "any ontology
    # lexically" instead of "no candidates" (different semantics).
    emitted_labels = entry.get("emitted_labels")
    canonicalization_candidates: (
        tuple[tuple[str, tuple[str, ...]], ...] | None
    )
    if emitted_labels is None:
        canonicalization_candidates = None
    else:
        # Sort by extractor's emitted order so the dialog shows the
        # adapter's natural priority. Each value becomes a tuple for
        # the frozen dataclass.
        canonicalization_candidates = tuple(
            (label, candidates_dict.get(label, ()))
            for label in emitted_labels
        )
    return InstallExtractorPlan(
        extractor_name=extractor_name,
        display_name=entry["display_name"],
        bundled=entry["bundled"],
        pip_extras=entry["pip_extras"],
        already_installed=already_installed,
        provenance=entry.get("provenance"),
        domain_tags=entry.get("domain_tags", ()),
        emitted_labels=emitted_labels,
        canonicalization_candidates=canonicalization_candidates,
    )


async def install_extractor_execute(
    plan: InstallExtractorPlan,
    *,
    distribution_name: str = "research-literature-agent",
) -> InstallExtractorResult:
    """Run `pip install <dist>[<extras>]` for non-bundled extractors.

    Bundled extractors and already-installed extractors short-circuit
    as no-op successes. `distribution_name` is the pip name of the app
    itself (the thing the user `pip install`d to get the base) - the
    `[<extras>]` suffix activates the optional dependency group.
    """
    if plan.bundled:
        return InstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_install=False, install_ok=True,
            restart_required=False, pip_output="",
        )
    if plan.already_installed:
        return InstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_install=False, install_ok=True,
            restart_required=False, pip_output="",
        )

    target = f"{distribution_name}[{plan.pip_extras}]"
    ok, output = await _run_pip(["install", target])
    return InstallExtractorResult(
        extractor_name=plan.extractor_name,
        did_install=True,
        install_ok=ok,
        restart_required=ok,  # only restart if the install actually landed
        pip_output=output,
    )


# ---- delete_extractor_cache ----


@dataclass(frozen=True)
class DeleteExtractorCachePlan:
    """Plan for freeing model-cache disk for an extractor.

    `model_packages` is the list of pip packages this extractor's
    model weights live in (e.g., `("en_ner_bc5cdr_md",)` for SciSpaCy).
    Uninstalling those frees the disk while leaving the extractor
    library so a fresh re-download is one button away.

    Empty `model_packages` (LLM) means "nothing to delete" and the
    summary reflects that.
    """

    extractor_name: str
    display_name: str
    model_packages: tuple[str, ...]
    installed: bool

    @property
    def summary(self) -> str:
        if not self.installed:
            return f"{self.display_name} is not installed - nothing to delete."
        if not self.model_packages:
            return (
                f"{self.display_name} has no separate model cache - "
                f"nothing to delete."
            )
        listed = ", ".join(self.model_packages)
        return (
            f"Delete cached model files for {self.display_name}? "
            f"Will uninstall: {listed}. The extractor library stays so "
            f"future re-download is a single click."
        )


@dataclass(frozen=True)
class DeleteExtractorCacheResult:
    """Outcome of `delete_extractor_cache_execute`."""

    extractor_name: str
    did_delete: bool
    delete_ok: bool
    pip_output: str


def delete_extractor_cache_plan(
    extractor_name: str,
) -> DeleteExtractorCachePlan:
    """Build a plan for deleting an extractor's model cache."""
    entry = _registry_entry(extractor_name)
    return DeleteExtractorCachePlan(
        extractor_name=extractor_name,
        display_name=entry["display_name"],
        model_packages=tuple(entry["model_packages"]),
        installed=bool(entry["is_installed_fn"]()),
    )


async def delete_extractor_cache_execute(
    plan: DeleteExtractorCachePlan,
) -> DeleteExtractorCacheResult:
    """Pip-uninstall the model packages declared in `plan.model_packages`."""
    if not plan.installed or not plan.model_packages:
        return DeleteExtractorCacheResult(
            extractor_name=plan.extractor_name,
            did_delete=False, delete_ok=True, pip_output="",
        )

    ok, output = await _run_pip(
        ["uninstall", "-y", *plan.model_packages]
    )
    return DeleteExtractorCacheResult(
        extractor_name=plan.extractor_name,
        did_delete=True,
        delete_ok=ok,
        pip_output=output,
    )


# ---- uninstall_extractor ----


@dataclass(frozen=True)
class UninstallExtractorPlan:
    """Plan for fully removing an extractor (package + cache).

    `bundled=True` blocks uninstall entirely (LLM ships with the base
    package; removing it would break the app). The `packages_to_remove`
    list is the union of the extras package name and any model
    packages - executed as one `pip uninstall` for atomicity.
    """

    extractor_name: str
    display_name: str
    bundled: bool
    pip_extras: str | None
    packages_to_remove: tuple[str, ...]
    installed: bool

    @property
    def summary(self) -> str:
        if self.bundled:
            return (
                f"{self.display_name} is bundled with the base package - "
                f"cannot be uninstalled."
            )
        if not self.installed:
            return f"{self.display_name} is not installed."
        return (
            f"Uninstall {self.display_name}? Will remove the library + "
            f"cached models ({', '.join(self.packages_to_remove)}). A "
            f"restart is required after uninstall."
        )


@dataclass(frozen=True)
class UninstallExtractorResult:
    """Outcome of `uninstall_extractor_execute`."""

    extractor_name: str
    did_uninstall: bool
    uninstall_ok: bool
    restart_required: bool
    pip_output: str


# Each extractor module names the pip-uninstall target for its main
# library separately from the model packages so the lifecycle module
# doesn't have to know e.g. "scispacy is the library, en_ner_* is the
# model". For now mirror this via the `pip_extras`-derived library
# name plus the model_packages list.
_EXTRACTOR_LIBRARY_PACKAGES: dict[str, tuple[str, ...]] = {
    "gliner": ("gliner",),
    "gliner_biomed": ("gliner",),
    "hunflair2": ("flair", "huggingface-hub"),
}


def uninstall_extractor_plan(extractor_name: str) -> UninstallExtractorPlan:
    """Build a plan for fully uninstalling an extractor."""
    entry = _registry_entry(extractor_name)
    library_pkgs = _EXTRACTOR_LIBRARY_PACKAGES.get(extractor_name, ())
    model_pkgs = tuple(entry["model_packages"])
    return UninstallExtractorPlan(
        extractor_name=extractor_name,
        display_name=entry["display_name"],
        bundled=entry["bundled"],
        pip_extras=entry["pip_extras"],
        packages_to_remove=tuple(library_pkgs) + model_pkgs,
        installed=bool(entry["is_installed_fn"]()),
    )


async def uninstall_extractor_execute(
    plan: UninstallExtractorPlan,
) -> UninstallExtractorResult:
    """Pip-uninstall the library + model packages in one command.

    Bundled extractors and not-installed extractors short-circuit as
    no-op successes. On a real uninstall, `restart_required=True` -
    Python can't reliably forget a loaded module mid-process.
    """
    if plan.bundled:
        return UninstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_uninstall=False, uninstall_ok=False,
            restart_required=False,
            pip_output="bundled extractor; uninstall is not allowed",
        )
    if not plan.installed:
        return UninstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_uninstall=False, uninstall_ok=True,
            restart_required=False, pip_output="",
        )

    ok, output = await _run_pip(
        ["uninstall", "-y", *plan.packages_to_remove]
    )
    return UninstallExtractorResult(
        extractor_name=plan.extractor_name,
        did_uninstall=True,
        uninstall_ok=ok,
        restart_required=ok,
        pip_output=output,
    )
