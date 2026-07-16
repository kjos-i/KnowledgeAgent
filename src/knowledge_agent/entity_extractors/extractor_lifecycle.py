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
        "medicine",
        "biology",
        "chemistry",
        "proteins",
        "cell biology",
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
    "DISEASE",
    "CHEMICAL",
    "GENE",
    "SPECIES",
    "CELL_LINE",
)


EXTRACTOR_REGISTRY: dict[str, dict[str, Any]] = {
    "llm": {
        "display_name": "LLM",
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


async def _run_pip(args: list[str], timeout: float = 600.0) -> tuple[bool, str]:
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
    canonicalization_candidates: tuple[tuple[str, tuple[str, ...]], ...] | None = None
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
            return f"{self.display_name} is already installed.{self._coverage_clause()}"
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
            parts.append(f"Domain tags: {', '.join(self.domain_tags)}.")
        else:
            # Empty tuple is meaningful for LLM ("domain-agnostic"); the
            # absence of the field altogether is what we want silent.
            if self.emitted_labels is None and self.bundled:
                parts.append("Domain tags: (domain-agnostic — open vocabulary).")
        if self.emitted_labels is None:
            parts.append(
                "Emitted labels: open vocabulary "
                "(LLM-chosen at runtime; constrain via "
                "`entity_types` in corpus.toml)."
            )
        elif self.emitted_labels:
            parts.append(f"Emitted labels: {', '.join(self.emitted_labels)}.")

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
                    target_parts.append(f"{label} → {', '.join(ont_names)}")
                else:
                    target_parts.append(f"{label} → (no shipped ontology covers this)")
            if target_parts:
                parts.append("Canonicalisation targets: " + "; ".join(target_parts) + ".")

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
            f"Unknown extractor {extractor_name!r}. Known extractors: {sorted(EXTRACTOR_REGISTRY)}."
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
    canonicalization_candidates: tuple[tuple[str, tuple[str, ...]], ...] | None
    if emitted_labels is None:
        canonicalization_candidates = None
    else:
        # Sort by extractor's emitted order so the dialog shows the
        # adapter's natural priority. Each value becomes a tuple for
        # the frozen dataclass.
        canonicalization_candidates = tuple(
            (label, candidates_dict.get(label, ())) for label in emitted_labels
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
            did_install=False,
            install_ok=True,
            restart_required=False,
            pip_output="",
        )
    if plan.already_installed:
        return InstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_install=False,
            install_ok=True,
            restart_required=False,
            pip_output="",
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
            return f"{self.display_name} has no separate model cache - nothing to delete."
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
            did_delete=False,
            delete_ok=True,
            pip_output="",
        )

    ok, output = await _run_pip(["uninstall", "-y", *plan.model_packages])
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
            return f"{self.display_name} is bundled with the base package - cannot be uninstalled."
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
            did_uninstall=False,
            uninstall_ok=False,
            restart_required=False,
            pip_output="bundled extractor; uninstall is not allowed",
        )
    if not plan.installed:
        return UninstallExtractorResult(
            extractor_name=plan.extractor_name,
            did_uninstall=False,
            uninstall_ok=True,
            restart_required=False,
            pip_output="",
        )

    ok, output = await _run_pip(["uninstall", "-y", *plan.packages_to_remove])
    return UninstallExtractorResult(
        extractor_name=plan.extractor_name,
        did_uninstall=True,
        uninstall_ok=ok,
        restart_required=ok,
        pip_output=output,
    )


# ---- weight download + delete (HF hub cache) ----
#
# Split 2026-07-03: model weights are a separate axis from pip install.
# `install_extractor` handles the pip package (library code + Python
# deps). `download_extractor_weights` handles the HF-cached model files
# (~1 GB each). Rule: no auto-download on first inference — adapters
# raise `WeightsNotDownloadedError` if weights are missing. Users must
# explicitly download via Library → Installs.


class WeightsNotDownloadedError(RuntimeError):
    """Raised by an adapter's model loader when weights are not on disk.

    Ingest catches this at the per-doc L6 boundary and surfaces a
    typed error rather than crashing the run. The GUI corpus-config
    editor grays out extractors whose weights aren't downloaded, so
    the user hits this error only when a race happens (config edited
    between plan and run) or the ingest was kicked off outside the
    GUI. Message includes a hint pointing at the Library → Installs
    tab so recovery is a single navigation.
    """

    def __init__(
        self,
        extractor_name: str,
        model_name: str,
        hint: str,
    ) -> None:
        super().__init__(f"{extractor_name}: model {model_name!r} weights not downloaded. {hint}")
        self.extractor_name = extractor_name
        self.model_name = model_name


def _weights_cache_dir(provenance: ModelProvenance):
    """Return the HF hub cache directory for this extractor's model.

    HF hub caches every downloaded repo under
    `<HF_HUB_CACHE>/models--<org>--<model>/`. Splitting on "/" gives
    us the two components. `HF_HUB_CACHE` is huggingface_hub's own
    resolved default (honours HF_HOME + HF_HUB_CACHE env vars) —
    reusing it means we never disagree with the library about where
    weights actually land.

    Returns None when `provenance` is None (bundled extractors have
    no downloadable weights) or when the model name doesn't split
    cleanly (defensive; every shipped provenance uses the canonical
    `org/model` form).
    """
    from pathlib import Path

    if provenance is None:
        return None
    parts = provenance.model_name.split("/")
    if len(parts) != 2:
        return None
    org, model = parts
    try:
        from huggingface_hub import constants as _hf_constants

        hub_cache = Path(_hf_constants.HF_HUB_CACHE)
    except ImportError:
        # huggingface_hub not installed — no weights could exist locally
        # since nothing could download them. Fall back to the documented
        # default so the caller can still report "not downloaded" without
        # blowing up.
        hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    return hub_cache / f"models--{org}--{model}"


def _is_weights_downloaded(provenance: ModelProvenance) -> bool:
    """True iff the pinned revision's snapshot is materialised on disk.

    HF snapshots live at `<cache_dir>/snapshots/<revision>/`. We check
    for the specific pinned revision directory rather than "any
    snapshot" so an older cached checkpoint doesn't masquerade as
    ready — the adapter loads via the pinned SHA and would trigger
    an unwanted download if the pin doesn't match.

    Returns False when provenance is None (bundled) or when the cache
    dir doesn't exist.
    """
    cache_dir = _weights_cache_dir(provenance)
    if cache_dir is None or not cache_dir.exists():
        return False
    snapshot_dir = cache_dir / "snapshots" / provenance.pinned_revision
    return snapshot_dir.exists()


def _weights_size_bytes(provenance: ModelProvenance) -> int:
    """Total on-disk bytes for this extractor's weights (0 when absent).

    Sums every file under the cache dir (blobs + snapshots + refs).
    Used by the Installs tab's status chip to show "downloaded
    (245 MB)" per row.
    """
    cache_dir = _weights_cache_dir(provenance)
    if cache_dir is None or not cache_dir.exists():
        return 0
    total = 0
    for f in cache_dir.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                # Symlink target moved, permissions changed — skip.
                continue
    return total


def get_weights_downloaded_bytes(extractor_name: str) -> int:
    """Public probe: total on-disk bytes for `extractor_name`'s weights.

    Returns 0 for bundled/unknown extractors and for adapters whose
    weights haven't been downloaded. Used by the Installs tab status
    chip. Never raises — falls back to 0 on any lookup miss so a
    corrupt registry entry doesn't break the whole tab render.
    """
    entry = EXTRACTOR_REGISTRY.get(extractor_name)
    if entry is None:
        return 0
    return _weights_size_bytes(entry.get("provenance"))


def is_extractor_ready(extractor_name: str) -> bool:
    """True iff the extractor is fully ready to run inference.

    Compound: pip package installed AND weights downloaded. Bundled
    extractors are always ready. Used by the corpus-config editor
    to gray out entries in the extractor Dropdown.
    """
    entry = EXTRACTOR_REGISTRY.get(extractor_name)
    if entry is None:
        return False
    if entry.get("bundled"):
        return True
    if not entry["is_installed_fn"]():
        return False
    provenance = entry.get("provenance")
    if provenance is None:
        # Non-bundled adapter with no weights (edge case — no shipped
        # adapter fits this today). If we ever add one, treat as ready
        # once the pip package is installed.
        return True
    return _is_weights_downloaded(provenance)


# ---- download_extractor_weights ----


@dataclass(frozen=True)
class DownloadExtractorWeightsPlan:
    """Plan for downloading an extractor's model weights.

    `provenance` is None for bundled extractors (LLM) — the plan's
    summary reports "no weights to download" and execute is a no-op.
    `already_downloaded` short-circuits the same way. `on_disk_bytes`
    is the current cache size; 0 when not downloaded, useful in the
    summary when already downloaded (shows what will be redundant).
    """

    extractor_name: str
    display_name: str
    provenance: ModelProvenance | None
    already_downloaded: bool
    on_disk_bytes: int

    @property
    def summary(self) -> str:
        if self.provenance is None:
            return f"{self.display_name} has no downloadable weights."
        if self.already_downloaded:
            mb = self.on_disk_bytes / (1024 * 1024)
            return (
                f"{self.display_name} weights already downloaded "
                f"({mb:.0f} MB on disk, pinned to "
                f"{self.provenance.pinned_revision[:12]})."
            )
        return (
            f"Download {self.display_name} weights: "
            f"{self.provenance.model_name} "
            f"(~{self.provenance.download_size_mb} MB) from "
            f"{self.provenance.publisher} under "
            f"{self.provenance.license}. Pinned to "
            f"{self.provenance.pinned_revision[:12]}."
        )


@dataclass(frozen=True)
class DownloadExtractorWeightsResult:
    """Outcome of `download_extractor_weights_execute`.

    `did_download` is False when the extractor had no weights (bundled)
    or when weights were already on disk. `download_ok=False` means
    huggingface_hub raised — `download_error` carries the message.
    `on_disk_bytes` is the post-download size so the GUI can update
    its status chip in one round-trip.
    """

    extractor_name: str
    did_download: bool
    download_ok: bool
    download_error: str | None
    on_disk_bytes: int


def download_extractor_weights_plan(
    extractor_name: str,
) -> DownloadExtractorWeightsPlan:
    """Build a plan for downloading the named extractor's model weights."""
    entry = _registry_entry(extractor_name)
    provenance = entry.get("provenance")
    on_disk = _weights_size_bytes(provenance)
    return DownloadExtractorWeightsPlan(
        extractor_name=extractor_name,
        display_name=entry["display_name"],
        provenance=provenance,
        already_downloaded=(provenance is not None and _is_weights_downloaded(provenance)),
        on_disk_bytes=on_disk,
    )


async def download_extractor_weights_execute(
    plan: DownloadExtractorWeightsPlan,
) -> DownloadExtractorWeightsResult:
    """Fetch weights via huggingface_hub.snapshot_download.

    Runs in a worker thread (snapshot_download is sync + blocking on
    network + disk). Pins to `provenance.pinned_revision` — never
    `main`.
    """
    if plan.provenance is None:
        return DownloadExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_download=False,
            download_ok=True,
            download_error=None,
            on_disk_bytes=plan.on_disk_bytes,
        )
    if plan.already_downloaded:
        return DownloadExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_download=False,
            download_ok=True,
            download_error=None,
            on_disk_bytes=plan.on_disk_bytes,
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return DownloadExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_download=False,
            download_ok=False,
            download_error=(
                f"huggingface_hub not installed: {exc!r}. Install the "
                f"pip extras for this extractor first."
            ),
            on_disk_bytes=plan.on_disk_bytes,
        )

    def _do_download() -> None:
        snapshot_download(
            repo_id=plan.provenance.model_name,
            revision=plan.provenance.pinned_revision,
        )

    try:
        await asyncio.to_thread(_do_download)
    except Exception as exc:
        logger.warning(
            "download_extractor_weights_execute (%s) failed: %r",
            plan.extractor_name,
            exc,
        )
        return DownloadExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_download=True,
            download_ok=False,
            download_error=repr(exc),
            on_disk_bytes=_weights_size_bytes(plan.provenance),
        )
    return DownloadExtractorWeightsResult(
        extractor_name=plan.extractor_name,
        did_download=True,
        download_ok=True,
        download_error=None,
        on_disk_bytes=_weights_size_bytes(plan.provenance),
    )


# ---- delete_extractor_weights ----


@dataclass(frozen=True)
class DeleteExtractorWeightsPlan:
    """Plan for wiping an extractor's HF-cached weights.

    `weights_dir` is None for bundled / no-weights extractors — the
    op is a no-op in that case. `is_downloaded` distinguishes "cache
    dir exists but wrong revision" (partial download) from "nothing
    to delete" — both currently treat delete as a no-op success; the
    partial case is left to a future recovery op.
    """

    extractor_name: str
    display_name: str
    provenance: ModelProvenance | None
    is_downloaded: bool
    on_disk_bytes: int


@dataclass(frozen=True)
class DeleteExtractorWeightsResult:
    """Outcome of `delete_extractor_weights_execute`.

    `freed_bytes` is what was on disk BEFORE the wipe (equals the
    plan's `on_disk_bytes` when the delete succeeds; 0 when nothing
    was there). `delete_error` carries the shutil.rmtree exception
    message when `delete_ok=False`.
    """

    extractor_name: str
    did_delete: bool
    delete_ok: bool
    delete_error: str | None
    freed_bytes: int


def delete_extractor_weights_plan(
    extractor_name: str,
) -> DeleteExtractorWeightsPlan:
    """Build a plan for deleting the named extractor's weights."""
    entry = _registry_entry(extractor_name)
    provenance = entry.get("provenance")
    on_disk = _weights_size_bytes(provenance)
    return DeleteExtractorWeightsPlan(
        extractor_name=extractor_name,
        display_name=entry["display_name"],
        provenance=provenance,
        is_downloaded=(provenance is not None and _is_weights_downloaded(provenance)),
        on_disk_bytes=on_disk,
    )


async def delete_extractor_weights_execute(
    plan: DeleteExtractorWeightsPlan,
) -> DeleteExtractorWeightsResult:
    """Remove the HF cache dir for this extractor's model.

    Bundled / no-weights extractors short-circuit as no-op successes.
    Delete runs in a worker thread — rmtree of a multi-GB dir can
    take seconds and blocks disk IO.
    """
    if plan.provenance is None:
        return DeleteExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_delete=False,
            delete_ok=True,
            delete_error=None,
            freed_bytes=0,
        )
    cache_dir = _weights_cache_dir(plan.provenance)
    if cache_dir is None or not cache_dir.exists():
        return DeleteExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_delete=False,
            delete_ok=True,
            delete_error=None,
            freed_bytes=0,
        )

    import shutil

    def _do_delete() -> None:
        shutil.rmtree(cache_dir)

    freed = plan.on_disk_bytes
    try:
        await asyncio.to_thread(_do_delete)
    except Exception as exc:
        logger.warning(
            "delete_extractor_weights_execute (%s) failed: %r",
            plan.extractor_name,
            exc,
        )
        return DeleteExtractorWeightsResult(
            extractor_name=plan.extractor_name,
            did_delete=True,
            delete_ok=False,
            delete_error=repr(exc),
            freed_bytes=0,
        )
    return DeleteExtractorWeightsResult(
        extractor_name=plan.extractor_name,
        did_delete=True,
        delete_ok=True,
        delete_error=None,
        freed_bytes=freed,
    )
