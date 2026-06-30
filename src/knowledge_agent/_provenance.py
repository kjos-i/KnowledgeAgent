"""Shared provenance / security warning helpers.

`security_warning_text(safetensors, trust_remote_code)` returns the
prominent warning string that install + download dialogs append to
their summaries when a model has known elevated risk. One source of
truth — extractor + embedder lifecycle both import it so the wording
stays consistent + a future audit only has to find one place.

Two concrete risks surfaced:

  - **Pickle execution.** When `safetensors=False` the model loads from
    `pytorch_model.bin` (or sibling `.bin` shards), which is Python's
    `pickle` format. Loading runs arbitrary code from the file. A
    tampered model on HuggingFace would execute that code on the
    user's machine. The mitigation is the pinned commit SHA — the
    adapter loads ONLY that revision — but the user should still know
    pickle is in play.

  - **`trust_remote_code=True`.** Some models require executing custom
    Python from the HF repo (typically `modeling_*.py` shipped
    alongside the weights) just to instantiate the model class. This
    is strictly more dangerous than pickle because it runs at import
    time, before any inference happens. Adapter source files MUST
    document which commit's code was audited when this flag is True.

The strings live here so the dialog wording is consistent across the
GUI, CLI install summary, and smoke tests. The 0d security review
(2026-06-30) extracted them after observing that both lifecycle
modules previously printed only the raw flag value (e.g.
`safetensors=False`) without explaining why the user should care.
"""
from __future__ import annotations


def security_warning_text(
    *,
    safetensors: bool,
    trust_remote_code: bool = False,
) -> str:
    """Return a prominent warning string for elevated-risk models.

    Empty string when both flags are safe (`safetensors=True,
    trust_remote_code=False`). Non-empty otherwise — the dialog
    appends it after the rest of the summary so the user sees the
    risk before clicking install/download.

    Format keeps the warnings on separate sentences, each starting
    with a stark word ("SECURITY:" prefix), so they read distinctly
    from the surrounding factual provenance lines.
    """
    parts: list[str] = []
    if not safetensors:
        parts.append(
            "SECURITY: this model ships pickle-format weights "
            "(`pytorch_model.bin`, no safetensors). Loading executes "
            "arbitrary code from the file. The pinned commit SHA "
            "mitigates tampering risk, but only as long as the SHA "
            "itself is verified."
        )
    if trust_remote_code:
        parts.append(
            "SECURITY: this model requires `trust_remote_code=True`, "
            "meaning custom Python from the HF repo runs at load time. "
            "Audit the adapter's source for the commit it pins before "
            "installing."
        )
    if not parts:
        return ""
    return " " + " ".join(parts)
