"""GLiNER zero-shot entity extractor (L6, open vocabulary).

KNOWN_LABELS = None: this adapter does NOT constrain the user's
`entity_types` against any closed set. GLiNER is zero-shot — at
inference time it accepts a list of label strings and returns matches
for whichever ones occur in the text. Whatever labels the corpus puts
in `corpus.toml [entities].entity_types` flow into the inference call
directly.

Model: `urchade/gliner_multi-v2.1` (multilingual general-purpose
GLiNER, 209M parameters). Published by Urchade Zaratiana (Sorbonne,
GLiNER paper author) under Apache-2.0.

Why GLiNER on the menu: this is our ONLY local-deterministic zero-shot
NER. Useful for fields where we have no domain specialist (math,
physics, humanities, computer science, environment, etc.) — without it
the only fallback for those fields is the LLM, which costs API tokens
and is non-deterministic. GLiNER covers them all in one model with
sensible accuracy.

Default labels semantics (empty `entity_types`):
  - GLiNER MUST be called with a non-empty `labels` argument; the
    library has no built-in default. Our adapter defaults to
    `DEFAULT_LABELS` (PERSON / ORGANIZATION / LOCATION / EVENT / DATE
    / MISC) when the user provides no labels — sensible general-NER
    coverage without forcing every corpus to enumerate them. Non-empty
    `entity_types` is passed through verbatim.

Security disclosure (locked in ModelProvenance — surfaced by the
install dialog before any download):
  - safetensors only — no pickle execution risk.
  - trust_remote_code = False — the gliner library is plain Python
    on PyPI; we never execute custom code from the HF repo.
  - Pinned to a specific commit SHA — never `main` (which auto-updates
    without our re-verification).

Lazy import: `from gliner import GLiNER` only fires inside
`_get_model()` at first extract() call. Module import is dependency-
free so the dispatcher can read KNOWN_LABELS / pass tests without
PyTorch being importable.

Confidence: GLiNER exposes a per-mention score; we surface it on
`Mention.confidence`. Offsets come from GLiNER's `start` field.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from knowledge_agent.entity_extractors.base import Mention

# Pinned model identity. Bump pinned_revision only after manually
# re-verifying license, safetensors availability, and
# trust_remote_code requirement on the HF model page. The provenance
# entry in extractor_lifecycle.EXTRACTOR_REGISTRY mirrors these values
# so the install dialog and the actual loader stay in sync.
MODEL_NAME = "urchade/gliner_multi-v2.1"

MODEL_REVISION = "443d26d654e0324125a96bebd8e796c14ff2efe6"
"""HuggingFace commit SHA pinned 2026-06-23. Never load `main` —
that auto-updates and bypasses our verification."""

# Open vocabulary - the user's entity_types flow through unvalidated.
# GLiNER is zero-shot so it accepts arbitrary label strings.
KNOWN_LABELS: tuple[str, ...] | None = None

# Defaults used when corpus.toml provides no `entity_types`. Chosen
# to cover the common general-NER categories without forcing every
# corpus to write them out. Override per corpus by setting
# `entity_types` in corpus.toml.
DEFAULT_LABELS: tuple[str, ...] = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "DATE",
    "MISC",
)

# Threshold for accepting a mention. GLiNER returns scores in [0, 1];
# below ~0.4 the predictions get noisy. The library's default is also
# 0.5; we surface ours as a constant so a future tuning knob has a
# single home.
_SCORE_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _get_model():
    """Load the GLiNER model once per process.

    Heavy: ~1.1 GB resident (model.safetensors), ~5-15s first-call
    load. Cached for the rest of the process lifetime via lru_cache.
    The `from gliner import GLiNER` lives inside the function (not at
    module top-level) so the dispatcher's `get_known_labels("gliner")`
    can answer without paying the import cost.

    NO auto-download: raises `WeightsNotDownloadedError` if the pinned
    revision isn't on disk (2026-07-03 rule — weights are an explicit
    user action via Library → Installs). This keeps ingest runs
    predictable: no surprise ~1 GB HF fetch mid-corpus.

    Pinned revision: passes `revision=MODEL_REVISION` so the loader
    always pulls the exact SHA we vetted, regardless of upstream
    branch movement. The gliner library forwards this kwarg to the
    HuggingFace loader.
    """
    from gliner import GLiNER

    from knowledge_agent.entity_extractors.extractor_lifecycle import (
        WeightsNotDownloadedError,
        _GLINER_PROVENANCE,
        _is_weights_downloaded,
    )

    if not _is_weights_downloaded(_GLINER_PROVENANCE):
        raise WeightsNotDownloadedError(
            "gliner", MODEL_NAME,
            "Open Library → Installs and press Download weights on "
            "the GLiNER row before running extraction.",
        )
    return GLiNER.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def _run_inference(text: str, entity_types: list[str]) -> list[Mention]:
    """Run GLiNER inference synchronously. Shared by `extract` (sync
    direct call) and `aextract` (async, wrapped in `asyncio.to_thread`).

    Args + return shape identical to the public `extract` contract.
    """
    model = _get_model()
    labels = list(entity_types) if entity_types else list(DEFAULT_LABELS)

    predictions = model.predict_entities(
        text, labels, threshold=_SCORE_THRESHOLD,
    )

    return [
        Mention(
            raw_text=p["text"],
            entity_type=p["label"],
            offset=p["start"],
            confidence=p["score"],
        )
        for p in predictions
    ]


async def extract(text: str, entity_types: list[str]) -> list[Mention]:
    """Extract entity mentions from `text` using GLiNER zero-shot.

    Args:
      text:         the chunk's raw text content.
      entity_types: empty list -> use DEFAULT_LABELS (general-NER
                    categories). Non-empty list -> GLiNER predicts
                    against those labels verbatim.

    Returns:
      List of `Mention` with `offset` set from GLiNER's `start` field
      and `confidence` set from GLiNER's `score`. Empty list when
      GLiNER finds no matching entities above `_SCORE_THRESHOLD`.

    Offloads inference to a worker thread via `asyncio.to_thread`.
    GLiNER inference is CPU/GPU bound; the PyTorch forward pass
    releases the GIL while running, so the worker thread lets the
    event loop progress concurrent tasks (other chunks embedding,
    KG writes, agent LLM calls) while the model runs.
    """
    return await asyncio.to_thread(_run_inference, text, entity_types)
