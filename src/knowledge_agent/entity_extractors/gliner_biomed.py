"""GLiNER-BioMed zero-shot biomedical entity extractor (L6a, open vocabulary).

KNOWN_LABELS = None: same shape as the general `gliner` adapter — this
is also a zero-shot model, no closed label set. Whatever `entity_types`
the corpus.toml provides flow into the inference call directly. Falls
back to `DEFAULT_LABELS` (biomedical categories) when corpus provides
none.

Model: `Ihor/gliner-biomed-bi-large-v1.0` — bi-encoder GLiNER variant
domain-adapted on synthetic biomedical annotations distilled from large
generative biomedical LLMs. Published in Bioinformatics (Oxford
Academic) 2025 by Ihor Stepanov in collaboration with DS4DH (University
of Geneva). Apache-2.0.

Why bi-large of the GLiNER-BioMed family (locked 2026-06-23):
  - Bi-encoder paradigm best on BC5CDR + CHIA + NCBI Disease (the
    standard biomedical NER benchmarks).
  - Smaller (~1.1 GB) than the cross-encoder large variant (~1.5 GB).
  - Same gliner library as the general `gliner` adapter — single
    library family to maintain.

KNOWN SECURITY DISCLOSURE — surfaced by the install dialog before any
download:
  - safetensors = False (no .safetensors file in this checkpoint; only
    `pytorch_model.bin`). Loading uses pickle deserialisation — the
    known PyTorch attack vector. The install dialog flags this
    prominently. Provenance (academic publisher + peer-reviewed
    Bioinformatics paper + Apache-2.0 + pinned SHA) is solid; the user
    opts in with full disclosure.
  - trust_remote_code = False (gliner library is plain Python on PyPI;
    no custom code executed from the HF repo).
  - Pinned to a specific commit SHA — never `main`.

Lazy import: `from gliner import GLiNER` only fires inside `_get_model()`
at first extract() call. Module import is dependency-free.

Confidence: GLiNER exposes a per-mention score; we surface it on
`Mention.confidence`. Offsets come from GLiNER's `start` field. Same
shape as the general `gliner` adapter.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from knowledge_agent.entity_extractors.base import Mention

# Pinned model identity. Bump pinned_revision only after manually
# re-verifying license, file format, and trust_remote_code requirement
# on the HF model page. The provenance entry in
# extractor_lifecycle.EXTRACTOR_REGISTRY mirrors these values so the
# install dialog and the actual loader stay in sync.
MODEL_NAME = "Ihor/gliner-biomed-bi-large-v1.0"

MODEL_REVISION = "75fb10d6d5500c5e98c285493578116540ec47d3"
"""HuggingFace commit SHA pinned 2026-06-23. Never load `main` —
that auto-updates and bypasses our verification."""

# Open vocabulary - same shape as the general gliner adapter.
KNOWN_LABELS: tuple[str, ...] | None = None

# Defaults used when corpus.toml provides no `entity_types`. Covers
# the biomedical NER labels most commonly used in research papers
# (drawn from BC5CDR / JNLPBA / NCBI Disease label conventions).
# Override per corpus by setting `entity_types` in corpus.toml.
DEFAULT_LABELS: tuple[str, ...] = (
    "DISEASE",
    "CHEMICAL",
    "GENE",
    "PROTEIN",
    "SPECIES",
    "CELL_LINE",
    "CELL_TYPE",
    "ANATOMY",
)

# Threshold for accepting a mention. Same default as the general
# gliner adapter; GLiNER returns scores in [0, 1].
_SCORE_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _get_model():
    """Load the GLiNER-BioMed model once per process.

    Heavy: ~1.1 GB resident (pytorch_model.bin), ~10-20s first-call
    load + (optional) one-time HF download. Cached for the rest of
    the process lifetime via lru_cache. The `from gliner import GLiNER`
    lives inside the function (not at module top-level) so the
    dispatcher's `get_known_labels("gliner_biomed")` can answer without
    paying the import cost.

    Pinned revision: passes `revision=MODEL_REVISION` so the loader
    always pulls the exact SHA we vetted, regardless of upstream
    branch movement. The gliner library forwards this kwarg to the
    HuggingFace download.

    Pickle format: this checkpoint has no .safetensors file — loading
    inevitably uses pickle deserialisation. Install dialog warns the
    user before download.
    """
    from gliner import GLiNER

    return GLiNER.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)


def _run_inference(text: str, entity_types: list[str]) -> list[Mention]:
    """Run GLiNER-BioMed inference synchronously. Shared by `extract`
    (sync direct call) and `aextract` (async, wrapped in
    `asyncio.to_thread`).

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


def extract(text: str, entity_types: list[str]) -> list[Mention]:
    """Extract biomedical entity mentions from `text` using GLiNER-BioMed (sync).

    Args:
      text:         the chunk's raw text content.
      entity_types: empty list -> use DEFAULT_LABELS (biomedical
                    categories). Non-empty list -> GLiNER predicts
                    against those labels verbatim.

    Returns:
      List of `Mention` with `offset` set from GLiNER's `start` field
      and `confidence` set from GLiNER's `score`. Empty list when
      GLiNER finds no matching entities above `_SCORE_THRESHOLD`.

    Sync path retained for callers that haven't migrated to async yet.
    Event-loop callers use `aextract` instead.
    """
    return _run_inference(text, entity_types)


async def aextract(text: str, entity_types: list[str]) -> list[Mention]:
    """Async sibling of `extract`. Offloads inference to a worker thread.

    GLiNER inference is CPU/GPU bound; the PyTorch forward pass
    releases the GIL while running, so `asyncio.to_thread` lets the
    event loop progress concurrent tasks while the model runs.
    """
    return await asyncio.to_thread(_run_inference, text, entity_types)
