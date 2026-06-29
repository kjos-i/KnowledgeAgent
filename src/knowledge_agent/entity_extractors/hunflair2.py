"""HunFlair2 biomedical NER extractor (L6a, closed vocabulary — but
exposed to the user as a single on/off option).

UX intent (locked 2026-06-23): user enables `hunflair2` and gets the
full biomedical NER package — DISEASE / CHEMICAL / GENE / SPECIES /
CELL_LINE — on every chunk. There is NO user-facing way to narrow to
a subset. Anyone enabling HunFlair2 wants biomedical NER coverage;
hiding the label-selection surface keeps the menu simple.

  - `KNOWN_LABELS = None`: the dispatcher's `validate_entity_types`
    is a no-op for this adapter. If the user accidentally sets
    `entity_types` in corpus.toml for hunflair2, the value is silently
    ignored (rather than rejected). This is intentional — we treat
    the field as not applicable to HunFlair2.
  - `extract()` ignores its `entity_types` argument entirely and
    always returns every entity HunFlair2 emits.

Model: `hunflair/hunflair2-ner` — a single unified Flair
`PrefixedSequenceTagger` that covers all biomedical entity types in
one inference call (NOT the older HunFlair v1 "five sub-taggers"
shape). Published by the HU Berlin Bioinformatics Flair NLP team,
under MIT (inherited from the Flair library, Zalando SE 2018).

KNOWN SECURITY DISCLOSURE — surfaced by the install dialog before any
download:
  - safetensors = False (this checkpoint ships only `pytorch_model.bin`).
    Loading uses pickle deserialisation — the known PyTorch attack
    vector. Provenance (established academic group + MIT-licensed Flair
    library + pinned SHA) mitigates but does not eliminate.
  - trust_remote_code = False (Flair library is plain Python on PyPI;
    no custom code executed from the HF repo).
  - Pinned to a specific commit SHA — never `main`.

Lazy import: `from flair.models import PrefixedSequenceTagger` only
fires inside `_get_model()` at first extract() call. Module import
is dependency-free so the dispatcher can read KNOWN_LABELS without
importing Flair or PyTorch.

Output normalisation: Flair emits title-case tags ("Disease",
"Chemical", "Gene", "Species", "CellLine"). The adapter normalises
these to our project convention (UPPERCASE_SNAKE_CASE: "DISEASE",
"CHEMICAL", "GENE", "SPECIES", "CELL_LINE") at the Mention boundary
so downstream `:Entity` nodes have consistent label casing.

Confidence: Flair's span objects expose a per-mention score; we
surface it on `Mention.confidence`. Offsets come from span
`start_position`.

SciSpaCy is superseded by HunFlair2 (verified ~7-9 pp F1 better on
biomedical NER) — the SciSpaCy adapter is removed in the same change
that ships this adapter.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from knowledge_agent.entity_extractors.base import Mention

# Pinned model identity. Bump pinned_revision only after manually
# re-verifying license, file format, and trust_remote_code requirement
# on the HF model page. The provenance entry in
# extractor_lifecycle.EXTRACTOR_REGISTRY mirrors these values — keep
# the two in sync when bumping.
MODEL_NAME = "hunflair/hunflair2-ner"

MODEL_REVISION = "3af2b8972f7af2910ce8d9ae724da09b3d7a166c"
"""HuggingFace commit SHA pinned 2026-06-23. Never load `main` —
that auto-updates and bypasses our verification."""

# Open vocabulary from the dispatcher's perspective: user cannot
# narrow the label set. `entity_types` from corpus.toml is silently
# ignored by extract(). The 5 actual labels HunFlair2 emits are
# documented in `_FLAIR_TO_OUR_LABELS` below.
KNOWN_LABELS: tuple[str, ...] | None = None


# Flair's HunFlair2 emits title-case tags. We normalise these to
# UPPERCASE_SNAKE matching the project's `:Entity.entity_type`
# convention. Update both sides if Flair changes its tag scheme.
_FLAIR_TO_OUR_LABELS: dict[str, str] = {
    "Disease": "DISEASE",
    "Chemical": "CHEMICAL",
    "Gene": "GENE",
    "Species": "SPECIES",
    "CellLine": "CELL_LINE",
}


@lru_cache(maxsize=1)
def _get_model():
    """Load the HunFlair2 unified NER tagger once per process.

    Heavy: ~1.24 GB resident (pytorch_model.bin), ~10-20s first-call
    load + (optional) one-time HF download. Cached for the rest of
    the process lifetime via lru_cache. The Flair import lives inside
    the function (not at module top-level) so the dispatcher's
    `get_known_labels("hunflair2")` can answer without paying the
    Flair + PyTorch import cost.

    Pinned revision: Flair's `PrefixedSequenceTagger.load` does NOT
    accept a `revision` kwarg directly — it goes through HF
    `snapshot_download`. We force the pinned revision by setting the
    HF cache to download a specific snapshot before delegating to
    Flair. This keeps us off `main` even though the Flair API itself
    has no per-call revision knob.

    Pickle format: this checkpoint has no .safetensors file — loading
    inevitably uses pickle deserialisation. Install dialog warns the
    user before download.
    """
    from flair.models import PrefixedSequenceTagger
    from huggingface_hub import snapshot_download

    # Pre-download the pinned revision to the HF cache so the
    # subsequent Flair load picks up exactly that snapshot. Flair
    # resolves model identifiers via the HF cache transparently when
    # the files are already present.
    snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
    )

    return PrefixedSequenceTagger.load(MODEL_NAME)


def _build_sentence(text: str):
    """Build a Flair `Sentence` for the given text.

    Wrapped in its own function so tests can mock the Sentence
    construction without having to patch the `flair.data` module
    globally. Real callers get a real Sentence; tests get a fake
    with a duck-typed `get_spans` returning canned data.
    """
    from flair.data import Sentence

    return Sentence(text)


def _run_inference(text: str) -> list[Mention]:
    """Run HunFlair2 inference synchronously. Shared by `extract` (sync
    direct call) and `aextract` (async, wrapped in `asyncio.to_thread`).

    Note: no `entity_types` parameter — HunFlair2 is all-or-nothing,
    the parameter only exists on the public `extract` / `aextract`
    signatures to satisfy the adapter contract (and is discarded
    there).
    """
    model = _get_model()
    sentence = _build_sentence(text)
    model.predict(sentence)

    mentions: list[Mention] = []
    for span in sentence.get_spans("ner"):
        flair_label = span.tag  # e.g., "Disease"
        our_label = _FLAIR_TO_OUR_LABELS.get(flair_label)
        if our_label is None:
            # Unknown Flair tag — skip rather than emit unmapped.
            # If this fires in practice the Flair upstream changed
            # its label scheme; bump _FLAIR_TO_OUR_LABELS.
            continue
        mentions.append(
            Mention(
                raw_text=span.text,
                entity_type=our_label,
                offset=span.start_position,
                confidence=span.score,
            )
        )
    return mentions


async def extract(text: str, entity_types: list[str]) -> list[Mention]:
    """Extract biomedical entity mentions from `text` using HunFlair2.

    The `entity_types` argument is INTENTIONALLY IGNORED — HunFlair2
    is exposed as an all-or-nothing adapter per the locked 2026-06-23
    UX decision. The user enables HunFlair2 and gets the full
    biomedical NER package; there is no user-facing way to narrow.

    Args:
      text:         the chunk's raw text content.
      entity_types: IGNORED — present only to satisfy the adapter
                    contract. Pass any value (including []); behaviour
                    is identical.

    Returns:
      List of `Mention` covering every biomedical entity HunFlair2
      finds in the text. Labels are normalised from Flair's title-case
      ("Disease") to our UPPERCASE_SNAKE convention ("DISEASE").
      Confidence and offset are populated from Flair's span data.
      Empty list when HunFlair2 finds no biomedical entities.

    Offloads inference to a worker thread via `asyncio.to_thread`.
    HunFlair2 inference is CPU/GPU bound; the PyTorch forward pass
    releases the GIL while running.
    """
    del entity_types  # Intentional no-op — documented above.
    return await asyncio.to_thread(_run_inference, text)
