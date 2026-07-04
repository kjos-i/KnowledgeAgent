"""Dispatch table for entity-extractor adapters (L6).

Each backend (LLM, SciSpaCy, future general spaCy / BERT NER / etc.)
lives in its own module so adding a new one is mechanical and does
not load existing adapters' dependencies. Importing this package
alone has zero ML / SDK side effects - each adapter's heavy imports
fire only inside `get_extractor(name)`.

Public API:
  - `get_extractor(name) -> module`
      Returns the adapter module. Caller invokes `module.extract(text,
      entity_types) -> list[Mention]`. Raises ValueError on unknown name.
  - `get_known_labels(name) -> tuple[str, ...] | None`
      Surfaces the adapter's label set. None = open vocabulary (LLM).
      Used by `validate_entity_types` and by the future GUI's entity-type
      dropdown.
  - `validate_entity_types(name, entity_types) -> None`
      Cross-checks the user's entity_types list against the adapter's
      KNOWN_LABELS. No-op for open-vocabulary adapters. Raises ValueError
      with the offending labels + the valid set on mismatch.

Adding a new adapter:
  1. New file `entity_extractors/<name>.py` exporting
     `KNOWN_LABELS: tuple[str, ...] | None` and
     `extract(text: str, entity_types: list[str]) -> list[Mention]`.
  2. Add `<name>` to `_ADAPTER_NAMES` below + one branch in
     `get_extractor`.
  3. Optional dep in `pyproject.toml` under
     `[project.optional-dependencies]` keyed `entities-<name>`.
  4. Tests in `tests/unit/test_entity_extractors_<name>.py` matching the
     existing adapter test pattern (mock the backend; assert prompt /
     filter behaviour against the Mention output).
"""

import asyncio
import re
from dataclasses import replace

from knowledge_agent.entity_extractors.base import Mention

# Whitespace collapse for overlap detection in the priority-ordered
# union. Used ONLY to decide whether two adapters found "the same span"
# - the STORED entity key is still `raw_text.lower()` in the write path,
# so this normalization never changes what lands in Neo4j.
_WS_RE = re.compile(r"\s+")

# Source of truth for known adapter names. Used in error messages so the
# user sees the valid set when they typo. Update alongside the
# `get_extractor` branches below.
_ADAPTER_NAMES: tuple[str, ...] = (
    "llm",
    "gliner",
    "gliner_biomed",
    "hunflair2",
)


def get_extractor(name: str):
    """Return the adapter module for `name`.

    Lazy imports so the chosen adapter is the only one whose
    dependencies load. Caller treats the returned module as having
    `KNOWN_LABELS` and `extract(text, entity_types)`.
    """
    if name == "llm":
        from knowledge_agent.entity_extractors import llm

        return llm
    if name == "gliner":
        from knowledge_agent.entity_extractors import gliner

        return gliner
    if name == "gliner_biomed":
        from knowledge_agent.entity_extractors import gliner_biomed

        return gliner_biomed
    if name == "hunflair2":
        from knowledge_agent.entity_extractors import hunflair2

        return hunflair2
    raise ValueError(f"Unknown entity extractor: {name!r}. Known adapters: {_ADAPTER_NAMES}.")


def get_known_labels(name: str) -> tuple[str, ...] | None:
    """Adapter's KNOWN_LABELS constant. None means open vocabulary."""
    return get_extractor(name).KNOWN_LABELS


def validate_entity_types(name: str, entity_types: list[str]) -> None:
    """Cross-check `entity_types` against the adapter's KNOWN_LABELS.

    No-op when the adapter declares open vocabulary (KNOWN_LABELS is
    None). For closed-vocabulary adapters, raises ValueError listing
    unknown labels + the valid set so the user can fix the typo.

    Called by the pipeline before any extraction so a misconfigured
    corpus fails fast, not after partial writes.
    """
    known = get_known_labels(name)
    if known is None:
        return
    unknown = [t for t in entity_types if t not in known]
    if unknown:
        raise ValueError(
            f"entity_types contains labels unknown to extractor {name!r}: "
            f"{unknown}. Known labels for {name!r}: {sorted(known)}."
        )


def _normalize_span(text: str) -> str:
    """Normalized form of a mention span for UNION overlap detection.

    Lowercase + collapse internal whitespace + strip. Used only to
    decide "did two adapters find the same span?" - NOT the stored key
    (that stays `raw_text.lower()` in `entity_writes`). Kept
    intentionally conservative so it never merges genuinely different
    spans.
    """
    return _WS_RE.sub(" ", text.strip().lower())


def effective_entity_types(
    name: str,
    entity_types: list[str],
    mode: str,
) -> list[str]:
    """Resolve the entity_types an adapter should receive under the
    corpus's `entity_types_mode`.

    - Empty `entity_types` -> passthrough (the adapter falls back to its
      own default behaviour, whatever that is).
    - `mode == "replace"` (default) -> the user's list, unchanged.
    - `mode == "add"` -> the user's list merged with this adapter's
      `DEFAULT_LABELS` (order-preserving dedup, user labels first). For
      adapters with no `DEFAULT_LABELS` constant (the LLM, HunFlair2)
      this is a no-op so "add" == "replace" for them.
    """
    if not entity_types or mode != "add":
        return list(entity_types)
    defaults = list(getattr(get_extractor(name), "DEFAULT_LABELS", ()) or ())
    return list(dict.fromkeys([*entity_types, *defaults]))


async def extract_union(
    text: str,
    extractor_names: list[str],
    entity_types: list[str],
    *,
    entity_types_mode: str = "replace",
    llm_kwargs: dict | None = None,
) -> list[Mention]:
    """Run several extractors on one chunk and merge by PRIORITY UNION.

    `extractor_names` is ordered: index 0 is the base (owns overlaps);
    each later extractor contributes only spans no higher-priority
    extractor already found (compared via `_normalize_span`). The
    surviving mention keeps the base owner's type; every extractor that
    found the span is recorded on `Mention.sources` for provenance.

    Fragmentation-free by construction: one owner per normalized span,
    so two adapters never write competing `(key, type)` nodes for the
    same text.

    `entity_types_mode` is applied per adapter (see
    `effective_entity_types`). `llm_kwargs` (model / temperature) is
    forwarded ONLY to the "llm" adapter; other adapters ignore those
    kwargs so we don't pass them.

    The N adapters for a chunk run concurrently; the merge is then
    applied deterministically in priority order. A single-element
    `extractor_names` degenerates to "run one extractor" with each
    mention's `sources` set to that one adapter.
    """
    if not extractor_names:
        return []

    async def _run_one(name: str) -> list[Mention]:
        module = get_extractor(name)
        eff_types = effective_entity_types(name, entity_types, entity_types_mode)
        kwargs = dict(llm_kwargs) if (llm_kwargs and name == "llm") else {}
        return await module.extract(text, eff_types, **kwargs)

    results = await asyncio.gather(*(_run_one(n) for n in extractor_names))

    claimed: dict[str, Mention] = {}
    sources: dict[str, list[str]] = {}
    # `results` preserves `extractor_names` order (asyncio.gather does),
    # so iterating here honours priority.
    for name, mentions in zip(extractor_names, results, strict=True):
        for m in mentions:
            span = _normalize_span(m.raw_text)
            if span not in claimed:
                claimed[span] = m
                sources[span] = [name]
            elif name not in sources[span]:
                sources[span].append(name)

    return [replace(m, sources=tuple(sources[span])) for span, m in claimed.items()]


__all__ = [
    "Mention",
    "effective_entity_types",
    "extract_union",
    "get_extractor",
    "get_known_labels",
    "validate_entity_types",
]
