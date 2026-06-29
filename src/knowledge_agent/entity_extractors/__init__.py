"""Dispatch table for entity-extractor adapters (L6a).

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

from knowledge_agent.entity_extractors.base import Mention

# Source of truth for known adapter names. Used in error messages so the
# user sees the valid set when they typo. Update alongside the
# `get_extractor` branches below.
_ADAPTER_NAMES: tuple[str, ...] = (
    "llm", "gliner", "gliner_biomed", "hunflair2",
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
    raise ValueError(
        f"Unknown entity extractor: {name!r}. "
        f"Known adapters: {_ADAPTER_NAMES}."
    )


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


__all__ = [
    "Mention",
    "get_extractor",
    "get_known_labels",
    "validate_entity_types",
]
