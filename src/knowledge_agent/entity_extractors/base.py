"""Shared types + contract for entity-extractor adapters.

Every adapter under `entity_extractors/` exposes the same shape on the
outside even though the bodies differ wildly (LLM SDK call, spaCy
pipeline, BERT inference, ...). Concretely each adapter module exports:

  - `KNOWN_LABELS: tuple[str, ...] | None`
      * `None`           = open vocabulary. The LLM-style adapters set this;
                           `entity_types` in `corpus.toml` flows into the
                           prompt as guidance but is NOT cross-checked
                           against any closed set.
      * `tuple[str, ...]` = closed label set. NER-style adapters declare the
                           labels their trained model can emit. The
                           dispatcher's `validate_entity_types` cross-checks
                           the user's `entity_types` against this tuple and
                           raises on unknown labels.
  - `extract(text: str, entity_types: list[str]) -> list[Mention]`
      A pure function. Text in, mentions out. NO database side effects, NO
      I/O outside the model call itself.

`entity_types` semantics across adapters:
  - Non-empty list: closed mode. LLM adapter constrains its output to those
                    types via the prompt; NER adapter filters its native
                    output down to those types.
  - Empty list:     adapter default. LLM picks the categories itself
                    (unpredictable vocabulary across docs); NER returns
                    every label its training emits.

Adapter responsibilities:
  - The extractor preserves the ORIGINAL spelling of each span (`raw_text`).
    Lowercasing happens in the write path (kg/entity_writes.py) when
    computing the `:Entity` node's `key`. Splitting the responsibility this
    way means downstream consumers can still recover original spelling for
    NER mentions (via the offset + chunk text in LanceDB).
  - The extractor populates `offset` and `confidence` when the underlying
    backend provides them for free. For LLMs they stay `None` (LLMs
    hallucinate offsets; no native per-mention confidence). For NER they
    typically get the char-start and (when available) the model's score.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mention:
    """One extracted entity mention from a chunk's text.

    Frozen so adapter output is immutable - the same Mention object can be
    safely reused across batching / write paths without aliasing surprises.
    """

    raw_text: str
    """Verbatim span as it appears in the source chunk text. NOT
    lowercased - that happens in the write path when computing the
    :Entity node's `key` for the MERGE."""

    entity_type: str
    """Category label. For closed-vocabulary adapters this is one of the
    adapter's KNOWN_LABELS; for open-vocabulary adapters it's free-form
    (LLM-chosen, typically uppercase snake-case)."""

    offset: int | None = None
    """Char position of the mention's start in the chunk text. NER
    adapters set this from the spaCy / model token offsets. LLM
    adapters leave it None (we intentionally do NOT ask the LLM for
    offsets - hallucination prone)."""

    confidence: float | None = None
    """Extractor-reported confidence in [0, 1]. NER adapters set this
    when the underlying model exposes a score; spaCy NER pipelines
    typically do NOT expose per-entity confidence via doc.ents, so
    even NER adapters may legitimately set this to None. LLM adapters
    leave it None (no native confidence signal)."""
