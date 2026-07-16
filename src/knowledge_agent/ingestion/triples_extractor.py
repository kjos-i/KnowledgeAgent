"""LLM-based triple extractor for L8 (typed entity-to-entity relations).

Per-chunk LLM call. Reads chunk text + the L6 entity
vocabulary mined from this chunk, and emits triples constrained to
that vocabulary and to the 15 fixed predicates in
`schema.TRIPLE_PREDICATE_RELS`.

Key design (see [[researchliteratureagent-l8-l9-specs]] for the locked
design rationale):

  - **Constrained subject + object vocabulary.** The chunk's L6
    entity keys are passed into the prompt; the LLM is told to use ONLY
    those keys. This keeps L6 as the single source of truth for
    "what entities exist" and prevents L8 from silently introducing
    new entities.

  - **Constrained predicate vocabulary.** The 15 predicates are
    enumerated in the prompt; the LLM must pick one or emit nothing.
    Open-ended predicates were rejected because they fragment the
    graph ("EDITS"/"MODIFIES"/"CHANGES" become three queries instead
    of one).

  - **Evidence span** carries a short verbatim snippet from the chunk
    that supports the assertion - lets the agent show citations like
    "asserted at: '...CRISPR was used to edit BRCA1...'".

  - **Open vocabulary entity types are fine.** Subject/object types
    flow through from the L6 entity vocabulary unchanged - whatever
    L6 returned (GENE, DISEASE, PERSON, ...) is what the LLM uses.

Cost: one call per chunk. Input ~1000-1500 tokens (system prompt +
chunk + entity vocab), output small — per-chunk cost tracks the
configured provider/model's input price. Same order as L6.

Defensive policies:
  - Triples referencing keys not in the chunk's L6 vocab are dropped
    + logged (LLM hallucinated an entity).
  - Triples with predicates outside the 15 are dropped + logged
    (LLM ignored the constraint).
  - The whole call is wrapped at the pipeline level in try/except;
    a chunk that crashes the LLM doesn't poison the whole doc.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from knowledge_agent.kg.schema import TRIPLE_PREDICATE_RELS
from knowledge_agent.kg.triples_writes import ExtractedTriple
from knowledge_agent.llm_factory import get_llm_ref as _get_llm
from knowledge_agent.llm_factory import with_retry as _with_retry

# ---------------------------------------------------------------------------
# Structured-output schema. Pydantic models are LLM-facing only; the public
# ExtractedTriple dataclass in triples_writes.py is what the rest of the
# codebase consumes.
# ---------------------------------------------------------------------------


class _LLMTriple(BaseModel):
    """One triple as returned by the LLM's structured output.

    Keys flow as lowercase already (matches the L6 vocabulary the
    prompt hands the model). Predicate is a free string at the schema
    level so structured-output enforcement stays simple; we validate
    it post-hoc against TRIPLE_PREDICATE_RELS.
    """

    subject_key: str = Field(
        description=(
            "Subject entity key (lowercased). MUST be one of the keys "
            "from the supplied 'available entities' list - do not "
            "invent new entity names."
        )
    )
    predicate: str = Field(
        description=("Relation type. MUST be one of the allowed predicates (SCREAMING_SNAKE_CASE).")
    )
    object_key: str = Field(
        description=(
            "Object entity key (lowercased). MUST be one of the keys "
            "from the supplied 'available entities' list."
        )
    )
    evidence_span: str = Field(
        description=(
            "Short verbatim snippet from the chunk text (one sentence "
            "or clause) that supports the triple. Do not paraphrase."
        )
    )


class _LLMTriples(BaseModel):
    """Top-level wrapper for the structured-output call."""

    triples: list[_LLMTriple] = Field(
        default_factory=list,
        description=(
            "All relation triples found in the text. Empty list when "
            "no supported relations between supplied entities are "
            "present."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt.
# ---------------------------------------------------------------------------


_PREDICATE_LIST = ", ".join(TRIPLE_PREDICATE_RELS)


_SYSTEM_PROMPT_TEMPLATE = """\
You extract typed entity-to-entity relations from text.

Available entities in this chunk (use these and ONLY these as subject
and object - do not invent new entity names):

{entity_vocab}

Allowed predicates (use one of these as the relation type - do not
invent new predicates):

{predicate_list}

For each relation explicitly stated in the chunk text, return:
  - subject_key   : one of the available entity keys (lowercased)
  - predicate     : one of the allowed predicates
  - object_key    : one of the available entity keys (lowercased)
  - evidence_span : a short verbatim snippet from the chunk that
                    supports this relation (one sentence or clause)

Rules:
  - Only emit a triple when the chunk text DIRECTLY states the
    relation. Do not infer from outside knowledge.
  - Both subject_key and object_key MUST come from the available
    entities list above. If a relation involves an entity not in the
    list, skip it (do not invent a key).
  - The predicate MUST be one of the allowed list. If none of the
    allowed predicates fits, skip the relation (do not invent a
    predicate).
  - evidence_span is verbatim from the chunk - do not paraphrase or
    summarise.
  - Direction matters: (a INHIBITS b) is different from (b INHIBITS a).
  - If no qualifying relations are present, return an empty list.
"""


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------


# _get_llm is imported from llm_factory above — provider-agnostic
# dispatch shared across all LLM call sites.


def _build_system_prompt(entity_vocab: list[tuple[str, str]]) -> str:
    """Render the prompt template with this chunk's entity vocabulary.

    `entity_vocab` is `[(key, entity_type), ...]` from the L6 mentions
    found in this chunk. Rendered as one bullet per entity so the LLM
    sees both the key and its type.
    """
    if not entity_vocab:
        # Defensive - the pipeline guards against this too, but if it
        # slips through the prompt should still parse cleanly.
        rendered = "  (none)"
    else:
        rendered = "\n".join(
            f"  - {key} (type: {entity_type})" for key, entity_type in entity_vocab
        )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        entity_vocab=rendered,
        predicate_list=_PREDICATE_LIST,
    )


def _build_runnable(
    entity_vocab: list[tuple[str, str]],
    model: str,
    temperature: float,
):
    """Build the per-call retryable structured-output runnable.

    `model` + `temperature` come from the corpus's `CorpusConfig` via
    the pipeline call site, per-corpus since 2026-07-02 (previously
    read from global Settings).

    The retry wrapper sits OUTSIDE `.with_structured_output(...)` per
    LangChain's composition order: structured-output first, retry
    second.
    """
    llm = _get_llm(model, temperature)
    structured = llm.with_structured_output(_LLMTriples)
    return _with_retry(structured), _build_system_prompt(entity_vocab)


def _filter_and_convert_triples(
    result: object, entity_vocab: list[tuple[str, str]]
) -> list[ExtractedTriple]:
    """Drop hallucinated entities/predicates; convert to ExtractedTriple.

    Defensive `isinstance` first (same rationale as in entity
    extractor). Then drop triples whose subject/object isn't in the
    L6 vocab (LLM hallucinated an entity) or whose predicate isn't
    in the allowed 15 (LLM ignored the constraint). Caller logs
    aggregate counts via the write-side mismatch check.
    """
    if not isinstance(result, _LLMTriples):
        return []

    vocab_lookup: dict[str, str] = {key: entity_type for key, entity_type in entity_vocab}
    allowed_predicates: set[str] = set(TRIPLE_PREDICATE_RELS)

    out: list[ExtractedTriple] = []
    for t in result.triples:
        if t.subject_key not in vocab_lookup:
            continue
        if t.object_key not in vocab_lookup:
            continue
        if t.predicate not in allowed_predicates:
            continue
        out.append(
            ExtractedTriple(
                subject_key=t.subject_key,
                subject_entity_type=vocab_lookup[t.subject_key],
                predicate=t.predicate,
                object_key=t.object_key,
                object_entity_type=vocab_lookup[t.object_key],
                evidence_span=t.evidence_span,
            )
        )
    return out


async def extract(
    text: str,
    entity_vocab: list[tuple[str, str]],
    *,
    model: str,
    temperature: float,
) -> list[ExtractedTriple]:
    """Extract typed triples from `text`, constrained to `entity_vocab`.

    Args:
      text:          chunk text content.
      entity_vocab:  list of (key, entity_type) pairs from L6 for this
                     chunk. The LLM is constrained to these as subject +
                     object identities.
      model:         LLM model identifier (per-corpus
                     `CorpusConfig.triples_extractor_model`).
      temperature:   LLM temperature (per-corpus
                     `CorpusConfig.triples_extractor_temperature`).

    Returns:
      List of `ExtractedTriple`. Empty when the chunk has no qualifying
      relations OR when `entity_vocab` is empty (no entities = nothing
      to relate). LLM hallucinations referencing out-of-vocabulary keys
      or predicates are dropped (caller logs the totals via the
      write-side mismatch check).

    The structured-output call is wrapped in the standard retry policy
    via `with_retry` (covers transient 429s after the per-provider
    InMemoryRateLimiter does its job).
    """
    # Fast-path: a chunk with no L6 entities can't yield any triples
    # (since subject + object must come from L6). Skip the LLM call.
    if not entity_vocab:
        return []

    runnable, system_prompt = _build_runnable(entity_vocab, model, temperature)
    result = await runnable.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]
    )
    return _filter_and_convert_triples(result, entity_vocab)
