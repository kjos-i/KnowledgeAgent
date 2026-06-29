"""Anthropic-Haiku entity extractor (L6a, open vocabulary).

KNOWN_LABELS = None: this adapter does NOT constrain the user's
`entity_types` against any closed set. Whatever labels the corpus puts
in `corpus.toml [entities].entity_types` flow into the LLM prompt as
guidance; an empty list means the model categorises freely.

Cost (Haiku at 2026 pricing): one call per chunk. Input ~800-1000
tokens (system prompt + chunk + types). Output small. ~$0.05 per
200-chunk paper, ~$0.0003 per chunk.

Offsets + confidence are intentionally NOT requested from the LLM:
  - Offsets: LLMs hallucinate character positions badly; the offset is
    recoverable from chunk text via the raw_text anyway, just not at
    extraction time.
  - Confidence: no native per-mention score; asking for a self-reported
    confidence produces noise.
Both fields stay None on every Mention returned from this adapter.

Per-call LLM client is cached via lru_cache keyed by (model, temperature),
mirroring `nodes._get_llm`.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from knowledge_agent.config import get_settings
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.llm_factory import get_llm as _get_llm
from knowledge_agent.llm_factory import with_retry as _with_retry

# Open vocabulary - the user's entity_types flow through unvalidated.
KNOWN_LABELS: tuple[str, ...] | None = None


# ---------------------------------------------------------------------------
# Structured-output schema. Pydantic models are LLM-facing only; the public
# Mention dataclass in base.py is what the rest of the codebase consumes.
# ---------------------------------------------------------------------------


class _ExtractedMention(BaseModel):
    """One mention as returned by the LLM's structured output."""

    raw_text: str = Field(
        description=(
            "Verbatim span as it appears in the input text. Original "
            "spelling preserved (do not lowercase / normalise)."
        )
    )
    entity_type: str = Field(
        description=(
            "Short uppercase category label for this mention "
            "(e.g. GENE, DISEASE, PERSON). Use SCREAMING_SNAKE_CASE for "
            "multi-word types (e.g. CELL_TYPE)."
        )
    )


class _ExtractedMentions(BaseModel):
    """Top-level wrapper for the structured-output call."""

    mentions: list[_ExtractedMention] = Field(
        default_factory=list,
        description=(
            "All entity mentions found in the text. Empty list when no "
            "matching entities are present."
        ),
    )


# ---------------------------------------------------------------------------
# Prompts.
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT_OPEN = """\
You extract named entity mentions from text.

Find any named entities you can identify (genes, proteins, diseases,
chemicals, people, places, organisations, ...) and for each return:

  - raw_text   : the verbatim span as it appears in the input text
  - entity_type: a short uppercase category label that fits (e.g. GENE,
                 DISEASE, PERSON, LOCATION). Use SCREAMING_SNAKE_CASE
                 for multi-word categories (e.g. CELL_TYPE).

Rules:
  - DO NOT paraphrase or normalise spelling - copy the span verbatim.
  - DO NOT invent mentions that are not in the text.
  - If no entities are present, return an empty list.
"""


_SYSTEM_PROMPT_CLOSED_TEMPLATE = """\
You extract named entity mentions from text.

ONLY extract entities of these types: {types}.

For each mention return:
  - raw_text   : the verbatim span as it appears in the input text
  - entity_type: which of the allowed types this mention is. The value
                 MUST exactly match one of: {types}.

Rules:
  - DO NOT paraphrase or normalise spelling - copy the span verbatim.
  - DO NOT invent mentions that are not in the text.
  - DO NOT return mentions of types outside the allowed list.
  - If no matching entities are present, return an empty list.
"""


# ---------------------------------------------------------------------------
# Adapter.
# ---------------------------------------------------------------------------


# _get_llm is imported from llm_factory above — provider-agnostic
# dispatch shared across all LLM call sites.


def _build_system_prompt(entity_types: list[str]) -> str:
    """Closed-mode prompt when types provided; open-mode prompt when not."""
    if entity_types:
        return _SYSTEM_PROMPT_CLOSED_TEMPLATE.format(
            types=", ".join(entity_types)
        )
    return _SYSTEM_PROMPT_OPEN


def _mentions_to_list(result: object) -> list[Mention]:
    """Convert structured-output result into the public Mention list.

    Defensive `isinstance` check: `with_structured_output` should
    always return `_ExtractedMentions` but if the LLM somehow returns
    a dict (shape mismatch, future SDK change) we want to fail closed
    with `[]` rather than crash partway through the pipeline.
    """
    if not isinstance(result, _ExtractedMentions):
        return []
    return [
        Mention(
            raw_text=m.raw_text,
            entity_type=m.entity_type,
            offset=None,
            confidence=None,
        )
        for m in result.mentions
    ]


def _build_runnable(entity_types: list[str]):
    """Build the per-call retryable structured-output runnable.

    The retry wrapper sits OUTSIDE `.with_structured_output(...)` per
    LangChain's composition order: structured-output first, retry
    second.
    """
    settings = get_settings()
    llm = _get_llm(
        settings.entity_extractor_model,
        settings.entity_extractor_temperature,
    )
    structured = llm.with_structured_output(_ExtractedMentions)
    return _with_retry(structured), _build_system_prompt(entity_types)


async def extract(text: str, entity_types: list[str]) -> list[Mention]:
    """Extract entity mentions from `text`.

    Args:
      text:         the chunk's raw text content.
      entity_types: empty list -> LLM categorises freely (open vocab).
                    Non-empty list -> LLM constrained to those types.

    Returns:
      List of `Mention` with `offset` and `confidence` always None.
      Empty list when the LLM finds no matching entities.

    The structured-output call is wrapped in the standard retry policy
    via `with_retry` (covers transient 429s after the per-provider
    InMemoryRateLimiter does its job).
    """
    runnable, system_prompt = _build_runnable(entity_types)
    result = await runnable.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]
    )
    return _mentions_to_list(result)
