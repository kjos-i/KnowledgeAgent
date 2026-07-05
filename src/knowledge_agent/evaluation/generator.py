"""LLM eval-case generator — draft candidate `EvalCase`s from corpus text.

Slice 5 of the eval-dataset authoring UI. Samples text passages from the
active corpus, asks the configured LLM to write one self-contained
question + key answer facts + salient keywords per passage, and returns
them as `EvalCase`s flagged `origin="llm"`.

These are CANDIDATES, NOT gold. Treating an LLM's own output as ground
truth is circular — especially if the generator shares a model family with
the agent's synthesizer — so every generated case lands flagged for human
review (the Dataset tab's Edit form is the promote step). Best practice:
keep the generator model distinct from the judge and the synthesizer.

Provider-agnostic: builds its LLM via `llm_factory.get_llm` (the active
provider) defaulting to the mode-classifier model (a cheap tier), the same
choose-your-provider pattern as `judge.resolve_judge_models` — no hardcoded
cross-provider model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from knowledge_agent.evaluation.models import EvalCase
from knowledge_agent.llm_factory import get_llm, with_retry

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from knowledge_agent.search.client import LanceClient

logger = logging.getLogger(__name__)


@dataclass
class Passage:
    """One corpus text passage to generate a case from."""

    doc_id: str
    text: str


class GeneratedCase(BaseModel):
    """The LLM's structured draft for one passage."""

    question: str = Field(description="A specific question answerable using ONLY this passage.")
    answer_points: list[str] = Field(
        default_factory=list,
        description="The key facts a correct answer must contain.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="A few salient terms that should appear in a correct answer.",
    )


_GEN_SYSTEM = (
    "You write evaluation cases for a retrieval-augmented question-answering "
    "system. You are given ONE passage from a document. Produce:\n"
    "  - question: a single, specific question answerable using ONLY this "
    "passage — not general knowledge, not other documents;\n"
    "  - answer_points: the key facts a correct answer must contain;\n"
    "  - keywords: a few salient terms that should appear in a correct answer.\n"
    "Do NOT invent anything beyond the passage. Make the question natural and "
    "self-contained — do NOT say 'according to the passage' or 'in the text'."
)


def _slug(text: str, *, max_len: int = 40) -> str:
    """A short lowercase-hyphenated slug from the question, for a readable id."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "case"


async def generate_cases(
    passages: list[Passage],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    llm: BaseChatModel | None = None,
) -> list[EvalCase]:
    """Draft one `origin="llm"` `EvalCase` per passage.

    `llm` is injectable (tests pass a fake); when None, one is built from
    the active provider via `get_llm(model, temperature)` with `model`
    defaulting to the mode-classifier model. A passage whose generation
    fails (rate limit, malformed structured output, …) is skipped
    best-effort — one bad passage doesn't abort the batch.
    """
    if llm is None:
        from knowledge_agent.config import get_settings

        model = model or get_settings().mode_classifier_model
        llm = get_llm(model, temperature)
    structured = with_retry(llm.with_structured_output(GeneratedCase))

    cases: list[EvalCase] = []
    for i, passage in enumerate(passages):
        try:
            gen: GeneratedCase = await structured.ainvoke(
                [SystemMessage(content=_GEN_SYSTEM), HumanMessage(content=passage.text)]
            )
        except Exception as exc:
            # Best-effort: skip a passage the LLM couldn't turn into a case
            # rather than aborting the whole batch.
            logger.warning(
                "generate_cases: skipping passage %d (doc %s): %r", i, passage.doc_id, exc
            )
            continue
        question = (gen.question or "").strip()
        if not question:
            continue
        cases.append(
            EvalCase(
                id=f"gen-{i:02d}-{_slug(question)}",
                question=question,
                expected_sources=[passage.doc_id],
                expected_answer_points=list(gen.answer_points),
                required_keywords=list(gen.keywords),
                origin="llm",
                category="generated",
                notes="LLM-generated candidate — review before trusting.",
            )
        )
    return cases


async def sample_passages(
    n: int,
    *,
    client: LanceClient | None = None,
    min_chars: int = 200,
) -> list[Passage]:
    """Pull up to `n` text passages (each >= `min_chars`) from the corpus.

    Takes the FIRST chunk per document that clears `min_chars`, so the
    sample spreads ACROSS documents rather than concentrating in one — one
    passage per doc, up to `n`. `client` is injectable for tests; None uses
    the active `get_search_client()`.
    """
    if n <= 0:
        return []
    if client is None:
        from knowledge_agent.search.client import get_search_client

        client = get_search_client()
    passages: list[Passage] = []
    for doc in await client.list_indexed_docs():
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        for chunk in await client.get_chunks_by_doc_id(doc_id):
            text = (chunk.get("text") or "").strip()
            if len(text) >= min_chars:
                passages.append(Passage(doc_id=doc_id, text=text))
                break  # one passage per doc → spread coverage across the corpus
        if len(passages) >= n:
            break
    return passages[:n]


async def generate_from_corpus(
    n: int,
    *,
    model: str | None = None,
    temperature: float = 0.3,
) -> list[EvalCase]:
    """Sample up to `n` passages from the active corpus and draft a case
    per passage. Live glue over `sample_passages` + `generate_cases`."""
    passages = await sample_passages(n)
    return await generate_cases(passages, model=model, temperature=temperature)
