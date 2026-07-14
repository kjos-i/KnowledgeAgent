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

    from knowledge_agent.config import Settings
    from knowledge_agent.search.client import LanceClient

logger = logging.getLogger(__name__)


class EvalGenerationConnectionError(RuntimeError):
    """Raised when case generation can't reach the LLM API.

    A network/connection failure is NOT passage-specific — every call would
    fail identically until the connection is back — so the batch aborts with
    one clear, actionable message instead of grinding through N identical
    failures and reporting "0 cases". The message is user-facing (the GUI
    shows it as-is), so it names the problem and says to retry."""


def _is_connection_error(exc: BaseException) -> bool:
    """Best-effort, provider-agnostic check for a transport/network failure.

    Walks the exception's cause/context chain and matches on class name so we
    don't have to import each provider SDK — anthropic/openai `APIConnectionError`
    and `APITimeoutError`, httpx `ConnectError` / `TimeoutException`, and the
    builtin `ConnectionError` family all qualify. Rate-limit / auth / bad-request
    errors (which name the real problem) deliberately do NOT match."""
    seen: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and len(seen) < 6:
        seen.append(cur)
        cur = cur.__cause__ or cur.__context__
    for e in seen:
        for cls in type(e).__mro__:
            name = cls.__name__
            if (
                "APIConnectionError" in name
                or "APITimeoutError" in name
                or "ConnectError" in name
                or "ConnectionError" in name
                or "TimeoutException" in name
            ):
                return True
    return False


def _pinned_retrieval(settings: Settings) -> dict[str, object]:
    """A fully-pinned per-case retrieval config from the active GLOBAL defaults.

    Generated cases must pin every knob (not leave it None) or the runner
    rejects them as non-reproducible (see `models.validate_case`). We read the
    same global Settings the manual Dataset form seeds its defaults from, so
    generated and hand-authored cases share one source of truth and both run
    out of the box."""
    return {
        "retrieval_mode": settings.default_retrieval_mode,
        "lancedb_search_mode": settings.lancedb_search_mode,
        "top_k": settings.top_k,
        "num_candidates": settings.num_candidates,
        "rrf_rank_constant": settings.rrf_rank_constant,
        "mmr_lambda": settings.mmr_lambda,
        "use_mmr": settings.default_use_mmr,
        "kg_max_rows": settings.kg_max_rows,
    }


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
    from knowledge_agent.config import get_settings

    settings = get_settings()
    if llm is None:
        model = model or settings.mode_classifier_model
        llm = get_llm(model, temperature)
    structured = with_retry(llm.with_structured_output(GeneratedCase))
    # Pin every retrieval knob from the active global defaults so each
    # generated case is runnable + reproducible out of the box.
    retrieval = _pinned_retrieval(settings)

    cases: list[EvalCase] = []
    for i, passage in enumerate(passages):
        try:
            gen: GeneratedCase = await structured.ainvoke(
                [SystemMessage(content=_GEN_SYSTEM), HumanMessage(content=passage.text)]
            )
        except Exception as exc:
            if _is_connection_error(exc):
                # Not this passage's fault — the LLM API is unreachable, so
                # every remaining call would fail the same way. Abort the whole
                # batch with one clear, retryable message instead of N skips.
                raise EvalGenerationConnectionError(
                    "Couldn't reach the LLM API — this looks like a network / "
                    "connection problem, not your corpus. Check your internet "
                    "connection (and VPN / proxy / firewall), then try again."
                ) from exc
            # Best-effort: skip a passage the LLM couldn't turn into a case
            # (e.g. malformed structured output) rather than aborting the batch.
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
                retrieval=retrieval,
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


class GeneratedGold(BaseModel):
    """The LLM's gold (answer facts + keywords) for a GIVEN question.

    Unlike `GeneratedCase`, the question is fixed (e.g. a chat router's
    distilled query) — the LLM only writes the gold FOR it, grounded in the
    passages that were retrieved for that question. Still a CANDIDATE for human
    review, not trusted truth (same caveat as `GeneratedCase`)."""

    answer_points: list[str] = Field(
        default_factory=list,
        description="The key facts a correct answer to the question must contain.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="A few salient terms that should appear in a correct answer.",
    )


_GOLD_SYSTEM = (
    "You write the GOLD answer for an evaluation case. You are given a QUESTION "
    "and the passages that were retrieved for it. Produce:\n"
    "  - answer_points: the key facts a correct answer to the question must "
    "contain, grounded ONLY in the passages;\n"
    "  - keywords: a few salient terms that should appear in a correct answer.\n"
    "Do NOT invent anything beyond the passages. If the passages don't fully "
    "answer the question, return only the points they DO support (possibly none)."
)


async def passages_from_sources(
    chunk_sources: list,
    *,
    client: LanceClient | None = None,
) -> list[Passage]:
    """Re-fetch the full text of the chunks a chat/search actually cited.

    An `AgentAnswer`'s `chunk_sources` carry only doc_id + chunk_id + an
    optional short `quote`, not the full chunk text. To ground LLM gold in what
    the chat really retrieved, re-fetch each cited chunk's text from LanceDB (by
    doc_id, filtered to the cited chunk_ids), falling back to the citation quote
    when a chunk can't be re-fetched. Citation order + de-dup preserved. Best
    effort: a failed re-fetch degrades to the quote, never raises."""
    if client is None:
        from knowledge_agent.search.client import get_search_client

        client = get_search_client()
    # (doc_id, chunk_id) in citation order, de-duped; + quote fallbacks by chunk.
    order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    quote_by_chunk: dict[str, str] = {}
    for cs in chunk_sources or []:
        doc_id = getattr(cs, "doc_id", None)
        chunk_id = getattr(cs, "chunk_id", None)
        if not doc_id or not chunk_id or (doc_id, chunk_id) in seen:
            continue
        seen.add((doc_id, chunk_id))
        order.append((doc_id, chunk_id))
        quote = getattr(cs, "quote", None)
        if quote:
            quote_by_chunk[chunk_id] = quote
    # Re-fetch full text once per cited doc, keep the cited chunks' text.
    text_by_chunk: dict[str, str] = {}
    for doc_id in {d for d, _ in order}:
        try:
            for chunk in await client.get_chunks_by_doc_id(doc_id):
                cid = chunk.get("chunk_id")
                text = (chunk.get("text") or "").strip()
                if cid and text:
                    text_by_chunk[cid] = text
        except Exception as exc:  # best-effort re-fetch; fall back to the quotes
            logger.warning("passages_from_sources: re-fetch failed for %s: %r", doc_id, exc)
    passages: list[Passage] = []
    for doc_id, chunk_id in order:
        text = text_by_chunk.get(chunk_id) or quote_by_chunk.get(chunk_id, "")
        if text.strip():
            passages.append(Passage(doc_id=doc_id, text=text))
    return passages


async def generate_gold_for_question(
    question: str,
    passages: list[Passage],
    *,
    model: str | None = None,
    temperature: float = 0.3,
    llm: BaseChatModel | None = None,
) -> GeneratedGold:
    """Draft the gold (answer_points + keywords) for a GIVEN question from the
    passages retrieved for it.

    Used when 'Query from chat' is on and the LLM seed fills the gold for the
    chat's distilled question (rather than inventing a fresh one). `llm` is
    injectable for tests; None builds one from the active provider. Returns
    empty lists when the passages don't ground an answer. A network failure
    raises `EvalGenerationConnectionError` (clear, retryable)."""
    from knowledge_agent.config import get_settings

    if llm is None:
        settings = get_settings()
        model = model or settings.mode_classifier_model
        llm = get_llm(model, temperature)
    structured = with_retry(llm.with_structured_output(GeneratedGold))
    context = "\n\n".join(f"[{p.doc_id}]\n{p.text}" for p in passages if p.text.strip())
    human = f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{context or '(none)'}"
    try:
        return await structured.ainvoke(
            [SystemMessage(content=_GOLD_SYSTEM), HumanMessage(content=human)]
        )
    except Exception as exc:
        if _is_connection_error(exc):
            raise EvalGenerationConnectionError(
                "Couldn't reach the LLM API — this looks like a network / "
                "connection problem, not your corpus. Check your internet "
                "connection (and VPN / proxy / firewall), then try again."
            ) from exc
        raise
