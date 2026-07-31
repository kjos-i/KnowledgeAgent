"""LLM eval-case generator: draft candidate `EvalCase`s from a corpus.

Grounds each case in a whole document (its full text) and, where the corpus
has a knowledge graph, that document's entities and relationships, then asks
the configured LLM to write a question + key answer facts + salient keywords,
returning them as `EvalCase`s flagged `origin="llm"`. It can also write the
gold for a fixed question (the "Query from chat" flow).

These are CANDIDATES, NOT gold. Treating an LLM's own output as ground truth
is circular (especially if the generator shares a model family with the
agent's synthesizer), so every generated case lands flagged for human review
(the Dataset tab's Edit form is the promote step). Best practice: keep the
generator model distinct from the judge and the synthesizer.

Provider-agnostic: builds its LLM via `llm_factory.get_llm_ref` (the active
provider) defaulting to the mode-classifier model (a cheap tier), the same
choose-your-provider pattern as `judge.resolve_judge_models`, no hardcoded
cross-provider model.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from knowledge_agent.evaluation.models import EvalCase
from knowledge_agent.llm_factory import get_llm_ref, with_retry

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from knowledge_agent.kg.client import Neo4jClient
    from knowledge_agent.search.client import LanceClient

logger = logging.getLogger(__name__)


class EvalGenerationConnectionError(RuntimeError):
    """Raised when case generation can't reach the LLM API.

    A network/connection failure is NOT passage-specific — every call would
    fail identically until the connection is back — so the batch aborts with
    one clear, actionable message instead of grinding through N identical
    failures and reporting "0 cases". The message is user-facing (the GUI
    shows it as-is), so it names the problem and says to retry."""


_CONN_MSG = (
    "Couldn't reach the LLM API — this looks like a network / connection "
    "problem, not your corpus. Check your internet connection (and VPN / proxy "
    "/ firewall), then try again."
)


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


def _pinned_retrieval() -> dict[str, object]:
    """A fully-pinned per-case retrieval config from the fixed config defaults.

    Generated cases must pin every knob (not leave it None) or the runner
    rejects them as non-reproducible (see `models.validate_case`). Reads the
    FIXED config field defaults (`config.retrieval_defaults`), the same baseline
    the manual Dataset form seeds a new case with, not the user's live settings,
    so generated and hand-authored cases share one source of truth and stay
    reproducible."""
    from knowledge_agent.config import retrieval_defaults

    return retrieval_defaults()


@dataclass
class Passage:
    """One corpus text passage to generate a case from."""

    doc_id: str
    text: str


def _slug(text: str, *, max_len: int = 40) -> str:
    """A short lowercase-hyphenated slug from the question, for a readable id."""
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:max_len].strip("-") or "case"


# ==================== Case generator ====================
# Grounds each case in the WHOLE document (all its chunks) and, where the corpus
# has a knowledge graph, in that document's entities and their relationships, so
# it can write BOTH hybrid (text-answerable) and KG (relationship-answerable)
# cases, plus a few CROSS-DOCUMENT cases from entities that appear in more than
# one document. Graph reads go through the existing Neo4j client (read_query),
# no new Cypher layer, and degrade to full-doc hybrid cases when there's no
# graph. Because it grounds in the ACTUAL sampled docs/entities, it fills
# expected_sources / expected_entities (the "paste into ChatGPT" flow can't).


class GeneratedAdvancedCase(BaseModel):
    """The LLM's structured draft: picks hybrid vs KG and fills the gold fields
    (question, answer points, keywords, and expected entities for KG cases)."""

    case_type: Literal["hybrid", "kg"] = Field(
        default="hybrid",
        description="'kg' if best answered from an entity RELATIONSHIP; 'hybrid' if from the TEXT.",
    )
    question: str = Field(
        description="A specific, self-contained question grounded in the material."
    )
    answer_points: list[str] = Field(
        default_factory=list, description="Key facts a correct answer must contain."
    )
    keywords: list[str] = Field(
        default_factory=list, description="Salient terms a correct answer should include."
    )
    expected_entities: list[str] = Field(
        default_factory=list,
        description="For a 'kg' case: entity / relationship names expected in the graph rows.",
    )
    category: str = Field(
        default="", description="Short grouping tag, e.g. Mechanism / Relationship."
    )


_ADVANCED_SYSTEM = (
    "You write evaluation cases for a retrieval-augmented QA system that searches "
    "BOTH a text store and a knowledge graph. You are given a document's full "
    "text and (when available) graph facts — entity relationships drawn from that "
    "document. Write ONE high-quality case:\n"
    "  - case_type: 'kg' when the best answer depends on a RELATIONSHIP between "
    "entities (use the graph facts); 'hybrid' when it depends on a fact stated in "
    "the TEXT.\n"
    "  - question: one specific, self-contained question — natural, not "
    "'according to the document'.\n"
    "  - answer_points: the key facts a correct answer must contain.\n"
    "  - keywords: a few salient terms a correct answer should include.\n"
    "  - expected_entities: for a 'kg' case, the entity / relationship names the "
    "graph rows should contain; leave empty for 'hybrid'.\n"
    "  - category: a short grouping tag (Mechanism, Definition, Relationship, …).\n"
    "Ground everything ONLY in the material provided — invent nothing. Prefer "
    "'kg' when strong graph facts are present, otherwise 'hybrid'."
)

_ADVANCED_CROSS_SYSTEM = (
    "You write a CROSS-DOCUMENT evaluation case: a question answerable ONLY by "
    "COMBINING facts from the TWO documents provided (they share an entity). Same "
    "output fields; set case_type='hybrid'. The question must genuinely require "
    "BOTH documents — not be answerable from either alone. Ground only in the "
    "provided text; invent nothing."
)


async def _sample_doc_ids(client: LanceClient, n: int) -> list[str]:
    """Up to `n` indexed doc_ids (spread across the corpus, one per doc)."""
    ids: list[str] = []
    for doc in await client.list_indexed_docs():
        did = doc.get("doc_id")
        if did:
            ids.append(did)
        if len(ids) >= n:
            break
    return ids


async def _full_doc_text(client: LanceClient, doc_id: str, *, max_chars: int = 8000) -> str:
    """The document's chunks concatenated (capped): full-doc grounding for a
    generated case."""
    parts: list[str] = []
    total = 0
    for chunk in await client.get_chunks_by_doc_id(doc_id):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts)


async def _doc_graph_facts(
    kg_client: Neo4jClient, doc_id: str, *, max_entities: int = 8, max_facts: int = 15
) -> list[str]:
    """`A --REL--> B` relationship facts for the entities this document mentions.
    Best-effort: any KG error (no graph / Neo4j down) yields []."""
    facts: list[str] = []
    try:
        by_chunk = await kg_client.get_entities_by_chunk(doc_id)
    except Exception as exc:  # broad: no graph / unreachable → hybrid-only
        logger.info("advanced: no graph facts for %s: %r", doc_id, exc)
        return facts
    keys: list[str] = []
    for pairs in by_chunk.values():
        for key, _type in pairs:
            if key and key not in keys:
                keys.append(key)
    for key in keys[:max_entities]:
        try:
            rows = await kg_client.read_query(
                "MATCH (a:Entity {key: $key})-[r]->(b:Entity) "
                "RETURN type(r) AS rel, b.key AS target LIMIT 5",
                key=key,
            )
        except Exception as exc:  # broad: skip this entity's facts
            logger.info("advanced: triple read failed for %s: %r", key, exc)
            continue
        for row in rows:
            rel, target = row.get("rel"), row.get("target")
            if rel and target:
                facts.append(f"{key} --{rel}--> {target}")
        if len(facts) >= max_facts:
            break
    return facts[:max_facts]


async def _cross_doc_pairs(kg_client: Neo4jClient, *, limit: int) -> list[tuple[str, str, str]]:
    """(entity_key, doc_a, doc_b) for entities mentioned in >= 2 documents — the
    seeds for cross-document cases. Best-effort: [] on any KG error."""
    if limit <= 0:
        return []
    try:
        rows = await kg_client.read_query(
            "MATCH (e:Entity)<-[:MENTIONS]-(:Chunk)-[:PART_OF]->(d) "
            "WITH e, collect(DISTINCT d.doc_id) AS docs "
            "WHERE size(docs) >= 2 "
            "RETURN e.key AS key, docs[0] AS a, docs[1] AS b "
            "ORDER BY rand() LIMIT $limit",
            limit=limit,
        )
    except Exception as exc:  # broad: no graph / pattern absent → no cross-doc
        logger.info("advanced: cross-doc pair read failed: %r", exc)
        return []
    return [
        (r["key"], r["a"], r["b"])
        for r in rows
        if r.get("key") and r.get("a") and r.get("b") and r["a"] != r["b"]
    ]


def _doc_context(doc_id: str, text: str, facts: list[str]) -> str:
    graph = "\n".join(f"- {f}" for f in facts) if facts else "(none for this document)"
    return f"DOCUMENT [{doc_id}]:\n{text}\n\nGRAPH FACTS (entity relationships):\n{graph}"


def _cross_context(key: str, a: str, ta: str, b: str, tb: str) -> str:
    return f"SHARED ENTITY: {key}\n\nDOCUMENT A [{a}]:\n{ta}\n\nDOCUMENT B [{b}]:\n{tb}"


def _advanced_to_case(i: int, doc_ids: list[str], gen: GeneratedAdvancedCase) -> EvalCase | None:
    """Map an LLM draft to a runnable `EvalCase`, pinning retrieval to the KG or
    text leg per the chosen type (retrieval stays fully pinned, reproducible)."""
    question = (gen.question or "").strip()
    if not question:
        return None
    is_kg = gen.case_type == "kg" and bool(gen.expected_entities)
    retrieval = {
        **_pinned_retrieval(),
        "retrieval_mode": "neo4j_only" if is_kg else "lancedb_only",
    }
    return EvalCase(
        id=f"adv-{i:02d}-{_slug(question)}",
        question=question,
        expected_sources=list(doc_ids),
        expected_answer_points=list(gen.answer_points),
        required_keywords=list(gen.keywords),
        expected_entities=list(gen.expected_entities) if is_kg else [],
        origin="llm",
        category=gen.category.strip() or ("kg" if is_kg else "generated"),
        notes="Advanced LLM candidate — review before trusting.",
        retrieval=retrieval,
    )


async def _draft(structured, system: str, human: str, label: str) -> GeneratedAdvancedCase | None:
    """One structured LLM draft; a connection failure aborts the batch (retryable),
    any other error skips this unit (best-effort)."""
    try:
        return await structured.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
    except Exception as exc:
        if _is_connection_error(exc):
            raise EvalGenerationConnectionError(_CONN_MSG) from exc
        logger.warning("advanced: skipping unit %s: %r", label, exc)
        return None


async def generate_advanced(
    n: int,
    *,
    model: str | None = None,
    temperature: float = 0.3,
    lance_client: LanceClient | None = None,
    kg_client: Neo4jClient | None = None,
    llm: BaseChatModel | None = None,
) -> list[EvalCase]:
    """Draft up to `n` richer `origin="llm"` cases with full-document + knowledge-
    graph grounding.

    Per document: the WHOLE document text + that document's entity relationships
    (from Neo4j). The LLM writes either a hybrid (text-answerable) or a KG
    (relationship-answerable) case. When entities are shared across documents, a
    slice of the batch is CROSS-DOCUMENT cases. Degrades to full-doc hybrid cases
    when there's no graph. `lance_client` / `kg_client` / `llm` are injectable for
    tests. A network failure raises `EvalGenerationConnectionError` (retryable)."""
    from knowledge_agent.config import get_settings

    if n <= 0:
        return []
    if llm is None:
        model = model or get_settings().mode_classifier_model
        llm = get_llm_ref(model, temperature)
    if lance_client is None:
        from knowledge_agent.search.client import get_search_client

        lance_client = get_search_client()
    if kg_client is None:
        try:
            from knowledge_agent.kg.client import get_kg_client

            kg_client = get_kg_client()
        except Exception as exc:  # broad: no graph configured → hybrid-only
            logger.info("advanced: KG client unavailable, hybrid-only: %r", exc)

    per_case = with_retry(llm.with_structured_output(GeneratedAdvancedCase))

    # Reserve a slice for cross-document cases when the graph supports them.
    pairs = await _cross_doc_pairs(kg_client, limit=n // 4) if kg_client else []
    doc_ids = await _sample_doc_ids(lance_client, n - len(pairs))

    cases: list[EvalCase] = []
    for doc_id in doc_ids:
        text = await _full_doc_text(lance_client, doc_id)
        if not text:
            continue
        facts = await _doc_graph_facts(kg_client, doc_id) if kg_client else []
        gen = await _draft(per_case, _ADVANCED_SYSTEM, _doc_context(doc_id, text, facts), doc_id)
        if gen is None:
            continue
        case = _advanced_to_case(len(cases), [doc_id], gen)
        if case:
            cases.append(case)

    for key, a, b in pairs:
        ta, tb = await _full_doc_text(lance_client, a), await _full_doc_text(lance_client, b)
        if not (ta and tb):
            continue
        gen = await _draft(
            per_case, _ADVANCED_CROSS_SYSTEM, _cross_context(key, a, ta, b, tb), f"{a}+{b}"
        )
        if gen is None:
            continue
        case = _advanced_to_case(len(cases), [a, b], gen)
        if case:
            cases.append(case)

    return cases


class GeneratedGold(BaseModel):
    """The LLM's gold (answer facts + keywords) for a GIVEN question.

    Here the question is fixed (e.g. a chat router's distilled query): the LLM
    only writes the gold FOR it, grounded in the passages that were retrieved
    for that question. Still a CANDIDATE for human review, not trusted truth."""

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
        llm = get_llm_ref(model, temperature)
    structured = with_retry(llm.with_structured_output(GeneratedGold))
    context = "\n\n".join(f"[{p.doc_id}]\n{p.text}" for p in passages if p.text.strip())
    human = f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{context or '(none)'}"
    try:
        return await structured.ainvoke(
            [SystemMessage(content=_GOLD_SYSTEM), HumanMessage(content=human)]
        )
    except Exception as exc:
        if _is_connection_error(exc):
            raise EvalGenerationConnectionError(_CONN_MSG) from exc
        raise
