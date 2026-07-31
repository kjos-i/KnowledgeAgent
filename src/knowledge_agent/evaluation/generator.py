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
import random
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


_ADVANCED_CROSS_SYSTEM = (
    "You write a CROSS-DOCUMENT evaluation case: a question answerable ONLY by "
    "COMBINING facts from the TWO documents provided (they share an entity). Same "
    "output fields; set case_type='hybrid'. The question must genuinely require "
    "BOTH documents — not be answerable from either alone. Ground only in the "
    "provided text; invent nothing."
)


async def _all_doc_ids(client: LanceClient) -> list[str]:
    """Every indexed doc_id — the eligible pool for the text modes."""
    return [doc["doc_id"] for doc in await client.list_indexed_docs() if doc.get("doc_id")]


async def _graph_eligible_docs(kg_client: Neo4jClient) -> list[str]:
    """doc_ids that mention at least one entity — the eligible pool for the graph
    modes (a graph case needs entities to ground a relationship question).
    Best-effort: [] on any KG error."""
    try:
        rows = await kg_client.read_query(
            "MATCH (:Entity)<-[:MENTIONS]-(:Chunk)-[:PART_OF]->(d) "
            "RETURN DISTINCT d.doc_id AS doc_id"
        )
    except Exception as exc:  # broad: no graph / unreachable → no graph docs
        logger.info("targeted: graph-eligible docs read failed: %r", exc)
        return []
    return [r["doc_id"] for r in rows if r.get("doc_id")]


async def _l9_pairs(kg_client: Neo4jClient) -> list[tuple[str, str, list[str]]]:
    """(doc_a, doc_b, shared_entities) from the L9 :RELATED_TO edges (docs that
    share entities). Best-effort: [] on any KG error."""
    try:
        rows = await kg_client.read_query(
            "MATCH (a)-[r:RELATED_TO]-(b) WHERE a.doc_id < b.doc_id "
            "RETURN a.doc_id AS a, b.doc_id AS b, r.shared_entities AS shared"
        )
    except Exception as exc:  # broad: no L9 layer / unreachable
        logger.info("targeted: L9 pair read failed: %r", exc)
        return []
    return [
        (r["a"], r["b"], list(r.get("shared") or []))
        for r in rows
        if r.get("a") and r.get("b") and r["a"] != r["b"]
    ]


async def _l10_pairs(kg_client: Neo4jClient) -> list[tuple[str, str, list[str]]]:
    """(doc_a, doc_b, shared_concepts) from the L10 :RELATED_BY_XREF edges (docs
    linked because their entities share an ontology concept). Best-effort: []."""
    try:
        rows = await kg_client.read_query(
            "MATCH (a)-[r:RELATED_BY_XREF]-(b) WHERE a.doc_id < b.doc_id "
            "RETURN a.doc_id AS a, b.doc_id AS b, r.shared_concepts AS shared"
        )
    except Exception as exc:  # broad: no L10 layer / unreachable
        logger.info("targeted: L10 pair read failed: %r", exc)
        return []
    return [
        (r["a"], r["b"], list(r.get("shared") or []))
        for r in rows
        if r.get("a") and r.get("b") and r["a"] != r["b"]
    ]


async def generation_capabilities(kg_client: Neo4jClient | None = None) -> dict[str, bool]:
    """Which generation targets the active corpus can support, for the GUI
    checkbox grey-out. The text modes (lancedb_only, auto) are always available;
    the graph modes need entities; the two cross-doc kinds need their edges.
    Best-effort: a missing / unreachable graph leaves the graph targets off."""
    caps = {"text": True, "graph": False, "cross_related": False, "cross_xref": False}
    if kg_client is None:
        try:
            from knowledge_agent.kg.client import get_kg_client

            kg_client = get_kg_client()
        except Exception as exc:  # broad: no graph configured → text-only
            logger.info("generation_capabilities: no KG client: %r", exc)
            return caps
    try:
        caps["graph"] = (await kg_client.count_mentions()) > 0
        caps["cross_related"] = (await kg_client.count_related_to_edges()) > 0
        caps["cross_xref"] = (await kg_client.count_related_by_xref_edges()) > 0
    except Exception as exc:  # broad: probe failed → leave graph targets off
        logger.info("generation_capabilities: KG probe failed: %r", exc)
    return caps


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


def _doc_context(doc_id: str, text: str, facts: list[str]) -> str:
    graph = "\n".join(f"- {f}" for f in facts) if facts else "(none for this document)"
    return f"DOCUMENT [{doc_id}]:\n{text}\n\nGRAPH FACTS (entity relationships):\n{graph}"


def _cross_context(shared: list[str], a: str, ta: str, b: str, tb: str) -> str:
    shared_line = ", ".join(shared) if shared else "(shared context)"
    return f"SHARED: {shared_line}\n\nDOCUMENT A [{a}]:\n{ta}\n\nDOCUMENT B [{b}]:\n{tb}"


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
        logger.warning("targeted: skipping unit %s: %r", label, exc)
        return None


# The six retrieval modes selectable as generation targets, split by their
# eligible document pool: text modes draw from every doc; graph modes draw only
# from docs that mention entities (so a relationship question can ground).
_TEXT_MODES = ("lancedb_only", "auto")
_GRAPH_MODES = ("neo4j_only", "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused")

# Per-mode guidance inserted into the case-writing prompt so each generated
# question suits the store(s) that mode runs.
_MODE_GUIDANCE = {
    "lancedb_only": (
        "answerable from a fact stated in the document TEXT; leave expected_entities empty."
    ),
    "auto": (
        "a natural question about the document (text- or relationship-based); "
        "fill expected_entities only if it hinges on an entity relationship."
    ),
    "neo4j_only": (
        "answerable from an entity RELATIONSHIP in the graph facts; fill "
        "expected_entities with the relevant entity / relationship names."
    ),
    "lancedb_then_neo4j": (
        "needing BOTH a fact from the text AND an entity relationship from the "
        "graph facts; fill expected_entities."
    ),
    "neo4j_then_lancedb": (
        "needing BOTH an entity relationship from the graph facts AND a fact "
        "from the text; fill expected_entities."
    ),
    "parallel_fused": (
        "drawing on BOTH the document text and an entity relationship from the "
        "graph facts; fill expected_entities."
    ),
}


def _mode_system(guidance: str) -> str:
    """The case-writing system prompt for a mode target, with its guidance."""
    return (
        "You write ONE evaluation case for a retrieval-augmented QA system, from "
        "the document (and any graph facts) provided. Write a question that is "
        f"{guidance}\n"
        "Also produce: answer_points (the key facts a correct answer must "
        "contain), keywords (a few salient terms), expected_entities (entity / "
        "relationship names for a graph-based question, empty otherwise), and a "
        "short category tag. Ground everything ONLY in the material provided; "
        "invent nothing. Make the question natural and self-contained (do NOT say "
        "'according to the document')."
    )


def _targeted_to_case(
    i: int, doc_ids: list[str], gen: GeneratedAdvancedCase, retrieval_mode: str
) -> EvalCase | None:
    """Map an LLM draft to a runnable `EvalCase`, pinning retrieval to the target
    mode (the checkbox the case was generated for); every other knob comes from
    the config baseline, so the case is fully pinned + reproducible."""
    question = (gen.question or "").strip()
    if not question:
        return None
    retrieval = {**_pinned_retrieval(), "retrieval_mode": retrieval_mode}
    return EvalCase(
        id=f"gen-{i:02d}-{_slug(question)}",
        question=question,
        expected_sources=list(doc_ids),
        expected_answer_points=list(gen.answer_points),
        required_keywords=list(gen.keywords),
        expected_entities=list(gen.expected_entities),
        origin="llm",
        category=gen.category.strip() or retrieval_mode,
        notes="LLM-generated candidate — review before trusting.",
        retrieval=retrieval,
    )


async def _make_targeted_case(
    target: dict, item, idx: int, per_case, lance_client, kg_client
) -> EvalCase | None:
    """Generate one case for a target from its dealt pool item: a doc_id for a
    mode target, a (doc_a, doc_b, shared) tuple for a cross-doc target."""
    if target["kind"] == "mode":
        mode = target["mode"]
        text = await _full_doc_text(lance_client, item)
        if not text:
            return None
        facts = (
            await _doc_graph_facts(kg_client, item)
            if kg_client is not None and mode in _GRAPH_MODES
            else []
        )
        gen = await _draft(
            per_case, _mode_system(_MODE_GUIDANCE[mode]), _doc_context(item, text, facts), item
        )
        return _targeted_to_case(idx, [item], gen, mode) if gen else None
    a, b, shared = item
    ta, tb = await _full_doc_text(lance_client, a), await _full_doc_text(lance_client, b)
    if not (ta and tb):
        return None
    gen = await _draft(
        per_case, _ADVANCED_CROSS_SYSTEM, _cross_context(shared, a, ta, b, tb), f"{a}+{b}"
    )
    return _targeted_to_case(idx, [a, b], gen, "parallel_fused") if gen else None


async def generate_targeted(
    n: int,
    *,
    modes: list[str] | tuple[str, ...] = (),
    cross_doc: list[str] | tuple[str, ...] = (),
    model: str | None = None,
    temperature: float = 0.3,
    lance_client: LanceClient | None = None,
    kg_client: Neo4jClient | None = None,
    llm: BaseChatModel | None = None,
    rng: random.Random | None = None,
) -> list[EvalCase]:
    """Draft up to `n` `origin="llm"` cases across the chosen generation targets.

    `modes` is a subset of the six retrieval modes; `cross_doc` a subset of
    {"related" (L9), "xref" (L10)}. Each target has its own eligible pool (all
    docs for the text modes, entity-bearing docs for the graph modes, shared-doc
    pairs for the cross-doc kinds), shuffled with `rng` (injectable for tests) and
    dealt one item per case, round-robin across targets, so N splits evenly and a
    target that runs dry rolls its remainder onto the others (coverage-first). A
    doc can serve one case per target (independent pools), never twice in one
    target. Each case pins `retrieval_mode` to its target (cross-doc ->
    parallel_fused); other knobs come from the config baseline. Best-effort per
    draft; a network failure raises `EvalGenerationConnectionError` (retryable)."""
    from knowledge_agent.config import get_settings

    modes = list(dict.fromkeys(modes))
    cross_doc = list(dict.fromkeys(cross_doc))
    if n <= 0 or not (modes or cross_doc):
        return []
    if rng is None:
        rng = random.Random()
    if llm is None:
        model = model or get_settings().mode_classifier_model
        llm = get_llm_ref(model, temperature)
    if lance_client is None:
        from knowledge_agent.search.client import get_search_client

        lance_client = get_search_client()
    need_graph = any(m in _GRAPH_MODES for m in modes) or bool(cross_doc)
    if kg_client is None and need_graph:
        try:
            from knowledge_agent.kg.client import get_kg_client

            kg_client = get_kg_client()
        except Exception as exc:  # broad: no graph → graph targets get empty pools
            logger.info("targeted: KG client unavailable: %r", exc)

    per_case = with_retry(llm.with_structured_output(GeneratedAdvancedCase))

    # Assemble the targets, each a shuffled pool dealt round-robin below.
    targets: list[dict] = []
    all_docs: list[str] | None = None
    for mode in modes:
        if mode in _TEXT_MODES:
            if all_docs is None:
                all_docs = await _all_doc_ids(lance_client)
            pool: list = list(all_docs)
        elif mode in _GRAPH_MODES:
            pool = await _graph_eligible_docs(kg_client) if kg_client is not None else []
        else:
            continue
        rng.shuffle(pool)
        if pool:
            targets.append({"kind": "mode", "mode": mode, "pool": pool, "i": 0})
    for kind in cross_doc:
        if kg_client is None:
            continue
        pairs = await (_l9_pairs(kg_client) if kind == "related" else _l10_pairs(kg_client))
        rng.shuffle(pairs)
        if pairs:
            targets.append({"kind": "cross", "cross": kind, "pool": pairs, "i": 0})

    cases: list[EvalCase] = []
    while len(cases) < n and targets:
        for t in list(targets):
            if len(cases) >= n:
                break
            item = t["pool"][t["i"]]
            t["i"] += 1
            case = await _make_targeted_case(t, item, len(cases), per_case, lance_client, kg_client)
            if case:
                cases.append(case)
        targets = [t for t in targets if t["i"] < len(t["pool"])]
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
