"""LangGraph node functions for the agent.

Each node takes the agent state, does its work, returns a partial state
update dict. LangGraph merges the returned dict back into state and routes
to the next node.

Nodes:

  - mode_classifier_node  : LLM picks one of the 5 concrete retrieval modes
                            (only runs in auto mode); fail-soft to lancedb_only
  - query_builder_node    : LLM rewrites raw query into search-optimised form
                            (LanceDB hybrid search)
  - cypher_builder_node   : LLM writes Cypher against the KG schema; in cross-
                            store modes the system prompt gains a doc_id rule
                            and (mode 3) the user message carries Lance hits
  - lancedb_retriever_node: hybrid / fts / vector search in LanceDB; in mode
                            neo4j_then_lancedb applies a doc_id IN [...] filter
                            from state.kg_hits, falls back to unfiltered on
                            empty doc_id list. Catches Lance / Voyage
                            failures and populates state.lancedb_retrieval_error.
  - neo4j_retriever_node  : runs the Cypher against Neo4j with three-layer
                            safety (keyword validation, CALL-wrap LIMIT,
                            read-only session). Catches Cypher / driver
                            failures and populates state.kg_retrieval_error.
  - synthesizer_node      : LLM produces final answer with citations from
                            chunks and/or kg rows

Toggles honoured at the per-invocation state field, falling back to the
matching Settings default:

  - skip_query_builder : bypass the rewrite LLM, use raw query verbatim
  - direct_retrieval   : bypass the synthesizer LLM, return retriever output

Structured-output contracts (see `models.py`):

  - ModeChoice          : mode_classifier's structured output
  - SearchQueryRewrite  : query_builder's structured output
  - CypherQueryRewrite  : cypher_builder's structured output
  - RetrievedChunk      : one hit returned by LanceDB search
  - KGHit               : one row returned by the Neo4j retriever
  - AgentAnswer         : synthesizer's structured output
                          (answer + chunk_sources + kg_sources)
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge_agent.config import get_settings
from knowledge_agent.kg.client import get_kg_client
from knowledge_agent.llm_factory import get_llm as _get_llm
from knowledge_agent.kg.cypher_safety import (
    is_cypher_read_only,
    wrap_with_limit,
)
from knowledge_agent.kg.schema_as_prompt import (
    format_schema_for_prompt,
)
from knowledge_agent.models import (
    AgentAnswer,
    ChunkSource,
    CypherQueryRewrite,
    KGHit,
    KGSource,
    ModeChoice,
    RetrievedChunk,
    SearchQueryRewrite,
)
from knowledge_agent.errors import ErrorDetail
from knowledge_agent.search.client import get_search_client
from knowledge_agent.state import AgentState, effective_mode

logger = logging.getLogger(__name__)


# _get_llm is imported from llm_factory above — provider-agnostic
# dispatch keyed by (model, temperature) under the active
# `settings.llm_provider`. Cache lives inside the factory so all four
# node call sites share one set of clients.


# =========================================================================
# mode_classifier_node
# =========================================================================

_MODE_CLASSIFIER_SYSTEM = """\
You are a retrieval-mode classifier for a research literature agent.

Read the user's question and pick the ONE retrieval mode that best fits.
Output ONLY the mode name from the allowed list.

Available modes:

- lancedb_only       - Semantic/content question. Lance hybrid search over
                       paper chunks. Pick for "what does the literature say
                       about X" style questions where the answer lives in
                       paper text, not graph relationships.

- neo4j_only         - Pure structural/graph question. Cypher over the KG.
                       Pick for "who cites / who authored / how many authors"
                       style questions where the answer lives in graph
                       relationships, not paper text.

- lancedb_then_neo4j - Content question + want graph enrichment of the
                       relevant papers. Pick when the user asks about a
                       topic AND about authors/citations of papers on that
                       topic.

- neo4j_then_lancedb - Want chunks scoped to a KG-defined subset of papers.
                       Pick when the user asks for content (paper text)
                       restricted to specific authors / years / sources.

- parallel_fused     - Question genuinely needs both stores and there's no
                       clear sequential dependency. Pick for broad /
                       exploratory questions where both paper text AND
                       graph structure inform the answer.

Examples:
  "What does the literature say about CRISPR?"               -> lancedb_only
  "What are the main themes in cancer immunotherapy?"        -> lancedb_only
  "Who cites paper W123456?"                                 -> neo4j_only
  "How many corresponding authors per paper on average?"     -> neo4j_only
  "Find chunks about CRISPR and show me who cites those"     -> lancedb_then_neo4j
  "What do Smith's papers say about CRISPR?"                 -> neo4j_then_lancedb
  "Comprehensive overview of CRISPR research"                -> parallel_fused
"""


async def mode_classifier_node(state: AgentState) -> dict[str, Any]:
    """Classify the user's question into one of the 5 concrete retrieval modes.

    Runs only when `retrieval_mode == "auto"` (the graph router gates this).
    Writes the choice to `state["routed_mode"]`; the router then dispatches
    accordingly.

    Fail-soft: any exception logs + defaults to `lancedb_only` (safest
    fallback - Lance hybrid search always works against the chunk corpus
    regardless of KG state).
    """
    settings = get_settings()
    llm = _get_llm(
        settings.mode_classifier_model,
        settings.mode_classifier_temperature,
    )
    structured = llm.with_structured_output(ModeChoice)
    try:
        result = await structured.ainvoke(
            [
                SystemMessage(content=_MODE_CLASSIFIER_SYSTEM),
                HumanMessage(content=state["query"]),
            ]
        )
    except Exception as exc:
        logger.warning(
            "mode_classifier: failed (%r); falling back to lancedb_only",
            exc,
        )
        return {"routed_mode": "lancedb_only"}
    logger.info("mode_classifier: %r -> %s", state["query"], result.mode)
    return {"routed_mode": result.mode}


# =========================================================================
# query_builder_node
# =========================================================================

_QUERY_BUILDER_SYSTEM = """\
You are a search-query rewriter for a research literature corpus.

Take the user's question and rewrite it as a keyword-dense search query
optimised for hybrid (BM25 + vector) retrieval over scientific papers:

- Extract the domain terms and key concepts.
- Drop conversational filler ("can you tell me", "I'd like to know").
- Drop question marks and other punctuation.
- Keep important qualifiers (year ranges, authors, organism, technique).
- Output ONLY the rewritten query - no preamble, no explanation.

Examples:
  User: "What does the literature say about apoptosis in cancer cells?"
  -> "apoptosis cancer cells"

  User: "Can you find papers about CRISPR screens for drug resistance from 2020 onwards?"
  -> "CRISPR screens drug resistance 2020"

  User: "I want to learn about how BRCA1 mutations affect DNA repair pathways."
  -> "BRCA1 mutations DNA repair pathways"
"""


async def query_builder_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the user's question into a keyword-dense search query.

    Honours `state["skip_query_builder"]` (per-invocation) and
    `settings.skip_query_builder` (default). When True, the raw query is
    used verbatim - no LLM call, no cost.
    """
    settings = get_settings()
    skip = state.get("skip_query_builder")
    if skip is None:
        skip = settings.skip_query_builder
    if skip:
        logger.info("query_builder: skipped (using raw query verbatim)")
        return {"search_query": state["query"]}

    llm = _get_llm(
        settings.query_builder_model, settings.query_builder_temperature
    )
    structured = llm.with_structured_output(SearchQueryRewrite)
    result = await structured.ainvoke(
        [
            SystemMessage(content=_QUERY_BUILDER_SYSTEM),
            HumanMessage(content=state["query"]),
        ]
    )
    logger.info("query_builder: %r -> %r", state["query"], result.search_query)
    return {"search_query": result.search_query}


# =========================================================================
# cypher_builder_node
# =========================================================================

# `<SCHEMA>` and `<MODE_RULES>` are literal placeholder strings replaced at
# call time. Plain string + .replace() (not f-string) because Cypher syntax
# uses `{}` heavily and an f-string would need every brace doubled.
_CYPHER_BUILDER_SYSTEM_TEMPLATE = """\
You are a Cypher query writer for a Neo4j knowledge graph of research
literature. Read the user's question and produce a Cypher query that
returns the rows most relevant to the question. Output ONLY the Cypher
string - no preamble, no explanation, no markdown fences.

<SCHEMA>

Rules:
- READ-ONLY queries only. NEVER use CREATE, MERGE, DELETE, DROP, SET, or
  REMOVE. The runtime rejects any of those keywords.
- Use the labels, relationships, and properties listed above exactly.
  Do NOT invent labels, relationship types, or property names.
- Always include a LIMIT clause.<MODE_RULES>
- If the question cannot be answered from this schema (e.g., the user
  asks about paper content, abstracts, or topics that aren't stored in
  the KG), return a Cypher query that yields no rows:
    MATCH (x) WHERE false RETURN x LIMIT 0

Examples:

  Question: "Who cites paper W123456?"
  -> MATCH (p:Document)-[:CITES]->(c:Document {openalex_id: 'W123456'})
     RETURN p.openalex_id, p.in_corpus LIMIT 50

  Question: "Who are the authors of paper W123456?"
  -> MATCH (a:Author)-[:AUTHORED]->(d:Document {openalex_id: 'W123456'})
     RETURN a.display_name, a.openalex_id LIMIT 50

  Question: "Which authors have the most papers in our corpus?"
  -> MATCH (a:Author)-[:AUTHORED]->(d:Document)
     WHERE d.in_corpus = true
     RETURN a.display_name, count(d) AS paper_count
     ORDER BY paper_count DESC LIMIT 20
"""

# Inserted into `<MODE_RULES>` for cross-store modes where the next leg
# needs doc_ids to correlate with LanceDB (mode 3) or apply a doc_id
# filter on Lance (mode 4). For modes 1+2 this placeholder is empty.
_MODE_RULES_DOC_ID = """
- Your RETURN clause MUST include `doc_id` so the next retrieval step can
  correlate results with the LanceDB store. Add `d.doc_id` (or whichever
  variable carries the Document) to your RETURN. For aggregate queries
  where doc_id has no meaningful slot, return per-document rows with
  doc_id rather than fully aggregating away."""

_CROSS_STORE_MODES = ("lancedb_then_neo4j", "neo4j_then_lancedb")


async def cypher_builder_node(state: AgentState) -> dict[str, Any]:
    """Write a Cypher query from the user's question + the KG schema.

    Sibling of `query_builder_node` but for the Neo4j side. No skip toggle:
    natural language can't run as Cypher, so there's no raw-query fallback.

    Mode-aware behaviour:
    - Modes `lancedb_then_neo4j` + `neo4j_then_lancedb`: the system prompt
      gains the "RETURN must include doc_id" rule (`<MODE_RULES>` slot) so
      the next leg can correlate or filter by doc_id.
    - Mode `lancedb_then_neo4j` (Lance ran first): the user message
      prepends a "context from LanceDB" block with the retrieved chunks so
      the LLM can reference their doc_ids directly in its Cypher.
    """
    settings = get_settings()
    mode = effective_mode(state, settings)

    corpus_config = state.get("corpus_config")
    if corpus_config is None:
        raise ValueError(
            "cypher_builder_node requires `corpus_config` in state - the "
            "schema description shown to the Cypher LLM is per-corpus and "
            "cannot be inferred. Set it on the initial state before "
            "invoking the graph."
        )

    mode_rules = (
        _MODE_RULES_DOC_ID if mode in _CROSS_STORE_MODES else ""
    )
    system_msg = (
        _CYPHER_BUILDER_SYSTEM_TEMPLATE
        .replace("<SCHEMA>", format_schema_for_prompt(corpus_config))
        .replace("<MODE_RULES>", mode_rules)
    )

    chunks = state.get("retrieved_chunks") or []
    if mode == "lancedb_then_neo4j" and chunks:
        user_msg = (
            "The LanceDB retriever returned these chunks for the user's "
            "question. Use their doc_ids in your Cypher to find related "
            "rows in the knowledge graph.\n\n"
            f"{_format_chunks_for_prompt(chunks)}\n\n"
            f"User question: {state['query']}"
        )
    else:
        user_msg = state["query"]

    llm = _get_llm(
        settings.cypher_builder_model, settings.cypher_builder_temperature
    )
    structured = llm.with_structured_output(CypherQueryRewrite)
    result = await structured.ainvoke(
        [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ]
    )
    logger.info(
        "cypher_builder: mode=%s, %r -> %r",
        mode, state["query"], result.cypher_query,
    )
    return {"cypher_query": result.cypher_query}


# =========================================================================
# neo4j_retriever_node
# =========================================================================


async def neo4j_retriever_node(state: AgentState) -> dict[str, Any]:
    """Run the cypher_builder's Cypher against Neo4j; populate `kg_hits`.

    Three-layer safety around the LLM-generated Cypher:
      1. Keyword validation (`is_cypher_read_only`) rejects writes up front.
      2. `wrap_with_limit` caps row count at the database level (LLM may
         have forgotten LIMIT or used too high a value).
      3. `Neo4jClient.read_query` uses `session.execute_read(...)`, which
         Neo4j itself refuses to honour for write queries.

    Fail-soft: any exception (server error, connection failure, invalid
    Cypher) logs + returns empty `kg_hits`. The synthesizer treats empty
    `kg_hits` the same as a zero-row query result.
    """
    cypher = state.get("cypher_query")
    if not cypher:
        logger.info("neo4j_retriever: no cypher_query in state; skipping")
        return {}

    if not is_cypher_read_only(cypher):
        logger.warning(
            "neo4j_retriever: rejected cypher (write keyword found): %r",
            cypher,
        )
        return {"kg_hits": []}

    settings = get_settings()
    wrapped = wrap_with_limit(cypher, settings.kg_max_rows)

    try:
        client = get_kg_client()
        rows = client.read_query(wrapped)
    except Exception as exc:
        logger.warning(
            "neo4j_retriever: query failed: %r; cypher=%r",
            exc, wrapped,
        )
        return {
            "kg_hits": [],
            "kg_retrieval_error": ErrorDetail.from_exception(exc),
        }

    hits = [KGHit(data=row) for row in rows]
    logger.info(
        "neo4j_retriever: %d row(s) for cypher=%r", len(hits), cypher
    )
    return {"kg_hits": hits}


# =========================================================================
# lancedb_retriever_node
# =========================================================================


def _extract_doc_ids_from_kg_hits(kg_hits: list[KGHit]) -> list[str]:
    """Collect every `doc_id` present in any kg_hit's data dict, deduped.

    Used by `lancedb_retriever_node` in mode `neo4j_then_lancedb` to build
    the `doc_id IN (...)` filter. Order is the order of first appearance
    in `kg_hits`. Missing or empty `doc_id` values are dropped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for hit in kg_hits:
        doc_id = hit.data.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


async def lancedb_retriever_node(state: AgentState) -> dict[str, Any]:
    """Hybrid (or fts/vector) search in LanceDB; populate `retrieved_chunks`.

    Uses `state["search_query"]` (set by query_builder) or falls back to the
    raw `state["query"]` if the query builder didn't run. `top_k` and within-
    store mode come from settings, overridable on state.

    Mode `neo4j_then_lancedb`: reads `kg_hits` from state, extracts every
    `doc_id`, and applies a `doc_id IN (...)` filter so Lance only searches
    within the KG-scoped subset. If no doc_ids are available (KG returned
    nothing useful, or no `doc_id` key in any hit), logs a warning and
    falls back to UNFILTERED Lance so the user still gets something.
    """
    settings = get_settings()
    query = state.get("search_query") or state["query"]
    top_k = state.get("top_k") or settings.top_k

    filters: dict[str, Any] | None = None
    if effective_mode(state, settings) == "neo4j_then_lancedb":
        kg_hits: list[KGHit] = state.get("kg_hits") or []
        doc_ids = _extract_doc_ids_from_kg_hits(kg_hits)
        if doc_ids:
            filters = {"doc_id": doc_ids}
        else:
            logger.warning(
                "lancedb_retriever: mode=neo4j_then_lancedb but no doc_ids "
                "found in kg_hits (count=%d); falling back to unfiltered "
                "Lance search",
                len(kg_hits),
            )

    client = get_search_client()
    try:
        hits = client.retrieve(query=query, top_k=top_k, filters=filters)
    except Exception as exc:
        # Search-path methods now raise on failure (typed-errors
        # contract); catch at the node boundary so the agent's response
        # carries a typed `lancedb_retrieval_error` and the synthesizer
        # can distinguish 'Lance failed' from 'no chunks matched'.
        logger.warning(
            "lancedb_retriever: search failed: %r; query=%r filters=%r",
            exc, query, filters,
        )
        return {
            "retrieved_chunks": [],
            "lancedb_retrieval_error": ErrorDetail.from_exception(exc),
        }
    logger.info(
        "lancedb_retriever: %r -> %d hits (filters=%r)",
        query, len(hits), filters,
    )
    return {"retrieved_chunks": hits}


# =========================================================================
# synthesizer_node
# =========================================================================

_SYNTHESIZER_SYSTEM = """\
You are a research literature assistant. Answer the user's question using
ONLY the evidence provided below. Two evidence sources may appear:

- Chunks: passages of paper text. Each has a chunk_id, doc_id, optional
  title/year/authors, and a body of text. Listed with [1], [2], ... markers.
- KG rows: structured rows from the knowledge graph. Each is a dict of
  fields the Cypher query returned. Listed with [K0], [K1], ... markers.

Either list may be empty in a given query.

Requirements:
- Cite EVERY non-trivial claim with the appropriate bracket marker.
- For chunk citations:
  - Use [1], [2], ... in the answer text.
  - Add a ChunkSource to `chunk_sources` (in marker order) with the
    chunk_id and doc_id taken from the shown chunk. Do NOT invent IDs.
- For KG row citations:
  - Use [K0], [K1], ... in the answer text.
  - Add a KGSource to `kg_sources` (in marker order) with `hit_index`
    equal to the row's number (e.g., [K2] -> hit_index=2).
- If the evidence doesn't contain enough information to answer, say so
  plainly in `answer` and return empty `chunk_sources` and `kg_sources`.
- Be concise and factual. Don't add information not supported by the evidence.

Output the structured AgentAnswer with all fields populated correctly.
"""


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered context for the synthesizer prompt.

    Numbering starts at 1 to match the [1], [2], ... markers the LLM is
    asked to emit. chunk_id and doc_id are always shown; metadata fields
    (title, year, authors_display) are included when present.
    """
    if not chunks:
        return "(no chunks retrieved)"
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header_bits = [f"chunk_id={c.chunk_id}", f"doc_id={c.doc_id}"]
        if c.title:
            header_bits.append(f"title={c.title!r}")
        if c.year:
            header_bits.append(f"year={c.year}")
        if c.authors_display:
            header_bits.append(f"authors={c.authors_display!r}")
        header = f"[{i}] " + " ".join(header_bits)
        parts.append(f"{header}\n{c.text}")
    return "\n\n".join(parts)


def _format_kg_hits_for_prompt(kg_hits: list[KGHit]) -> str:
    """Render kg_hits as numbered context for the synthesizer prompt.

    Numbering is zero-based to match the `hit_index` field on KGSource the
    LLM emits. Markers use the [K<n>] form to keep them distinguishable
    from chunk markers ([1], [2], ...) in the answer text. Each row's data
    dict is rendered as key=value pairs.
    """
    if not kg_hits:
        return "(no kg hits retrieved)"
    parts: list[str] = []
    for i, hit in enumerate(kg_hits):
        body_bits = [f"{k}={v!r}" for k, v in hit.data.items()]
        parts.append(f"[K{i}] " + ", ".join(body_bits))
    return "\n".join(parts)


async def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Produce the final AgentAnswer from retrieved chunks and/or kg hits.

    Honours `state["direct_retrieval"]` (per-invocation) and
    `settings.direct_retrieval` (default). When True, no LLM call -
    the chunks and kg hits become ChunkSource/KGSource entries with
    empty answer text.
    """
    settings = get_settings()
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks") or []
    kg_hits: list[KGHit] = state.get("kg_hits") or []

    direct = state.get("direct_retrieval")
    if direct is None:
        direct = settings.direct_retrieval

    if direct:
        logger.info(
            "synthesizer: skipped (direct_retrieval; %d chunks, %d kg_hits -> sources)",
            len(chunks), len(kg_hits),
        )
        chunk_sources = [
            ChunkSource(chunk_id=c.chunk_id, doc_id=c.doc_id)
            for c in chunks
        ]
        kg_sources = [KGSource(hit_index=i) for i in range(len(kg_hits))]
        return {
            "final_answer": AgentAnswer(
                answer="",
                chunk_sources=chunk_sources,
                kg_sources=kg_sources,
            )
        }

    llm = _get_llm(
        settings.synthesizer_model, settings.synthesizer_temperature
    )
    structured = llm.with_structured_output(AgentAnswer)
    user_msg = (
        f"Question: {state['query']}\n\n"
        f"Chunks:\n{_format_chunks_for_prompt(chunks)}\n\n"
        f"KG rows:\n{_format_kg_hits_for_prompt(kg_hits)}"
    )
    result = await structured.ainvoke(
        [
            SystemMessage(content=_SYNTHESIZER_SYSTEM),
            HumanMessage(content=user_msg),
        ]
    )
    logger.info(
        "synthesizer: produced %d-char answer with %d chunk sources, %d kg sources",
        len(result.answer), len(result.chunk_sources), len(result.kg_sources),
    )
    return {"final_answer": result}
