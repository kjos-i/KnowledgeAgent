"""Graph state for the agent.

LangGraph's state is the data structure that threads through every node
in the graph. Per invocation: the query goes in, intermediate fields are
filled by each node, and the final answer comes out.

Six-mode topology (see `settings.default_retrieval_mode`). The state shape
is the same across every mode - different nodes populate different fields,
and the synthesizer reads whichever evidence is present:

  Mode                Populated by
  ------------------  ----------------------------------------------------
  lancedb_only        search_query, retrieved_chunks
  neo4j_only          cypher_query, kg_hits
  lancedb_then_neo4j  search_query, retrieved_chunks, cypher_query, kg_hits
                      (Cypher sees Lance hits as context)
  neo4j_then_lancedb  cypher_query, kg_hits, search_query, retrieved_chunks
                      (Lance scoped by doc_ids from kg_hits)
  parallel_fused      all four (legs run in parallel, synthesizer fuses)
  auto                routed_mode (set by classifier) + whichever fields
                      the chosen concrete mode populates

`routed_mode` carries the classifier's pick when `retrieval_mode == "auto"`;
otherwise it stays None and the router dispatches on `retrieval_mode`.

`TypedDict(total=False)` means every field is optional - nodes set fields
as they execute, downstream nodes read what's been set. The contract for
each node is documented at the field, not enforced by the type.
"""

from typing import TypedDict

from knowledge_agent.config import Settings
from knowledge_agent.errors import ErrorDetail
from knowledge_agent.kg.corpus_config import CorpusConfig
from knowledge_agent.models import AgentAnswer, KGHit, RetrievedChunk


class AgentState(TypedDict, total=False):
    """Graph state shape. All fields optional; nodes fill what they own."""

    # ---- input fields (set by the caller before invoke) ----

    query: str
    """The user's question. Required input - every invocation must set it."""

    corpus_config: CorpusConfig | None
    """The active corpus's `CorpusConfig`. Required input for any node
    that needs to know which KG layers are present in this corpus -
    notably `cypher_builder_node`, which uses it to scope the schema
    description shown to the Cypher LLM. Loaded from the corpus folder's
    `corpus.yaml` (or constructed in-memory by the caller) before invoke.
    Nodes that need it raise if missing - no silent fallback."""

    retrieval_mode: str | None
    """Per-invocation mode override. When None, the graph falls back to
    `settings.default_retrieval_mode`. Allowed values match the Literal
    declared in `config.Settings.default_retrieval_mode`."""

    top_k: int | None
    """Per-invocation top-k override for the retriever. When None, the
    retriever node uses its own default."""

    skip_query_builder: bool | None
    """Per-invocation override for the query-rewrite LLM step. None falls
    back to `settings.skip_query_builder`. When True, the user's raw
    `query` is used verbatim as the search query."""

    direct_retrieval: bool | None
    """Per-invocation override for the synthesizer LLM step. None falls
    back to `settings.direct_retrieval`. When True, the synthesizer
    is skipped - the agent returns retrieved chunks without a synthesised
    answer."""

    # ---- intermediate fields (set by nodes during graph execution) ----

    routed_mode: str | None
    """The concrete retrieval mode actually being run. Set by the
    `mode_classifier` node when `retrieval_mode == "auto"`; left None
    otherwise. The graph router prefers `routed_mode` when present, falling
    back to `retrieval_mode`. Kept separate so logs / UI can distinguish
    'user picked auto' from 'user picked X' even after dispatch."""

    search_query: str | None
    """Query string passed to the retriever. May be the user's raw `query`
    (when skip_query_builder is True), or a rewritten keyword-dense version
    produced by the query_builder node. Set by query_builder; read by the
    retriever."""

    cypher_query: str | None
    """Cypher query produced by the cypher_builder node. Set by
    cypher_builder; read by the neo4j_retriever. Read-only constraints are
    enforced at execution time, not on this field."""

    retrieved_chunks: list[RetrievedChunk]
    """Hits from the LanceDB retriever, typed. Set by lancedb_retriever
    (and later by the fuser in cross-store modes); read by the synthesizer."""

    kg_hits: list[KGHit]
    """Hits from the Neo4j retriever, typed. Set by neo4j_retriever; read by
    the synthesizer. Sibling of `retrieved_chunks` for the graph-store side.
    `KGHit.data` is a flexible dict because Cypher row shape depends on the
    LLM-generated query."""

    kg_retrieval_error: ErrorDetail | None
    """Typed-error detail when the `neo4j_retriever_node`'s read query
    failed (Cypher error, server error, connection failure). Set
    alongside an empty `kg_hits` so the synthesizer / GUI can
    distinguish 'KG was unreachable' from 'KG returned no rows'. None on
    success and when the node didn't run."""

    lancedb_retrieval_error: ErrorDetail | None
    """Typed-error detail when the `lancedb_retriever_node`'s search
    call raised (Lance corruption, Voyage outage, schema mismatch). Set
    alongside an empty `retrieved_chunks` so the synthesizer / GUI can
    distinguish 'LanceDB failed' from 'no chunks matched'. None on
    success and when the node didn't run."""

    # ---- output fields (set by the synthesizer) ----

    final_answer: AgentAnswer | None
    """Synthesizer output - the structured answer + sources. The GUI
    renders `answer` as text and `sources` as clickable references. Set by
    the synthesizer node; what the caller ultimately reads."""


def effective_mode(state: AgentState, settings: Settings) -> str:
    """Resolve which retrieval mode is actually running this turn.

    Precedence:
      1. `state["routed_mode"]`     - set by the mode_classifier in auto mode.
      2. `state["retrieval_mode"]`  - the caller's per-invocation choice.
      3. `settings.default_retrieval_mode` - the config default.

    Single source of truth so every node (retrievers, cypher_builder, graph
    router) reads the active mode the same way.
    """
    return (
        state.get("routed_mode") or state.get("retrieval_mode") or settings.default_retrieval_mode
    )
