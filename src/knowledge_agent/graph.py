"""LangGraph topology for the agent.

Six retrieval modes wired through one graph, dispatched by conditional
edges that read `effective_mode(state)`:

  lancedb_only        START -> query_builder  -> lancedb_retriever -> synthesizer
  neo4j_only          START -> cypher_builder -> neo4j_retriever   -> synthesizer
  lancedb_then_neo4j  START -> query_builder  -> lancedb_retriever
                            -> cypher_builder -> neo4j_retriever   -> synthesizer
  neo4j_then_lancedb  START -> cypher_builder -> neo4j_retriever
                            -> query_builder  -> lancedb_retriever -> synthesizer
  parallel_fused      START -> [query_builder, cypher_builder]  (fan-out)
                            -> [lancedb_retriever, neo4j_retriever]
                            -> synthesizer (fan-in)
  auto                START -> mode_classifier -> dispatch as one of the 5 above

`synthesizer` is the convergence point for every mode; it handles both
`retrieved_chunks` and `kg_hits` in one cited answer (the fusion in
parallel_fused is implicit in the synthesizer, no separate fuser node).

The compiled graph is exposed as the module-level `graph`:

    from knowledge_agent.graph import graph
    final_state = await graph.ainvoke({"query": "..."})

`final_state["final_answer"]` is the structured `AgentAnswer`.
"""

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge_agent.config import get_settings
from knowledge_agent.nodes import (
    cypher_builder_node,
    lancedb_retriever_node,
    mode_classifier_node,
    neo4j_retriever_node,
    query_builder_node,
    synthesizer_node,
)
from knowledge_agent.state import AgentState, effective_mode

logger = logging.getLogger(__name__)

# ---- Node name constants. Anywhere code references a node-by-string-name
# (edges, conditional dispatch, traces, tests) uses these so renames are
# one-line changes.

NODE_MODE_CLASSIFIER = "mode_classifier"
NODE_QUERY_BUILDER = "query_builder"
NODE_CYPHER_BUILDER = "cypher_builder"
NODE_LANCEDB_RETRIEVER = "lancedb_retriever"
NODE_NEO4J_RETRIEVER = "neo4j_retriever"
NODE_SYNTHESIZER = "synthesizer"


def _entry_for_mode(mode: str) -> str | list[str]:
    """Map a concrete retrieval mode to its entry node(s).

    Returns a single node name for sequential modes; a LIST of node names
    for `parallel_fused` (LangGraph fans out when a conditional edge
    function returns a list).

    Unknown modes (including the meta-mode "auto" which should have been
    resolved by the classifier upstream) fall back to `lancedb_only` as
    the safest default - Lance hybrid search always works against the
    chunk corpus regardless of KG state.
    """
    if mode in ("lancedb_only", "lancedb_then_neo4j"):
        return NODE_QUERY_BUILDER
    if mode in ("neo4j_only", "neo4j_then_lancedb"):
        return NODE_CYPHER_BUILDER
    if mode == "parallel_fused":
        return [NODE_QUERY_BUILDER, NODE_CYPHER_BUILDER]
    logger.warning(
        "graph router: unknown mode %r; falling back to lancedb_only",
        mode,
    )
    return NODE_QUERY_BUILDER


def _route_from_start(state: AgentState) -> str | list[str]:
    """From START: pick the first node(s) based on the caller's mode choice.

    Auto mode -> goes to the classifier first. All other modes dispatch
    directly. This reads `retrieval_mode` (or settings default) WITHOUT
    using `effective_mode`, because routed_mode hasn't been set yet at
    this point in the run.
    """
    raw_mode = state.get("retrieval_mode") or get_settings().default_retrieval_mode
    if raw_mode == "auto":
        return NODE_MODE_CLASSIFIER
    return _entry_for_mode(raw_mode)


def _route_from_classifier(state: AgentState) -> str | list[str]:
    """After mode_classifier has set routed_mode, dispatch to the chosen mode."""
    mode = effective_mode(state, get_settings())
    return _entry_for_mode(mode)


def _route_after_lance(state: AgentState) -> str:
    """After lancedb_retriever, continue to cypher_builder (mode 3) or end."""
    mode = effective_mode(state, get_settings())
    if mode == "lancedb_then_neo4j":
        return NODE_CYPHER_BUILDER
    return NODE_SYNTHESIZER


def _route_after_neo(state: AgentState) -> str:
    """After neo4j_retriever, continue to query_builder (mode 4) or end."""
    mode = effective_mode(state, get_settings())
    if mode == "neo4j_then_lancedb":
        return NODE_QUERY_BUILDER
    return NODE_SYNTHESIZER


def _build_graph() -> CompiledStateGraph:
    """Construct and compile the agent graph. Called once at import time."""
    builder = StateGraph(AgentState)

    # ---- Register every node.
    builder.add_node(NODE_MODE_CLASSIFIER, mode_classifier_node)
    builder.add_node(NODE_QUERY_BUILDER, query_builder_node)
    builder.add_node(NODE_CYPHER_BUILDER, cypher_builder_node)
    builder.add_node(NODE_LANCEDB_RETRIEVER, lancedb_retriever_node)
    builder.add_node(NODE_NEO4J_RETRIEVER, neo4j_retriever_node)
    builder.add_node(NODE_SYNTHESIZER, synthesizer_node)

    # ---- Entry routing. START -> conditional dispatch by retrieval_mode.
    builder.add_conditional_edges(START, _route_from_start)

    # ---- After classifier: conditional dispatch by routed_mode.
    builder.add_conditional_edges(NODE_MODE_CLASSIFIER, _route_from_classifier)

    # ---- Static "builder -> retriever" edges (always fire).
    builder.add_edge(NODE_QUERY_BUILDER, NODE_LANCEDB_RETRIEVER)
    builder.add_edge(NODE_CYPHER_BUILDER, NODE_NEO4J_RETRIEVER)

    # ---- Post-retriever conditional edges (continue or converge).
    builder.add_conditional_edges(NODE_LANCEDB_RETRIEVER, _route_after_lance)
    builder.add_conditional_edges(NODE_NEO4J_RETRIEVER, _route_after_neo)

    # ---- Synthesizer is the only exit.
    builder.add_edge(NODE_SYNTHESIZER, END)

    return builder.compile()


graph: CompiledStateGraph = _build_graph()
