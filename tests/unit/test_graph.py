"""Tests for the compiled LangGraph topology + the per-mode routing logic.

Two layers:

- Compiled-graph shape tests: node-name constants, all 6 nodes registered,
  graph is async- and sync-invocable.
- Routing-function tests: each `_route_*` and `_entry_for_mode` is called
  directly with synthetic state dicts and mocked settings, asserting the
  next-node decision for each of the 6 modes. End-to-end execution tests
  (mocked nodes fired via `graph.ainvoke`) live in the smoke scripts.
"""

from unittest.mock import patch

from knowledge_agent.graph import (
    NODE_CYPHER_BUILDER,
    NODE_LANCEDB_RETRIEVER,
    NODE_MODE_CLASSIFIER,
    NODE_NEO4J_RETRIEVER,
    NODE_QUERY_BUILDER,
    NODE_SYNTHESIZER,
    _entry_for_mode,
    _route_after_lance,
    _route_after_neo,
    _route_from_classifier,
    _route_from_start,
    graph,
)

# ---- helpers ----


def _mock_default_mode(default: str = "lancedb_only"):
    """Patch context: `graph.get_settings().default_retrieval_mode = <default>`."""
    return patch(
        "knowledge_agent.graph.get_settings",
        return_value=type("S", (), {"default_retrieval_mode": default})(),
    )


# ---- compiled-graph shape ----


def test_graph_is_compiled_at_module_import():
    """The module-level `graph` is constructed when graph.py is imported."""
    assert graph is not None


def test_graph_exposes_async_invoke():
    assert hasattr(graph, "ainvoke")
    assert callable(graph.ainvoke)


def test_graph_exposes_sync_invoke():
    assert hasattr(graph, "invoke")
    assert callable(graph.invoke)


def test_node_name_constants_match_literal_strings():
    """Node-name constants are stable - changing them is a deliberate move."""
    assert NODE_MODE_CLASSIFIER == "mode_classifier"
    assert NODE_QUERY_BUILDER == "query_builder"
    assert NODE_CYPHER_BUILDER == "cypher_builder"
    assert NODE_LANCEDB_RETRIEVER == "lancedb_retriever"
    assert NODE_NEO4J_RETRIEVER == "neo4j_retriever"
    assert NODE_SYNTHESIZER == "synthesizer"


def test_graph_registers_all_six_nodes():
    """Every node referenced by the routing logic is present in the graph."""
    nodes = graph.get_graph().nodes
    node_names = set(nodes.keys()) if hasattr(nodes, "keys") else set(nodes)
    for expected in (
        NODE_MODE_CLASSIFIER,
        NODE_QUERY_BUILDER,
        NODE_CYPHER_BUILDER,
        NODE_LANCEDB_RETRIEVER,
        NODE_NEO4J_RETRIEVER,
        NODE_SYNTHESIZER,
    ):
        assert expected in node_names, f"node {expected!r} missing from compiled graph"


# ---- _entry_for_mode (concrete mode -> entry node[s]) ----


def test_entry_for_mode_lancedb_only_starts_query_builder():
    assert _entry_for_mode("lancedb_only") == NODE_QUERY_BUILDER


def test_entry_for_mode_neo4j_only_starts_cypher_builder():
    assert _entry_for_mode("neo4j_only") == NODE_CYPHER_BUILDER


def test_entry_for_mode_lancedb_then_neo4j_starts_query_builder():
    """Mode 3 starts with Lance, so entry is query_builder."""
    assert _entry_for_mode("lancedb_then_neo4j") == NODE_QUERY_BUILDER


def test_entry_for_mode_neo4j_then_lancedb_starts_cypher_builder():
    """Mode 4 starts with KG, so entry is cypher_builder."""
    assert _entry_for_mode("neo4j_then_lancedb") == NODE_CYPHER_BUILDER


def test_entry_for_mode_parallel_fused_returns_both_entries():
    """Mode 5 fans out: a list return value triggers LangGraph parallel."""
    entry = _entry_for_mode("parallel_fused")
    assert isinstance(entry, list)
    assert set(entry) == {NODE_QUERY_BUILDER, NODE_CYPHER_BUILDER}


def test_entry_for_mode_unknown_falls_back_to_query_builder():
    """Unknown / 'auto' / bogus modes default to lancedb_only entry."""
    assert _entry_for_mode("auto") == NODE_QUERY_BUILDER
    assert _entry_for_mode("not_a_real_mode") == NODE_QUERY_BUILDER


# ---- _route_from_start (START -> first node[s]) ----


def test_route_from_start_auto_mode_goes_to_classifier():
    with _mock_default_mode("auto"):
        assert _route_from_start({"retrieval_mode": "auto"}) == NODE_MODE_CLASSIFIER


def test_route_from_start_auto_via_settings_default_goes_to_classifier():
    """No retrieval_mode in state -> settings default kicks in."""
    with _mock_default_mode("auto"):
        assert _route_from_start({}) == NODE_MODE_CLASSIFIER


def test_route_from_start_lancedb_only_dispatches_to_query_builder():
    with _mock_default_mode("lancedb_only"):
        assert _route_from_start({"retrieval_mode": "lancedb_only"}) == NODE_QUERY_BUILDER


def test_route_from_start_neo4j_only_dispatches_to_cypher_builder():
    with _mock_default_mode("lancedb_only"):
        assert _route_from_start({"retrieval_mode": "neo4j_only"}) == NODE_CYPHER_BUILDER


def test_route_from_start_mode3_dispatches_to_query_builder():
    with _mock_default_mode("lancedb_only"):
        assert _route_from_start({"retrieval_mode": "lancedb_then_neo4j"}) == NODE_QUERY_BUILDER


def test_route_from_start_mode4_dispatches_to_cypher_builder():
    with _mock_default_mode("lancedb_only"):
        assert _route_from_start({"retrieval_mode": "neo4j_then_lancedb"}) == NODE_CYPHER_BUILDER


def test_route_from_start_parallel_fused_returns_list():
    with _mock_default_mode("lancedb_only"):
        result = _route_from_start({"retrieval_mode": "parallel_fused"})
        assert isinstance(result, list)
        assert set(result) == {NODE_QUERY_BUILDER, NODE_CYPHER_BUILDER}


# ---- _route_from_classifier (after mode_classifier set routed_mode) ----


def test_route_from_classifier_uses_routed_mode():
    """After classifier, dispatch reads routed_mode (effective_mode prefers it)."""
    with _mock_default_mode("lancedb_only"):
        state = {
            "retrieval_mode": "auto",
            "routed_mode": "neo4j_only",
        }
        assert _route_from_classifier(state) == NODE_CYPHER_BUILDER


def test_route_from_classifier_parallel_fused_fans_out():
    with _mock_default_mode("lancedb_only"):
        state = {
            "retrieval_mode": "auto",
            "routed_mode": "parallel_fused",
        }
        result = _route_from_classifier(state)
        assert isinstance(result, list)
        assert set(result) == {NODE_QUERY_BUILDER, NODE_CYPHER_BUILDER}


# ---- _route_after_lance ----


def test_route_after_lance_mode3_goes_to_cypher_builder():
    """Mode 3 is the only mode where Lance is followed by KG."""
    with _mock_default_mode("lancedb_only"):
        assert _route_after_lance({"retrieval_mode": "lancedb_then_neo4j"}) == NODE_CYPHER_BUILDER


def test_route_after_lance_mode1_goes_to_synthesizer():
    with _mock_default_mode("lancedb_only"):
        assert _route_after_lance({"retrieval_mode": "lancedb_only"}) == NODE_SYNTHESIZER


def test_route_after_lance_mode4_goes_to_synthesizer():
    """Mode 4: Lance ran second; the next step is synthesizer."""
    with _mock_default_mode("lancedb_only"):
        assert _route_after_lance({"retrieval_mode": "neo4j_then_lancedb"}) == NODE_SYNTHESIZER


def test_route_after_lance_mode5_goes_to_synthesizer():
    """Mode 5: parallel branches converge at synthesizer."""
    with _mock_default_mode("lancedb_only"):
        assert _route_after_lance({"retrieval_mode": "parallel_fused"}) == NODE_SYNTHESIZER


# ---- _route_after_neo ----


def test_route_after_neo_mode4_goes_to_query_builder():
    """Mode 4 is the only mode where KG is followed by Lance."""
    with _mock_default_mode("lancedb_only"):
        assert _route_after_neo({"retrieval_mode": "neo4j_then_lancedb"}) == NODE_QUERY_BUILDER


def test_route_after_neo_mode2_goes_to_synthesizer():
    with _mock_default_mode("lancedb_only"):
        assert _route_after_neo({"retrieval_mode": "neo4j_only"}) == NODE_SYNTHESIZER


def test_route_after_neo_mode3_goes_to_synthesizer():
    """Mode 3: Neo4j ran second; next is synthesizer."""
    with _mock_default_mode("lancedb_only"):
        assert _route_after_neo({"retrieval_mode": "lancedb_then_neo4j"}) == NODE_SYNTHESIZER


def test_route_after_neo_mode5_goes_to_synthesizer():
    with _mock_default_mode("lancedb_only"):
        assert _route_after_neo({"retrieval_mode": "parallel_fused"}) == NODE_SYNTHESIZER


# ---- routed_mode mid-flight precedence ----


def test_post_classifier_routing_honours_routed_mode_not_retrieval_mode():
    """After classifier, retrieval_mode is still 'auto' but routed_mode is
    the real choice. _route_after_lance must dispatch on routed_mode."""
    with _mock_default_mode("lancedb_only"):
        state = {
            "retrieval_mode": "auto",
            "routed_mode": "lancedb_then_neo4j",
        }
        assert _route_after_lance(state) == NODE_CYPHER_BUILDER
