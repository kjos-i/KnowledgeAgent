"""E2E backend journey — write a node, then run the compiled agent graph
end to end and read the result back.

Seeds the e2e tier with a REAL full-stack journey that drives the same
backend the GUI calls, per this tier's convention (no headless-Flet driver
exists — see conftest). The chosen path is deliberately LLM-FREE and
key-free so it exercises the whole graph against nothing but the test Neo4j
instance:

    retrieval_mode = "neo4j_only"   -> START routes to cypher_builder
    user_cypher    = "MATCH ..."    -> cypher_builder returns it VERBATIM
                                       (no LLM, no corpus_config; nodes.py)
    direct_retrieval = True         -> synthesizer returns retriever output
                                       (no LLM)

So the journey is: MERGE an :Entity -> graph.ainvoke -> neo4j_retriever runs
the read-only Cypher -> synthesizer packages kg_hits -> assert the node
comes back. Everything runs in ONE event loop (this async test) to sidestep
cross-loop driver reuse.

Opt-in: marked `e2e`, skipped by the default run. Needs the test Neo4j
instance up (`.env.test`, loaded autouse by the conftest). Run with:
    pytest -m e2e tests/e2e/test_backend_journey.py
"""

from __future__ import annotations

import pytest

from knowledge_agent.graph import graph
from knowledge_agent.kg.client import Neo4jClient
from knowledge_agent.kg.schema import ENTITY_LABEL

pytestmark = pytest.mark.e2e

_KEY = "e2e-journey-entity"
_TYPE = "Concept"


async def test_neo4j_only_direct_cypher_journey() -> None:
    """Full graph run over the test instance, LLM-free, and read back."""
    client = Neo4jClient()
    try:
        # ---- clean slate + seed one known node (setup writes are ours, not
        # the graph's — the graph only ever runs the read-only user_cypher).
        async with client.driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
            await session.run(
                f"MERGE (e:{ENTITY_LABEL} {{key: $key}}) SET e.entity_type = $etype",
                key=_KEY,
                etype=_TYPE,
            )

        # ---- run the whole graph: neo4j_only + verbatim Cypher + no synth.
        final_state = await graph.ainvoke(
            {
                "query": "find the seeded entity",
                "retrieval_mode": "neo4j_only",
                "user_cypher": (
                    f"MATCH (e:{ENTITY_LABEL} {{key: '{_KEY}'}}) "
                    "RETURN e.key AS key, e.entity_type AS entity_type"
                ),
                "direct_retrieval": True,
            }
        )

        # ---- the read-only Cypher ran cleanly ...
        assert final_state.get("kg_retrieval_error") is None
        # ... and the seeded node came back through the retriever.
        kg_hits = final_state.get("kg_hits") or []
        assert any(h.data.get("key") == _KEY for h in kg_hits)
        # ... and the synthesizer packaged it (direct_retrieval → no LLM,
        # kg hits surface as kg_sources on the answer).
        answer = final_state.get("final_answer")
        assert answer is not None
        assert answer.kg_sources
    finally:
        await client.close()
