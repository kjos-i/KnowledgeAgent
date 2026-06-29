"""Tests for ingestion-agnostic node functions in `nodes.py`.

LLM calls are mocked via `unittest.mock.patch` on `_get_llm`; the LanceDB
client is mocked via `patch` on `get_search_client`. Tests use
`asyncio.run` for the async nodes rather than pytest-asyncio (no extra
dep, simple invocation in the test body).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.kg.corpus_config import CorpusConfig, LayerFlags
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
from knowledge_agent.nodes import (
    _extract_doc_ids_from_kg_hits,
    _format_chunks_for_prompt,
    _format_kg_hits_for_prompt,
    cypher_builder_node,
    lancedb_retriever_node,
    mode_classifier_node,
    neo4j_retriever_node,
    query_builder_node,
    synthesizer_node,
)

# Default test corpus config - every layer on, so the rendered schema
# matches what cypher_builder tests historically asserted (:Document,
# :Author, :CITES, :AUTHORED, etc.). Tests that exercise layer-gating
# in the schema renderer live in test_schema_as_prompt.py; here we just
# need a valid config for cypher_builder to do its work.
_TEST_CONFIG = CorpusConfig(
    domain="test",
    layers=LayerFlags(openalex_papers=True, chunks=True),
)


# ---- helpers ----


def _chunk(
    index: int,
    text: str = "body text",
    title: str | None = None,
    year: int | None = None,
    authors_display: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc#{index}",
        doc_id="doc",
        text=text,
        title=title,
        year=year,
        authors_display=authors_display,
    )


def _mock_llm_returning(structured_output):
    """Build the mock chain: llm.with_structured_output(...).ainvoke(...) -> output.

    `with_retry` is wired so that the production code's
    `_with_retry(structured)` call resolves to the same mock — the retry
    wrapper is the identity in tests. Retry behaviour is covered in
    isolation by `test_llm_factory.with_retry` tests.
    """
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(return_value=structured_output)
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    return mock_llm


# ---- mode_classifier_node ----


def test_mode_classifier_returns_routed_mode_from_llm():
    choice = ModeChoice(mode="neo4j_only")
    mock_llm = _mock_llm_returning(choice)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.mode_classifier_model = "claude-haiku"
        mock_settings.return_value.mode_classifier_temperature = 0.0
        result = asyncio.run(
            mode_classifier_node({"query": "Who cites paper W123?"})
        )
    assert result == {"routed_mode": "neo4j_only"}


def test_mode_classifier_calls_llm_with_mode_choice_schema():
    choice = ModeChoice(mode="lancedb_only")
    mock_llm = _mock_llm_returning(choice)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.mode_classifier_model = "claude-haiku"
        mock_settings.return_value.mode_classifier_temperature = 0.0
        asyncio.run(mode_classifier_node({"query": "q"}))
    mock_llm.with_structured_output.assert_called_once_with(ModeChoice)


def test_mode_classifier_passes_user_query_in_human_message():
    choice = ModeChoice(mode="lancedb_only")
    mock_llm = _mock_llm_returning(choice)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.mode_classifier_model = "claude-haiku"
        mock_settings.return_value.mode_classifier_temperature = 0.0
        asyncio.run(
            mode_classifier_node({"query": "Who is the most cited author?"})
        )
    human_content = mock_structured.ainvoke.call_args.args[0][1].content
    assert human_content == "Who is the most cited author?"


def test_mode_classifier_uses_settings_model_and_temperature():
    choice = ModeChoice(mode="lancedb_only")
    mock_llm = _mock_llm_returning(choice)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ) as mock_get_llm,
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.mode_classifier_model = "test-model"
        mock_settings.return_value.mode_classifier_temperature = 0.5
        asyncio.run(mode_classifier_node({"query": "q"}))
    mock_get_llm.assert_called_once_with("test-model", 0.5)


def test_mode_classifier_fail_soft_to_lancedb_only_on_exception():
    """LLM call raises -> log + return routed_mode='lancedb_only'."""
    mock_structured = MagicMock()
    mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("api down"))
    mock_structured.with_retry = MagicMock(return_value=mock_structured)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_structured)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.mode_classifier_model = "claude-haiku"
        mock_settings.return_value.mode_classifier_temperature = 0.0
        result = asyncio.run(mode_classifier_node({"query": "q"}))
    assert result == {"routed_mode": "lancedb_only"}


# ---- query_builder_node ----


def test_query_builder_skip_via_state_uses_raw_query():
    state = {"query": "raw question", "skip_query_builder": True}
    result = asyncio.run(query_builder_node(state))
    assert result == {"search_query": "raw question"}


def test_query_builder_skip_via_settings_when_state_silent():
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.skip_query_builder = True
        result = asyncio.run(query_builder_node({"query": "raw"}))
    assert result == {"search_query": "raw"}


def test_query_builder_state_false_beats_settings_true():
    """Explicit per-invocation override wins over the settings default."""
    with (
        patch("knowledge_agent.nodes._get_llm") as mock_get_llm,
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.skip_query_builder = True
        mock_settings.return_value.query_builder_model = "claude-haiku"
        mock_settings.return_value.query_builder_temperature = 0.0
        mock_get_llm.return_value = _mock_llm_returning(
            SearchQueryRewrite(search_query="rewritten")
        )
        state = {"query": "raw", "skip_query_builder": False}
        result = asyncio.run(query_builder_node(state))
    assert result == {"search_query": "rewritten"}


def test_query_builder_calls_llm_with_structured_output():
    rewrite = SearchQueryRewrite(search_query="apoptosis cancer cells")
    mock_llm = _mock_llm_returning(rewrite)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.skip_query_builder = False
        mock_settings.return_value.query_builder_model = "claude-haiku"
        mock_settings.return_value.query_builder_temperature = 0.0
        result = asyncio.run(
            query_builder_node(
                {"query": "What does the literature say about apoptosis?"}
            )
        )
    mock_llm.with_structured_output.assert_called_once_with(
        SearchQueryRewrite
    )
    assert result == {"search_query": "apoptosis cancer cells"}


# ---- cypher_builder_node ----


def test_cypher_builder_calls_llm_with_structured_output():
    rewrite = CypherQueryRewrite(
        cypher_query="MATCH (d:Document) RETURN d LIMIT 5"
    )
    mock_llm = _mock_llm_returning(rewrite)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        result = asyncio.run(
            cypher_builder_node(
                {"query": "Who authored paper W123?", "corpus_config": _TEST_CONFIG}
            )
        )
    mock_llm.with_structured_output.assert_called_once_with(CypherQueryRewrite)
    assert result == {"cypher_query": "MATCH (d:Document) RETURN d LIMIT 5"}


def test_cypher_builder_raises_when_corpus_config_missing():
    """No silent default - if the caller forgets to seed corpus_config in
    the state, cypher_builder fails loudly rather than rendering a schema
    for layers that may not exist in this corpus."""
    with pytest.raises(ValueError, match="corpus_config"):
        asyncio.run(cypher_builder_node({"query": "q"}))


def test_cypher_builder_passes_schema_in_system_message():
    """KG schema must reach the system prompt - and the placeholder must go."""
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        asyncio.run(
            cypher_builder_node({"query": "q", "corpus_config": _TEST_CONFIG})
        )
    messages = mock_structured.ainvoke.call_args.args[0]
    system_msg_content = messages[0].content
    # Schema constants must appear (proves format_schema_for_prompt was injected).
    assert ":Document" in system_msg_content
    assert ":Author" in system_msg_content
    assert ":CITES" in system_msg_content
    assert ":AUTHORED" in system_msg_content
    # Placeholder must be gone (proves .replace ran).
    assert "<SCHEMA>" not in system_msg_content


def test_cypher_builder_passes_user_query_in_human_message():
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        asyncio.run(
            cypher_builder_node(
                {"query": "Who authored paper W999?", "corpus_config": _TEST_CONFIG}
            )
        )
    messages = mock_structured.ainvoke.call_args.args[0]
    human_msg_content = messages[1].content
    assert "Who authored paper W999?" in human_msg_content


def test_cypher_builder_uses_settings_model_and_temperature():
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ) as mock_get_llm,
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "test-model"
        mock_settings.return_value.cypher_builder_temperature = 0.3
        asyncio.run(
            cypher_builder_node({"query": "q", "corpus_config": _TEST_CONFIG})
        )
    mock_get_llm.assert_called_once_with("test-model", 0.3)


def test_cypher_builder_non_cross_store_omits_doc_id_rule():
    """Modes 1+2: system prompt does NOT contain the doc_id RETURN rule."""
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "neo4j_only",
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    system_content = mock_structured.ainvoke.call_args.args[0][0].content
    assert "MUST include `doc_id`" not in system_content


def test_cypher_builder_mode3_injects_doc_id_rule():
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "lancedb_then_neo4j",
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    system_content = mock_structured.ainvoke.call_args.args[0][0].content
    assert "MUST include `doc_id`" in system_content


def test_cypher_builder_mode4_injects_doc_id_rule():
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "neo4j_then_lancedb",
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    system_content = mock_structured.ainvoke.call_args.args[0][0].content
    assert "MUST include `doc_id`" in system_content


def test_cypher_builder_mode3_prepends_lance_hits_to_user_message():
    """Mode 3 + chunks present -> user msg has the chunks + the question."""
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    chunks = [_chunk(0)]
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "What about CRISPR?",
            "retrieval_mode": "lancedb_then_neo4j",
            "retrieved_chunks": chunks,
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    user_content = mock_structured.ainvoke.call_args.args[0][1].content
    assert "doc_id=doc" in user_content
    assert "User question: What about CRISPR?" in user_content


def test_cypher_builder_mode3_with_no_chunks_uses_plain_user_query():
    """Mode 3 but Lance returned nothing -> user msg is just the question."""
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "What about CRISPR?",
            "retrieval_mode": "lancedb_then_neo4j",
            "retrieved_chunks": [],
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    user_content = mock_structured.ainvoke.call_args.args[0][1].content
    assert user_content == "What about CRISPR?"


def test_cypher_builder_mode4_does_not_prepend_lance_hits():
    """Mode 4: only system rule changes; user msg is plain question even if
    retrieved_chunks happens to be set (in mode 4 Lance hasn't run yet)."""
    rewrite = CypherQueryRewrite(cypher_query="RETURN 1")
    mock_llm = _mock_llm_returning(rewrite)
    mock_structured = mock_llm.with_structured_output.return_value
    chunks = [_chunk(0)]
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.cypher_builder_model = "claude-sonnet"
        mock_settings.return_value.cypher_builder_temperature = 0.0
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "What about CRISPR?",
            "retrieval_mode": "neo4j_then_lancedb",
            "retrieved_chunks": chunks,
            "corpus_config": _TEST_CONFIG,
        }
        asyncio.run(cypher_builder_node(state))
    user_content = mock_structured.ainvoke.call_args.args[0][1].content
    assert user_content == "What about CRISPR?"


# ---- neo4j_retriever_node ----


def test_neo4j_retriever_no_cypher_query_is_noop():
    """No cypher_query in state -> return {} without touching the KG."""
    mock_client = MagicMock()
    with patch(
        "knowledge_agent.nodes.get_kg_client",
        return_value=mock_client,
    ):
        result = asyncio.run(neo4j_retriever_node({"query": "q"}))
    assert result == {}
    mock_client.read_query.assert_not_called()


def test_neo4j_retriever_empty_cypher_query_is_noop():
    mock_client = MagicMock()
    with patch(
        "knowledge_agent.nodes.get_kg_client",
        return_value=mock_client,
    ):
        result = asyncio.run(
            neo4j_retriever_node({"query": "q", "cypher_query": ""})
        )
    assert result == {}
    mock_client.read_query.assert_not_called()


def test_neo4j_retriever_rejects_unsafe_cypher():
    """Forbidden keyword in cypher -> empty kg_hits without touching the KG."""
    mock_client = MagicMock()
    with patch(
        "knowledge_agent.nodes.get_kg_client",
        return_value=mock_client,
    ):
        result = asyncio.run(
            neo4j_retriever_node(
                {"query": "q", "cypher_query": "MATCH (n) DELETE n"}
            )
        )
    assert result == {"kg_hits": []}
    mock_client.read_query.assert_not_called()


def test_neo4j_retriever_wraps_cypher_with_limit():
    """The wrapped Cypher (CALL { ... } RETURN * LIMIT N) is sent to read_query."""
    mock_client = MagicMock()
    mock_client.read_query = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_kg_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.kg_max_rows = 50
        asyncio.run(
            neo4j_retriever_node(
                {
                    "query": "q",
                    "cypher_query": "MATCH (d:Document) RETURN d",
                }
            )
        )
    cypher_sent = mock_client.read_query.call_args.args[0]
    assert "CALL {" in cypher_sent
    assert "MATCH (d:Document) RETURN d" in cypher_sent
    assert "RETURN * LIMIT 50" in cypher_sent


def test_neo4j_retriever_uses_settings_kg_max_rows():
    mock_client = MagicMock()
    mock_client.read_query = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_kg_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.kg_max_rows = 7
        asyncio.run(
            neo4j_retriever_node(
                {"query": "q", "cypher_query": "MATCH (n) RETURN n"}
            )
        )
    cypher_sent = mock_client.read_query.call_args.args[0]
    assert "LIMIT 7" in cypher_sent


def test_neo4j_retriever_returns_rows_as_kg_hits():
    """read_query rows (dicts) become KGHit objects in state["kg_hits"]."""
    mock_client = MagicMock()
    mock_client.read_query = AsyncMock(
        return_value=[
            {"name": "Alice", "papers": 5},
            {"name": "Bob", "papers": 3},
        ]
    )
    with (
        patch(
            "knowledge_agent.nodes.get_kg_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.kg_max_rows = 50
        result = asyncio.run(
            neo4j_retriever_node(
                {"query": "q", "cypher_query": "MATCH (a:Author) RETURN a"}
            )
        )
    hits = result["kg_hits"]
    assert len(hits) == 2
    assert all(isinstance(h, KGHit) for h in hits)
    assert hits[0].data == {"name": "Alice", "papers": 5}
    assert hits[1].data == {"name": "Bob", "papers": 3}


def test_neo4j_retriever_fail_soft_on_exception():
    """Any read_query exception is swallowed -> empty kg_hits."""
    mock_client = MagicMock()
    mock_client.read_query = AsyncMock(
        side_effect=RuntimeError("connection refused")
    )
    with (
        patch(
            "knowledge_agent.nodes.get_kg_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.kg_max_rows = 50
        result = asyncio.run(
            neo4j_retriever_node(
                {"query": "q", "cypher_query": "MATCH (n) RETURN n"}
            )
        )
    # Empty hits + typed-error detail populated for the synthesizer / UI.
    assert result["kg_hits"] == []
    err = result["kg_retrieval_error"]
    assert err is not None
    assert "connection refused" in err.message
    assert err.exception_type == "builtins.RuntimeError"


# ---- lancedb_retriever_node ----


def test_retriever_prefers_rewritten_search_query():
    hits = [_chunk(0)]
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=hits)
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {"query": "raw", "search_query": "rewritten"}
        result = asyncio.run(lancedb_retriever_node(state))
    mock_client.retrieve.assert_called_once_with(
        query="rewritten", top_k=5, filters=None,
    )
    assert result == {"retrieved_chunks": hits}


def test_retriever_falls_back_to_raw_query_when_no_search_query():
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        asyncio.run(lancedb_retriever_node({"query": "raw"}))
    mock_client.retrieve.assert_called_once_with(
        query="raw", top_k=5, filters=None,
    )


def test_retriever_honours_state_top_k_override():
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {"query": "raw", "search_query": "rewritten", "top_k": 20}
        asyncio.run(lancedb_retriever_node(state))
    call_kwargs = mock_client.retrieve.call_args.kwargs
    assert call_kwargs["query"] == "rewritten"
    assert call_kwargs["top_k"] == 20


def test_retriever_mode4_extracts_doc_ids_from_kg_hits_and_filters():
    """Mode neo4j_then_lancedb -> doc_id IN (...) filter applied."""
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    kg_hits = [
        KGHit(data={"doc_id": "W1", "x": 1}),
        KGHit(data={"doc_id": "W2"}),
    ]
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "search_query": "rewritten",
            "retrieval_mode": "neo4j_then_lancedb",
            "kg_hits": kg_hits,
        }
        asyncio.run(lancedb_retriever_node(state))
    call_kwargs = mock_client.retrieve.call_args.kwargs
    assert call_kwargs["filters"] == {"doc_id": ["W1", "W2"]}


def test_retriever_mode4_empty_kg_hits_falls_back_to_unfiltered():
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "neo4j_then_lancedb",
            "kg_hits": [],
        }
        asyncio.run(lancedb_retriever_node(state))
    call_kwargs = mock_client.retrieve.call_args.kwargs
    assert call_kwargs["filters"] is None


def test_retriever_mode4_no_doc_ids_in_hits_falls_back_to_unfiltered():
    """KG returned rows but none have doc_id key -> log + unfiltered."""
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    kg_hits = [
        KGHit(data={"author_name": "Smith"}),
        KGHit(data={"paper_count": 5}),
    ]
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "neo4j_then_lancedb",
            "kg_hits": kg_hits,
        }
        asyncio.run(lancedb_retriever_node(state))
    call_kwargs = mock_client.retrieve.call_args.kwargs
    assert call_kwargs["filters"] is None


def test_retriever_non_mode4_does_not_filter_even_with_kg_hits():
    """Mode lancedb_only: kg_hits in state is ignored, no filter applied."""
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(return_value=[])
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.top_k = 5
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        state = {
            "query": "q",
            "retrieval_mode": "lancedb_only",
            "kg_hits": [KGHit(data={"doc_id": "W1"})],
        }
        asyncio.run(lancedb_retriever_node(state))
    call_kwargs = mock_client.retrieve.call_args.kwargs
    assert call_kwargs["filters"] is None


def test_lancedb_retriever_captures_typed_error_on_failure():
    """Lance / Voyage failures now propagate from `client.retrieve()`;
    the node catches and populates `lancedb_retrieval_error`."""
    mock_client = MagicMock()
    mock_client.retrieve = AsyncMock(
        side_effect=RuntimeError("voyage outage")
    )
    with (
        patch(
            "knowledge_agent.nodes.get_search_client",
            return_value=mock_client,
        ),
        patch("knowledge_agent.nodes.get_settings") as mock_settings,
    ):
        mock_settings.return_value.top_k = 10
        mock_settings.return_value.default_retrieval_mode = "lancedb_only"
        result = asyncio.run(lancedb_retriever_node({"query": "q"}))
    assert result["retrieved_chunks"] == []
    err = result["lancedb_retrieval_error"]
    assert err is not None
    assert "voyage outage" in err.message
    assert err.exception_type == "builtins.RuntimeError"


# ---- synthesizer_node ----


def test_synthesizer_direct_via_state_returns_empty_answer_with_sources():
    chunks = [_chunk(0), _chunk(1)]
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.direct_retrieval = False
        state = {
            "query": "q",
            "retrieved_chunks": chunks,
            "direct_retrieval": True,
        }
        result = asyncio.run(synthesizer_node(state))
    answer = result["final_answer"]
    assert isinstance(answer, AgentAnswer)
    assert answer.answer == ""
    assert len(answer.chunk_sources) == 2
    assert answer.chunk_sources[0].chunk_id == "doc#0"
    assert answer.chunk_sources[1].chunk_id == "doc#1"


def test_synthesizer_direct_via_settings_when_state_silent():
    chunks = [_chunk(0)]
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.direct_retrieval = True
        result = asyncio.run(
            synthesizer_node({"query": "q", "retrieved_chunks": chunks})
        )
    assert result["final_answer"].answer == ""
    assert len(result["final_answer"].chunk_sources) == 1


def test_synthesizer_calls_llm_with_agent_answer_schema():
    answer_obj = AgentAnswer(
        answer="The literature shows [1].",
        chunk_sources=[ChunkSource(chunk_id="doc#0", doc_id="doc")],
    )
    mock_llm = _mock_llm_returning(answer_obj)
    chunks = [_chunk(0)]
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.direct_retrieval = False
        mock_settings.return_value.synthesizer_model = "claude-sonnet"
        mock_settings.return_value.synthesizer_temperature = 0.0
        result = asyncio.run(
            synthesizer_node({"query": "q", "retrieved_chunks": chunks})
        )
    mock_llm.with_structured_output.assert_called_once_with(AgentAnswer)
    assert result == {"final_answer": answer_obj}


def test_synthesizer_direct_with_no_chunks_returns_empty_sources():
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.direct_retrieval = True
        result = asyncio.run(synthesizer_node({"query": "q"}))
    assert result["final_answer"].answer == ""
    assert result["final_answer"].chunk_sources == []
    assert result["final_answer"].kg_sources == []


def test_synthesizer_direct_with_only_kg_hits():
    kg_hits = [KGHit(data={"x": 1}), KGHit(data={"y": 2})]
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.direct_retrieval = True
        result = asyncio.run(
            synthesizer_node({"query": "q", "kg_hits": kg_hits})
        )
    answer = result["final_answer"]
    assert answer.answer == ""
    assert answer.chunk_sources == []
    assert len(answer.kg_sources) == 2
    assert answer.kg_sources[0].hit_index == 0
    assert answer.kg_sources[1].hit_index == 1


def test_synthesizer_direct_with_both_chunks_and_kg_hits():
    chunks = [_chunk(0)]
    kg_hits = [KGHit(data={"x": 1})]
    with patch(
        "knowledge_agent.nodes.get_settings"
    ) as mock_settings:
        mock_settings.return_value.direct_retrieval = True
        state = {
            "query": "q",
            "retrieved_chunks": chunks,
            "kg_hits": kg_hits,
        }
        result = asyncio.run(synthesizer_node(state))
    answer = result["final_answer"]
    assert len(answer.chunk_sources) == 1
    assert answer.chunk_sources[0].chunk_id == "doc#0"
    assert len(answer.kg_sources) == 1
    assert answer.kg_sources[0].hit_index == 0


def test_synthesizer_llm_branch_passes_kg_hits_to_user_message():
    """kg_hits must reach the LLM prompt so it can cite them."""
    answer_obj = AgentAnswer(
        answer="The KG shows [K0].",
        chunk_sources=[],
        kg_sources=[KGSource(hit_index=0)],
    )
    mock_llm = _mock_llm_returning(answer_obj)
    mock_structured = mock_llm.with_structured_output.return_value
    kg_hits = [KGHit(data={"title": "Unique Title XYZ"})]
    with (
        patch(
            "knowledge_agent.nodes._get_llm", return_value=mock_llm
        ),
        patch(
            "knowledge_agent.nodes.get_settings"
        ) as mock_settings,
    ):
        mock_settings.return_value.direct_retrieval = False
        mock_settings.return_value.synthesizer_model = "claude-sonnet"
        mock_settings.return_value.synthesizer_temperature = 0.0
        result = asyncio.run(
            synthesizer_node({"query": "q", "kg_hits": kg_hits})
        )
    # Verify the row data actually reached the prompt.
    messages = mock_structured.ainvoke.call_args.args[0]
    human_msg_content = messages[1].content
    assert "Unique Title XYZ" in human_msg_content
    assert "[K0]" in human_msg_content
    # And the LLM's structured output flowed back through unchanged.
    assert result["final_answer"] is answer_obj
    assert result["final_answer"].kg_sources[0].hit_index == 0


# ---- _format_chunks_for_prompt ----


def test_format_chunks_empty_returns_placeholder():
    assert _format_chunks_for_prompt([]) == "(no chunks retrieved)"


def test_format_chunks_numbers_from_one():
    out = _format_chunks_for_prompt([_chunk(0, "first"), _chunk(1, "second")])
    assert "[1]" in out
    assert "[2]" in out


def test_format_chunks_includes_required_identity_fields():
    out = _format_chunks_for_prompt([_chunk(0)])
    assert "chunk_id=doc#0" in out
    assert "doc_id=doc" in out


def test_format_chunks_includes_optional_metadata_when_set():
    out = _format_chunks_for_prompt(
        [_chunk(0, title="Test Paper", year=2024, authors_display="Smith, J.")]
    )
    assert "title=" in out
    assert "year=2024" in out
    assert "authors=" in out


def test_format_chunks_omits_optional_metadata_when_missing():
    out = _format_chunks_for_prompt([_chunk(0)])
    assert "title=" not in out
    assert "year=" not in out
    assert "authors=" not in out


def test_format_chunks_includes_body_text():
    out = _format_chunks_for_prompt([_chunk(0, text="body content")])
    assert "body content" in out


# ---- _format_kg_hits_for_prompt ----


def test_format_kg_hits_empty_returns_placeholder():
    assert _format_kg_hits_for_prompt([]) == "(no kg hits retrieved)"


def test_format_kg_hits_numbers_from_zero():
    out = _format_kg_hits_for_prompt(
        [KGHit(data={"x": 1}), KGHit(data={"y": 2})]
    )
    assert "[K0]" in out
    assert "[K1]" in out


def test_format_kg_hits_renders_data_as_key_value_pairs():
    out = _format_kg_hits_for_prompt(
        [KGHit(data={"title": "Paper A", "year": 2024})]
    )
    assert "title=" in out
    assert "Paper A" in out
    assert "year=2024" in out


# ---- _extract_doc_ids_from_kg_hits ----


def test_extract_doc_ids_empty_returns_empty():
    assert _extract_doc_ids_from_kg_hits([]) == []


def test_extract_doc_ids_dedupes_preserving_first_appearance_order():
    hits = [
        KGHit(data={"doc_id": "W1"}),
        KGHit(data={"doc_id": "W2"}),
        KGHit(data={"doc_id": "W1"}),
        KGHit(data={"doc_id": "W3"}),
    ]
    assert _extract_doc_ids_from_kg_hits(hits) == ["W1", "W2", "W3"]


def test_extract_doc_ids_skips_missing_or_empty():
    hits = [
        KGHit(data={"doc_id": "W1"}),
        KGHit(data={"author": "Smith"}),
        KGHit(data={"doc_id": ""}),
        KGHit(data={"doc_id": "W2"}),
    ]
    assert _extract_doc_ids_from_kg_hits(hits) == ["W1", "W2"]
