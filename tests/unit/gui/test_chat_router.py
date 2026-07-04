"""Tests for `knowledge_agent.gui.chat_router`.

Just the contract bits — the actual LLM call goes through llm_factory
which has its own tests. Here we confirm: (a) the structured-output
schema is what the GUI expects, (b) the cache works, (c) the system
prompt mentions both retrieval stores so the router routes correctly
for KG-only questions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge_agent.gui.chat_router import (
    CHAT_SYSTEM_PROMPT,
    ChatTurnOutput,
    get_chat_router,
)


def test_chat_turn_output_has_response_and_ready_to_retrieve_fields():
    out = ChatTurnOutput(response="ok", ready_to_retrieve=True)
    assert out.response == "ok"
    assert out.ready_to_retrieve is True


def test_chat_turn_output_search_query_defaults_empty_and_is_settable():
    """search_query is empty when not retrieving; the RFA-style router
    fills it with a distilled query when ready_to_retrieve is True."""
    assert ChatTurnOutput(response="hi", ready_to_retrieve=False).search_query == ""
    out = ChatTurnOutput(
        response="looking it up",
        ready_to_retrieve=True,
        search_query="effects of CRISPR on X",
    )
    assert out.search_query == "effects of CRISPR on X"


def test_system_prompt_mentions_both_retrieval_stores():
    """KA uses LanceDB (hybrid) + Neo4j (graph). The router must know
    both exist so it doesn't ask follow-ups like 'I can only answer
    text questions' when a graph question arrives."""
    assert "hybrid" in CHAT_SYSTEM_PROMPT.lower()
    assert "graph" in CHAT_SYSTEM_PROMPT.lower()


def test_system_prompt_instructs_distilled_search_query():
    """RFA-style: the router must produce a self-contained search_query
    distilled from the conversation, not just gate yes/no."""
    low = CHAT_SYSTEM_PROMPT.lower()
    assert "search_query" in low
    assert "self-contained" in low or "distilled" in low


def test_get_chat_router_is_cached_by_temperature():
    """Two calls at the same temperature must share the same Runnable
    — that's the point of the LRU cache (avoid re-binding the
    structured-output schema per turn)."""
    fake_llm = MagicMock()
    fake_runnable = MagicMock(name="bound_runnable")
    fake_llm.with_structured_output.return_value = fake_runnable

    get_chat_router.cache_clear()
    with patch(
        "knowledge_agent.gui.chat_router.get_llm",
        return_value=fake_llm,
    ) as mock_get:
        first = get_chat_router("claude-haiku-4-5-20251001", 0.0)
        second = get_chat_router("claude-haiku-4-5-20251001", 0.0)

    assert first is second
    # The factory was only called once even though we asked for the
    # router twice → cache hit.
    mock_get.assert_called_once()


def test_get_chat_router_dispatches_via_llm_factory():
    """Slice 1 contract: chat router builds on top of llm_factory.get_llm
    so it works under whichever provider the user has configured."""
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = MagicMock()

    get_chat_router.cache_clear()
    with patch(
        "knowledge_agent.gui.chat_router.get_llm",
        return_value=fake_llm,
    ) as mock_get:
        get_chat_router("claude-haiku-4-5-20251001", 0.0)

    mock_get.assert_called_once_with("claude-haiku-4-5-20251001", 0.0)
    fake_llm.with_structured_output.assert_called_once_with(ChatTurnOutput)
