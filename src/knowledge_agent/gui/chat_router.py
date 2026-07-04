"""Dialogue-routing layer for the desktop GUI.

A small LLM that sits *above* the agent graph: reads the running
conversation and decides whether to (a) answer/clarify directly or
(b) trigger a retrieval via the full graph. The graph is unchanged —
the router is purely about *when* to invoke it.

Lives in `gui/` because the GUI is its only consumer (the CLI is
one-shot per command; eval harness drives the graph directly).

Mirrors `research_articles_agent/gui/chat_router.py` so the two apps'
chat UX feels the same.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from knowledge_agent.config import get_settings
from knowledge_agent.llm_factory import get_llm

CHAT_SYSTEM_PROMPT = """\
You are a research literature assistant. The user asks questions that
are answered from a corpus of research papers indexed in a hybrid
retrieval store (vector + BM25 over chunks) and a knowledge graph
(citations, authors, venues, topics, entities, triples). You decide,
each turn, whether to retrieve from the corpus or to reply without
retrieving.

Each turn you produce two fields:
- `response`: what to say to the user (concise).
- `ready_to_retrieve`: bool — whether to run a retrieval this turn.

Set `ready_to_retrieve=True` when the user has asked a substantive
question that the corpus + graph could answer (the common case).
Briefly acknowledge that you're looking it up in `response`.

Set `ready_to_retrieve=False` only when retrieval doesn't apply:
- The message is greeting/meta/chit-chat ("hi", "thanks", "what can
  you do?").
- The question is too vague or ambiguous to retrieve usefully → ask
  ONE focused clarifying question.
- The user is discussing or following up on the previous answer
  without asking something new that needs fresh retrieval.

Guidelines: be brief, no long preambles, ask at most one clarifying
question at a time, and prefer retrieving when in doubt about a
genuine question.
"""

CHAT_ROUTER_TIMEOUT_SECONDS = 30


class ChatTurnOutput(BaseModel):
    """Structured output of the chat router (its contract, not graph data)."""

    response: str = Field(description="What to say to the user this turn.")
    ready_to_retrieve: bool = Field(
        description=("True when this turn should run a retrieval over the corpus.")
    )


@lru_cache(maxsize=8)
def get_chat_router(temperature: float = 0.0) -> Any:
    """Return the cached chat-router Runnable (LLM + structured-output).

    Lazy + cached so the module imports without API keys (test
    discovery). Uses the project's `llm_factory.get_llm` so the router
    runs under whichever provider the user has configured — works on
    Anthropic, OpenAI, Google, or Ollama identically.

    Keyed by temperature so different temperatures get distinct cached
    clients. The chat router uses `mode_classifier_model` (the cheap
    tier) — routing is a light decision task.
    """
    settings = get_settings()
    llm = get_llm(settings.mode_classifier_model, temperature)
    return llm.with_structured_output(ChatTurnOutput)
