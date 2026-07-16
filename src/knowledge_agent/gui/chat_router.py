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

from knowledge_agent.llm_factory import get_llm_ref

CHAT_SYSTEM_PROMPT = """\
You are a research-literature assistant with a corpus of research
papers indexed in a hybrid retrieval store (vector + BM25 over paper
chunks) and a knowledge graph (citations, authors, venues, topics,
entities, triples). You are the conversational layer above the
retrieval pipeline: each turn you decide whether to answer directly or
run a retrieval, and when you retrieve you formulate the search query
the rest of the pipeline runs. (Think of yourself as a stand-in for a
supervisor agent that will one day drive this tool.)

Each turn you produce three fields:
- `response`: what to say to the user (concise).
- `ready_to_retrieve`: bool — whether to run a retrieval this turn.
- `search_query`: when `ready_to_retrieve` is True, a single
  self-contained search query distilled from the WHOLE conversation —
  resolve pronouns ("it", "those", "that paper"), fold in details the
  user gave in earlier turns, and phrase it as a clear standalone
  question. Do NOT just echo the user's raw words. Leave it EMPTY when
  not retrieving.

Set `ready_to_retrieve=True` when the user has a substantive question
the corpus + graph could answer AND it is specific enough to search.
Briefly acknowledge in `response` that you're looking it up, and put
the distilled query in `search_query`.

Set `ready_to_retrieve=False` (and leave `search_query` empty) when:
- The message is greeting / meta / chit-chat ("hi", "thanks", "what
  can you do?").
- The question is too vague or ambiguous to search usefully → ask ONE
  focused clarifying question in `response`.
- The user is discussing or following up on the previous answer
  without asking something new that needs fresh retrieval.

Do NOT try to choose which store to search (vector vs graph) — the
pipeline classifies that downstream. Your job is the conversation plus
a clean, self-contained query. Be brief, ask at most one clarifying
question at a time, and prefer retrieving when a genuine question is
clear.
"""

CHAT_ROUTER_TIMEOUT_SECONDS = 30


class ChatTurnOutput(BaseModel):
    """Structured output of the chat router (its contract, not graph data)."""

    response: str = Field(description="What to say to the user this turn.")
    ready_to_retrieve: bool = Field(
        description=("True when this turn should run a retrieval over the corpus.")
    )
    search_query: str = Field(
        default="",
        description=(
            "When ready_to_retrieve is True, a single self-contained search "
            "query distilled from the whole conversation (pronouns resolved, "
            "earlier context folded in). Empty when not retrieving."
        ),
    )


@lru_cache(maxsize=8)
def get_chat_router(model: str, temperature: float = 0.0) -> Any:
    """Return the cached chat-router Runnable (LLM + structured-output).

    `model` + `temperature` are the router's OWN config, passed in by the
    GUI from `GuiConfig`. The router is a GUI-only stand-in for a future
    supervisor agent, so its model choice lives in the GUI, NOT in the
    backend Settings (which is the graph/tool's config — the tool must not
    carry a piece of its own caller). Builds on the shared
    `llm_factory.get_llm_ref`, which routes to the provider named in the model
    ref (or the active provider for a bare / legacy ref).

    Keyed by (model, temperature) so changing either yields a distinct
    cached client.
    """
    llm = get_llm_ref(model, temperature)
    return llm.with_structured_output(ChatTurnOutput)
