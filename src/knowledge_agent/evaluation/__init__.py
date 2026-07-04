"""Evaluation harness for the KnowledgeAgent retrieval graph.

A registry-driven RAG-evaluation harness ported + adapted from the
HybridSearchAgent reference. Three layers, provider-agnostic by
construction:

  1. Node layer (already in `knowledge_agent`): every graph node runs
     through `init_chat_model` + `.with_structured_output(...)`, so the
     agent's output reaches this harness as TYPED state — never raw
     provider JSON.
  2. Adapter (`adapter.py`): invokes the graph, attaches a usage
     callback, and reads the typed final state into one normalized
     per-case dict.
  3. Metrics: a deterministic track (`metrics.py`) and — from Phase 3 —
     a DeepEval judge track (`judge.py`), both consuming that dict.

Results land in an auto-migrating SQLite ledger (`ledger.py`) plus
timestamped JSON/CSV reports (`report.py`). Run via the CLI:

    python -m knowledge_agent.evaluation.runner

Everything reads its metric names / columns / formats from the single
source of truth in `registry.py`.
"""

from __future__ import annotations
