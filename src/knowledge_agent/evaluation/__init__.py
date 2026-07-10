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

import os

# DeepEval (the LLM-judge metric backend) sends anonymous usage stats to
# PostHog and crash reports to Sentry BY DEFAULT. We opt out on the user's
# behalf here — the eval package root, which is imported before any submodule
# (incl. `judge.py`) can import deepeval — so nothing is ever sent off-machine.
# `setdefault` means a user who genuinely wants to contribute telemetry can
# still opt back in by exporting these before launch; the default is off.
# (deepeval reads these as truthy/falsy: OPT_OUT truthy => telemetry off;
# ERROR_REPORTING falsy => Sentry off. Verified against deepeval 4.0.8.)
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")
