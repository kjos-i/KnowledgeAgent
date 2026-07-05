"""Evaluation top-tab GUI — a full-window coordinator + 5 sub-tabs.

Renders the evaluation harness: the Run sub-tab calls
`knowledge_agent.evaluation.runner.run(cfg)` in-process, and the result
sub-tabs read the SQLite ledger via the `EvalLedger` readers. This package
is presentation only — it never reimplements harness logic.

NB: distinct from `knowledge_agent.evaluation` (the backend harness). This
is `knowledge_agent.gui.evaluation` — the GUI that renders it.
"""

from __future__ import annotations

from knowledge_agent.gui.evaluation.evaluation_view import EvaluationView

__all__ = ["EvaluationView"]
