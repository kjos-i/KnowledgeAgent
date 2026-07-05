"""Shared helpers for the Evaluation GUI sub-tabs.

One definition of "which corpus is active" and "its eval ledger", so the Run
tab (writer) and the Trends / Run Summary / Deep Analysis tabs (readers)
always resolve to the SAME per-corpus `eval_output/` — the folder beside the
corpus's `lancedb`, which `EvalConfig` derives from `corpus_config_path`. If
these drifted, the readers would show an empty CWD ledger while runs wrote
into the corpus folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knowledge_agent.evaluation.ledger import EvalLedger
    from knowledge_agent.gui.app import GuiApp


def active_corpus_config_path(app: GuiApp) -> Path | None:
    """The active corpus's `corpus.toml` path, or None when no corpus is set.

    Single source for wiring the active corpus into an `EvalConfig`; the eval
    output dir is derived from its parent folder (beside `lancedb`).
    """
    raw = getattr(app.gui_config, "corpus_config_path", None)
    return Path(raw) if raw else None


def active_eval_ledger(app: GuiApp) -> EvalLedger:
    """The `EvalLedger` for the active corpus's `eval_output/` (CWD fallback
    when no corpus is set). Backs the three read tabs so they see exactly the
    runs the Run tab wrote — same corpus-folder derivation, one code path.
    """
    from knowledge_agent.evaluation.config import load_eval_config
    from knowledge_agent.evaluation.ledger import EvalLedger

    corpus = active_corpus_config_path(app)
    overrides = {"corpus_config_path": corpus} if corpus else {}
    return EvalLedger(load_eval_config(**overrides).ledger_path)
