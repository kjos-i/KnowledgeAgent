"""Shared helpers for the Evaluation GUI sub-tabs.

One definition of "which corpus is active" and "its eval ledger", so the Run
tab (writer) and the Trends / Run Summary / Run Charts tabs (readers)
always resolve to the SAME per-corpus `eval_output/` — the folder beside the
corpus's `lancedb`, which `EvalConfig` derives from `corpus_config_path`. If
these drifted, the readers would show an empty CWD ledger while runs wrote
into the corpus folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.evaluation.ledger import EvalLedger
    from knowledge_agent.gui.app import GuiApp

# Case-origin filter (the dashboard's shared left-rail checkboxes). The four
# provenance buckets a case can carry; ticking a subset restricts every per-run
# view (Run Summary / Run Charts / Compare — NOT Trends) to those cases, and the
# run's metrics are recomputed over just them. All ticked = no filter.
ALL_ORIGINS: tuple[str, ...] = ("manual", "llm", "search", "chat")
ORIGIN_LABELS: dict[str, str] = {
    "manual": "Manual",
    "llm": "LLM",
    "search": "Search",
    "chat": "Chat",
}


def filtered_run(
    app: GuiApp, run_id: int, origins: set[str] | None
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """A run's summary row + its per-case rows, restricted to cases whose origin
    is in `origins`.

    `origins=None` or all-selected → the STORED run row + all cases (no recompute
    — fast and exact). Otherwise the run-level metrics are recomputed over only
    the filtered cases via `report.build_summary` (SSOT: the SAME aggregation that
    produced the stored averages, run over a subset), overlaid onto the run row
    so its provenance columns (dataset, hashes, groups, thresholds…) stay intact.
    A legacy case with no recorded origin counts as 'manual' (the default). None
    run → (None, []).
    """
    ledger = active_eval_ledger(app)
    run = ledger.get_run(run_id)
    if run is None:
        return None, []
    cases = ledger.get_run_cases(run_id)
    if origins is None or set(origins) >= set(ALL_ORIGINS):
        return run, cases
    keep = set(origins)
    filtered = [c for c in cases if (c.get("origin") or "manual") in keep]
    from knowledge_agent.evaluation.report import build_summary

    return {**run, **build_summary(filtered)}, filtered


def confirm_dialog(
    app: GuiApp,
    *,
    title: str,
    message: str,
    confirm_label: str,
    on_confirm: Callable[[], None],
) -> None:
    """Pop a modal yes/no confirm. `on_confirm` runs after the dialog closes on
    the confirm button; Cancel just closes it. Shared so both eval tabs raise an
    identical confirm (e.g. the Unfreeze warning) instead of building it twice.
    """
    import flet as ft

    def _cancel(_e: ft.Event) -> None:
        app.page.pop_dialog()

    def _ok(_e: ft.Event) -> None:
        app.page.pop_dialog()
        on_confirm()

    dialog = ft.AlertDialog(
        modal=True,
        title=ft.Text(title),
        content=ft.Text(message, size=12),
        actions=[
            ft.TextButton("Cancel", on_click=_cancel),
            ft.Button(content=confirm_label, on_click=_ok),
        ],
    )
    app.page.show_dialog(dialog)
    app.page.update()


def active_corpus_config_path(app: GuiApp) -> Path | None:
    """The active corpus's `corpus.toml` path, or None when no corpus is set.

    Single source for wiring the active corpus into an `EvalConfig`; the eval
    output dir is derived from its parent folder (beside `lancedb`).
    """
    raw = getattr(app.gui_config, "corpus_config_path", None)
    return Path(raw) if raw else None


def active_corpus_dir(app: GuiApp) -> Path | None:
    """The active corpus's home folder (parent of its `corpus.toml`), or None
    when no corpus is set. Used to open the dataset file pickers in the corpus's
    own folder and to place a newly-created dataset beside it.
    """
    cfg = active_corpus_config_path(app)
    return cfg.parent if cfg else None


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


def active_output_dir(app: GuiApp) -> Path:
    """Where this run's results (ledger + JSON/CSV reports) will be written:
    `<active corpus folder>/eval_output`, or `<cwd>/eval_output` when no corpus
    is set. Resolves the SAME `EvalConfig.output_dir` the Run tab builds — but
    WITHOUT constructing an `EvalLedger`, so reading it for a read-only display
    never creates the folder on disk.
    """
    from knowledge_agent.evaluation.config import load_eval_config

    corpus = active_corpus_config_path(app)
    overrides = {"corpus_config_path": corpus} if corpus else {}
    output_dir = load_eval_config(**overrides).output_dir
    assert output_dir is not None  # always resolved in EvalConfig.__post_init__
    return output_dir
