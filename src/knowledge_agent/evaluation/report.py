"""Report building + persistence + run provenance.

`build_report` turns the engine's per-case results into the report dict
the ledger persists and the JSON file records: a run-level `summary`
(pass rate + registry-driven averages), the per-case `results`, and
PROVENANCE — the git commit, the graph's model IDs, and a snapshot of
the node system prompts. Provenance is what makes "did my prompt/model
change move the numbers?" answerable across runs.

`write_report` writes a timestamped JSON (full) + CSV (per-case summary)
to the git-ignored output dir.
"""

from __future__ import annotations

import csv
import json
import re
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation.metrics import safe_mean
from knowledge_agent.evaluation.registry import csv_fieldnames, summary_avg_pairs

if TYPE_CHECKING:
    from pathlib import Path

    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.evaluation.models import EvalRecipe


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    """Short HEAD commit, or None when git isn't available / not a repo."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _model_config() -> dict[str, Any]:
    """The graph's active model IDs — half of 'what config ran'."""
    try:
        from knowledge_agent.config import get_settings

        s = get_settings()
        return {
            "llm_provider": s.llm_provider,
            "mode_classifier_model": s.mode_classifier_model,
            "query_builder_model": s.query_builder_model,
            "cypher_builder_model": s.cypher_builder_model,
            "synthesizer_model": s.synthesizer_model,
            "embedding_provider": s.embedding_provider,
            "embedding_model": s.embedding_model,
        }
    except Exception:
        return {}


def _prompt_snapshot() -> dict[str, str]:
    """Snapshot the graph's node system prompts (auto-discovers every
    `*_SYSTEM` string constant in `nodes`, so it survives prompt renames)
    plus the chat-router prompt. Best-effort — degrades to what it finds."""
    out: dict[str, str] = {}
    try:
        from knowledge_agent import nodes as nodes_mod

        for name in dir(nodes_mod):
            if name.endswith("_SYSTEM"):
                val = getattr(nodes_mod, name, None)
                if isinstance(val, str):
                    out[name.strip("_").lower()] = val
    except Exception:
        pass
    try:
        from knowledge_agent.gui.chat_router import CHAT_SYSTEM_PROMPT

        out["chat_router"] = CHAT_SYSTEM_PROMPT
    except Exception:
        pass
    return out


def capture_provenance() -> dict[str, Any]:
    """git commit + model IDs + node-prompt snapshot for the run row."""
    return {
        "git_commit": _git_commit(),
        "model_config": _model_config(),
        "prompts": _prompt_snapshot(),
    }


# ---------------------------------------------------------------------------
# Summary + report assembly
# ---------------------------------------------------------------------------


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Run-level rollup: pass rate + a mean per registry metric (None-safe, so
    cases lacking a metric's gold don't drag its average), plus a per-metric
    `n_<key>` — how many cases actually fed each mean. Because a no-gold case is
    dropped (scored `None`) rather than counted, the denominator varies per
    metric; `n` records it so a mean over 1 case is distinguishable from one
    over all of them."""
    case_count = len(results)
    pass_count = sum(1 for r in results if r.get("status") == "PASS")
    summary: dict[str, Any] = {
        "case_count": case_count,
        "pass_count": pass_count,
        "pass_rate": (pass_count / case_count) if case_count else 0.0,
    }
    for avg_key, source_key in summary_avg_pairs():
        values = [r.get(source_key) for r in results]
        summary[avg_key] = safe_mean(values)
        summary[f"n_{source_key}"] = sum(1 for v in values if v is not None)
    return summary


def build_report(
    cfg: EvalConfig,
    results: list[dict[str, Any]],
    run_timestamp: str,
    *,
    dataset_hash: str | None = None,
    facts_hash: str | None = None,
    recipe: EvalRecipe | None = None,
    suite: str | None = None,
    suite_run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the full report dict (what the ledger + JSON consume).

    `dataset_hash` (a SHA-256 of the scored dataset's cases) + `facts_hash` (a
    SHA-256 of the gold content only, shared across a suite) are recorded as run
    provenance next to `git_commit` — so a trend view can tell when the gold
    itself changed. `recipe` (the dataset's saved recipe, if any) contributes
    `recipe_hash` (its fingerprint — the twin of dataset_hash, so a run records
    WHICH recipe it ran under). `suite_run_id` groups the members of one suite
    execution.
    """
    # Lazy imports: keep report's import path free of the judge module's
    # (potentially heavy) deps for the non-judge surfaces that import report.
    from knowledge_agent.evaluation.judge import resolve_judge_models
    from knowledge_agent.evaluation.models import compute_recipe_hash

    prov = capture_provenance()
    model_config = prov["model_config"]
    # The judge PANEL that actually ran (resolved to the provider default when
    # the user left it empty). Empty when the judge group is off — no judge ran.
    judge_models = resolve_judge_models(cfg.judge_models) if "judge" in cfg.enabled_groups else []
    return {
        "run_timestamp": run_timestamp,
        "dataset_path": str(cfg.dataset_path),
        # Filter keys lifted to the top level so the ledger can column them
        # (previously only reachable inside prompts_snapshot / the paths).
        "dataset_name": cfg.dataset_path.stem if cfg.dataset_path else None,
        "corpus_name": (cfg.corpus_config_path.parent.name if cfg.corpus_config_path else None),
        "dataset_hash": dataset_hash,
        # Suite identity: the gold-content hash SHARED by every member of a
        # test-dataset suite (twin of dataset_hash, which is per-file).
        "facts_hash": facts_hash,
        "git_commit": prov["git_commit"],
        "llm_provider": model_config.get("llm_provider"),
        "synthesizer_model": model_config.get("synthesizer_model"),
        "judge_models": judge_models,
        # Dataset-recipe provenance (None when the dataset has no saved recipe).
        # Filter/grouping tag for the dashboard — never read by scoring or the
        # pass-gate (the gates that ran are in `gate_thresholds` above).
        "recipe_hash": compute_recipe_hash(recipe),
        # The named suite this run was executed as (None for a single-file run) +
        # the shared launch timestamp stamped on all members of one suite run.
        # Grouping only — never scored.
        "suite": suite,
        "suite_run_id": suite_run_id,
        "prompts_snapshot": {
            "model_config": prov["model_config"],
            "prompts": prov["prompts"],
        },
        "enabled_groups": sorted(cfg.enabled_groups),
        "gate_thresholds": cfg.gate_thresholds(),
        "summary": build_summary(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Persistence — JSON (full) + CSV (per-case summary)
# ---------------------------------------------------------------------------


def _filename_stamp(run_timestamp: str) -> str:
    return re.sub(r"[^0-9]", "", run_timestamp)[:14] or "unknown"


def _dataset_slug(dataset_name: Any) -> str:
    """A filename-safe `_<dataset>` fragment for the report name (empty when the
    run has no dataset name). Keeps a SUITE's per-member reports from colliding
    when two members finish in the same second (same timestamp)."""
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", str(dataset_name or "").strip()).strip("-")
    return f"_{safe}" if safe else ""


def _unique_stem(output_dir: Path, base: str) -> str:
    """`base`, or `base_2` / `base_3` … so neither the .json nor the .csv for
    this run overwrites an existing report (guards the rare same-dataset,
    same-second re-run, where the timestamp + name alone aren't unique)."""
    stem, n = base, 1
    while (output_dir / f"eval_report_{stem}.json").exists() or (
        output_dir / f"eval_summary_{stem}.csv"
    ).exists():
        n += 1
        stem = f"{base}_{n}"
    return stem


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write `eval_report_<ts>_<dataset>.json` + `eval_summary_<ts>_<dataset>.csv`;
    return paths. The dataset name (+ a uniqueness counter) keeps a suite's
    per-member reports distinct even when they share a run timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _filename_stamp(report["run_timestamp"]) + _dataset_slug(report.get("dataset_name"))
    stem = _unique_stem(output_dir, base)
    json_path = output_dir / f"eval_report_{stem}.json"
    csv_path = output_dir / f"eval_summary_{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, report["results"])
    return json_path, csv_path


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = csv_fieldnames()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "error_count": len(r.get("errors", []))})
