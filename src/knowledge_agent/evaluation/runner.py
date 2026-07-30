"""Programmatic + CLI orchestration for the evaluation harness.

`run(cfg)` is the single, presentation-free entry point — it evaluates a
gold dataset end-to-end and returns a `RunResult` (the report + where it was
persisted), printing nothing. Both callers reuse it:

  - the GUI Evaluation tab calls `run(cfg)` in-process (its environment is
    already bridged from the keyring + settings.json at GUI startup);
  - the CLI runs it via `ka eval ...` (a subcommand of the main agent CLI,
    which reads `.env` — a dev/testing path) or `python -m
    knowledge_agent.evaluation.runner`.

The harness is self-contained in `evaluation/` and has no GUI dependency;
the callers differ only in how their environment got populated. Evaluation
only READS the corpus (LanceDB + Neo4j), never writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation.config import DEFAULT_LANGSMITH_PROJECT, load_eval_config
from knowledge_agent.evaluation.registry import metric_fmts, metric_labels, summary_avg_pairs

if TYPE_CHECKING:
    from collections.abc import Callable

    from knowledge_agent.corpus_config import CorpusConfig
    from knowledge_agent.evaluation.config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunResult:
    """Outcome of one eval run — the report plus where it was persisted.

    Returned by `run()` so callers (CLI, GUI) can render results without
    `run()` doing any stdout I/O of its own.
    """

    report: dict[str, Any]
    json_path: Path | None
    csv_path: Path | None
    run_id: int


@dataclass(slots=True)
class SuiteRunResult:
    """Outcome of one suite execution — the shared `suite_run_id` (launch
    timestamp) plus each member's `RunResult`, in run order. Returned by
    `run_suite` so the GUI can hop to Compare scoped to this suite-run."""

    suite_run_id: str
    results: list[RunResult]


def _load_corpus_config(cfg: EvalConfig) -> CorpusConfig | None:
    if cfg.corpus_config_path is None:
        return None
    from knowledge_agent.corpus_config import load_corpus_config

    return load_corpus_config(cfg.corpus_config_path)


async def run(
    cfg: EvalConfig,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    trace: bool = False,
    langsmith_project: str | None = None,
    suite: str | None = None,
    suite_run_id: str | None = None,
    save_json: bool = True,
    save_csv: bool = True,
) -> RunResult:
    """Execute one evaluation run end-to-end; return a `RunResult`.

    Presentation-free: writes the ledger row (always) plus the JSON/CSV report
    files and returns the report + paths, but prints nothing (the CLI wrapper
    prints; the GUI renders). `save_json` / `save_csv` gate the two optional
    file exports (default on; the Run tab's Save-options toggles); a skipped
    file yields a `None` path on the result. The ledger write is unconditional,
    so `run_id` always exists and the dashboards can always read the run.
    `on_progress(done, total)` is passed through to `evaluate_cases`
    so a caller can drive a progress bar / print line. Heavy deps (engine ->
    adapter -> langchain) import lazily so merely importing this module stays
    cheap for `ka`'s other subcommands.

    `trace=True` sends this run's graph steps (queries, retrieved chunks,
    answers) to LangSmith for step-by-step debugging. It is scoped to THIS
    run via a context manager — never the global `LANGSMITH_TRACING` env flag
    — so nothing is ever uploaded unless a caller explicitly opts in per run.
    Auth uses `LANGSMITH_API_KEY` already in the environment (the GUI bridges
    it from the OS keyring; the CLI from `.env`); `langsmith_project` names the
    bucket traces land in (default `DEFAULT_LANGSMITH_PROJECT`).
    """
    from knowledge_agent.evaluation import report as report_mod
    from knowledge_agent.evaluation.engine import evaluate_cases
    from knowledge_agent.evaluation.ledger import EvalLedger
    from knowledge_agent.evaluation.models import (
        compute_dataset_hash,
        compute_facts_hash,
        compute_knob_hash,
        load_dataset,
        validate_dataset,
    )

    if cfg.dataset_path is None:
        raise ValueError(
            "No dataset selected — choose a gold dataset first "
            "(GUI: Browse in the corpus folder; CLI: --dataset / KA_EVAL_DATASET)."
        )
    # Load the FULL dataset (header + cases): the header carries the recipe
    # whose `recipe_hash` is stamped as run provenance.
    dataset = load_dataset(cfg.dataset_path)
    cases = dataset.cases
    # Refuse up-front (before spending tokens) if any case leaves a required
    # retrieval knob blank — otherwise it would silently fall back to the
    # shifting global setting and the run wouldn't be reproducible. Catches
    # datasets hand-edited or imported from elsewhere; the GUI form prevents
    # it for cases authored in-app.
    invalid = validate_dataset(cases)
    if invalid:
        detail = "\n".join(f"  - {cid}: {'; '.join(problems)}" for cid, problems in invalid.items())
        raise ValueError(
            "Cannot run: the dataset has case(s) with missing or invalid "
            "retrieval settings — fix them before running:\n" + detail
        )
    # Hash the FULL dataset (before max_cases truncation / any filter) — the
    # fingerprint is the dataset's identity, independent of how many ran.
    dataset_hash = compute_dataset_hash(cases)
    # facts_hash = gold content only (knobs + id excluded) → the suite identity,
    # shared by every member that carries the same facts. Same full-dataset scope
    # as dataset_hash (before max_cases truncation).
    facts_hash = compute_facts_hash(cases)
    # knob_hash = per-case retrieval knobs only → what tells two suite members
    # apart (same facts, different knobs). Same full-dataset scope.
    knob_hash = compute_knob_hash(cases)
    if cfg.max_cases is not None:
        cases = cases[: cfg.max_cases]
    corpus_config = _load_corpus_config(cfg)
    if corpus_config is not None:
        # Resolve the embedder from the corpus under evaluation (per-corpus;
        # query vectors must match the model the corpus was ingested with),
        # overriding any global / .env embedder.
        from knowledge_agent.config import reset_after_settings_change
        from knowledge_agent.corpus_config import apply_corpus_embedding_to_env

        apply_corpus_embedding_to_env(corpus_config)
        reset_after_settings_change()

    logger.info("evaluating %d case(s), groups=%s", len(cases), sorted(cfg.enabled_groups))
    if trace:
        # Lazy import keeps langchain's tracer off the default (untraced) path.
        from langchain_core.tracers.context import tracing_v2_enabled

        project = (langsmith_project or "").strip() or DEFAULT_LANGSMITH_PROJECT
        logger.info("LangSmith tracing ON (project=%s)", project)
        with tracing_v2_enabled(project_name=project):
            results = await evaluate_cases(cases, corpus_config, cfg, on_progress=on_progress)
    else:
        results = await evaluate_cases(cases, corpus_config, cfg, on_progress=on_progress)

    run_timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    # build_report captures provenance via a blocking `git rev-parse` subprocess.
    # run() is async and the GUI drives it on the Flet event loop, so assemble
    # the report off-loop in a worker thread — otherwise the git call (a few ms,
    # up to its 5s timeout on a git hang) stalls the UI. The CLI is unaffected.
    report = await asyncio.to_thread(
        report_mod.build_report,
        cfg,
        results,
        run_timestamp,
        dataset_hash=dataset_hash,
        facts_hash=facts_hash,
        knob_hash=knob_hash,
        recipe=dataset.recipe,
        suite=suite,
        suite_run_id=suite_run_id,
    )
    json_path, csv_path = report_mod.write_report(
        report, cfg.output_dir, save_json=save_json, save_csv=save_csv
    )
    run_id = EvalLedger(cfg.ledger_path).save_run(report)

    return RunResult(report=report, json_path=json_path, csv_path=csv_path, run_id=run_id)


async def run_suite(
    cfgs: list[EvalConfig],
    on_run_complete: Callable[[int, int, RunResult], None] | None = None,
    *,
    suite: str | None = None,
    trace: bool = False,
    langsmith_project: str | None = None,
    save_json: bool = True,
    save_csv: bool = True,
) -> SuiteRunResult:
    """Run every member of a test-dataset suite under ONE shared `suite_run_id`.

    Each `cfg` is one suite member — the SAME facts, a DIFFERENT pinned knob
    setting. All N runs are stamped with a single launch timestamp (and the
    `suite` name they were run as) so the dashboard loads + compares them as one
    unit. Members run SEQUENTIALLY on purpose: the graph is already concurrent
    within a run, and back-to-back keeps the machine + code identical across
    members, so a metric difference between them is attributable to the knob, not
    to drift. `on_run_complete(done, total, result)` fires as each member finishes
    so a caller can advance a progress bar.
    """
    suite_run_id = datetime.now(UTC).isoformat(timespec="seconds")
    total = len(cfgs)
    results: list[RunResult] = []
    for cfg in cfgs:
        result = await run(
            cfg,
            trace=trace,
            langsmith_project=langsmith_project,
            suite=suite,
            suite_run_id=suite_run_id,
            save_json=save_json,
            save_csv=save_csv,
        )
        results.append(result)
        if on_run_complete is not None:
            on_run_complete(len(results), total, result)
    return SuiteRunResult(suite_run_id=suite_run_id, results=results)


def _print_summary(result: RunResult) -> None:
    report = result.report
    s = report["summary"]
    labels, fmts = metric_labels(), metric_fmts()
    print(f"\n=== KA eval run {result.run_id} @ {report['run_timestamp']} ===")
    print(
        f"cases: {s['case_count']}   pass: {s['pass_count']}   "
        f"pass_rate: {s['pass_rate']:.0%}   groups: {', '.join(report['enabled_groups'])}"
    )
    for avg_key, _ in summary_avg_pairs():
        val = s.get(avg_key)
        if val is None:
            continue
        print(f"  {labels.get(avg_key, avg_key):26s} {format(val, fmts.get(avg_key, '.2f'))}")
    if result.json_path is not None:
        print(f"report: {result.json_path}")
    if result.csv_path is not None:
        print(f"csv:    {result.csv_path}")
    print(f"ledger: run_id={result.run_id}")


def _cli_progress(done: int, total: int) -> None:
    """In-place terminal progress line for `ka eval` (stderr, so it doesn't
    pollute the stdout summary); newline once the last case lands."""
    end = "\n" if done >= total else ""
    print(f"\r  evaluating case {done}/{total} ...", end=end, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Ledger read views — `ka eval --history` / `--show`. The same
# EvalLedger readers back the GUI history table; here they render to the
# terminal (+ optional file export). Rendering stays consistent with the
# GUI because both pull column labels/formats from the registry.
# ---------------------------------------------------------------------------


def _fmt_int(value: Any) -> str:
    return "-" if value is None else str(value)


def _fmt_rate(value: Any) -> str:
    return "-" if value is None else f"{value:.0%}"


def _export_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write `rows` to `path` — CSV when the suffix is .csv, else JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        import csv

        fieldnames = list(rows[0].keys()) if rows else []
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    else:
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _show_history(cfg: EvalConfig, export: Path | None) -> int:
    """List past runs (newest first) as a terminal table; optionally export."""
    from knowledge_agent.evaluation.ledger import EvalLedger

    runs = EvalLedger(cfg.ledger_path).list_runs()
    if not runs:
        print("No eval runs recorded yet.")
        return 0
    print(f"{'run':>4}  {'when':19}  {'cases':>5}  {'pass':>4}  {'pass_rate':>9}")
    for r in runs:
        print(
            f"{r['run_id']:>4}  {(r.get('run_timestamp') or '')[:19]:19}  "
            f"{_fmt_int(r.get('case_count')):>5}  {_fmt_int(r.get('pass_count')):>4}  "
            f"{_fmt_rate(r.get('pass_rate')):>9}"
        )
    if export is not None:
        _export_rows(runs, export)
        print(f"exported {len(runs)} run(s) -> {export}")
    return 0


def _show_run(cfg: EvalConfig, run_id: int, export: Path | None) -> int:
    """Show one run's per-case results; optionally export. 1 if not found."""
    from knowledge_agent.evaluation.ledger import EvalLedger

    ledger = EvalLedger(cfg.ledger_path)
    run = ledger.get_run(run_id)
    if run is None:
        print(f"error: no run with id {run_id}", file=sys.stderr)
        return 1
    cases = ledger.get_run_cases(run_id)
    print(
        f"=== run {run_id} @ {run.get('run_timestamp')} — "
        f"{_fmt_int(run.get('pass_count'))}/{_fmt_int(run.get('case_count'))} pass "
        f"({_fmt_rate(run.get('pass_rate'))}) ==="
    )
    print(f"{'status':7}  {'category':14}  case")
    for c in cases:
        print(f"{(c.get('status') or ''):7}  {(c.get('category') or ''):14}  {c.get('case_id')}")
    if export is not None:
        _export_rows(cases, export)
        print(f"exported {len(cases)} case(s) -> {export}")
    return 0


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Register the harness's CLI flags on `parser`.

    Single source for the eval CLI surface — used by the standalone parser
    (`python -m ...runner`) and the `ka eval` subparser alike. With no read
    flag the command RUNS an eval; `--history` / `--show` read the ledger.
    """
    parser.add_argument("--dataset", type=Path, help="Gold queryset JSON.")
    parser.add_argument("--corpus", type=Path, help="corpus.toml of the corpus to evaluate.")
    parser.add_argument("--groups", help="Comma-separated metric groups (e.g. source,chunk).")
    parser.add_argument("--max-cases", type=int, help="Truncate the queryset (dev runs).")
    parser.add_argument("--output-dir", type=Path, help="Where to write ledger + reports.")
    # ---- LangSmith tracing (opt-in per run; uploads run data to the cloud) ----
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Trace this run to LangSmith (needs LANGSMITH_API_KEY in env/.env). "
            "Uploads queries + retrieved chunks + answers; use only on a "
            "non-sensitive corpus."
        ),
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="LangSmith project for --trace (default: knowledge-agent-eval).",
    )
    # ---- read views (don't run an eval) ----
    parser.add_argument(
        "--history",
        action="store_true",
        help="List past runs (newest first) instead of running.",
    )
    parser.add_argument(
        "--show",
        type=int,
        metavar="RUN_ID",
        help="Show one past run's per-case results instead of running.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="PATH",
        help="With --history/--show: also write the rows to PATH (.csv or .json by suffix).",
    )


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.dataset:
        overrides["dataset_path"] = args.dataset
    if args.corpus:
        overrides["corpus_config_path"] = args.corpus
    if args.groups:
        overrides["enabled_groups"] = frozenset(
            g.strip() for g in args.groups.split(",") if g.strip()
        )
    if args.max_cases is not None:
        overrides["max_cases"] = args.max_cases
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    return overrides


async def run_from_args(args: argparse.Namespace) -> int:
    """Build config from parsed CLI args, dispatch, return an exit code.

    `--history` / `--show` read the ledger (no eval runs); otherwise run an
    eval and print its summary. Shared by the `ka eval` subcommand and
    `python -m ...runner`.
    """
    cfg = load_eval_config(**_overrides_from_args(args))
    if args.history:
        return _show_history(cfg, args.export)
    if args.show is not None:
        return _show_run(cfg, args.show, args.export)
    if args.trace and not (
        os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    ):
        print(
            "error: --trace needs a LangSmith API key — set LANGSMITH_API_KEY in .env.",
            file=sys.stderr,
        )
        return 1
    result = await run(
        cfg, on_progress=_cli_progress, trace=args.trace, langsmith_project=args.project
    )
    _print_summary(result)
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="knowledge_agent.evaluation.runner",
        description="Run the KnowledgeAgent evaluation harness.",
    )
    add_eval_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    return asyncio.run(run_from_args(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
