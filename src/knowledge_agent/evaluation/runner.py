"""CLI entry point for the evaluation harness.

    python -m knowledge_agent.evaluation.runner [--dataset ...] [--groups source,chunk] ...

Wires the pieces: load config → load the gold cases → load the corpus
config → evaluate every case (async) → build the report (+ provenance) →
write JSON/CSV → persist to the SQLite ledger → print a summary.

A real run reads the corpus + models from the ambient environment — point
it at the EVAL instance (`.env.eval`, Phase 3) or, for the bootstrap, the
`test_corpus/` ingestion (`--corpus test_corpus/corpus.toml` with the
matching `LANCEDB_PATH`/Neo4j env). Evaluation only READS the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation import report as report_mod
from knowledge_agent.evaluation.config import load_eval_config
from knowledge_agent.evaluation.engine import evaluate_cases
from knowledge_agent.evaluation.ledger import EvalLedger
from knowledge_agent.evaluation.models import load_cases
from knowledge_agent.evaluation.registry import metric_fmts, metric_labels, summary_avg_pairs

if TYPE_CHECKING:
    from knowledge_agent.evaluation.config import EvalConfig
    from knowledge_agent.kg.corpus_config import CorpusConfig

logger = logging.getLogger(__name__)


def _load_corpus_config(cfg: EvalConfig) -> CorpusConfig | None:
    if cfg.corpus_config_path is None:
        return None
    from knowledge_agent.kg.corpus_config import load_corpus_config

    return load_corpus_config(cfg.corpus_config_path)


async def run(cfg: EvalConfig) -> dict[str, Any]:
    """Execute one evaluation run end-to-end; return the report dict."""
    cases = load_cases(cfg.dataset_path)
    if cfg.max_cases is not None:
        cases = cases[: cfg.max_cases]
    corpus_config = _load_corpus_config(cfg)

    logger.info("evaluating %d case(s), groups=%s", len(cases), sorted(cfg.enabled_groups))
    results = await evaluate_cases(cases, corpus_config, cfg)

    run_timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    report = report_mod.build_report(cfg, results, run_timestamp)
    json_path, csv_path = report_mod.write_report(report, cfg.output_dir)
    run_id = EvalLedger(cfg.ledger_path).save_run(report)

    _print_summary(report, json_path, csv_path, run_id)
    return report


def _print_summary(report: dict[str, Any], json_path: Path, csv_path: Path, run_id: int) -> None:
    s = report["summary"]
    labels, fmts = metric_labels(), metric_fmts()
    print(f"\n=== KA eval run {run_id} @ {report['run_timestamp']} ===")
    print(
        f"cases: {s['case_count']}   pass: {s['pass_count']}   "
        f"pass_rate: {s['pass_rate']:.0%}   groups: {', '.join(report['enabled_groups'])}"
    )
    for avg_key, _ in summary_avg_pairs():
        val = s.get(avg_key)
        if val is None:
            continue
        print(f"  {labels.get(avg_key, avg_key):26s} {format(val, fmts.get(avg_key, '.2f'))}")
    print(f"report: {json_path}")
    print(f"csv:    {csv_path}")
    print(f"ledger: run_id={run_id}")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="knowledge_agent.evaluation.runner",
        description="Run the KnowledgeAgent evaluation harness.",
    )
    p.add_argument("--dataset", type=Path, help="Gold queryset JSON.")
    p.add_argument("--corpus", type=Path, help="corpus.toml of the corpus to evaluate.")
    p.add_argument("--groups", help="Comma-separated metric groups (e.g. source,chunk).")
    p.add_argument("--max-cases", type=int, help="Truncate the queryset (dev runs).")
    p.add_argument("--output-dir", type=Path, help="Where to write ledger + reports.")
    return p


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


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_eval_config(**_overrides_from_args(args))
    asyncio.run(run(cfg))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
