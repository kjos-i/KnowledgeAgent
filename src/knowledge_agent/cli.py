"""CLI entry point — headless ops for ingest / query / health.

Three subcommands wired to the same async machinery the GUI calls:

  - `ka ingest <folder>` — recursively scan, plan, ingest every supported
    file via `ingestion.bulk_ops.ingest_folder_execute`. Failures
    fail-soft per-file (same contract as the GUI's Sync).
  - `ka query "..."` — invoke the compiled agent graph
    (`graph.ainvoke({"query": "..."})`) and print the resulting
    `AgentAnswer.answer` to stdout. Source markers are kept in the
    answer; structured `chunk_sources` / `kg_sources` are emitted as
    JSON only when `--json` is set.
  - `ka health` — basic liveness probe: Neo4j ping, LanceDB open,
    active LLM + embedder provider keys present. Exit 0 = all OK; non-
    zero = at least one failure (the message names which).

The CLI is intentionally thin — it composes existing async
orchestrators rather than reimplementing logic, so the eval harness,
cron jobs, and the GUI all hit the same code paths. No subcommand
runs more than ~30 LOC of glue.

Health is deliberately minimal here; sprint 0c lands
`system_status() → StatusReport` (the rich diagnostic surface used by
the GUI's Settings → Diagnostics panel) and this command will swap
to render that report instead.

Exit codes:
  - 0 — success
  - 1 — operational failure (ingest had failures, query failed, health
        check failed)
  - 2 — usage error (handled by argparse)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# ingest
# =============================================================================


async def _cmd_ingest(args: argparse.Namespace) -> int:
    """Build a plan via `bulk_ops.ingest_folder_plan`, then execute."""
    from knowledge_agent.ingestion import bulk_ops
    from knowledge_agent.kg.corpus_config import load_corpus_config

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 1

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"error: corpus config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_corpus_config(config_path)
    plan = await bulk_ops.ingest_folder_plan(
        folder,
        main_label=args.main_label,
        sub_label=args.sub_label,
    )
    print(plan.summary)

    if plan.n_files == 0:
        print("Nothing to ingest.")
        return 0

    if not args.yes:
        # Stdin prompt — the CLI is the only layer allowed to ask.
        # Backend per [[backend-no-ui-prompts]] never prompts.
        reply = input("Continue? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Cancelled.")
            return 0

    result = await bulk_ops.ingest_folder_execute(plan, config)
    print(
        f"Done: {result.n_succeeded} succeeded, {result.n_failed} failed.",
    )
    if result.failures:
        print("Failures:")
        for name, err in result.failures:
            print(f"  - {name}: {err}")
    return 0 if result.n_failed == 0 else 1


# =============================================================================
# query
# =============================================================================


async def _cmd_query(args: argparse.Namespace) -> int:
    """Invoke the compiled agent graph and print the answer."""
    from knowledge_agent.graph import graph
    from knowledge_agent.kg.corpus_config import load_corpus_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"error: corpus config not found: {config_path}", file=sys.stderr)
        return 1
    corpus_config = load_corpus_config(config_path)

    initial_state: dict[str, Any] = {
        "query": args.query,
        "corpus_config": corpus_config,
    }
    if args.mode != "auto":
        initial_state["retrieval_mode"] = args.mode

    final_state = await graph.ainvoke(initial_state)
    answer = final_state.get("final_answer")
    if answer is None:
        print("error: graph returned no final_answer", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(answer.model_dump(), indent=2, default=str))
    else:
        print(answer.answer)
        if answer.chunk_sources:
            print("\nChunk sources:")
            for i, src in enumerate(answer.chunk_sources, start=1):
                print(f"  [{i}] doc_id={src.doc_id} chunk_id={src.chunk_id}")
        if answer.kg_sources:
            print("\nKG sources:")
            for src in answer.kg_sources:
                print(f"  [K{src.hit_index}]")
    return 0


# =============================================================================
# health
# =============================================================================


async def _cmd_health(_args: argparse.Namespace) -> int:
    """Run `system_status()` and render the StatusReport to stdout.

    Exit 0 iff every component reports `ok=True`. The GUI's Settings →
    Diagnostics panel (sprint 1) calls the SAME `system_status()` and
    renders the same dataclass with its own widgets.
    """
    from knowledge_agent.health import render_report_text, system_status

    report = await system_status()
    print(render_report_text(report))
    return 0 if report.all_ok else 1


# =============================================================================
# Top-level parser + dispatcher
# =============================================================================


_DEFAULT_CONFIG = "corpus.toml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ka",
        description=(
            "knowledge-agent CLI: headless ingest / query / health "
            "for cron, CI, eval harness, and scripting."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ingest a folder of documents.")
    p_ingest.add_argument("folder", help="Path to the folder to ingest.")
    p_ingest.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to corpus.toml (default: ./corpus.toml).",
    )
    p_ingest.add_argument(
        "--main-label",
        default="Document",
        help="Main KG label for each ingested file (default: Document).",
    )
    p_ingest.add_argument(
        "--sub-label",
        default=None,
        help="Optional KG sub-label (e.g. Paper, Note).",
    )
    p_ingest.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt; ingest immediately.",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    p_query = subparsers.add_parser("query", help="Run one agent query end-to-end.")
    p_query.add_argument("query", help="Natural-language question.")
    p_query.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to corpus.toml (default: ./corpus.toml).",
    )
    p_query.add_argument(
        "--mode",
        default="auto",
        choices=(
            "auto",
            "lancedb_only",
            "neo4j_only",
            "lancedb_then_neo4j",
            "neo4j_then_lancedb",
            "parallel_fused",
        ),
        help="Force a specific retrieval mode (default: auto = classifier picks).",
    )
    p_query.add_argument(
        "--json",
        action="store_true",
        help="Print the full AgentAnswer as JSON instead of just the answer text.",
    )
    p_query.set_defaults(func=_cmd_query)

    p_health = subparsers.add_parser(
        "health",
        help="Liveness probe: Neo4j + LanceDB + active provider keys.",
    )
    p_health.set_defaults(func=_cmd_health)

    return parser


async def _async_main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return await args.func(args)


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — `pyproject.toml [project.scripts]` calls this.

    Wraps the async machinery in `asyncio.run`. Returns the exit code so
    the wheel-installed script can propagate it via `sys.exit(main())`.
    """
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    sys.exit(main())
