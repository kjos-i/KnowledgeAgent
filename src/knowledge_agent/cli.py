"""CLI entry point — headless ops for ingest / query / health / eval.

Four subcommands wired to the same async machinery the GUI calls:

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
  - `ka eval` — run the evaluation harness over a gold dataset; write the
    JSON/CSV report + ledger row and print a summary. A dev/testing path
    (reads `.env` like the other subcommands); the GUI Evaluation tab
    calls the same `evaluation.runner.run()` in-process. Thin wrapper —
    all logic lives self-contained in `evaluation/`.

The CLI is intentionally thin — it composes existing async
orchestrators rather than reimplementing logic, so the eval harness,
cron jobs, and the GUI all hit the same code paths. No subcommand
runs more than ~30 LOC of glue.

Configuration — `.env` vs. flags (THE convention; kept here as the
single source so it's easy to rediscover):

  The split is by *stability + sensitivity* — environment vs. invocation.

  - `.env` (→ backend `Settings` in `config.py`, `env_file=.env`) holds
    the ENVIRONMENT the run executes in — things that rarely change
    run-to-run and/or are secret:
      * secrets — `NEO4J_PASSWORD`, provider keys (`ANTHROPIC/OPENAI/
        GOOGLE/VOYAGE/LANGSMITH_API_KEY`);
      * connections — `NEO4J_URI/USER`, `LANCEDB_PATH` (all
        required-no-default: a forgotten value fails fast instead of
        silently hitting the wrong instance);
      * provider + model + tuning defaults — `LLM_PROVIDER`,
        `EMBEDDING_PROVIDER`, per-node models/temps, retrieval knobs
        (`TOP_K`, `DEFAULT_RETRIEVAL_MODE`, `LANCEDB_SEARCH_MODE`,
        RRF/MMR, `KG_MAX_ROWS`), rate limits, `ONTOLOGY_DOWNLOADS_DIR`;
      * optional eval defaults — `KA_EVAL_*`.
    Secrets live here and NEVER as flags: a password in `argv` leaks into
    shell history / `ps` / CI logs.

  - argparse flags say WHAT this invocation does — the verb + its object
    (`ingest <folder>`, `query "<q>"`, `eval --dataset`), per-run
    overrides (`--mode`, `--max-cases`, `--trace`), and output routing
    (`--json`, `--export`, `--output-dir`). These change every run.

  Overlap zone (flag wins): a few knobs sit in env AND can be overridden
  per-run by a flag — `--mode` over `DEFAULT_RETRIEVAL_MODE`; the eval
  flags over their `KA_EVAL_*` twins (precedence in `load_eval_config`:
  defaults < `KA_EVAL_*` env < explicit flags). The flag is the most
  immediate signal, so it should win.

  Clean split worth remembering: WHICH corpus is a per-run choice
  (`--config corpus.toml`, a flag), but WHERE its DB lives + WHO you are
  is `.env` — so you point at different corpora without touching secrets.

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
from typing import TYPE_CHECKING, Any

from knowledge_agent.logging_setup import init_logging

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# =============================================================================
# ingest
# =============================================================================


async def _cmd_ingest(args: argparse.Namespace) -> int:
    """Build a plan via `bulk_ops.ingest_folder_plan`, then execute."""
    from knowledge_agent.config import reset_after_key_change
    from knowledge_agent.corpus_config import (
        apply_corpus_embedding_to_env,
        load_corpus_config,
    )
    from knowledge_agent.ingestion import bulk_ops

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"error: not a directory: {folder}", file=sys.stderr)
        return 1

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"error: corpus config not found: {config_path}", file=sys.stderr)
        return 1

    config = load_corpus_config(config_path)
    # Resolve the embedder from THIS corpus (per-corpus; LanceDB pins the
    # vector dim at ingest), overriding any global / .env embedder.
    apply_corpus_embedding_to_env(config)
    reset_after_key_change()
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
        # The backend never prompts.
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
    from knowledge_agent.config import reset_after_key_change
    from knowledge_agent.corpus_config import (
        apply_corpus_embedding_to_env,
        load_corpus_config,
    )
    from knowledge_agent.graph import graph

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        print(f"error: corpus config not found: {config_path}", file=sys.stderr)
        return 1
    corpus_config = load_corpus_config(config_path)
    # Resolve the embedder from THIS corpus (per-corpus; query vectors must
    # come from the same model the corpus was ingested with).
    apply_corpus_embedding_to_env(corpus_config)
    reset_after_key_change()

    initial_state: dict[str, Any] = {
        "query": args.query,
        "corpus_config": corpus_config,
        # Always forward the mode, INCLUDING "auto": the graph router runs the
        # mode classifier only when retrieval_mode == "auto". Skipping it left
        # retrieval_mode unset, so the router fell back to
        # settings.default_retrieval_mode (lancedb_only) and the classifier
        # never ran — `--mode auto` silently behaved like `--mode lancedb_only`.
        "retrieval_mode": args.mode,
    }

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

    if args.output:
        from knowledge_agent.artifacts import SaveError, save_answer

        fmts = [f.strip() for f in args.format.split(",") if f.strip()]
        try:
            paths = save_answer(answer, args.query, args.output, fmts)
        except SaveError as exc:
            print(f"error: could not save: {exc}", file=sys.stderr)
            return 1
        # Confirmations to stderr so stdout stays the answer (pipeable).
        for saved in paths:
            print(f"saved: {saved}", file=sys.stderr)
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
# eval
# =============================================================================


async def _cmd_eval(args: argparse.Namespace) -> int:
    """Run the evaluation harness (dev/testing path — reads `.env`).

    Delegates to the self-contained harness in `evaluation/`; the GUI
    Evaluation tab calls the same `run()` in-process. Heavy imports stay
    lazy so `ka`'s other subcommands don't pull the eval stack.
    """
    from knowledge_agent.evaluation import runner as eval_runner

    return await eval_runner.run_from_args(args)


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
    p_query.add_argument(
        "--output",
        type=Path,
        metavar="DIR",
        help="Also save the answer to this folder (files named by timestamp + query slug).",
    )
    p_query.add_argument(
        "--format",
        default="md",
        metavar="FMTS",
        help="Comma-separated save formats for --output: md,txt,docx,json (default: md).",
    )
    p_query.set_defaults(func=_cmd_query)

    p_health = subparsers.add_parser(
        "health",
        help="Liveness probe: Neo4j + LanceDB + active provider keys.",
    )
    p_health.set_defaults(func=_cmd_health)

    # `eval` — hosted here, but its flags + logic live in `evaluation/`
    # (single source). Import is light (the heavy engine/adapter deps load
    # lazily inside `run()`), so `ka health` etc. don't pull the eval stack.
    from knowledge_agent.evaluation.runner import add_eval_args

    p_eval = subparsers.add_parser(
        "eval",
        help="Run the evaluation harness over a gold dataset (dev/testing).",
    )
    add_eval_args(p_eval)
    p_eval.set_defaults(func=_cmd_eval)

    return parser


async def _async_main(
    argv: list[str] | None = None,
    asyncio_handler: Callable[[Any, dict[str, Any]], None] | None = None,
) -> int:
    # Install the logging asyncio-exception handler on the loop asyncio.run()
    # created (asyncio binds handlers to a specific loop) so coroutine crashes
    # land in the crash log.
    if asyncio_handler is not None:
        asyncio.get_running_loop().set_exception_handler(asyncio_handler)
    parser = _build_parser()
    args = parser.parse_args(argv)
    return await args.func(args)


def main(argv: list[str] | None = None) -> int:
    """Sync entry point — `pyproject.toml [project.scripts]` calls this.

    Wraps the async machinery in `asyncio.run`. Returns the exit code so
    the wheel-installed script can propagate it via `sys.exit(main())`.

    init_logging() sets up the full logging system (rotating file log, crash +
    faulthandler hooks, library-noise clamp) and returns asyncio's exception
    handler, installed on the loop inside `_async_main`. `asyncio.run` runs the
    normal atexit on teardown, so the QueueListener drains cleanly — no os._exit
    drain is needed here (unlike the GUI's close-hang path).
    """
    asyncio_handler = init_logging()
    return asyncio.run(_async_main(argv, asyncio_handler))


if __name__ == "__main__":
    sys.exit(main())
