"""Tests for the `kg` CLI dispatcher (`knowledge_agent.cli`).

Each subcommand's body is exercised with mocks for the heavy deps it
composes (bulk_ops, graph, kg_client, search_client). The parser is
also smoke-tested — argparse routing + the `--help` exit-0 contract.

These tests don't cover the underlying bulk_ops / graph behaviour —
those have their own suites. The CLI is glue; the tests verify the
glue is wired correctly + the exit codes match the contract.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knowledge_agent.cli import (
    _build_parser,
    _cmd_eval,
    _cmd_health,
    _cmd_ingest,
    _cmd_query,
    main,
)
from knowledge_agent.models import AgentAnswer

# ---- parser ----


def test_build_parser_requires_subcommand():
    """No subcommand → argparse exits 2 via SystemExit."""
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args([])
    assert exc.value.code == 2


def test_build_parser_routes_ingest():
    parser = _build_parser()
    args = parser.parse_args(["ingest", "/tmp/folder"])
    assert args.command == "ingest"
    assert args.folder == "/tmp/folder"


def test_build_parser_routes_query():
    parser = _build_parser()
    args = parser.parse_args(["query", "what is X?"])
    assert args.command == "query"
    assert args.query == "what is X?"
    assert args.mode == "auto"


def test_build_parser_query_mode_validated():
    """Unknown --mode is a usage error."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["query", "x", "--mode", "definitely_not_a_mode"])


def test_build_parser_routes_health():
    parser = _build_parser()
    args = parser.parse_args(["health"])
    assert args.command == "health"


def test_build_parser_routes_eval():
    parser = _build_parser()
    args = parser.parse_args(["eval", "--groups", "source,chunk", "--max-cases", "5"])
    assert args.command == "eval"
    assert args.groups == "source,chunk"
    assert args.max_cases == 5
    assert args.func is _cmd_eval


def test_build_parser_eval_read_flags():
    """`ka eval --history` / `--show <id>` / `--export <path>` parse."""
    parser = _build_parser()
    hist = parser.parse_args(["eval", "--history"])
    assert hist.history is True and hist.show is None
    shown = parser.parse_args(["eval", "--show", "3", "--export", "out.csv"])
    assert shown.show == 3
    assert str(shown.export) == "out.csv"


def test_main_help_exits_zero():
    """`ka --help` is a successful no-op (logging init mocked out so the test
    never touches the real log dir / installs global excepthooks)."""
    with (
        patch("knowledge_agent.cli.init_logging") as init,
        pytest.raises(SystemExit) as exc,
        redirect_stdout(io.StringIO()),
    ):
        main(["--help"])
    assert exc.value.code == 0
    init.assert_called_once()  # entry point wires the logging system


def test_main_initializes_logging_and_installs_asyncio_handler():
    """`ka` entry point calls init_logging() and installs the returned asyncio
    exception handler on the loop asyncio.run() creates."""

    def handler(_loop, _context):  # must be callable — the real loop validates it
        pass

    captured: dict[str, object] = {}

    async def _fake_func(_args: argparse.Namespace) -> int:
        captured["handler"] = asyncio.get_running_loop().get_exception_handler()
        return 0

    parser = argparse.ArgumentParser()
    parser.set_defaults(func=_fake_func)
    with (
        patch("knowledge_agent.cli.init_logging", return_value=handler) as init,
        patch("knowledge_agent.cli._build_parser", return_value=parser),
    ):
        rc = main([])
    assert rc == 0
    init.assert_called_once()
    assert captured["handler"] is handler


# ---- ingest ----


async def test_cmd_ingest_invalid_folder_returns_one(tmp_path):
    args = argparse.Namespace(
        folder=str(tmp_path / "does-not-exist"),
        config="corpus.toml",
        main_label="Document",
        sub_label=None,
        yes=True,
    )
    with redirect_stderr(io.StringIO()) as err:
        rc = await _cmd_ingest(args)
    assert rc == 1
    assert "not a directory" in err.getvalue()


async def test_cmd_ingest_missing_config_returns_one(tmp_path):
    args = argparse.Namespace(
        folder=str(tmp_path),
        config=str(tmp_path / "missing.toml"),
        main_label="Document",
        sub_label=None,
        yes=True,
    )
    with redirect_stderr(io.StringIO()) as err:
        rc = await _cmd_ingest(args)
    assert rc == 1
    assert "corpus config not found" in err.getvalue()


async def test_cmd_ingest_empty_folder_returns_zero(tmp_path):
    """A plan with n_files=0 exits 0 with a friendly message."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")

    fake_plan = MagicMock(n_files=0, summary="Ingest 0 files.")
    args = argparse.Namespace(
        folder=str(tmp_path),
        config=str(config_path),
        main_label="Document",
        sub_label=None,
        yes=True,
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.ingest_folder_plan",
            new_callable=AsyncMock,
            return_value=fake_plan,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.ingest_folder_execute",
            new_callable=AsyncMock,
        ) as exec_mock,
        redirect_stdout(io.StringIO()),
    ):
        rc = await _cmd_ingest(args)

    assert rc == 0
    exec_mock.assert_not_called()


async def test_cmd_ingest_propagates_failure_count_to_exit_code(tmp_path):
    """n_failed > 0 → exit code 1."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")

    fake_plan = MagicMock(n_files=3, summary="Ingest 3 files.")
    fake_result = MagicMock(
        n_succeeded=2,
        n_failed=1,
        failures=(("doc.pdf", "RuntimeError('boom')"),),
    )
    args = argparse.Namespace(
        folder=str(tmp_path),
        config=str(config_path),
        main_label="Document",
        sub_label=None,
        yes=True,
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.ingestion.bulk_ops.ingest_folder_plan",
            new_callable=AsyncMock,
            return_value=fake_plan,
        ),
        patch(
            "knowledge_agent.ingestion.bulk_ops.ingest_folder_execute",
            new_callable=AsyncMock,
            return_value=fake_result,
        ),
        redirect_stdout(io.StringIO()),
    ):
        rc = await _cmd_ingest(args)
    assert rc == 1


# ---- query ----


async def test_cmd_query_prints_answer_text(tmp_path):
    """Default mode (no --json) prints just the answer text + sources."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")

    fake_answer = MagicMock(
        answer="The answer is 42.",
        chunk_sources=[],
        kg_sources=[],
    )
    args = argparse.Namespace(
        query="what is X?",
        config=str(config_path),
        mode="auto",
        json=False,
        output=None,
        format="md",
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": fake_answer},
        ),
        redirect_stdout(io.StringIO()) as out,
    ):
        rc = await _cmd_query(args)

    assert rc == 0
    assert "The answer is 42." in out.getvalue()


async def test_cmd_query_forwards_auto_mode_so_classifier_runs(tmp_path):
    """`--mode auto` must forward retrieval_mode="auto" so the graph router runs
    the mode classifier. Regression: the CLI set retrieval_mode ONLY when mode
    != "auto", so "auto" left it unset -> the router fell back to
    settings.default_retrieval_mode (lancedb_only) and never classified."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")
    args = argparse.Namespace(
        query="what is X?",
        config=str(config_path),
        mode="auto",
        json=False,
        output=None,
        format="md",
    )
    with (
        patch("knowledge_agent.corpus_config.load_corpus_config", return_value=MagicMock()),
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": MagicMock(chunk_sources=[], kg_sources=[])},
        ) as mock_ainvoke,
        redirect_stdout(io.StringIO()),
    ):
        rc = await _cmd_query(args)
    assert rc == 0
    initial_state = mock_ainvoke.call_args.args[0]
    assert initial_state["retrieval_mode"] == "auto"  # forwarded -> router classifies


async def test_cmd_query_forwards_concrete_mode(tmp_path):
    """A concrete --mode reaches the graph state verbatim (unchanged path)."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")
    args = argparse.Namespace(
        query="q",
        config=str(config_path),
        mode="neo4j_only",
        json=False,
        output=None,
        format="md",
    )
    with (
        patch("knowledge_agent.corpus_config.load_corpus_config", return_value=MagicMock()),
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": MagicMock(chunk_sources=[], kg_sources=[])},
        ) as mock_ainvoke,
        redirect_stdout(io.StringIO()),
    ):
        await _cmd_query(args)
    assert mock_ainvoke.call_args.args[0]["retrieval_mode"] == "neo4j_only"


async def test_cmd_query_output_saves_answer_files(tmp_path):
    """--output DIR saves the answer via the backend renderers (--format list)."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")
    out_dir = tmp_path / "out"
    answer = AgentAnswer(answer="The answer is 42.", chunk_sources=[], kg_sources=[])
    args = argparse.Namespace(
        query="what is X?",
        config=str(config_path),
        mode="auto",
        json=False,
        output=out_dir,
        format="md,txt",
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": answer},
        ),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()) as err,
    ):
        rc = await _cmd_query(args)

    assert rc == 0
    assert list(out_dir.glob("*.md"))
    assert list(out_dir.glob("*.txt"))
    assert "saved:" in err.getvalue()  # confirmations go to stderr


async def test_cmd_query_output_invalid_format_returns_one(tmp_path):
    """--output with an all-invalid --format → SaveError → exit 1 (not a crash)."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")
    answer = AgentAnswer(answer="x", chunk_sources=[], kg_sources=[])
    args = argparse.Namespace(
        query="q",
        config=str(config_path),
        mode="auto",
        json=False,
        output=tmp_path / "out",
        format="pdf,rtf",  # neither is a valid save format
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": answer},
        ),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()) as err,
    ):
        rc = await _cmd_query(args)

    assert rc == 1
    assert "could not save" in err.getvalue()


async def test_cmd_query_returns_one_when_no_answer(tmp_path):
    """Graph that returns no final_answer → exit 1."""
    config_path = tmp_path / "corpus.toml"
    config_path.write_text("dummy", encoding="utf-8")

    args = argparse.Namespace(
        query="x",
        config=str(config_path),
        mode="auto",
        json=False,
    )
    with (
        patch(
            "knowledge_agent.corpus_config.load_corpus_config",
            return_value=MagicMock(),
        ),
        # Per-corpus embedder bridge + cache reset are glue the CLI now
        # composes; no-op them here (covered by their own suites).
        patch("knowledge_agent.corpus_config.apply_corpus_embedding_to_env"),
        patch("knowledge_agent.config.reset_after_settings_change"),
        patch(
            "knowledge_agent.graph.graph.ainvoke",
            new_callable=AsyncMock,
            return_value={"final_answer": None},
        ),
        redirect_stderr(io.StringIO()),
    ):
        rc = await _cmd_query(args)
    assert rc == 1


# ---- health ----


async def test_cmd_health_all_ok_returns_zero():
    """All-OK report → exit 0, text contains every component."""
    from knowledge_agent.health import ComponentStatus, StatusReport

    fake_report = StatusReport(
        components=(
            ComponentStatus("neo4j", True, "ok"),
            ComponentStatus("lancedb", True, "ok"),
            ComponentStatus("llm_key", True, "anthropic: set"),
            ComponentStatus("embed_key", True, "voyage: set"),
        )
    )
    args = argparse.Namespace()
    with (
        patch(
            "knowledge_agent.health.system_status",
            new_callable=AsyncMock,
            return_value=fake_report,
        ),
        redirect_stdout(io.StringIO()) as out,
    ):
        rc = await _cmd_health(args)
    assert rc == 0
    assert "neo4j" in out.getvalue()
    assert "lancedb" in out.getvalue()


async def test_cmd_health_any_component_fail_returns_one():
    """Any component reporting ok=False → exit 1."""
    from knowledge_agent.health import ComponentStatus, StatusReport

    fake_report = StatusReport(
        components=(
            ComponentStatus("neo4j", False, "RuntimeError('conn refused')"),
            ComponentStatus("lancedb", True, "ok"),
            ComponentStatus("llm_key", True, "anthropic: set"),
            ComponentStatus("embed_key", True, "voyage: set"),
        )
    )
    args = argparse.Namespace()
    with (
        patch(
            "knowledge_agent.health.system_status",
            new_callable=AsyncMock,
            return_value=fake_report,
        ),
        redirect_stdout(io.StringIO()) as out,
    ):
        rc = await _cmd_health(args)
    assert rc == 1
    assert "FAIL" in out.getvalue()


# ---- eval ----


async def test_cmd_eval_delegates_to_runner():
    """`ka eval` is a thin wrapper — it forwards the parsed args to the
    harness's `run_from_args` and returns its exit code. The eval logic
    itself lives self-contained in `evaluation/`."""
    args = argparse.Namespace(
        dataset=None, corpus=None, groups=None, max_cases=None, output_dir=None
    )
    with patch(
        "knowledge_agent.evaluation.runner.run_from_args",
        new_callable=AsyncMock,
        return_value=0,
    ) as run_mock:
        rc = await _cmd_eval(args)
    assert rc == 0
    run_mock.assert_awaited_once_with(args)
