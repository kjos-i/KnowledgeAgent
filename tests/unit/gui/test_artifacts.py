"""Tests for `knowledge_agent.gui.artifacts` — Markdown rendering + save.

Pure-function tests; no Flet involvement. File-IO tests use tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from knowledge_agent.gui.artifacts import (
    SaveError,
    _slugify,
    render_answer_markdown,
    render_chat_markdown,
    save_answer,
    save_chat,
)
from knowledge_agent.models import AgentAnswer, ChunkSource, KGSource


def _answer(text: str, *, chunks: int = 0, kgs: int = 0) -> AgentAnswer:
    return AgentAnswer(
        answer=text,
        chunk_sources=[
            ChunkSource(chunk_id=f"d{i}#{i}", doc_id=f"d{i}")
            for i in range(chunks)
        ],
        kg_sources=[KGSource(hit_index=i) for i in range(kgs)],
    )


# ---- _slugify ----


def test_slugify_lowercases_and_hyphenates():
    assert _slugify("Hello World") == "hello-world"


def test_slugify_strips_trailing_punctuation():
    assert _slugify("foo bar!?") == "foo-bar"


def test_slugify_empty_falls_back_to_answer():
    assert _slugify("") == "answer"
    assert _slugify("!!!") == "answer"


def test_slugify_respects_max_length():
    long = "word " * 30
    assert len(_slugify(long, max_length=20)) <= 20


# ---- render_answer_markdown ----


def test_render_answer_includes_question_and_body():
    md = render_answer_markdown(_answer("the answer"), "the question")
    assert "the question" in md
    assert "the answer" in md
    assert md.startswith("# Knowledge Agent Answer")


def test_render_answer_with_no_sources_shows_none():
    md = render_answer_markdown(_answer("body"), "q")
    assert "_(none)_" in md


def test_render_answer_lists_chunk_sources_with_indices():
    md = render_answer_markdown(
        _answer("body", chunks=2), "q",
    )
    assert "**[1]**" in md
    assert "**[2]**" in md
    assert "Chunk sources" in md


def test_render_answer_lists_kg_sources_with_K_prefix():
    md = render_answer_markdown(
        _answer("body", kgs=3), "q",
    )
    assert "**[K0]**" in md
    assert "**[K1]**" in md
    assert "**[K2]**" in md
    assert "KG sources" in md


# ---- save_answer ----


def test_save_answer_writes_md_and_json_sidecar(tmp_path: Path):
    md_path, json_path = save_answer(
        _answer("body", chunks=1), "what is X", tmp_path,
    )
    assert md_path.exists() and json_path.exists()
    assert md_path.suffix == ".md"
    assert json_path.suffix == ".json"
    # Stem includes the slugified query.
    assert "what-is-x" in md_path.stem


def test_save_answer_json_round_trips_to_AgentAnswer(tmp_path: Path):
    """The JSON sidecar must be parseable back into AgentAnswer so the
    GUI's loaded-file flow can deserialise it if we add that later."""
    original = _answer("body", chunks=2, kgs=1)
    _, json_path = save_answer(original, "q", tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    restored = AgentAnswer.model_validate(data)
    assert restored.answer == original.answer
    assert len(restored.chunk_sources) == 2
    assert len(restored.kg_sources) == 1


def test_save_answer_raises_save_error_when_dir_inaccessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """OS error during write surfaces as the explicit SaveError type so
    the GUI's catch path can distinguish it from coding bugs."""
    target = tmp_path / "results"
    target.mkdir()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(SaveError, match="disk full"):
        save_answer(_answer("body"), "q", target)


# ---- save_chat / render_chat_markdown ----


def test_render_chat_markdown_renders_roles():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        SystemMessage(content="sys"),
    ]
    md = render_chat_markdown(msgs, "2026-06-30_120000")
    assert "**you**" in md
    assert "**assistant**" in md
    # System messages fall back to class name so unexpected roles still
    # render — important for safety messages that get persisted.
    assert "SystemMessage" in md


def test_save_chat_uses_chat_slug_when_query_is_none(tmp_path: Path):
    msgs = [HumanMessage(content="hi")]
    path = save_chat(msgs, None, tmp_path)
    assert path.exists()
    assert "_chat_" in path.name
