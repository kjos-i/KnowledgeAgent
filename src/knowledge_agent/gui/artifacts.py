"""Persistence + Markdown rendering for the GUI's output artifacts.

GUI-only: turns an `AgentAnswer` (or a conversation) into a human-
facing artifact — a saved Markdown file (+ JSON sidecar for the
answer) or a Markdown string for display. The CLI doesn't persist
artifacts (it prints to stdout — different consumer, different
shape).

Mirrors `research_articles_agent/gui/artifacts.py` so saved files
from both apps have a consistent layout (title block, separator,
body, source list).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from pathlib import Path

    from knowledge_agent.models import AgentAnswer, ChunkSource


class SaveError(Exception):
    """Raised when a save can't write to disk — keeps the save APIs honest."""


def render_answer_markdown(answer: AgentAnswer, query: str) -> str:
    """Render an AgentAnswer (text + sources) as Markdown for display / disk.

    KnowledgeAgent's sources split into two lists (`chunk_sources`
    from LanceDB; `kg_sources` from Neo4j Cypher rows), so we render
    them in two distinct sections — the bracket markers in the
    answer text (`[1]`, `[K0]`) reference them by index, and the
    section here makes the index visible to the reader.
    """
    lines: list[str] = []
    lines.append("# Knowledge Agent Answer")
    lines.append("")
    lines.append(f"**Question:** {query}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(answer.answer)
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Sources ({len(answer.chunk_sources)} chunk, {len(answer.kg_sources)} KG)")
    lines.append("")
    if not answer.chunk_sources and not answer.kg_sources:
        lines.append("_(none)_")
        return "\n".join(lines)

    if answer.chunk_sources:
        lines.append("### Chunk sources")
        lines.append("")
        for i, c in enumerate(answer.chunk_sources, start=1):
            lines.append(_format_chunk_source_line(i, c))
            if c.content_type == "figure" and c.image_ref:
                # Embed the figure as a Markdown image so the saved .md
                # opens with the picture visible (offline reading of
                # cited figures). Absolute path so it resolves without
                # needing to be next to the .md.
                lines.append("")
                lines.append(f"![figure {i}]({c.image_ref})")
            if c.quote:
                lines.append("")
                # For figure chunks the quote IS the caption / OCR text
                # — label it as such so the reader knows what it is.
                label = "caption / OCR" if c.content_type == "figure" else "quote"
                lines.append(f"> _{label}:_ {c.quote}")
            lines.append("")

    if answer.kg_sources:
        lines.append("### KG sources")
        lines.append("")
        for k in answer.kg_sources:
            lines.append(f"**[K{k.hit_index}]**")
            if k.quote:
                lines.append("")
                lines.append(f"> {k.quote}")
            lines.append("")
    return "\n".join(lines)


def _format_chunk_source_line(index: int, c: ChunkSource) -> str:
    """Return the top line for one chunk source in the answer's Sources
    section — different shapes for text vs. figure per the multimodal
    citation format locked 2026-07-03.

    Format table:
      - **Text chunk (or content_type=None):**
        `**[i]** doc: <doc_id> · chunk: <chunk_id>`
      - **Figure extracted from PDF / Word / PPT** (content_type='figure'
        AND `page` set): `**[i]** Figure at page P of <doc_id>`
      - **Standalone image figure** (content_type='figure' AND no `page`):
        `**[i]** Image: <doc_id>`

    Doc title would render here in place of the doc_id if the
    synthesizer populated it — for now the doc_id is what's on the
    ChunkSource. Behavior falls back gracefully when
    `content_type` / `page` are missing (older ChunkSource shapes).
    """
    if c.content_type == "figure":
        if c.page is not None:
            return f"**[{index}]** Figure at page {c.page} of `{c.doc_id}` · chunk: `{c.chunk_id}`"
        return f"**[{index}]** Image: `{c.doc_id}` · chunk: `{c.chunk_id}`"
    return f"**[{index}]** doc: `{c.doc_id}` · chunk: `{c.chunk_id}`"


_SLUG_NONALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_length: int = 50) -> str:
    """Turn free text into a filesystem-safe slug (lowercased, hyphenated).

    Trailing hyphens are stripped; an empty result falls back to
    "answer" so the caller always gets a non-empty stem.
    """
    slug = _SLUG_NONALNUM.sub("-", text.lower()).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0] or slug[:max_length]
    return slug or "answer"


def save_answer(answer: AgentAnswer, query: str, results_dir: Path) -> tuple[Path, Path]:
    """Save the answer as Markdown + JSON in `results_dir`.

    Returns both paths (`.md` and `.json`). Filenames use a
    timestamp prefix + slugified query so a results directory stays
    chronologically ordered AND human-scannable.

    Raises `SaveError` on any filesystem failure.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    stem = f"{timestamp}_{_slugify(query)}"
    md_path = results_dir / f"{stem}.md"
    json_path = results_dir / f"{stem}.json"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        md_path.write_text(
            render_answer_markdown(answer, query),
            encoding="utf-8",
        )
        json_path.write_text(
            answer.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        raise SaveError(f"could not write to {results_dir}: {exc}") from exc
    return md_path, json_path


def render_chat_markdown(messages: list[BaseMessage], timestamp: str) -> str:
    """Render a conversation as a Markdown transcript.

    Roles: HumanMessage → "you", AIMessage → "assistant", anything
    else → the class name (system messages, tool messages — rarely
    saved but rendered safely).
    """
    lines: list[str] = []
    lines.append(f"# Knowledge Agent Chat — {timestamp}")
    lines.append("")
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "you"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = type(m).__name__
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"**{role}**")
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def save_chat(messages: list[BaseMessage], query: str | None, results_dir: Path) -> Path:
    """Save the conversation as a Markdown transcript in `results_dir`.

    `query` is used for the filename slug only — when None (the
    chat was Save-Chat'd without a final query), the slug falls
    back to "chat".
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(query) if query else "chat"
    path = results_dir / f"{timestamp}_chat_{slug}.md"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_chat_markdown(messages, timestamp),
            encoding="utf-8",
        )
    except OSError as exc:
        raise SaveError(f"could not write to {results_dir}: {exc}") from exc
    return path
