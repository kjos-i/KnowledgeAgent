"""Pydantic schema for a gold evaluation case + the queryset loader.

An `EvalCase` is one row of the gold dataset: a question, the retrieval
settings to run it under, and the expected outcomes each metric scores
against. Fields are grouped by which metric consumes them so a curator
only fills what they need:

  - retrieval metrics  → `expected_sources` (gold doc_ids)
  - chunk metrics      → `expected_chunks` (gold snippet substrings)
  - keyword checks     → `required_keywords` / `disallowed_keywords`
  - judge metrics (P3) → `expected_answer_points`
  - KG metrics (P2)    → `expected_entities` / `expected_mode`

Everything except `id` + `question` is optional, so a minimal case is
just those two.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal[
    "auto",
    "lancedb_only",
    "neo4j_only",
    "lancedb_then_neo4j",
    "neo4j_then_lancedb",
    "parallel_fused",
]
LanceDbSearchMode = Literal["hybrid", "fts", "vector"]


class RetrievalSettings(BaseModel):
    """Per-case retrieval knobs — the KA graph's invoke overrides."""

    retrieval_mode: RetrievalMode = Field(
        default="lancedb_only",
        description="Agent-level retrieval topology for this case.",
    )
    lancedb_search_mode: LanceDbSearchMode = Field(
        default="hybrid",
        description="Within-LanceDB search mode (hybrid / fts / vector).",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Final result depth.")


class EvalCase(BaseModel):
    """One gold evaluation case."""

    id: str = Field(..., min_length=1, description="Stable slug — survives reorders.")
    question: str = Field(..., min_length=1, description="The query to run.")

    # ---- retrieval / chunk gold ----
    expected_sources: list[str] = Field(
        default_factory=list,
        description="Gold doc_ids that should appear in the retrieved set.",
    )
    expected_chunks: list[str] = Field(
        default_factory=list,
        description="Gold snippet substrings that should appear in a retrieved chunk.",
    )

    # ---- keyword checks ----
    required_keywords: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)

    # ---- judge gold (Phase 3) ----
    expected_answer_points: list[str] = Field(
        default_factory=list,
        description="Facts the answer should contain (fed to the judge as expected output).",
    )

    # ---- KG gold (Phase 2) ----
    expected_entities: list[str] = Field(
        default_factory=list,
        description="Gold entity/relationship names expected in the KG rows.",
    )
    expected_mode: RetrievalMode | None = Field(
        default=None,
        description="For auto-mode cases: the leg the mode-classifier should route to.",
    )

    # ---- run config + metadata ----
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    category: str = Field(default="", description="Free-text tag for grouping.")
    notes: str = Field(default="", description="Curator documentation, ignored by metrics.")


def load_cases(path: Path | str) -> list[EvalCase]:
    """Load + validate a JSON queryset into `EvalCase`s.

    The file is a JSON array of case objects. Validation errors surface
    with the offending field so a malformed dataset fails loudly rather
    than scoring against garbage.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"eval dataset must be a JSON array of cases, got {type(raw).__name__}")
    return [EvalCase.model_validate(item) for item in raw]
