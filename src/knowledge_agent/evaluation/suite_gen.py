"""Clone-generator for test-dataset SUITES — one master fact-set → N knob-forced
files (a "matrix permutation" sweep).

A suite is N dataset files that carry the SAME facts (questions + gold) under
DIFFERENT pinned retrieval knobs, so running the suite profiles which strategy
retrieves best. `generate_suite` takes a master dataset and a list of members
(each a label + a `RetrievalSettings`) and clones the master's cases into one
file per member:

  - each case keeps its gold, takes the member's uniform knobs, and gets a
    `__<label>` suffix on its id (so a run's ledger rows don't collide);
  - every file is tagged into `suite_name` (its `suites` header), so the Run tab
    + dashboard group them as one suite;
  - the master's recipe rides along (all members run under one recipe).

Members come from two places in the UI: the premade preset catalog below
(`STRATEGIES`, grouped by `PRESET_GROUPS`) whose numbers are pinned to defaults,
or a custom knob-set captured from the retrieval form. Pure: returns
(stem, EvalDataset) pairs; the caller writes them to the corpus folder. The
preset knobs are pre-validated (each satisfies `required_knobs` for its leg), so
the generated files are runnable as-is.
"""

from __future__ import annotations

import re

from knowledge_agent.evaluation.models import EvalDataset, RetrievalSettings

# Premade retrieval presets (the curated menu). Discrete knobs define each
# preset; the numeric knobs are pinned to the global defaults (top_k 5,
# num_candidates 100, RRF 60, MMR λ 0.6, kg_max_rows 50) — a preset never tweaks
# numbers (that would be an endless matrix; a custom form knob-set covers it).
STRATEGIES: dict[str, RetrievalSettings] = {
    # ---- Hybrid group (LanceDB retrieval) ----
    "vector": RetrievalSettings(
        retrieval_mode="lancedb_only", lancedb_search_mode="vector", num_candidates=100
    ),
    "fts": RetrievalSettings(
        retrieval_mode="lancedb_only", lancedb_search_mode="fts", num_candidates=100
    ),
    "hybrid": RetrievalSettings(
        retrieval_mode="lancedb_only",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
    ),
    "hybrid_mmr": RetrievalSettings(
        retrieval_mode="lancedb_only",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        use_mmr=True,
        mmr_lambda=0.6,
    ),
    # ---- KG group (Neo4j leg involved) ----
    "kg_only": RetrievalSettings(retrieval_mode="neo4j_only", kg_max_rows=50),
    "vector_then_kg": RetrievalSettings(
        retrieval_mode="lancedb_then_neo4j",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        kg_max_rows=50,
    ),
    "kg_then_vector": RetrievalSettings(
        retrieval_mode="neo4j_then_lancedb",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        kg_max_rows=50,
    ),
    "parallel_fused": RetrievalSettings(
        retrieval_mode="parallel_fused",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        kg_max_rows=50,
    ),
}

# Human-readable labels for the picker (key → shown text).
STRATEGY_LABELS: dict[str, str] = {
    "vector": "Vector",
    "fts": "Full-text (FTS)",
    "hybrid": "Hybrid",
    "hybrid_mmr": "Hybrid + MMR",
    "kg_only": "KG only",
    "vector_then_kg": "Vector → KG",
    "kg_then_vector": "KG → Vector",
    "parallel_fused": "Parallel fused",
}

# Preset groups for the premade picker (like the judge panel's grouped list).
PRESET_GROUPS: dict[str, tuple[str, ...]] = {
    "Hybrid": ("vector", "fts", "hybrid", "hybrid_mmr"),
    "KG": ("kg_only", "vector_then_kg", "kg_then_vector", "parallel_fused"),
}


def _slug(text: str) -> str:
    """A filename-safe fragment (alnum / _ / - / .)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text.strip()).strip("-") or "suite"


def strategy_members(names: list[str]) -> list[tuple[str, RetrievalSettings]]:
    """Resolve premade preset keys into `(label, settings)` members. Raises on an
    unknown key so a typo can't silently drop a member."""
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown presets {unknown}; known: {sorted(STRATEGIES)}")
    return [(n, STRATEGIES[n]) for n in names]


def generate_suite(
    master: EvalDataset,
    suite_name: str,
    members: list[tuple[str, RetrievalSettings]],
) -> list[tuple[str, EvalDataset]]:
    """Clone `master` into one dataset per member — same facts, the member's
    uniform knobs, `__<label>` case-id suffixes, all tagged into `suite_name`.

    `members` is a list of `(label, RetrievalSettings)` — premade presets (via
    `strategy_members`) and/or custom knob-sets captured from the form. Returns
    (filename_stem, dataset) pairs; the caller writes each to
    `<corpus>/<stem>.json`. The master itself is not modified.
    """
    if not members:
        raise ValueError("a suite needs at least one member (knob-set).")
    suite_slug = _slug(suite_name)
    out: list[tuple[str, EvalDataset]] = []
    for label, settings in members:
        label_slug = _slug(label)
        cases = [
            case.model_copy(
                update={"id": f"{case.id}__{label_slug}", "retrieval": settings.model_copy()}
            )
            for case in master.cases
        ]
        ds = EvalDataset(
            status="draft",
            name=f"{master.name or suite_name} [{label}]",
            description=master.description,
            recipe=master.recipe,
            suites=[suite_name],
            cases=cases,
        )
        out.append((f"{suite_slug}__{label_slug}", ds))
    return out
