"""Clone-generator for test-dataset SUITES — one master fact-set → N knob-forced
files (a "matrix permutation" sweep).

A suite is N dataset files that carry the SAME facts (questions + gold) under
DIFFERENT pinned retrieval knobs, so running the suite profiles which strategy
retrieves best (the matrix-permutation approach). `generate_suite` takes a
master dataset and clones its cases into one file per strategy:

  - each case keeps its gold, takes the strategy's uniform knobs, and gets a
    `__<strategy>` suffix on its id (so a run's ledger rows don't collide);
  - every file is tagged into `suite_name` (its `suites` header), so the Run tab
    + dashboard group them as one suite;
  - the master's recipe rides along (all members run under one recipe).

Pure: returns (stem, EvalDataset) pairs; the caller writes them to the corpus
folder. The strategy knobs are pre-validated (each satisfies `required_knobs`
for its leg), so the generated files are runnable as-is.
"""

from __future__ import annotations

import re

from knowledge_agent.evaluation.models import EvalDataset, RetrievalSettings

# The canonical retrieval strategies (the matrix): each forces one leg with the
# knobs that leg needs, so a generated file is runnable AND isolates that path.
STRATEGIES: dict[str, RetrievalSettings] = {
    "vector": RetrievalSettings(
        retrieval_mode="lancedb_only", lancedb_search_mode="vector", num_candidates=100
    ),
    "hybrid": RetrievalSettings(
        retrieval_mode="lancedb_only",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
    ),
    "graph": RetrievalSettings(retrieval_mode="neo4j_only", kg_max_rows=50),
    "fused": RetrievalSettings(
        retrieval_mode="parallel_fused",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        kg_max_rows=50,
    ),
    "auto": RetrievalSettings(
        retrieval_mode="auto",
        lancedb_search_mode="hybrid",
        num_candidates=100,
        rrf_rank_constant=60,
        kg_max_rows=50,
    ),
}


def _slug(text: str) -> str:
    """A filename-safe fragment (alnum / _ / - / .)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", text.strip()).strip("-") or "suite"


def generate_suite(
    master: EvalDataset, suite_name: str, strategies: list[str] | None = None
) -> list[tuple[str, EvalDataset]]:
    """Clone `master` into one dataset per strategy — same facts, the strategy's
    uniform knobs, `__<strategy>` case-id suffixes, all tagged into `suite_name`.

    `strategies` picks a subset of `STRATEGIES` (default: all, in declared
    order). Returns (filename_stem, dataset) pairs; the caller writes each to
    `<corpus>/<stem>.json`. The master itself is not modified.
    """
    chosen = strategies if strategies is not None else list(STRATEGIES)
    unknown = [s for s in chosen if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies {unknown}; known: {sorted(STRATEGIES)}")
    suite_slug = _slug(suite_name)
    out: list[tuple[str, EvalDataset]] = []
    for name in chosen:
        settings = STRATEGIES[name]
        cases = [
            case.model_copy(update={"id": f"{case.id}__{name}", "retrieval": settings.model_copy()})
            for case in master.cases
        ]
        ds = EvalDataset(
            status="draft",
            name=f"{master.name or suite_name} [{name}]",
            description=master.description,
            recipe=master.recipe,
            suites=[suite_name],
            cases=cases,
        )
        out.append((f"{suite_slug}__{name}", ds))
    return out
