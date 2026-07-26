"""Single source of truth for the retrieval mode-to-leg rule.

Two questions have one answer that must match everywhere in the app:
  1. Which store legs does a given `retrieval_mode` run (LanceDB and/or Neo4j)?
  2. Given the leg AND the within-store `lancedb_search_mode`, which nullable
     tuning knobs does the search actually read?

Both the GUI (the Retrieval tab's gray-out in `gui/_widgets/retrieval_form.py`)
and the eval harness (`evaluation.models.required_knobs`) need that answer. They
used to each restate it, and drifted. Now they both import it from here, so they
can never disagree. This is a pure leaf: standard library only, no GUI and no
heavy backend deps, so either layer can import it. It mirrors the pattern of
`config.check_window_ordering` (one shared rule both layers call), applied to
the mode-to-leg rule.
"""

from __future__ import annotations

# Which retrieval legs each agent-level mode runs. `auto` is classified at run
# time, so it is treated as running BOTH legs (the union of their knobs): the
# value is already pinned whichever way it routes.
LANCE_MODES: frozenset[str] = frozenset(
    {"auto", "lancedb_only", "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused"}
)
NEO4J_MODES: frozenset[str] = frozenset(
    {"auto", "neo4j_only", "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused"}
)


def runs_lance_leg(retrieval_mode: str) -> bool:
    """True when the mode runs the LanceDB (chunk) leg."""
    return retrieval_mode in LANCE_MODES


def runs_neo4j_leg(retrieval_mode: str) -> bool:
    """True when the mode runs the Neo4j (graph) leg."""
    return retrieval_mode in NEO4J_MODES


def mmr_applicable(retrieval_mode: str, lancedb_search_mode: str) -> bool:
    """True when MMR can run at all: the LanceDB leg runs and it is not FTS.
    MMR needs vectors, so FTS (BM25 only) can never diversify."""
    return runs_lance_leg(retrieval_mode) and lancedb_search_mode != "fts"


def active_knobs(retrieval_mode: str, lancedb_search_mode: str, use_mmr: bool) -> set[str]:
    """The nullable tuning knobs the given retrieval config actually reads at
    search time. A knob is active only when the leg (and sub-mode) that consumes
    it runs:

      - num_candidates: the LanceDB candidate pool. Hybrid and vector fetch it;
        FTS does NOT (it limits straight to top_k), so FTS excludes it.
      - rrf_rank_constant: only Hybrid fuses with RRF.
      - mmr_lambda: only when MMR runs (LanceDB leg, not FTS, use_mmr on).
      - kg_max_rows: whenever the Neo4j leg runs.
    """
    knobs: set[str] = set()
    lance = runs_lance_leg(retrieval_mode)
    is_fts = lancedb_search_mode == "fts"
    if lance and not is_fts:
        knobs.add("num_candidates")
    if lance and lancedb_search_mode == "hybrid":
        knobs.add("rrf_rank_constant")
    if lance and not is_fts and use_mmr:
        knobs.add("mmr_lambda")
    if runs_neo4j_leg(retrieval_mode):
        knobs.add("kg_max_rows")
    return knobs
