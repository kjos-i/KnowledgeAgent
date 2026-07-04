"""Central configuration for the evaluation harness.

One place for thresholds, paths, metric-group toggles, judge config, and
run control — every other module reads an `EvalConfig` so there are no
magic numbers scattered around (mirrors the reference's single-config
policy).

Outputs (the SQLite ledger + JSON/CSV reports) default to a git-ignored
`eval_output/` under the current working directory: disposable trend
data, never committed — the same principle as `test_corpus/`. Override
any field via the constructor (tests pass `tmp_path`) or the `KA_EVAL_*`
env vars honoured by `load_eval_config()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent

# The bootstrap gold set shipped with the harness (3 ESCRT PDFs). A
# placeholder to make the harness runnable; real datasets are curated
# separately and pointed at via `KA_EVAL_DATASET` / the constructor.
DEFAULT_DATASET_PATH: Path = _PKG_DIR / "datasets" / "escrt_bootstrap.json"

# Toggleable metric groups. Deterministic groups run from Phase 1; "kg"
# arrives in Phase 2 and "judge" (DeepEval) in Phase 3. A metric with no
# `toggle_group` in the registry is always-on regardless of this set.
ALL_TOGGLE_GROUPS: frozenset[str] = frozenset({"source", "chunk", "kg", "judge"})
# "kg" is on by default — its metrics are None (skipped) for cases whose leg
# didn't run (e.g. lancedb_only), so it costs nothing on non-KG corpora.
DEFAULT_ENABLED_GROUPS: frozenset[str] = frozenset({"source", "chunk", "kg"})


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """Immutable run configuration. Build with defaults, override per field."""

    # ---- what to evaluate ------------------------------------------------
    dataset_path: Path = field(default_factory=lambda: DEFAULT_DATASET_PATH)
    """Gold queryset (JSON list of EvalCase)."""

    corpus_config_path: Path | None = None
    """`corpus.toml` of the corpus to evaluate against. None → the graph
    falls back to the ambient `Settings`/CWD corpus. For the bootstrap this
    points at `test_corpus/corpus.toml`."""

    # ---- metric toggles --------------------------------------------------
    enabled_groups: frozenset[str] = DEFAULT_ENABLED_GROUPS
    """Which toggleable metric groups run this session (see ALL_TOGGLE_GROUPS)."""

    # ---- gate thresholds -------------------------------------------------
    judge_threshold: float = 0.5
    metadata_match_threshold: float = 0.8
    required_keyword_threshold: float = 0.5

    # ---- judge (Phase 3; unused while "judge" is off) --------------------
    # OPEN DECISION (deferred to the Phase-3 judge build): who picks the
    # judge model(s) — user-set, fixed, or derived from the user's
    # configured providers — and the panel composition. The value below is
    # a PLACEHOLDER, not a decision. Note: best practice keeps the judge
    # DIFFERENT from the agent-under-test's model (self-preference bias),
    # so "reuse the search LLM" is not the default.
    judge_model: str = "claude-haiku-4-5-20251001"
    """PLACEHOLDER single judge model (see open decision above). Used when
    `judge_panel_models` is empty."""
    judge_panel_models: tuple[str, ...] = ()
    """Optional diverse-model judge panel (mean/majority). Empty = single
    `judge_model`. Configurable size is the agreed mitigation for judge
    noise/bias; membership is the open Phase-3 decision."""

    # ---- run control -----------------------------------------------------
    concurrency: int = 4
    """Max cases evaluated concurrently (asyncio bound)."""
    max_cases: int | None = None
    """Truncate the queryset (dev runs). None = all."""

    # ---- outputs (git-ignored, disposable) -------------------------------
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "eval_output")
    """Root for the ledger DB + timestamped JSON/CSV reports."""

    @property
    def ledger_path(self) -> Path:
        """SQLite ledger path, derived from `output_dir` (single knob)."""
        return self.output_dir / "eval_ledger.db"

    def gate_thresholds(self) -> dict[str, float]:
        """The pass-gate thresholds as a plain dict (persisted per run so a
        historical row records the gates it was judged under)."""
        return {
            "judge_threshold": self.judge_threshold,
            "metadata_match_threshold": self.metadata_match_threshold,
            "required_keyword_threshold": self.required_keyword_threshold,
        }


def load_eval_config(**overrides: object) -> EvalConfig:
    """Build an `EvalConfig` from defaults + `KA_EVAL_*` env vars, with
    explicit keyword `overrides` winning over both.

    Env vars honoured:
      - KA_EVAL_DATASET       → dataset_path
      - KA_EVAL_CORPUS        → corpus_config_path
      - KA_EVAL_OUTPUT_DIR    → output_dir
      - KA_EVAL_GROUPS        → enabled_groups (comma-separated)
      - KA_EVAL_MAX_CASES     → max_cases (int)
      - KA_EVAL_CONCURRENCY   → concurrency (int)
    """
    env: dict[str, object] = {}
    if v := os.environ.get("KA_EVAL_DATASET"):
        env["dataset_path"] = Path(v)
    if v := os.environ.get("KA_EVAL_CORPUS"):
        env["corpus_config_path"] = Path(v)
    if v := os.environ.get("KA_EVAL_OUTPUT_DIR"):
        env["output_dir"] = Path(v)
    if v := os.environ.get("KA_EVAL_GROUPS"):
        env["enabled_groups"] = frozenset(g.strip() for g in v.split(",") if g.strip())
    if v := os.environ.get("KA_EVAL_MAX_CASES"):
        env["max_cases"] = int(v)
    if v := os.environ.get("KA_EVAL_CONCURRENCY"):
        env["concurrency"] = int(v)

    return replace(EvalConfig(), **{**env, **overrides})  # type: ignore[arg-type]
