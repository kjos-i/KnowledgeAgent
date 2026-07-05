"""Central configuration for the evaluation harness.

One place for thresholds, paths, metric-group toggles, judge config, and
run control — every other module reads an `EvalConfig` so there are no
magic numbers scattered around (mirrors the reference's single-config
policy).

Outputs (the SQLite ledger + JSON/CSV reports) land in a git-ignored
`eval_output/` folder: beside the evaluated corpus (the folder holding
its `corpus.toml` + `lancedb`) when `corpus_config_path` is set, else
under the current working directory. Either way it's disposable trend
data, never committed — the same principle as `test_corpus/`. Colocating
with the corpus keeps a corpus self-contained: its eval history moves
with the folder (survives a Library → Relocate) instead of orphaning.
Override the derivation via the constructor (tests pass `tmp_path`),
`--output-dir`, or the `KA_EVAL_OUTPUT_DIR` env var honoured by
`load_eval_config()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

# LangSmith tracing default project — the bucket traced runs land in (inside
# whatever workspace the API key belongs to). Tracing is OPT-IN per run
# (`runner.run(trace=...)`, `ka eval --trace`, the GUI Run-tab checkbox); it
# is never enabled by config alone. The API key is NOT here — it lives in the
# environment (`LANGSMITH_API_KEY`: from `.env` for the CLI, from the OS
# keyring for the GUI), so nothing is ever uploaded unless a run opts in.
DEFAULT_LANGSMITH_PROJECT = "knowledge-agent-eval"


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

    # ---- judge (Phase 3; used only when "judge" is enabled) --------------
    judge_models: tuple[str, ...] = ()
    """The LLM-judge PANEL — a user-set list of model IDs, one per judge
    (any models, same or different; the count is the panel size). Runs on
    the user's configured LLM provider, same choose-your-provider pattern
    as the search LLM (no hardcoded default model). Empty → a single
    default judge resolved from the active provider (see
    `judge.resolve_judge_models`). The GUI Evaluation tab (Phase 4)
    populates this list. Best practice keeps a judge model DIFFERENT from
    the agent's synthesizer (self-preference bias); a multi-model panel is
    the stronger mitigation."""

    # ---- run control -----------------------------------------------------
    concurrency: int = 4
    """Max cases evaluated concurrently (asyncio bound)."""
    max_cases: int | None = None
    """Truncate the queryset (dev runs). None = all."""

    # ---- outputs (git-ignored, disposable) -------------------------------
    output_dir: Path | None = None
    """Root for the ledger DB + timestamped JSON/CSV reports. Left None (the
    default) it is resolved in `__post_init__`: `<corpus folder>/eval_output`
    when `corpus_config_path` is set — the corpus folder is its parent, the
    same folder that holds `lancedb` — else `<cwd>/eval_output`. Pass an
    explicit path (constructor / `--output-dir` / `KA_EVAL_OUTPUT_DIR`) to
    override the derivation. Always a concrete `Path` after construction."""

    def __post_init__(self) -> None:
        # Resolve the disposable-output root once, from the corpus being
        # evaluated — so its history sits beside its `lancedb` and travels
        # with the folder (a Library → Relocate). Only when the caller didn't
        # pin an explicit path; frozen dataclass → set via object.__setattr__.
        if self.output_dir is None:
            base = self.corpus_config_path.parent if self.corpus_config_path else Path.cwd()
            object.__setattr__(self, "output_dir", base / "eval_output")

    @property
    def ledger_path(self) -> Path:
        """SQLite ledger path, derived from `output_dir` (single knob)."""
        assert self.output_dir is not None  # always resolved in __post_init__
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
    explicit keyword `overrides` winning over both. With no `output_dir` from
    any source, it is derived from the corpus folder (see `EvalConfig`).

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

    # Construct directly (not `replace(EvalConfig(), ...)`): a pre-built base
    # would already have `output_dir` resolved to the CWD before the corpus is
    # known, defeating the corpus-folder derivation in `__post_init__`.
    return EvalConfig(**{**env, **overrides})  # type: ignore[arg-type]
