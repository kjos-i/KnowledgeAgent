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

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from knowledge_agent.config import check_window_ordering
from knowledge_agent.evaluation.config import (
    DEFAULT_ENABLED_GROUPS,
    DEFAULT_JUDGE_THRESHOLD,
    DEFAULT_REQUIRED_KEYWORD_THRESHOLD,
)
from knowledge_agent.retrieval_rules import active_knobs

RetrievalMode = Literal[
    "auto",
    "lancedb_only",
    "neo4j_only",
    "lancedb_then_neo4j",
    "neo4j_then_lancedb",
    "parallel_fused",
]
LanceDbSearchMode = Literal["hybrid", "fts", "vector"]
CaseOrigin = Literal["manual", "llm", "search", "chat"]
DatasetStatus = Literal["draft", "final"]


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
    skip_query_builder: bool = Field(
        default=False,
        description=(
            "Bypass the query-rewrite LLM — run the raw question verbatim as "
            "the search query. Maps to state['skip_query_builder']."
        ),
    )
    direct_retrieval: bool = Field(
        default=False,
        description=(
            "Bypass the synthesizer LLM — return retrieved chunks as sources "
            "with no synthesized answer. Maps to state['direct_retrieval']. "
            "A direct_retrieval case is scored on retrieval only (the judge "
            "and answer-keyword gates don't apply — there's no prose)."
        ),
    )
    # ---- tuning knobs (nullable: pinned per case, or blank → global) ----
    # These four default to None ("not pinned"). At run time a None knob
    # falls back to the live global Settings value — which can shift — so a
    # reproducible case must pin every knob its config actually reads. The
    # Dataset form enforces that (required when the consuming leg runs) and
    # the runner refuses a dataset that leaves one blank; see `required_knobs`
    # / `validate_case`. use_mmr is a plain bool (never blank), so it isn't
    # in that required set.
    num_candidates: int | None = Field(
        default=None,
        ge=1,
        description=(
            "LanceDB candidate pool fetched before truncation to top_k. "
            "Required when the LanceDB leg runs; None → global num_candidates. "
            "Maps to state['num_candidates']."
        ),
    )
    rrf_rank_constant: int | None = Field(
        default=None,
        ge=1,
        description=(
            "RRF fusion constant K for hybrid search. Required when the "
            "LanceDB leg runs in hybrid mode; None → global rrf_rank_constant. "
            "Maps to state['rrf_rank_constant']."
        ),
    )
    mmr_lambda: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "MMR relevance/diversity tradeoff (1.0 = pure relevance). Required "
            "when the LanceDB leg runs with use_mmr=True; None → global "
            "mmr_lambda. Maps to state['mmr_lambda']."
        ),
    )
    use_mmr: bool = Field(
        default=False,
        description=(
            "Apply MMR re-ranking to the LanceDB candidate pool. Maps to state['use_mmr']."
        ),
    )
    kg_max_rows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Row cap for the Neo4j Cypher leg. Required when the Neo4j leg "
            "runs; None → global kg_max_rows. Maps to state['kg_max_rows']."
        ),
    )


class ConversationTurn(BaseModel):
    """One turn of the chat that produced an `origin="chat"` case.

    Provenance only — kept so the exchange behind a chat-sourced case stays
    visible (in the dataset JSON and, snapshotted at run time, under Run
    Summary → Case Details). Never read by scoring or the pass-gate.
    """

    role: str = Field(description="Who spoke: 'user', 'assistant', or 'system'.")
    content: str = Field(description="The message text.")


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

    # ---- direct-Cypher input (optional) ----
    user_cypher: str | None = Field(
        default=None,
        description=(
            "Raw read-only Cypher to run verbatim, bypassing the cypher_builder "
            "LLM (Direct-Cypher pathway). Maps to state['user_cypher']; pair "
            "with a retrieval_mode that runs the Neo4j leg (e.g. neo4j_only). "
            "Read-only rails are still enforced at execution time."
        ),
    )

    # ---- run config + metadata ----
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    category: str = Field(default="", description="Free-text tag for grouping.")
    notes: str = Field(default="", description="Curator documentation, ignored by metrics.")
    origin: CaseOrigin = Field(
        default="manual",
        description=(
            "Provenance / trust level: 'manual' = human-authored; 'search' = "
            "approved from a real Search result; 'llm' = unverified candidate "
            "generated by an LLM (review before trusting); 'chat' = question "
            "distilled from a Search conversation by the chat router (see "
            "source_conversation). Enables run-time filtering by origin; "
            "ignored by metrics."
        ),
    )
    source_conversation: list[ConversationTurn] = Field(
        default_factory=list,
        description=(
            "For origin='chat' cases: the chat turns that produced the distilled "
            "question, kept as provenance. Empty otherwise. Ignored by scoring "
            "and the pass-gate (like `notes`); snapshotted into the ledger at run "
            "time so it shows under Run Summary → Case Details."
        ),
    )
    chat_router_model: str | None = Field(
        default=None,
        description=(
            "For origin='chat' cases: the chat-router model that distilled the "
            "question. Provenance only; ignored by scoring."
        ),
    )


def required_knobs(retrieval: RetrievalSettings) -> set[str]:
    """The nullable tuning knobs this case's config actually reads at search
    time, i.e. the ones that must be pinned for the case to be reproducible.

    Delegates to the shared `retrieval_rules.active_knobs`, so this backend
    check and the GUI's gray-out (`retrieval_form.apply_gray_out`) can never
    disagree about which knob a mode uses. A blank required knob would fall
    back to the shifting global setting, so `validate_case` flags it.
    """
    return active_knobs(
        retrieval.retrieval_mode,
        retrieval.lancedb_search_mode,
        retrieval.use_mmr,
    )


def validate_case(case: EvalCase) -> list[str]:
    """Return the reasons `case` can't be run reproducibly, or [] when it's
    complete + coherent.

    Two checks: (1) every knob `required_knobs` names is pinned (not None),
    and (2) num_candidates >= top_k when the pool is pinned. Kept OUT of the
    pydantic model so `load_dataset` still loads an incomplete case (the GUI
    can show + fix it, and hand-edited / imported datasets still load) —
    enforcement happens at Run, where `runner.run` refuses a dataset with any
    invalid case before spending tokens.
    """
    rs = case.retrieval
    problems: list[str] = []
    for knob in sorted(required_knobs(rs)):
        if getattr(rs, knob) is None:
            problems.append(f"{knob} is required for retrieval_mode={rs.retrieval_mode!r}")
    if rs.num_candidates is not None:
        message = check_window_ordering(rs.top_k, rs.num_candidates)
        if message is not None:
            problems.append(message)
    return problems


def validate_dataset(cases: list[EvalCase]) -> dict[str, list[str]]:
    """Map each invalid case's id → its problems ({} when every case is
    runnable). The runner calls this before spending any tokens."""
    invalid: dict[str, list[str]] = {}
    for case in cases:
        problems = validate_case(case)
        if problems:
            invalid[case.id] = problems
    return invalid


class EvalRecipe(BaseModel):
    """The run settings BOUND to a dataset — its canonical "how to run me".

    Lives in the EvalDataset HEADER, so it is excluded from
    `compute_dataset_hash` (which fingerprints only the cases): flipping a
    threshold or the judge panel doesn't invalidate run comparability against
    the gold. The GUI Run tab auto-loads it on open, renders it frozen, and
    offers Unfreeze → deviate for a one-off run, OR Save to update this block.

    `compute_recipe_hash` fingerprints it (the twin of the dataset hash), so a
    run records WHICH recipe it used and can detect a later re-save. Field
    defaults mirror `EvalConfig` — a fresh recipe == the harness defaults.

    NOTE: there is deliberately no "dataset kind" (fact/knob) here. Judge-on/off
    and gate strictness ARE the recipe (`enabled_groups` + thresholds); whether a
    run is a quality/regression run or a knob-profiling run is a RUN choice, not a
    property of the dataset. Knob sweeps are modelled as a SUITE of files that
    share the same facts but pin different retrieval knobs (see
    `compute_facts_hash`).
    """

    enabled_groups: list[str] = Field(
        default_factory=lambda: sorted(DEFAULT_ENABLED_GROUPS),
        description="Metric groups this dataset runs under (see ALL_TOGGLE_GROUPS).",
    )
    judge_models: list[str] = Field(
        default_factory=list,
        description=(
            "The LLM-judge panel (model IDs). Empty → a single default judge "
            "resolved from the active provider. Consulted only when 'judge' is "
            "in enabled_groups."
        ),
    )
    judge_threshold: float = Field(default=DEFAULT_JUDGE_THRESHOLD, ge=0.0, le=1.0)
    required_keyword_threshold: float = Field(
        default=DEFAULT_REQUIRED_KEYWORD_THRESHOLD, ge=0.0, le=1.0
    )


class EvalDataset(BaseModel):
    """A gold dataset: a header of human metadata + the cases.

    On disk a dataset is EITHER this object
    `{"status": ..., "name": ..., "cases": [...]}` OR (legacy) a bare JSON
    array of cases. `load_dataset` reads both shapes; `load_cases` returns
    just the cases (the backward-compatible entry point the engine/runner
    use — they don't need the header). The header is human-facing only;
    metrics never read it, and it is deliberately excluded from the
    dataset hash so flipping `status` doesn't invalidate run comparability.
    """

    status: DatasetStatus = Field(
        default="draft",
        description="User lifecycle label (draft / final) — visibility only.",
    )

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_legacy_status(cls, v: object) -> object:
        # "in progress" was retired 2026-07-13 (status is draft/final only).
        # Map any older dataset's value forward so it still loads instead of
        # failing the Literal check.
        return "draft" if v == "in progress" else v

    name: str = Field(default="", description="Human-readable dataset name.")
    description: str = Field(default="", description="Optional notes about the dataset.")
    recipe: EvalRecipe | None = Field(
        default=None,
        description=(
            "The canonical run settings for this dataset (metric groups, judge "
            "panel, gate thresholds, profile). None = no recipe saved yet "
            "(legacy datasets, or one never frozen). Header-only, so it's "
            "excluded from `compute_dataset_hash`; `compute_recipe_hash` is its "
            "own fingerprint."
        ),
    )
    frozen: bool = Field(
        default=False,
        description=(
            "Whether the recipe (run settings) is LOCKED. Only ever true when "
            "status == 'final' (the `_frozen_requires_final` validator clears a "
            "stale flag otherwise). Frozen ⇒ the recipe renders read-only in "
            "both eval tabs; the user sets it by ticking 'Freeze run settings' "
            "on a Run, and clears it via Unfreeze (with a confirm). Locks the "
            "recipe ONLY — cases stay editable. Header-only, so it's excluded "
            "from `compute_dataset_hash`."
        ),
    )
    suites: list[str] = Field(
        default_factory=list,
        description=(
            "The named suites this dataset belongs to. A suite is a knob-sweep — "
            "files that share the SAME facts under different pinned knobs (e.g. "
            "'escrt-mode-sweep'). A dataset can belong to SEVERAL suites (the same "
            "knob-config file can be the baseline of two sweeps), so this is a "
            "list. Empty = the dataset is in no suite (grouping is by this tag "
            "only; there is no automatic facts-hash grouping). MEMBERSHIP only — a single "
            "RUN records which one suite it was executed as (the ledger `suite` "
            "column). Header-only, so it's excluded from `compute_dataset_hash`."
        ),
    )
    cases: list[EvalCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _frozen_requires_final(self) -> EvalDataset:
        # Invariant: only a final dataset can be frozen. If the status isn't
        # final (e.g. set back to draft), any stale `frozen` flag is cleared so
        # the two can never disagree on disk.
        if self.status != "final" and self.frozen:
            self.frozen = False
        return self


def load_dataset(path: Path | str) -> EvalDataset:
    """Load + validate a gold dataset, accepting BOTH on-disk shapes.

    Object `{status, name, cases:[...]}` → validated as-is. Bare JSON array
    (legacy) → wrapped into a default-header dataset. Validation errors
    surface with the offending field so a malformed dataset fails loudly
    rather than scoring against garbage.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return EvalDataset(cases=[EvalCase.model_validate(item) for item in raw])
    if isinstance(raw, dict):
        return EvalDataset.model_validate(raw)
    raise ValueError(f"eval dataset must be a JSON array or object, got {type(raw).__name__}")


def load_cases(path: Path | str) -> list[EvalCase]:
    """The dataset's cases (both on-disk shapes). Backward-compatible entry
    point for the engine/runner, which don't need the header."""
    return load_dataset(path).cases


def save_dataset(dataset: EvalDataset, path: Path | str) -> None:
    """Write `dataset` to `path` in the object format.

    Atomic-ish: serialise to a temp sibling, then rename, so a crash
    mid-write can't corrupt the live file (`Path.replace` is the atomic
    rename primitive on every supported OS).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dataset.model_dump(mode="json"), indent=2), encoding="utf-8")
    tmp.replace(target)


def append_case(path: Path | str, case: EvalCase) -> EvalDataset:
    """Append one case to the dataset at `path` (loading either shape), save
    it back in the object format, and return the updated dataset. Creates a
    fresh default dataset when the file doesn't exist yet."""
    target = Path(path)
    dataset = load_dataset(target) if target.exists() else EvalDataset()
    dataset.cases.append(case)
    save_dataset(dataset, target)
    return dataset


def compute_dataset_hash(cases: list[EvalCase]) -> str:
    """A stable SHA-256 fingerprint of the CASES' content.

    Order-independent (cases sorted) and field-order-independent (keys
    sorted), so it moves only when a case is added, edited, or removed —
    NOT when the dataset header (status / name) changes or the cases are
    merely reordered. Stamped on each run (the ledger's `dataset_hash`)
    so trend comparisons can tell when the gold shifted underneath them.
    """
    items = sorted(json.dumps(c.model_dump(mode="json"), sort_keys=True) for c in cases)
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


# The gold-content fields that define a case's FACTS — everything a case tests,
# with `id`, retrieval knobs, notes, category, and origin deliberately left out.
_FACTS_FIELDS: tuple[str, ...] = (
    "question",
    "expected_sources",
    "expected_chunks",
    "required_keywords",
    "disallowed_keywords",
    "expected_answer_points",
    "expected_entities",
    "expected_mode",
    "user_cypher",
)


def compute_facts_hash(cases: list[EvalCase]) -> str:
    """A stable SHA-256 fingerprint of the cases' GOLD CONTENT only — the
    question plus every expected_* / gold field, with `id`, retrieval knobs,
    notes, category, and origin EXCLUDED (see `_FACTS_FIELDS`).

    This is the SUITE identity. The members of a test-dataset suite carry the
    SAME facts but different per-case knobs — and strategy-suffixed ids like
    `case1__vector` vs `case1__graph` — so a knob- AND id-independent hash is
    what makes those files recognisable as one suite (and lets the Run tab
    auto-detect siblings). Order-independent (cases sorted) and
    key-order-independent (sorted keys), so it moves only when the questions or
    gold change — not when a knob is swept or a case is renamed/reordered.
    Contrast `compute_dataset_hash`, which hashes the WHOLE case (knobs + id
    included) and so is unique per file.
    """

    def _gold(case: EvalCase) -> str:
        dumped = case.model_dump(mode="json")
        return json.dumps({f: dumped.get(f) for f in _FACTS_FIELDS}, sort_keys=True)

    items = sorted(_gold(c) for c in cases)
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def compute_knob_hash(cases: list[EvalCase]) -> str:
    """A stable SHA-256 fingerprint of the cases' RETRIEVAL KNOBS only — each
    case's `retrieval` settings (mode / search mode / top_k / tuning knobs), with
    the facts and everything else EXCLUDED.

    The counterpart to `compute_facts_hash`: facts_hash says "same questions +
    gold", knob_hash says "same retrieval settings". Together they are what tells
    two suite MEMBERS apart — same facts, same run-settings, DIFFERENT knobs (one
    forced to vector, one to graph…). Always defined: a case whose knobs are all
    defaults hashes to a deterministic "defaults" value; a pinned / swept knob
    moves it. Order-independent (cases sorted) and key-order-independent (sorted
    keys), so it moves only when a knob changes — not when a case is reordered or
    its facts edited. Contrast `compute_dataset_hash` (facts + knobs + id, unique
    per file) and `compute_facts_hash` (gold only, shared across a suite):
    conceptually `facts_hash ⊕ knob_hash ≈ dataset_hash`.
    """
    items = sorted(json.dumps(c.retrieval.model_dump(mode="json"), sort_keys=True) for c in cases)
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def compute_recipe_hash(recipe: EvalRecipe | None) -> str | None:
    """A stable SHA-256 fingerprint of a dataset's recipe, or None when the
    dataset has no recipe.

    The twin of `compute_dataset_hash`: `dataset_hash` says "did the gold
    change?", `recipe_hash` says "did the way we run it change?". Stamped on
    each run (the ledger's `recipe_hash`) so a trend view can tell a genuine
    quality shift from a mere threshold/judge re-tune, and detect when a run
    used a since-edited recipe. Key-order-independent (sorted keys) so only a
    real value change moves it."""
    if recipe is None:
        return None
    payload = json.dumps(recipe.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
