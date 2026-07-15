"""SQLite ledger — per-run + per-case evaluation history for trend tracking.

Two tables: `eval_runs` (one row per run, with provenance + run-level
averages) and `eval_cases` (one row per case per run). Both schemas are
generated from a SINGLE column declaration, so — unlike the reference —
there is no second hand-maintained column list to drift out of sync, and
therefore no import-time drift assertion. Each fixed column is declared
once as a (name, type) pair; the `CREATE TABLE` and the INSERT column
list both derive from that one list, and the auto-increment primary key
is the only column excluded from inserts (detected from its type).

Also, done to standard:
  - `PRAGMA foreign_keys = ON` per connection (SQLite ignores FKs
    otherwise), so `eval_cases.run_id` actually references a run.
  - Indexes on the columns the dashboard queries (`eval_cases.run_id`,
    `eval_runs.run_timestamp`).
  - Connections opened + CLOSED via `contextlib.closing`.
  - Add-only auto-migration (`ALTER TABLE ADD COLUMN`) so a registry that
    grows (KG / judge metrics in later phases) extends an existing DB
    without a wipe.

Growth & retention — the ledger is APPEND-ONLY: every `save_run` adds one
`eval_runs` row + one `eval_cases` row per case, and nothing prunes, caps,
or ages rows out. That is deliberate: the accumulated history IS the trend
data the dashboard plots, so auto-pruning would silently drop points. The
footprint stays small — a run is dominated by the per-case `answer` text,
so even hundreds of runs total only a few MB (SQLite scales comfortably to
GB). It is disposable, git-ignored data under `output_dir`, never
committed, and now lives ONE-PER-CORPUS (each corpus's `eval_output/`), so
each corpus's history is self-contained. To reset, delete `eval_ledger.db`
— only trend history is lost. There is no automatic retention policy and
no in-app "clear"; deleting the file is the wipe.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, Any

from knowledge_agent.evaluation.registry import case_sql_columns, run_n_columns, run_sql_columns

if TYPE_CHECKING:
    from collections.abc import Iterable

# ---------------------------------------------------------------------------
# Fixed columns — declared ONCE. CREATE + INSERT both derive from these.
# ---------------------------------------------------------------------------

_RUNS_PREAMBLE: list[tuple[str, str]] = [
    ("run_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("run_timestamp", "TEXT NOT NULL"),
    ("dataset_path", "TEXT"),
    ("dataset_hash", "TEXT"),  # SHA-256 of the scored dataset's cases (provenance)
    ("git_commit", "TEXT"),
    ("prompts_snapshot", "TEXT"),  # JSON — run provenance
    ("case_count", "INTEGER"),
    ("pass_count", "INTEGER"),
    ("pass_rate", "REAL"),
]
_RUNS_TRAILING: list[tuple[str, str]] = [
    ("enabled_groups", "TEXT"),  # JSON list
    ("gate_thresholds", "TEXT"),  # JSON dict
    # Filter columns (added 2026-07-13). Lifted out of prompts_snapshot /
    # derived from the run's paths so the dashboard can query runs by model /
    # dataset / corpus without parsing JSON. Add-only migration extends old
    # DBs (existing rows get NULL). `dataset_kind` + `recipe_hash` join here
    # with workstream C2 (the dataset recipe), which is their source.
    ("dataset_name", "TEXT"),
    ("corpus_name", "TEXT"),
    ("llm_provider", "TEXT"),
    ("synthesizer_model", "TEXT"),
    ("judge_models", "TEXT"),  # JSON list — the judge panel that actually ran
    # Dataset-recipe provenance (C2): the profile tag + the recipe fingerprint
    # (twin of dataset_hash). NULL when the dataset carries no saved recipe.
    # PROVENANCE / FILTER ONLY — the dashboard groups + compares runs by these;
    # they are NEVER read by scoring or the pass-gate (the run's actual gate
    # thresholds live in `gate_thresholds`).
    ("dataset_kind", "TEXT"),
    ("recipe_hash", "TEXT"),
]

_CASES_PREAMBLE: list[tuple[str, str]] = [
    ("case_row_id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("run_id", "INTEGER NOT NULL REFERENCES eval_runs(run_id)"),
    ("run_timestamp", "TEXT NOT NULL"),
    ("case_id", "TEXT NOT NULL"),
    ("category", "TEXT"),
    ("question", "TEXT"),
    ("status", "TEXT"),
]
_CASES_TRAILING: list[tuple[str, str]] = [
    ("answer", "TEXT"),
    ("expected_output", "TEXT"),  # gold answer: "\n"-joined expected_answer_points
    ("errors", "TEXT"),  # JSON list
    # Case provenance: manual / llm / search / chat (added 2026-07-13). FILTER
    # ONLY — lets the dashboard compare origins; never read by scoring.
    ("origin", "TEXT"),
    # Per-case retrieval settings this case ran under (added 2026-07-14): the
    # RetrievalSettings dump (mode / search mode / top_k / tuning knobs), so Run
    # Summary → Case Details shows exactly how each case was retrieved. JSON
    # dict; NULL for runs recorded before this column existed. DISPLAY ONLY —
    # never read by scoring or the pass-gate.
    ("case_settings", "TEXT"),
    # Chat provenance for origin="chat" cases: the conversation turns that
    # produced the distilled question, snapshotted at run time so Run Summary →
    # Case Details can show it. JSON list of {role, content}. DISPLAY ONLY —
    # never read by scoring or the pass-gate.
    ("source_conversation", "TEXT"),
]


def _run_columns() -> list[tuple[str, str]]:
    return _RUNS_PREAMBLE + run_sql_columns() + run_n_columns() + _RUNS_TRAILING


def _case_columns() -> list[tuple[str, str]]:
    return _CASES_PREAMBLE + case_sql_columns() + _CASES_TRAILING


def _create_sql(table: str, columns: Iterable[tuple[str, str]]) -> str:
    body = ",\n    ".join(f"{name} {sqltype}" for name, sqltype in columns)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {body}\n)"


def _insert_names(columns: Iterable[tuple[str, str]]) -> list[str]:
    """Insertable column names — everything except the auto-increment PK."""
    return [name for name, sqltype in columns if "AUTOINCREMENT" not in sqltype.upper()]


def _try_add_column(conn: sqlite3.Connection, table: str, name: str, sqltype: str) -> None:
    """Add-only migration: ALTER ADD COLUMN, swallowing only the
    'already exists' error so a real SQL problem still surfaces."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}")
    except sqlite3.OperationalError as exc:
        if "duplicate column name" not in str(exc).lower():
            raise


class EvalLedger:
    """Thin SQLite wrapper. `save_run` writes; `list_runs` / `get_run` /
    `get_run_cases` read. The readers back both the GUI history table and
    `ka eval --history` / `--show` — plain row dicts, rendered by whichever
    surface (widget / terminal / file export) consumes them."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(_create_sql("eval_runs", _run_columns()))
            conn.execute(_create_sql("eval_cases", _case_columns()))
            # Add-only migration: registry growth (P2/P3 metrics) AND fixed
            # column additions (e.g. dataset_hash) extend an existing DB. We
            # walk the FULL column list (not just the registry) so a new
            # preamble/trailing column also reaches an old DB; the
            # AUTOINCREMENT PK is skipped (ALTER can't add it), and existing
            # columns no-op via the swallowed 'duplicate column name'.
            for name, sqltype in _run_columns():
                if "AUTOINCREMENT" not in sqltype.upper():
                    _try_add_column(conn, "eval_runs", name, sqltype)
            for name, sqltype in _case_columns():
                if "AUTOINCREMENT" not in sqltype.upper():
                    _try_add_column(conn, "eval_cases", name, sqltype)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cases_run_id ON eval_cases(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON eval_runs(run_timestamp)")
            conn.commit()

    def save_run(self, report: dict[str, Any]) -> int:
        """Persist one run (summary + provenance) + its per-case rows.

        `report` is the dict `report.build_report()` produces. Returns the
        assigned `run_id`.
        """
        summary = report.get("summary", {})
        run_row: dict[str, Any] = {
            "run_timestamp": report.get("run_timestamp"),
            "dataset_path": report.get("dataset_path"),
            "dataset_hash": report.get("dataset_hash"),
            "git_commit": report.get("git_commit"),
            "prompts_snapshot": _json_or_none(report.get("prompts_snapshot")),
            "case_count": summary.get("case_count"),
            "pass_count": summary.get("pass_count"),
            "pass_rate": summary.get("pass_rate"),
            "enabled_groups": json.dumps(sorted(report.get("enabled_groups", []))),
            "gate_thresholds": json.dumps(report.get("gate_thresholds", {})),
            "dataset_name": report.get("dataset_name"),
            "corpus_name": report.get("corpus_name"),
            "llm_provider": report.get("llm_provider"),
            "synthesizer_model": report.get("synthesizer_model"),
            "judge_models": json.dumps(report.get("judge_models", [])),
            "dataset_kind": report.get("dataset_kind"),
            "recipe_hash": report.get("recipe_hash"),
        }
        for avg_key, _ in run_sql_columns():
            run_row[avg_key] = summary.get(avg_key)
        for n_key, _ in run_n_columns():
            run_row[n_key] = summary.get(n_key)

        with closing(self._connect()) as conn:
            run_id = _insert(conn, "eval_runs", _run_columns(), run_row)
            for case in report.get("results", []):
                case_row: dict[str, Any] = {
                    "run_id": run_id,
                    "run_timestamp": report.get("run_timestamp"),
                    "case_id": case.get("id"),
                    "category": case.get("category", ""),
                    "question": case.get("question"),
                    "status": case.get("status"),
                    "answer": case.get("answer"),
                    "expected_output": case.get("expected_output"),
                    "errors": json.dumps(case.get("errors", [])),
                    "origin": case.get("origin"),
                    "case_settings": _json_or_none(case.get("case_settings")),
                    "source_conversation": json.dumps(case.get("source_conversation") or []),
                }
                for col, _ in case_sql_columns():
                    case_row[col] = case.get(col)
                _insert(conn, "eval_cases", _case_columns(), case_row)
            conn.commit()
        return run_id

    def list_runs(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Recorded runs, newest first — one dict per `eval_runs` row.

        `limit` caps the count (None = all). Backs the GUI history table and
        `ka eval --history`.
        """
        sql = "SELECT * FROM eval_runs ORDER BY run_id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        """One run's summary row, or None when there's no such run."""
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM eval_runs WHERE run_id = ?", (int(run_id),)
            ).fetchone()
            return dict(row) if row is not None else None

    def get_run_cases(self, run_id: int) -> list[dict[str, Any]]:
        """Per-case rows for one run, in insertion order (empty if none)."""
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM eval_cases WHERE run_id = ? ORDER BY case_row_id",
                (int(run_id),),
            ).fetchall()
            return [dict(row) for row in rows]


def _json_or_none(value: Any) -> str | None:
    return json.dumps(value) if value is not None else None


def _insert(
    conn: sqlite3.Connection, table: str, columns: list[tuple[str, str]], row: dict[str, Any]
) -> int:
    names = _insert_names(columns)
    placeholders = ", ".join("?" for _ in names)
    values = [row.get(name) for name in names]
    cursor = conn.execute(
        f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})", values
    )
    return int(cursor.lastrowid)
