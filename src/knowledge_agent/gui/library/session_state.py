"""Per-corpus GUI session state — the "resume where you left off" sidecar.

The Ingest tab's working state — the user's *unsaved* config edits plus the
last folder / file they picked — isn't part of the corpus's committed config
(`corpus.toml`), so it needs its own home. We persist it in a small JSON
sidecar next to `corpus.toml` (`<corpus_dir>/.ka_session.json`) so it travels
with the corpus (survives Library → Relocate) and is restored on the next
launch.

Contract:
  * ``draft_config`` — the full in-memory ``CorpusConfig`` (as JSON) whenever it
    differs from the saved baseline; ``None`` when there are no unsaved edits.
    Restoring it re-lights the Discard button and the pending-changes summary.
    Cleared when the user Discards or ingests (ingest saves the draft into
    ``corpus.toml``, so draft == baseline again).
  * ``last_folder`` / ``last_file`` — the paths the folder / file pickers
    pre-fill on the Ingest tab.

Every read is fail-soft: a missing or corrupt sidecar yields an empty
``CorpusSession`` (a clean first-run state), never an exception — losing
resume state must never block ingestion. Writes are fail-soft too (logged,
not raised).

The sidecar is machine-local UI state, not corpus content: exclude it from
version control for shared / gold corpora (add ``.ka_session.json`` to the
corpus's ``.gitignore``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SIDECAR_NAME = ".ka_session.json"


@dataclass
class CorpusSession:
    """Resumable per-corpus UI state (see the module docstring)."""

    draft_config: dict[str, Any] | None = None
    last_folder: str | None = None
    last_file: str | None = None

    def is_empty(self) -> bool:
        """True when there's nothing worth persisting (so the sidecar can be
        removed rather than left as an empty husk)."""
        return self.draft_config is None and not self.last_folder and not self.last_file


def sidecar_path(corpus_toml_path: Path) -> Path:
    """Return the session sidecar path beside the given ``corpus.toml``."""
    return Path(corpus_toml_path).parent / _SIDECAR_NAME


def load_session(corpus_toml_path: Path) -> CorpusSession:
    """Load the sidecar beside ``corpus.toml``.

    A missing file, an unreadable file, or malformed JSON all yield an empty
    ``CorpusSession`` (fresh start) — never an error.
    """
    path = sidecar_path(corpus_toml_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return CorpusSession()
    except OSError as exc:
        logger.warning("session sidecar read failed at %s: %r", path, exc)
        return CorpusSession()
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("session sidecar is not valid JSON at %s: %r", path, exc)
        return CorpusSession()
    if not isinstance(data, dict):
        return CorpusSession()
    draft = data.get("draft_config")
    return CorpusSession(
        draft_config=draft if isinstance(draft, dict) else None,
        last_folder=_str_or_none(data.get("last_folder")),
        last_file=_str_or_none(data.get("last_file")),
    )


def save_session(corpus_toml_path: Path, session: CorpusSession) -> None:
    """Write the sidecar beside ``corpus.toml``.

    When ``session.is_empty()`` the sidecar is removed instead of left behind
    as an empty file. Fail-soft: a write / unlink error is logged, not raised.
    """
    path = sidecar_path(corpus_toml_path)
    if session.is_empty():
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("session sidecar unlink failed at %s: %r", path, exc)
        return
    payload = {
        "draft_config": session.draft_config,
        "last_folder": session.last_folder,
        "last_file": session.last_file,
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("session sidecar write failed at %s: %r", path, exc)


# ---- granular field updates (load → modify → save) ---------------------------
#
# The GUI event loop is single-threaded, so a read-modify-write per call is
# race-free and keeps call sites to one line. Each helper touches exactly one
# field and preserves the others.


def update_draft(corpus_toml_path: Path, draft_config: dict[str, Any] | None) -> None:
    """Persist the unsaved-config draft (or clear it with ``None``)."""
    session = load_session(corpus_toml_path)
    session.draft_config = draft_config if draft_config else None
    save_session(corpus_toml_path, session)


def update_last_folder(corpus_toml_path: Path, folder: str | None) -> None:
    """Persist the folder-picker path (or clear it with a blank / ``None``)."""
    session = load_session(corpus_toml_path)
    session.last_folder = _str_or_none(folder)
    save_session(corpus_toml_path, session)


def update_last_file(corpus_toml_path: Path, file: str | None) -> None:
    """Persist the file-picker path (or clear it with a blank / ``None``)."""
    session = load_session(corpus_toml_path)
    session.last_file = _str_or_none(file)
    save_session(corpus_toml_path, session)


def _str_or_none(value: Any) -> str | None:
    """Normalise to a non-empty stripped string, or ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
