"""Persistent GUI settings — API keys (OS keyring) + preferences (JSON file).

Two backends by sensitivity (mirrors ResearchArticlesAgent):

  - **Secrets** (provider API keys + Neo4j password) live in the OS
    keyring. Windows Credential Manager / macOS Keychain / Linux Secret
    Service — same store every credential-manager-aware tool uses.
  - **Everything else** (retrieval defaults, results dir, toggles)
    lives in a JSON file at the platform-conventional config dir
    (`platformdirs`).

`APP_ID` is the single source for both the keyring service name and
the config directory — kept distinct from sibling apps so a Voyage
key entered in one app doesn't surface in another.

The keyring → env bridge (`apply_keys_to_env`) is what lets
`pydantic-settings` in `config.py` pick up the secrets without ever
seeing the developer's `.env`. `app.py` calls `disable_env_file()`
at startup so the GUI process is forbidden from falling back to the
developer's `.env`.

Slice 1 scope: just the fields needed for the Search tab (retrieval
toggles + chat-router temperature + debug). Later slices extend
`GuiConfig` with corpus list, install-related toggles, etc.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Literal

import keyring
import keyring.errors
from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


# Single source for the keyring service + config-dir name. Distinct from
# sibling apps (ResearchArticlesAgent, ResearchFundingAgent) so secrets
# never collide.
APP_ID = "knowledge-agent"

# Keyring identifiers stored in the OS credential store and shown in
# Settings forms. Order is the display order in the Settings form.
API_KEY_NAMES = ("anthropic", "openai", "google", "voyage", "neo4j")

# Human-readable labels for the Settings form — decoupled from the
# keyring identifier so "neo4j" renders as "Neo4j password" rather than
# "neo4j API key".
SECRET_DISPLAY_LABELS = {
    "anthropic": "Anthropic API key",
    "openai": "OpenAI API key",
    "google": "Google API key",
    "voyage": "Voyage API key",
    "neo4j": "Neo4j password",
}

# Keyring identifier -> env var the agent's config layer reads.
# Bridging keyring -> env lets pydantic-settings pick them up (env
# overrides .env). Mirrors the env-var names declared on Settings in
# `config.py`.
KEYRING_TO_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "neo4j": "NEO4J_PASSWORD",
}


class ConfigError(Exception):
    """Raised when persistent settings can't be read or written."""


class GuiConfig(BaseModel):
    """Everything the GUI persists between sessions (excluding API keys).

    Slice 1 set. Later slices add:
      - corpora: list[Corpus] — saved dataset definitions
      - active_corpus: str | None — currently-selected dataset
      - per-provider install metadata, etc.
    """

    model_config = ConfigDict(populate_by_name=True)

    # ---- retrieval ----------------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default retrieval depth per query.",
    )
    retrieval_mode: Literal[
        "auto", "lancedb_only", "neo4j_only",
        "lancedb_then_neo4j", "neo4j_then_lancedb", "parallel_fused",
    ] = Field(
        default="auto",
        description=(
            "Retrieval mode for the agent graph. `auto` uses the "
            "mode-classifier LLM; the others force a specific mode."
        ),
    )
    skip_query_builder: bool = Field(
        default=False,
        description=(
            "When True, the query-builder LLM is skipped and the raw "
            "user message is used as the search query. Faster + cheaper."
        ),
    )
    direct_retrieve: bool = Field(
        default=False,
        description=(
            "When True, Send bypasses the chat router + synthesizer and "
            "renders the retrieved chunks directly in the Latest view. "
            "Useful for query tuning and corpus browsing without LLM cost."
        ),
    )

    # ---- chat router --------------------------------------------------
    chat_router_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the chat-router LLM. 0.0 = deterministic "
            "(prefer retrieving on substantive questions)."
        ),
    )

    # ---- app behaviour ------------------------------------------------
    keep_loaded_file_on_clear: bool = Field(
        default=True,
        description=(
            "When the Clear button is pressed, keep any answer file "
            "loaded via Open Result / paste-path visible."
        ),
    )
    debug_mode: bool = Field(
        default=False,
        description=(
            "When True, surfaces per-node progress + diagnostic details "
            "(search query, retrieved-chunk titles + scores) in the chat "
            "panel. Off = clean chat with only essential closure messages."
        ),
    )

    # ---- I/O paths ----------------------------------------------------
    results_dir: Path | None = Field(
        default=None,
        description=(
            "Where Save Answer / Save Chat write. None = "
            "`<config_dir>/results`."
        ),
    )
    corpus_config_path: Path | None = Field(
        default=None,
        description=(
            "Path to the active corpus's `corpus.toml`. None means the "
            "user hasn't picked one yet — agent calls that need a "
            "CorpusConfig will surface a banner asking them to set it."
        ),
    )


# =============================================================================
# Filesystem layout.
# =============================================================================


def _config_dir() -> Path:
    """Return the platform-conventional config dir for this app.

    Creates it if missing — `user_config_dir` resolves to a string;
    we materialise the directory so the JSON write below is safe.
    """
    path = Path(user_config_dir(APP_ID, appauthor=False, ensure_exists=True))
    return path


def _config_file() -> Path:
    return _config_dir() / "settings.json"


def active_results_dir(cfg: GuiConfig) -> Path:
    """Resolve the effective results directory.

    `cfg.results_dir` if set; otherwise `<config_dir>/results`. Created
    on demand so callers don't have to mkdir.
    """
    if cfg.results_dir is not None:
        target = cfg.results_dir
    else:
        target = _config_dir() / "results"
    target.mkdir(parents=True, exist_ok=True)
    return target


# =============================================================================
# JSON persistence (non-secret fields).
# =============================================================================


def load_config() -> GuiConfig:
    """Load the persisted GuiConfig, or default if missing/invalid.

    Defaults are silently returned when:
      - the file doesn't exist yet (first launch)
      - the file is malformed JSON (treat as corruption; let the user
        re-enter via the UI)
      - the file fails pydantic validation (schema drift across versions)
    """
    path = _config_file()
    if not path.exists():
        return GuiConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("settings.json unreadable (%r); using defaults", exc)
        return GuiConfig()
    try:
        return GuiConfig.model_validate(data)
    except ValidationError as exc:
        logger.warning("settings.json failed validation (%r); using defaults", exc)
        return GuiConfig()


def save_config(cfg: GuiConfig) -> None:
    """Persist `cfg` to `settings.json`.

    Atomic-ish write: serialise → write to `<file>.tmp` → rename. So a
    crash mid-write doesn't leave a partial file at the live path.
    `pathlib.Path.replace` is the atomic rename primitive on every
    supported OS.
    """
    path = _config_file()
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = cfg.model_dump(mode="json", exclude_none=False)
    try:
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        # Tidy up partial file so a retry starts clean.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise ConfigError(f"could not write settings.json: {exc}") from exc


# =============================================================================
# Keyring (secrets).
# =============================================================================


def get_api_key(name: str) -> str | None:
    """Read a secret from the OS keyring. None on miss.

    Keyring failures (no backend available, permission denied) are
    logged at warning level and surfaced as None — the GUI then shows
    the field as empty and the user re-enters it.
    """
    try:
        return keyring.get_password(APP_ID, name)
    except keyring.errors.KeyringError as exc:
        logger.warning("keyring read failed for %r: %r", name, exc)
        return None


def set_api_key(name: str, value: str) -> None:
    """Write a secret to the OS keyring. Empty string deletes."""
    if not value:
        try:
            keyring.delete_password(APP_ID, name)
        except keyring.errors.PasswordDeleteError:
            # Already absent — fine.
            pass
        except keyring.errors.KeyringError as exc:
            raise ConfigError(
                f"could not delete keyring entry {name!r}: {exc}"
            ) from exc
        return
    try:
        keyring.set_password(APP_ID, name, value)
    except keyring.errors.KeyringError as exc:
        raise ConfigError(
            f"could not save keyring entry {name!r}: {exc}"
        ) from exc


def apply_keys_to_env() -> None:
    """Bridge every stored keyring secret to the matching env var.

    Called once at GUI startup AFTER `disable_env_file()`. Lets
    `pydantic-settings` in `config.py` pick the secrets up via env
    without ever falling back to the developer's `.env`.

    Skips empty/missing entries — preserves whatever the shell may
    already have set (e.g. a CI environment that exports keys).
    """
    for name, env_var in KEYRING_TO_ENV.items():
        value = get_api_key(name)
        if value:
            os.environ[env_var] = value
