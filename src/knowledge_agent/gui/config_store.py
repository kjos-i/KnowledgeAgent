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
# Settings → Keys. Order is the display order in the Settings form.
# Neo4j password is NOT in this list — it's stored per-corpus under
# `neo4j-{corpus_name}` and managed via the Library tab (Create New
# Dataset writes it; corpus-switch handler reads it). See
# `get_corpus_password` / `set_corpus_password` below.
API_KEY_NAMES = ("anthropic", "openai", "google", "voyage")

# Human-readable labels for the Settings form — decoupled from the
# keyring identifier so labels can differ from keyring names.
SECRET_DISPLAY_LABELS = {
    "anthropic": "Anthropic API key",
    "openai": "OpenAI API key",
    "google": "Google API key",
    "voyage": "Voyage API key",
}

# Keyring identifier -> env var the agent's config layer reads.
# Bridging keyring -> env lets pydantic-settings pick them up (env
# overrides .env). Mirrors the env-var names declared on Settings in
# `config.py`. NEO4J_PASSWORD is bridged separately by the Library's
# corpus-switch handler (per-corpus password → env).
KEYRING_TO_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "voyage": "VOYAGE_API_KEY",
}


class ConfigError(Exception):
    """Raised when persistent settings can't be read or written."""


class CorpusEntry(BaseModel):
    """One row in the corpus registry — everything needed to switch to
    a corpus.

    Neo4j password does NOT live here — it goes in the OS keyring
    under `f"neo4j-{name}"` so it never touches disk in plain text.
    The corpus registry itself is written to `settings.json` (non-
    secret), which is why URI / user / paths are safe here.

    `corpus_config_path` points at the corpus's `corpus.toml` — the
    file that carries the corpus's ontology toggles, layer flags,
    entity extractor, etc.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1, description="Display name in picker.")
    neo4j_uri: str = Field(description="Bolt endpoint for this corpus's DBMS.")
    neo4j_user: str = Field(default="neo4j", description="DB user.")
    lancedb_path: Path = Field(description="Directory for LanceDB files.")
    corpus_config_path: Path = Field(description="Path to corpus.toml.")


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

    # ---- retrieval knobs (mirror backend Settings; bridged to env) -----
    # All seven default to the same value as the backend `Settings`
    # field of the same name. Bridged at startup so backend
    # `pydantic-settings` picks them up; the user edits them through
    # Settings → Retrieval.
    lancedb_search_mode: Literal["hybrid", "fts", "vector"] = Field(
        default="hybrid",
        description=(
            "Default search mode WITHIN LanceDB. Distinct from "
            "`retrieval_mode` (which picks store(s) at the agent level)."
        ),
    )
    num_candidates: int = Field(
        default=100,
        ge=1,
        description=(
            "Vector-search breadth (kNN candidate pool size). Must be "
            ">= rrf_rank_window_size."
        ),
    )
    rrf_rank_constant: int = Field(
        default=60,
        ge=1,
        description=(
            "RRF rank constant `k` in 1/(k + rank). Lower = top-ranked "
            "hits dominate more; higher = flattens contribution across "
            "ranks."
        ),
    )
    rrf_rank_window_size: int = Field(
        default=50,
        ge=1,
        description=(
            "How deep into each sub-retriever's list RRF fuses before "
            "truncating to top_k. Must satisfy "
            "top_k <= this <= num_candidates."
        ),
    )
    use_mmr: bool = Field(
        default=False,
        description=(
            "When True, hybrid/vector LanceDB searches MMR-rerank the "
            "candidate pool for diversity. Bridged to USE_MMR; backend "
            "Settings.default_use_mmr picks it up. Silently ignored in "
            "FTS mode (no vectors). Per-call override via graph state."
        ),
    )
    mmr_lambda: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "MMR relevance/diversity tradeoff. 1.0 = pure relevance, "
            "0.0 = pure diversity."
        ),
    )
    mmr_candidate_multiplier: int = Field(
        default=4,
        ge=1,
        description=(
            "Candidate-pool multiplier for MMR: underlying retriever "
            "is asked for top_k * this many candidates before re-rank."
        ),
    )
    kg_max_rows: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum rows returned from a single Neo4j query. Wrapped "
            "around the LLM-generated Cypher at retrieval time."
        ),
    )

    # ---- LLM provider + per-node models + rate limits ------------------
    # Mirrors backend `Settings`. Active provider + ollama URL bridge to
    # env so factories pick them up. The 4 query-time node model +
    # temperature pairs also bridge; rate limits bridge with the
    # "unset env var = None" convention (pydantic-settings default
    # path applies). Per-call overrides via invoke_state still win.
    llm_provider: Literal["anthropic", "openai", "google", "ollama"] = Field(
        default="anthropic",
        description=(
            "Active LLM provider. Switching saves the choice + bridges "
            "to env; install/uninstall remain manual per the "
            "no-auto-install rule."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Ollama daemon endpoint. Consulted when `llm_provider` is "
            "ollama. Daemon itself isn't pip-installable — install from "
            "ollama.com."
        ),
    )
    mode_classifier_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model used by the mode-classifier node.",
    )
    mode_classifier_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Temperature for the mode-classifier LLM.",
    )
    query_builder_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Model used by the query-builder node.",
    )
    query_builder_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Temperature for the query-builder LLM.",
    )
    cypher_builder_model: str = Field(
        default="claude-sonnet-4-6",
        description="Model used by the cypher-builder node.",
    )
    cypher_builder_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Temperature for the cypher-builder LLM.",
    )
    synthesizer_model: str = Field(
        default="claude-sonnet-4-6",
        description="Model used by the synthesizer node.",
    )
    synthesizer_temperature: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Temperature for the synthesizer LLM.",
    )
    anthropic_requests_per_second: float | None = Field(
        default=None, gt=0.0,
        description=(
            "Anthropic LLM rate cap (req/sec). None disables the "
            "InMemoryRateLimiter for that provider."
        ),
    )
    openai_requests_per_second: float | None = Field(
        default=None, gt=0.0,
        description="OpenAI LLM+embedding rate cap (req/sec).",
    )
    google_requests_per_second: float | None = Field(
        default=None, gt=0.0,
        description="Google (Gemini) LLM+embedding rate cap (req/sec).",
    )
    ollama_requests_per_second: float | None = Field(
        default=None, gt=0.0,
        description="Ollama LLM rate cap (req/sec).",
    )
    llm_max_retries: int = Field(
        default=3, ge=1, le=10,
        description=(
            "Max attempts (incl. the first) per LLM call. Applied via "
            "`.with_retry(stop_after_attempt=N, "
            "wait_exponential_jitter=True)` at every LLM call site."
        ),
    )

    # ---- Embedding provider + per-provider models + Voyage rate ---------
    # Same shape as LLM. `embedding_model` mirrors the active provider's
    # per-provider field (kept in sync by `on_active_provider_changed`
    # in the Embedding tab). Per-provider fields persist each provider's
    # model choice so a A→B→A switch restores A's previous choice.
    embedding_provider: Literal[
        "voyage", "openai", "google", "huggingface"
    ] = Field(
        default="voyage",
        description=(
            "Active embedding provider. Switching saves the choice + "
            "bridges to env; dimension guard fires in slice 4 (changing "
            "dim breaks the LanceDB chunks-table schema)."
        ),
    )
    embedding_model: str = Field(
        default="voyage-multimodal-3",
        description=(
            "Active embedding model name. Mirrors the active "
            "provider's per-provider field; the bridge writes this to "
            "EMBEDDING_MODEL env var."
        ),
    )
    voyage_embedding_model: str = Field(
        default="voyage-multimodal-3",
        description=(
            "Voyage model — persisted per-provider so a switch away "
            "and back restores this choice."
        ),
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model.",
    )
    google_embedding_model: str = Field(
        default="models/text-embedding-004",
        description="Google embedding model.",
    )
    hf_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace embedding model (downloaded on first use).",
    )
    voyage_requests_per_second: float | None = Field(
        default=None, gt=0.0,
        description=(
            "Voyage embedding rate cap (req/sec). None disables the "
            "limiter. Voyage uses its native client; the limiter wraps "
            "at the factory level."
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
    restore_last_corpus: bool = Field(
        default=True,
        description=(
            "When True, the GUI loads the corpus referenced by "
            "`corpus_config_path` on startup (and falls back to "
            "`<cwd>/corpus.toml`). When False, the explicit path is "
            "skipped — only the CWD fallback applies, so the user "
            "starts without a corpus unless one is sitting in the "
            "working directory. Mirrors RA's `restore_last_corpus`."
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
            "Last folder the user picked for Save Result / Save (chat). "
            "Internal state — the picker opens here by default; pressing "
            "Save and choosing a different folder updates it. NOT exposed "
            "as a Settings form field."
        ),
    )
    corpus_config_path: Path | None = Field(
        default=None,
        description=(
            "Path to the active corpus's `corpus.toml`. Mirror of the "
            "active `CorpusEntry.corpus_config_path` — updated by the "
            "corpus-switch handler. Agent calls read this to load the "
            "corpus's TOML at query time."
        ),
    )
    # Per-corpus registry. Each entry holds the storage-level info the
    # GUI needs to switch corpora. The active corpus's URI/user/path/
    # corpus.toml-path are mirrored to the top-level fields above so
    # the existing bridges (apply_connection_to_env) keep working.
    corpora: list["CorpusEntry"] = Field(
        default_factory=list,
        description=(
            "Registered corpora. Added via Library → Create New Dataset. "
            "The active one is identified by `active_corpus_name`."
        ),
    )
    active_corpus_name: str | None = Field(
        default=None,
        description=(
            "Name of the active corpus (matches a `CorpusEntry.name`). "
            "None means no corpus is registered / selected — first-run "
            "state before the user creates one."
        ),
    )

    # ---- Backend connection params (bridged to os.environ at startup) ----
    # These three mirror the backend `Settings` fields of the same name.
    # Storing them here + bridging to env lets the end user configure
    # them through the GUI form. Without this bridge an end user would
    # have to set OS env vars before launching `ka-gui` — not a real
    # end-user workflow. CLI continues to read from `.env` separately
    # (it doesn't call `disable_env_file()`).
    neo4j_uri: str = Field(
        default="neo4j://127.0.0.1:7687",
        description=(
            "Neo4j Bolt endpoint. Bridged to `NEO4J_URI` at GUI startup. "
            "Default targets a local Neo4j Desktop install on the "
            "standard Bolt port; override for a remote / cluster setup."
        ),
    )
    neo4j_user: str = Field(
        default="neo4j",
        description=(
            "Neo4j database user. Bridged to `NEO4J_USER` at GUI "
            "startup. `neo4j` is the universal default user; override "
            "if your DBMS uses a different account."
        ),
    )
    lancedb_path: Path | None = Field(
        default=None,
        description=(
            "Directory where LanceDB stores its dataset files. Bridged "
            "to `LANCEDB_PATH` at GUI startup. None resolves to "
            "`<user_data_dir>/lancedb` at apply time so the platform-"
            "conventional location wins."
        ),
    )
    ontology_downloads_dir: Path | None = Field(
        default=None,
        description=(
            "Directory where downloaded ontology source files live "
            "(MeSH RDF, GO OBO, ChEBI, etc.). Bridged to "
            "`ONTOLOGY_DOWNLOADS_DIR` at GUI startup. None keeps the "
            "backend Settings default "
            "(`~/.research-literature-agent/ontology-downloads`). "
            "Editable via Library → Installs."
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


def get_corpus_password(corpus_name: str) -> str | None:
    """Read a corpus-scoped Neo4j password from the keyring.

    Key namespace: `f"neo4j-{corpus_name}"`. Distinct from the
    top-level `"neo4j"` entry so each corpus's DBMS can have its
    own password.
    """
    return get_api_key(f"neo4j-{corpus_name}")


def set_corpus_password(corpus_name: str, value: str) -> None:
    """Write a corpus-scoped Neo4j password. Empty deletes."""
    set_api_key(f"neo4j-{corpus_name}", value)


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


def apply_retrieval_to_env(cfg: "GuiConfig") -> None:
    """Bridge GuiConfig's retrieval knobs to the matching env vars.

    Same shape as `apply_connection_to_env()` — copies the persisted
    retrieval settings into `os.environ` so backend `Settings` reads
    them on the next `get_settings()` call. Per-query overrides via
    invoke_state still win at retrieval time; this bridge only sets
    the persistent defaults the backend sees.

    Note: GuiConfig's `retrieval_mode` maps to backend Settings's
    `default_retrieval_mode` (different field names, same concept).
    """
    os.environ["TOP_K"] = str(cfg.top_k)
    os.environ["DEFAULT_RETRIEVAL_MODE"] = cfg.retrieval_mode
    os.environ["LANCEDB_SEARCH_MODE"] = cfg.lancedb_search_mode
    os.environ["NUM_CANDIDATES"] = str(cfg.num_candidates)
    os.environ["RRF_RANK_CONSTANT"] = str(cfg.rrf_rank_constant)
    os.environ["RRF_RANK_WINDOW_SIZE"] = str(cfg.rrf_rank_window_size)
    os.environ["DEFAULT_USE_MMR"] = str(cfg.use_mmr).lower()
    os.environ["MMR_LAMBDA"] = str(cfg.mmr_lambda)
    os.environ["MMR_CANDIDATE_MULTIPLIER"] = str(cfg.mmr_candidate_multiplier)
    os.environ["KG_MAX_ROWS"] = str(cfg.kg_max_rows)
    os.environ["SKIP_QUERY_BUILDER"] = str(cfg.skip_query_builder).lower()
    os.environ["DIRECT_RETRIEVAL"] = str(cfg.direct_retrieve).lower()


def apply_llm_to_env(cfg: "GuiConfig") -> None:
    """Bridge GuiConfig's LLM fields to the matching env vars.

    Active provider + ollama URL + per-node model + temperature pairs
    bridge unconditionally. Rate limits use the unset-env-var-=-None
    convention: when `cfg.<provider>_requests_per_second is None`, we
    DELETE the env var (not set it to "") so pydantic-settings falls
    back to the default (None → no limiter).
    """
    os.environ["LLM_PROVIDER"] = cfg.llm_provider
    os.environ["OLLAMA_BASE_URL"] = cfg.ollama_base_url
    os.environ["MODE_CLASSIFIER_MODEL"] = cfg.mode_classifier_model
    os.environ["MODE_CLASSIFIER_TEMPERATURE"] = str(cfg.mode_classifier_temperature)
    os.environ["QUERY_BUILDER_MODEL"] = cfg.query_builder_model
    os.environ["QUERY_BUILDER_TEMPERATURE"] = str(cfg.query_builder_temperature)
    os.environ["CYPHER_BUILDER_MODEL"] = cfg.cypher_builder_model
    os.environ["CYPHER_BUILDER_TEMPERATURE"] = str(cfg.cypher_builder_temperature)
    os.environ["SYNTHESIZER_MODEL"] = cfg.synthesizer_model
    os.environ["SYNTHESIZER_TEMPERATURE"] = str(cfg.synthesizer_temperature)
    os.environ["LLM_MAX_RETRIES"] = str(cfg.llm_max_retries)
    # Rate limits: None → drop env var so pydantic default (None) applies.
    for provider, value in (
        ("ANTHROPIC", cfg.anthropic_requests_per_second),
        ("OPENAI", cfg.openai_requests_per_second),
        ("GOOGLE", cfg.google_requests_per_second),
        ("OLLAMA", cfg.ollama_requests_per_second),
    ):
        var = f"{provider}_REQUESTS_PER_SECOND"
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = str(value)


def apply_embedding_to_env(cfg: "GuiConfig") -> None:
    """Bridge GuiConfig's embedding fields to the matching env vars.

    Active provider + per-provider models + active model bridge
    unconditionally. Voyage rate limit uses the unset-env-var-=-None
    convention (same as the LLM rate limits).
    """
    os.environ["EMBEDDING_PROVIDER"] = cfg.embedding_provider
    os.environ["EMBEDDING_MODEL"] = cfg.embedding_model
    os.environ["OPENAI_EMBEDDING_MODEL"] = cfg.openai_embedding_model
    os.environ["GOOGLE_EMBEDDING_MODEL"] = cfg.google_embedding_model
    os.environ["HF_EMBEDDING_MODEL"] = cfg.hf_embedding_model
    var = "VOYAGE_REQUESTS_PER_SECOND"
    if cfg.voyage_requests_per_second is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = str(cfg.voyage_requests_per_second)


def apply_connection_to_env(cfg: "GuiConfig") -> None:
    """Bridge GuiConfig's connection params to the matching env vars.

    Called at GUI startup AFTER `disable_env_file()`. Same role as
    `apply_keys_to_env()` but for the URI / user / path triple. Lets
    the end user configure backend connection through the GUI form
    instead of editing OS env vars.

    `lancedb_path` resolution: if `cfg.lancedb_path is None`, falls
    back to `<user_data_dir>/lancedb` (platform-conventional). The
    directory is NOT created here — LanceDB creates it on first write.
    """
    os.environ["NEO4J_URI"] = cfg.neo4j_uri
    os.environ["NEO4J_USER"] = cfg.neo4j_user
    if cfg.lancedb_path is not None:
        os.environ["LANCEDB_PATH"] = str(cfg.lancedb_path)
    else:
        # Platform-conventional fallback. Distinct from `_config_dir`
        # (which is config files) — data goes under user_data_dir.
        from platformdirs import user_data_dir

        os.environ["LANCEDB_PATH"] = str(
            Path(user_data_dir(APP_ID, appauthor=False)) / "lancedb"
        )


def apply_ontology_downloads_dir_to_env(cfg: "GuiConfig") -> None:
    """Bridge `GuiConfig.ontology_downloads_dir` to `ONTOLOGY_DOWNLOADS_DIR`.

    None (the default) — env var is DELETED so backend Settings falls
    back to its own default (`~/.research-literature-agent/ontology-
    downloads`). Set — env var carries the user's chosen path so
    `get_settings().ontology_downloads_dir` sees it.

    Call after `disable_env_file()` alongside the other apply_*_to_env
    at GUI startup, AND after any successful save from the Installs
    tab so the change takes effect in-process without a restart.
    """
    if cfg.ontology_downloads_dir is None:
        os.environ.pop("ONTOLOGY_DOWNLOADS_DIR", None)
    else:
        os.environ["ONTOLOGY_DOWNLOADS_DIR"] = str(cfg.ontology_downloads_dir)


