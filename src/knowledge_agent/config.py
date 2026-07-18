"""Settings loaded from environment / .env file.

Mirrors the convention used by the sibling projects in this workspace
(ResearchArticlesAgent, ResearchFundingAgent): the `.env` file lives OUTSIDE
the project tree so it cannot be accidentally committed, and required keys
fail-fast at first access rather than at import time.

Only the CLI is expected to read the .env file. A future GUI should call
`disable_env_file()` at startup so an unconfigured GUI process cannot silently
fall back to the developer's keys.
"""

import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from knowledge_agent.kg.client import Neo4jClient

logger = logging.getLogger(__name__)

# The valid LLM providers — the SINGLE SOURCE for this set. The `llm_provider`
# field below, the factory's `provider:model` parser, and any provider
# iteration all derive from `LLM_PROVIDERS` (kept in sync with the Literal via
# `get_args`, so the two can never drift).
LlmProvider = Literal["anthropic", "openai", "google", "ollama"]
LLM_PROVIDERS: tuple[LlmProvider, ...] = get_args(LlmProvider)

# Absolute path to the developer's keys file. Shared with the sibling projects
# in this workspace — one file holds all dev secrets so we don't duplicate
# them per-project.
_ENV_FILE = Path(r"C:\Users\kjosi\dotenv\.env")

# Companion file holding credentials for a SEPARATE Neo4j Desktop instance
# dedicated to smoke tests. Loaded by `load_test_env()` (NOT auto-loaded).
# Workflow: real-data instance + `.env` for normal app use; test instance
# + `.env.test` for smokes. Both instances share bolt port 7687 but
# different passwords, so a wrong-credentials cross-connect fails at
# authentication rather than silently corrupting real data.
_ENV_TEST_FILE = Path(r"C:\Users\kjosi\dotenv\.env.test")

# Process-level toggle. When True, `get_settings()` bypasses the .env file
# and consults only OS env vars (so a future keyring bridge still works).
_env_file_disabled: bool = False


def check_window_ordering(top_k: int, num_candidates: int) -> str | None:
    """Validate the retrieval candidate-pool ordering rule.

    `num_candidates` is the pre-truncation pool the LanceDB retriever
    fetches; it must be at least as large as `top_k` (the final result
    count), or the pool can't fill the requested results. Returns a
    human-readable error message when the rule is violated, else None.

    Single source of truth for this rule — the `Settings` validator, the
    GUI's Retrieval tab, and the eval-case resolver all call this so the
    check can never drift between layers.
    """
    if num_candidates < top_k:
        return f"num_candidates ({num_candidates}) must be >= top_k ({top_k})"
    return None


# Single source of truth for the default LLM model at each call site, per
# provider. Tier mapping: classifier/query-builder = cheap, cypher/synthesizer
# = smart, extractors = cheap (short, repeated, structured-output tasks).
#
# config.py is a leaf module (it imports nothing from this package), so every
# layer can import this without a circular import:
#   - Settings field defaults (below) — the CLI/headless fallback;
#   - GuiConfig / CorpusConfig field defaults — the fresh-user / per-corpus
#     defaults;
#   - the GUI provider-switch — copies a provider's whole column into the
#     node fields when the active provider changes.
# The LLM lifecycle registry no longer duplicates these — install/availability
# is its concern; the default VALUES live here.
PROVIDER_NODE_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "mode_classifier": "claude-haiku-4-5",
        "query_builder": "claude-haiku-4-5",
        "cypher_builder": "claude-sonnet-4-6",
        "synthesizer": "claude-sonnet-4-6",
        "entity_extractor": "claude-haiku-4-5",
        "triples_extractor": "claude-haiku-4-5",
    },
    "openai": {
        "mode_classifier": "gpt-4o-mini",
        "query_builder": "gpt-4o-mini",
        "cypher_builder": "gpt-4o",
        "synthesizer": "gpt-4o",
        "entity_extractor": "gpt-4o-mini",
        "triples_extractor": "gpt-4o-mini",
    },
    "google": {
        "mode_classifier": "gemini-1.5-flash",
        "query_builder": "gemini-1.5-flash",
        "cypher_builder": "gemini-1.5-pro",
        "synthesizer": "gemini-1.5-pro",
        "entity_extractor": "gemini-1.5-flash",
        "triples_extractor": "gemini-1.5-flash",
    },
    "ollama": {
        "mode_classifier": "qwen2.5:7b",
        "query_builder": "qwen2.5:7b",
        "cypher_builder": "qwen2.5:7b",
        "synthesizer": "qwen2.5:7b",
        "entity_extractor": "qwen2.5:7b",
        "triples_extractor": "qwen2.5:7b",
    },
}


class Settings(BaseSettings):
    """Runtime configuration for the research literature agent.

    Loaded from environment variables or the developer's `.env` file (path
    above, NOT inside the project tree). Required keys raise a validation
    error at first access if missing — by design (fail-fast at first real
    use, not at import time).
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ----- API keys.
    # ALL provider keys are optional. The factory validates the active
    # provider's key at first call (lazy) and raises a clear
    # `ConfigError` naming the exact env var to set if missing.
    # See `llm_factory.py` / `embedder_factory.py` for the check.
    #
    # No provider is privileged as "default that's always required" —
    # the first-launch GUI wizard picks which provider(s) to install
    # AND prompts for the matching API key. A user on pure-OpenAI never
    # needs an Anthropic/Voyage key; a user on pure-local-HF never
    # needs any cloud key.
    anthropic_api_key: str | None = Field(
        default=None,
        description=(
            "Anthropic API key. Required when `llm_provider='anthropic'`. "
            "Validated lazily at first call."
        ),
    )
    voyage_api_key: str | None = Field(
        default=None,
        description=(
            "Voyage AI API key. Required when `embedding_provider='voyage'`. Validated lazily."
        ),
    )
    openai_api_key: str | None = Field(
        default=None,
        description=(
            "OpenAI API key. Required only when `llm_provider='openai'` "
            "or `embedding_provider='openai'`. Validated lazily at first "
            "call so users who never touch OpenAI don't need to set it."
        ),
    )
    google_api_key: str | None = Field(
        default=None,
        description=(
            "Google Generative AI (Gemini) API key. Required only when "
            "`llm_provider='google'` or `embedding_provider='google'`. "
            "Validated lazily at first call."
        ),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description=(
            "Ollama daemon endpoint. Only consulted when "
            "`llm_provider='ollama'`. Default targets a local install; "
            "override for a remote daemon. Daemon is NOT pip-installable "
            "— user installs it manually from https://ollama.com."
        ),
    )

    # ----- Provider selection. Two independent toggles so users can mix
    # (Anthropic LLM + OpenAI embeddings is a real pairing). Defaults
    # match today's behaviour — Anthropic + Voyage — so no breakage on
    # upgrade for existing corpora.
    llm_provider: LlmProvider = Field(
        default="anthropic",
        description=(
            "Active LLM provider. Settings change ONLY — does NOT trigger "
            "an install. The GUI surfaces a [Install] / [Switch back] "
            "info banner when the active provider's adapter isn't "
            "installed. The factory raises a clear `ConfigError` if code "
            "invokes an uninstalled provider. See `llm_lifecycle.py`."
        ),
    )
    embedding_provider: Literal["voyage", "openai", "google", "huggingface"] = Field(
        default="voyage",
        description=(
            "Active embedding provider. Same no-auto-install rule as "
            "`llm_provider`. CAVEAT: switching embedding provider with a "
            "non-matching dimension breaks the LanceDB schema — the "
            "lifecycle's switch-provider step guards against this. See "
            "`embedder_lifecycle.py`."
        ),
    )

    # ----- Neo4j connection (knowledge-graph store). The ingestion pipeline
    # writes here at the metadata stage; the agent traverses here at query
    # time — both sides need the same config, so it lives in the shared
    # Settings.
    #
    # ALL DB settings (neo4j_uri / neo4j_user / neo4j_password / lancedb_path)
    # are required-no-default by design. Forcing the user to set them
    # explicitly in `.env` (and `.env.test`) eliminates the failure mode where
    # a forgotten config silently falls back to a default that points at the
    # wrong instance / wrong directory. See [[researchliteratureagent-test-
    # instance-setup]] for the test/real isolation pattern this enables.
    neo4j_uri: str = Field(
        ...,
        description=(
            "Neo4j Bolt endpoint (e.g. `neo4j://127.0.0.1:7687`). Required - "
            "no safe default. Set via NEO4J_URI in .env / .env.test."
        ),
    )
    neo4j_user: str = Field(
        ...,
        description=(
            "Neo4j database user. Required - no safe default. The universal "
            "Neo4j default is `neo4j`; set it explicitly via NEO4J_USER in "
            ".env / .env.test."
        ),
    )
    neo4j_password: str = Field(
        ...,
        description=(
            "Neo4j password. Required - no safe default. Set via "
            "NEO4J_PASSWORD in .env (real instance) / .env.test (test "
            "instance). You chose the value at instance creation in Neo4j "
            "Desktop."
        ),
    )
    neo4j_max_connection_pool_size: int = Field(
        default=100,
        ge=1,
        description=(
            "Max connections in the AsyncDriver's connection pool. Neo4j "
            "driver default is 100; appropriate for single-user desktop. "
            "Bump if you point at a Neo4j cluster from a multi-user "
            "deployment. Read once at driver construction; restart the "
            "process to change."
        ),
    )
    neo4j_connection_acquisition_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description=(
            "Seconds the AsyncDriver waits for a free connection from the "
            "pool before raising. 60s matches the Neo4j default; bump for "
            "slow networks or contention-heavy setups."
        ),
    )

    # ----- LanceDB connection (vector + BM25 hybrid retrieval store, local
    # file-based, no server). Path points at the folder where LanceDB will
    # create and manage its on-disk dataset files.
    lancedb_path: Path = Field(
        ...,
        description=(
            "Directory where LanceDB stores its dataset files. Required - no "
            "safe default. Set via LANCEDB_PATH in .env (real corpus) / "
            ".env.test (smoke isolation). Created on first use."
        ),
    )

    # ----- Ontology downloads (L7). Source ontology files (MeSH RDF, GO
    # OBO, ChEBI, etc.) are downloaded once and reused across ingestion runs.
    ontology_downloads_dir: Path = Field(
        default=(Path.home() / ".research-literature-agent" / "ontology-downloads"),
        description=(
            "Directory for downloaded ontology source files. Each enabled "
            "L7 ontology layer downloads its file here on first use; "
            "subsequent ingestions reuse the local copy. Safe to delete — "
            "files re-download on next ingest. Default sits under the "
            "user home directory (not in the repo); override via "
            "ONTOLOGY_DOWNLOADS_DIR for a custom location. Renamed from "
            "ontology_cache_dir on 2026-07-02 — 'cache' was misleading "
            "given the size (100s of MB) and lifetime (persistent)."
        ),
    )

    # ----- Embedding model. Read by the ingestion pipeline (writes) and the
    # agent (queries) - MUST be the same model on both sides, or query
    # vectors land in a different latent space from the indexed chunks.
    # `embedding_model` carries the active provider's model name; the
    # provider-specific fields below let users keep ALL their model
    # choices in `.env` and only the active provider's one matters at
    # runtime. The factory reads `embedding_model` directly (already the
    # active one); the provider-specific fields exist for the lifecycle
    # switch step which copies the right one in.
    embedding_model: str = Field(
        default="voyage-multimodal-3",
        description=(
            "Active embedding model name. Set automatically by the GUI's "
            "provider-switch step from the per-provider fields below; can "
            "also be overridden directly. MUST stay consistent across "
            "ingest + query for the same corpus (LanceDB pins the "
            "dimension at table creation)."
        ),
    )
    embedding_dims: int = Field(
        default=1024,
        description=(
            "Embedding vector dimension. FIXED by the active model "
            "(voyage-multimodal-3 = 1024). Changing it requires reindex + "
            "re-embed — the embedder lifecycle's switch step enforces "
            "this against the LanceDB schema before allowing a swap."
        ),
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "OpenAI embedding model used when `embedding_provider='openai'`. "
            "`text-embedding-3-small` is 1536-dim and 5x cheaper than the "
            "older ada-002; `text-embedding-3-large` is also 1536-dim by "
            "default (can be shortened via dimensions parameter)."
        ),
    )
    google_embedding_model: str = Field(
        default="models/text-embedding-004",
        description=(
            "Google Generative AI embedding model used when "
            "`embedding_provider='google'`. 768-dim by default."
        ),
    )
    hf_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace model used when `embedding_provider='huggingface'`. "
            "Default `BAAI/bge-m3` is multilingual 1024-dim (~2.3 GB). "
            "Curated menu in `embedder_lifecycle.py` — picking one from "
            "the GUI updates this field AND triggers the model download "
            "step. Manual edits skip the download; first inference will "
            "auto-download via HF cache."
        ),
    )

    # ----- Agent retrieval mode. Five retrieval topologies span the
    # two stores (LanceDB hybrid + Neo4j graph). The graph state carries
    # the per-invocation `retrieval_mode`; this default is what the agent
    # uses when the caller doesn't override.
    #
    # Build status (will move the default to 'auto' as later modes land):
    #   lancedb_only       - hybrid search in LanceDB only (PHASE 1)
    #   neo4j_only         - Cypher traversal in Neo4j only (PHASE 2)
    #   lancedb_then_neo4j - LanceDB hits, then Neo4j enrichment (PHASE 3)
    #   neo4j_then_lancedb - Neo4j filter, then LanceDB within scope (PHASE 3)
    #   parallel_fused     - both fire in parallel, results fused (PHASE 4)
    #   auto               - LLM router picks one of 1-5 per query (PHASE 5)
    default_retrieval_mode: Literal[
        "lancedb_only",
        "neo4j_only",
        "lancedb_then_neo4j",
        "neo4j_then_lancedb",
        "parallel_fused",
        "auto",
    ] = Field(
        default="lancedb_only",
        description=(
            "Default retrieval topology when the caller doesn't specify one. "
            "Per-invocation override lives on the graph state's "
            "`retrieval_mode` field."
        ),
    )

    # ----- LanceDB retrieval knobs. All query-time tunables - changing any of
    # these does NOT require re-ingest or re-embed (they only affect how
    # search runs). Mirrors the ResearchArticlesAgent ES retrieval set, with
    # one addition (lancedb_search_mode for within-store mode selection,
    # distinct from default_retrieval_mode which is the agent-level mode).
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Number of chunks returned per query (final result size). "
            "Per-invocation override lives on the graph state."
        ),
    )
    lancedb_search_mode: Literal["hybrid", "fts", "vector"] = Field(
        default="hybrid",
        description=(
            "Default search mode WITHIN LanceDB: 'hybrid' (BM25 + vector "
            "fused via RRF), 'fts' (BM25 only), or 'vector' (kNN cosine "
            "only). Distinct from `default_retrieval_mode` which picks the "
            "agent-level store(s)."
        ),
    )
    num_candidates: int = Field(
        default=100,
        ge=1,
        description=(
            "Candidate-pool size the LanceDB retriever fetches BEFORE "
            "truncating/re-ranking to top_k. In hybrid mode the BM25 + "
            "vector legs each retrieve this many rows, RRF fuses them, and "
            "the result is cut to top_k (or MMR-reranked to top_k). Higher "
            "= closer to exact nearest-neighbours + better fusion recall, "
            "slower. Must be >= top_k."
        ),
    )
    rrf_rank_constant: int = Field(
        default=60,
        ge=1,
        description=(
            "RRF rank constant `k` in the fusion score 1/(k + rank). Lower = "
            "top-ranked hits dominate more; higher = flattens the contribution "
            "across ranks. Applied to LanceDB's native hybrid RRF via "
            "`RRFReranker(K=...)`."
        ),
    )
    mmr_lambda: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "MMR relevance/diversity tradeoff: score = lambda * sim(query, c) "
            "- (1 - lambda) * max_sim(c, selected). 1.0 = pure relevance, "
            "0.0 = pure diversity. Used when the caller passes use_mmr=True "
            "and the mode supports it (hybrid/vector, not fts)."
        ),
    )
    default_use_mmr: bool = Field(
        default=False,
        description=(
            "Default for the `use_mmr` flag the LanceDB retriever node "
            "passes to `client.retrieve()`. When True, hybrid/vector "
            "searches Python-side MMR-rerank the candidate pool to "
            "boost diversity. Silently ignored for `fts` mode (no "
            "vectors). Per-invocation override lives on the graph "
            "state's `use_mmr` field."
        ),
    )
    optimize_indexes_per_ingest: bool = Field(
        default=True,
        description=(
            "When True, `ingest_document()` calls `LanceClient.ensure_indexes()` "
            "after each successful write so vector + FTS indexes stay current. "
            "INCREMENTAL optimize - not a full rebuild - typically <1s per doc. "
            "Turn off only for bulk loads where you want to defer all index "
            "work to a single optimize at the end (bulk_ops.py will do this "
            "automatically when it lands)."
        ),
    )
    min_rows_for_vector_index: int = Field(
        default=256,
        ge=1,
        description=(
            "Minimum row count before `ensure_indexes()` attempts vector "
            "index creation. LanceDB's default IVF_PQ index needs ~256 rows "
            "to train its clustering; below that, brute force scan is fine "
            "(milliseconds at small scale). FTS index has no threshold and "
            "is always created."
        ),
    )

    # ----- Neo4j retrieval knobs. Today just the row cap; grows as cross-
    # store fusion and more sophisticated graph queries land.
    kg_max_rows: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum number of rows returned from a single Neo4j query. "
            "Enforced by wrapping the LLM-generated Cypher in "
            "`CALL { ... } RETURN * LIMIT N` at retrieval time, so the cap "
            "holds even if the LLM forgets its own LIMIT clause. Distinct "
            "from `top_k` (which caps the final LanceDB chunk count) - "
            "graph queries legitimately return more rows (e.g., 50 citations "
            "of a paper) than a chunk retriever typically would."
        ),
    )

    @model_validator(mode="after")
    def _validate_retrieval_windows(self) -> "Settings":
        """Enforce the candidate-pool ordering: top_k <= num_candidates.

        `num_candidates` is the pool the retriever fetches before cutting
        to `top_k`, so a pool smaller than the final result count is a
        misconfiguration. LanceDB would surface it inconsistently at
        search time, so catch it up front with a clear message. Shares the
        rule with the GUI + eval-case resolvers via `check_window_ordering`.
        """
        message = check_window_ordering(self.top_k, self.num_candidates)
        if message is not None:
            raise ValueError(message)
        return self

    # ----- LLM nodes (mode classifier + query builders + synthesizer).
    # The agent pipeline uses up to four LLM call sites:
    #   - mode_classifier: Haiku, picks one of the 5 concrete retrieval
    #                      modes from the user question (only runs when
    #                      `retrieval_mode` is "auto").
    #   - query_builder  : Haiku, rewrites the user question into a hybrid-
    #                      search query for LanceDB.
    #   - cypher_builder : Sonnet, writes Cypher against the KG schema for
    #                      Neo4j (compositional reasoning, so Sonnet not Haiku).
    #   - synthesizer    : Sonnet, produces the final cited answer.
    # Toggles below switch nodes off for cost-saving / debugging paths.
    mode_classifier_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["mode_classifier"],
        description=(
            "Model used by the mode-classifier node (auto mode). Haiku is "
            "cheap + fast - classification into one of 5 modes is a small, "
            "repeated task that doesn't need Sonnet's depth."
        ),
    )
    query_builder_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["query_builder"],
        description=(
            "Model used by the query-builder node. Haiku is cheap + fast - "
            "the query rewrite is a small, repeated task that doesn't need "
            "the synthesizer's depth."
        ),
    )
    cypher_builder_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["cypher_builder"],
        description=(
            "Model used by the cypher-builder node. Sonnet (not Haiku) - "
            "writing schema-aware Cypher from natural language needs "
            "compositional reasoning that Haiku struggles with."
        ),
    )
    synthesizer_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["synthesizer"],
        description=(
            "Model used by the synthesizer node. Sonnet has better reasoning "
            "for citation generation and multi-source synthesis than Haiku, "
            "at ~10x the cost - but it's a one-shot per query so the absolute "
            "cost stays small."
        ),
    )
    mode_classifier_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the mode-classifier LLM. 0.0 = deterministic - "
            "structured Pydantic output (ModeChoice) works best at "
            "temperature 0."
        ),
    )
    query_builder_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the query-builder LLM. 0.0 = deterministic - "
            "recommended since the output is a structured Pydantic schema "
            "(SearchQueryRewrite) and structured output enforcement is "
            "strongest at temperature 0."
        ),
    )
    cypher_builder_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the cypher-builder LLM. 0.0 = deterministic - "
            "structured Pydantic output (CypherQueryRewrite) works best at "
            "temperature 0."
        ),
    )
    synthesizer_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the synthesizer LLM. 0.0 = deterministic - same "
            "reasoning as the query builder: structured Pydantic output works "
            "best at temperature 0."
        ),
    )
    entity_extractor_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["entity_extractor"],
        description=(
            "Model used by the LLM entity-extractor adapter (L6). Haiku is "
            "cheap + fast - one call per chunk, the input is short and the "
            "task is straightforward span extraction. ~$0.05 per 200-chunk "
            "paper at current Haiku pricing. Only consulted when "
            "`[entities].extractor = 'llm'` in the corpus's `corpus.toml`."
        ),
    )
    entity_extractor_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the LLM entity-extractor adapter. 0.0 = "
            "deterministic - structured Pydantic output (ExtractedMentions) "
            "works best at temperature 0."
        ),
    )
    triples_extractor_model: str = Field(
        default=PROVIDER_NODE_DEFAULTS["anthropic"]["triples_extractor"],
        description=(
            "Model used by the LLM triples-extractor (L8). Haiku - one "
            "call per chunk, vocabulary-constrained output (15 fixed "
            "predicates) is a small structured-output task. Only "
            "consulted when `layers.triples=true` in the corpus's "
            "`corpus.toml`. Same model as the entity extractor by "
            "default; bump independently if L8 quality needs Sonnet."
        ),
    )
    triples_extractor_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Temperature for the LLM triples-extractor. 0.0 = "
            "deterministic - the constrained predicate vocabulary + "
            "structured Pydantic output (ExtractedTriples) need stable "
            "output."
        ),
    )
    skip_query_builder: bool = Field(
        default=False,
        description=(
            "When True, the user's raw question is used verbatim as the "
            "search query - no Haiku call to rewrite it. Useful when you've "
            "crafted a search query yourself and don't want the LLM "
            "second-guessing. Per-invocation override lives on the graph "
            "state's `skip_query_builder` field."
        ),
    )
    direct_retrieval: bool = Field(
        default=False,
        description=(
            "When True, the synthesizer is skipped - the agent returns the "
            "raw retriever output (chunks and/or graph rows) wrapped as "
            "sources, with an empty `answer` field. Useful for retrieval-"
            "quality debugging, query tuning, and cases where you want to "
            "read the sources yourself. Applies across all retrieval modes "
            "(lancedb, neo4j, cross-store). Per-invocation override lives "
            "on the graph state's `direct_retrieval` field."
        ),
    )

    # ----- OpenAlex resolution. The "polite pool" gives faster + more reliable
    # responses; opt-in by providing an email here.
    openalex_mailto: str | None = Field(
        default=None,
        description=(
            "Email for OpenAlex's polite pool (faster, more reliable DOI "
            "lookups). Optional - lookups still work without it, just rate-"
            "limited more aggressively."
        ),
    )

    # ----- Central HTTP client (see `_http_client.py`).
    # Three knobs shared by every outbound HTTP call (OpenAlex, FIBO GitHub
    # tree API, Ollama daemon probe, ontology downloads). One place to tune,
    # one User-Agent string to identify ourselves to upstreams.
    http_default_timeout: float = Field(
        default=30.0,
        description=(
            "Default per-request timeout (seconds) for the central HTTP "
            "client's `get()`. Individual call sites can override (e.g. "
            "the Ollama daemon probe uses 1s; ontology downloads use None "
            "for unbounded streaming)."
        ),
    )
    http_max_retries: int = Field(
        default=3,
        description=(
            "Max retry attempts on retryable HTTP failures (429, 5xx, "
            "network errors). 0 disables retries. Stream() does NOT retry "
            "regardless — partial-download replays are unsafe."
        ),
    )
    http_user_agent: str = Field(
        default="knowledge-agent/0.x",
        description=(
            "User-Agent header on every outbound HTTP request. Identifies "
            "this app to upstream APIs; some (GitHub, OpenAlex) condition "
            "rate-limit pools on it. Override to add a contact email "
            "(e.g. 'knowledge-agent/0.x (you@example.com)') if you ship "
            "this app to others."
        ),
    )

    # ----- Parsing toggles (docling). Format support is broad by default
    # (PDF, DOCX, HTML, JATS XML, images, etc.); these knobs control OCR
    # and chunker behaviour.
    enable_pdf_ocr: bool = Field(
        default=False,
        description=(
            "OCR for PDF inputs. Default off - research papers are usually "
            "born-digital (text already embedded), so OCR adds latency for no "
            "gain. Turn on for scanned / image-only PDFs."
        ),
    )
    enable_image_ocr: bool = Field(
        default=True,
        description=(
            "OCR for standalone image-format inputs (PNG / JPG / TIFF as the "
            "whole document). Default on - without OCR, image-only inputs "
            "produce no searchable text."
        ),
    )
    chunker_strategy: Literal["hybrid", "hierarchical"] = Field(
        default="hybrid",
        description=(
            "Which docling chunker to use. 'hybrid' (default) is structure- "
            "AND token-aware - respects headings/paragraphs/tables AND splits "
            "anything over `chunk_max_tokens`. 'hierarchical' is structure- "
            "only with no token cap - chunks align to document sections "
            "exactly but can be arbitrarily large."
        ),
    )
    chunk_max_tokens: int = Field(
        default=512,
        ge=64,
        description=(
            "Max tokens per chunk (HybridChunker only - ignored by "
            "Hierarchical). Higher = broader context per chunk, fewer chunks "
            "total, more storage per chunk. Changing this requires re-chunking "
            "+ re-embedding existing documents to take effect."
        ),
    )

    # ----- Rate limiting + concurrency. The async refactor introduces
    # proactive request-rate throttling on the LLM/embedder side AND
    # bounded concurrency on the fan-out side. Both are needed: rate
    # limiter prevents 429s before they happen, semaphore prevents OOM
    # from infinite chunk-parallel awaits. The retry knob covers the
    # transient-network residual after the rate limiter does its job.
    #
    # Default `None` for `*_requests_per_second` means "no rate limit
    # wired" — LangChain's InMemoryRateLimiter is skipped for that
    # provider. Set to a positive float in `.env` when the provider's
    # documented rate limit is below your free-running throughput
    # (Anthropic free tier ~5 RPM ≈ 0.08; OpenAI tier 1 ≈ 3.0).
    anthropic_requests_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Anthropic LLM rate cap in requests/sec. None disables the "
            "limiter. Wire to your tier's documented limit (free tier "
            "≈ 0.08, paid tiers higher). Single token bucket per process."
        ),
    )
    openai_requests_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "OpenAI LLM+embedding rate cap in requests/sec. None disables "
            "the limiter. Wire to your tier's documented limit (tier 1 "
            "≈ 3.0). Single token bucket per process — applies to BOTH "
            "the chat models and embedding endpoint."
        ),
    )
    google_requests_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Google (Gemini) LLM+embedding rate cap in requests/sec. None "
            "disables the limiter. Single token bucket per process — "
            "applies to BOTH chat models and embedding endpoint."
        ),
    )
    voyage_requests_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Voyage AI embedding rate cap in requests/sec. None disables. "
            "Voyage uses its native client (not LangChain) so the limiter "
            "is applied at the factory wrapper level."
        ),
    )
    ollama_requests_per_second: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Ollama LLM rate cap in requests/sec. Usually None — local "
            "daemon has no upstream rate limit. Set only if you want to "
            "throttle against your GPU's saturation point."
        ),
    )
    llm_max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Max attempts (including the first) per LLM call. Applied via "
            "LangChain `.with_retry(stop_after_attempt=N, "
            "wait_exponential_jitter=True)` at every LLM call site. "
            "Together with the rate limiters this covers transient 429s + "
            "network blips without an external retry layer."
        ),
    )
    pipeline_max_concurrent_chunks: int = Field(
        default=8,
        ge=1,
        le=64,
        description=(
            "Max chunks the ingest pipeline processes in parallel within "
            "a single document. asyncio.Semaphore bound on the per-chunk "
            "fan-out (embedding + L6 entity extraction + L8 triples "
            "extraction). Higher = faster but more concurrent LLM calls "
            "(bumps up against rate limits)."
        ),
    )
    bulk_ops_max_concurrent_docs: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "Max documents bulk_ops processes in parallel for multi-doc "
            "operations (bulk re-embed, bulk re-extract). Each document "
            "then fans out across `pipeline_max_concurrent_chunks` of its "
            "own chunks, so effective concurrency at the API is roughly "
            "the product of the two."
        ),
    )


def disable_env_file() -> None:
    """Prevent `get_settings()` from reading the developer's .env file.

    Intended for a future GUI: cost-safety so an unconfigured GUI cannot
    silently fall back to the developer's keys. OS env vars still work
    (so a keyring-to-env bridge keeps functioning). One-way switch.
    """
    global _env_file_disabled
    _env_file_disabled = True
    get_settings.cache_clear()


# Scheduled Neo4j-driver closes that must outlive the sync reset call that
# spawned them. Holding a strong reference stops the event loop from garbage-
# collecting the task before it finishes (a bare create_task would be dropped
# mid-close — the exact hazard the installs.py audit item flagged).
_pending_aclose_tasks: set[asyncio.Task] = set()


def _drain_kg_client(client: "Neo4jClient") -> None:
    """Close an evicted KG client's Bolt driver so its connection pool is
    released instead of leaking for the rest of the process's life.

    `Neo4jClient.close()` is async (it awaits `AsyncDriver.close()`) but this
    reset is sync, so bridge the two:
      - a running loop (the GUI's Flet loop, where the driver was opened) →
        schedule the close on it and keep a reference so the task is not GC'd
        before it runs;
      - no running loop (CLI / a fresh sync context) → run it to completion.
    Best-effort: closing must never block a corpus switch, so any failure is
    logged and swallowed. This explicit close is the ONLY thing that frees the
    pool — neo4j 5.x's `AsyncDriver.__del__` merely warns for async drivers,
    so garbage collection does not drain it."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    try:
        if loop is not None:
            task = loop.create_task(client.close())
            _pending_aclose_tasks.add(task)
            task.add_done_callback(_pending_aclose_tasks.discard)
        else:
            asyncio.run(client.close())
    except Exception:
        logger.warning("failed to close evicted Neo4j driver on reset", exc_info=True)


def reset_after_key_change() -> None:
    """Drop cached state that captured an API key at construction.

    Call after a key reached `os.environ` (any path — GUI keyring
    bridge, a fresh CLI shell export, the first-launch wizard, etc.).
    Without this, the next factory call returns a client that was
    built against the OLD key:

      - `get_settings` is `lru_cache`d — the new value reaches
        `pydantic-settings` only after a clear.
      - `kg.client.get_kg_client` captures the Neo4j password at
        driver init (`AsyncDriver(auth=...)`).
      - `llm_factory._build_llm` passes the provider api_key at LLM
        build; the cache key is `(provider, model, temperature)` so
        the same (provider, model) returns the same stale client.
      - `embedder_factory._build_voyage_client` captures the Voyage
        api_key at native-client construction.
      - `embedder_factory._build_langchain_embedder` captures the
        openai/google api_key the same way.
      - `search.client.get_search_client` captures the LanceDB
        `lancedb_path` at construction. Not a key — but this also runs
        on every corpus switch, where the active corpus's lancedb_path
        changes, so the LanceDB client must rebuild too or reads +
        writes keep hitting the previous corpus's store.

    Lazy imports so the function is free to call from contexts that
    don't have the heavy provider deps installed (e.g. the CLI
    `health` check loads this module but not necessarily neo4j /
    langchain).
    """
    get_settings.cache_clear()
    from knowledge_agent.embedder_factory import (
        _build_langchain_embedder,
        _build_voyage_client,
    )
    from knowledge_agent.kg.client import get_kg_client
    from knowledge_agent.llm_factory import _build_llm
    from knowledge_agent.search.client import get_search_client

    # Grab the live KG client BEFORE evicting it so its Bolt pool can be drained.
    # `cache_clear()` only drops the reference; the AsyncDriver would otherwise
    # leak per corpus switch (its __del__ does not close async pools).
    old_kg_client = get_kg_client() if get_kg_client.cache_info().currsize else None

    _build_llm.cache_clear()
    _build_voyage_client.cache_clear()
    _build_langchain_embedder.cache_clear()
    get_kg_client.cache_clear()
    get_search_client.cache_clear()

    if old_kg_client is not None:
        _drain_kg_client(old_kg_client)


def load_test_env() -> None:
    """Load `.env.test` so smokes target the test Neo4j instance.

    Call this at the very top of a smoke script, BEFORE any other RLA
    imports that might trigger `get_settings()`. `.env.test` should
    carry credentials for a SEPARATE Neo4j Desktop instance dedicated
    to smoke runs - if you accidentally have the wrong instance
    running, authentication fails (rather than silently writing test
    artefacts into the real corpus).

    Populates `os.environ`; `override=True` so the test values win
    over anything that pydantic-settings would later read from `.env`.
    Also clears the `get_settings()` cache so any prior call gets
    re-resolved against the test creds.
    """
    from dotenv import load_dotenv

    load_dotenv(_ENV_TEST_FILE, override=True)
    get_settings.cache_clear()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Instantiated lazily so module import does not require env vars —
    important for tests that exercise structure without hitting external
    services. Honors `disable_env_file()`: when set, the .env file is
    bypassed and only OS env vars are consulted.
    """
    if _env_file_disabled:
        return Settings(_env_file=None)
    return Settings()
