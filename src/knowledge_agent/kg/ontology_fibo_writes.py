"""L7 ontology writes for FIBO (Financial Industry Business Ontology).

FIBO is the canonical OWL/RDF finance ontology - defines financial
instruments, transactions, parties, agreements, corporations, market
infrastructure. Maintained by the EDM Council; MIT license, quarterly
releases. ~2,500 classes spread across ~70 modules organized by
financial sub-domain (FBC, BE, DER, SEC, LOAN, etc.).

**Distribution + reader path.** FIBO does NOT ship as a single file.
The repo contains ~70 modular `.rdf` files wired together via a
catalog (`AboutFIBOProd.rdf`) that `owl:imports` each module. The
EDM Council also ships a consolidated ZIP, but it's gated behind
EDMConnect community registration. We use the public GitHub repo
directly:

  1. **Walker** — list every `.rdf` path in the repo via the GitHub
     trees API (`api.github.com/repos/edmcouncil/fibo/git/trees/master
     ?recursive=1`).
  2. **Cache step** — `ensure_cached` each module URL into a nested
     subdirectory under the ontology cache (`fibo/<repo_path>`).
  3. **Multi-file parse** — load every cached `.rdf` into a single
     `rdflib.Graph`, then call the shared `extract_terms_owl` helper.

On a fresh install the walker fetches ~70 files over the network
(~1-2 min); subsequent imports re-use the local cache and skip the
walker. Re-pinning to a new release = `force=True` re-import.

What we write to Neo4j (multi-label):
  (:OntologyTerm:FIBOTerm {id, label, synonyms, definition})
  (:FIBOTerm)-[:FIBO_IS_A]->(:FIBOTerm)

**IDs.** FIBO URIs follow `https://spec.edmcouncil.org/fibo/ontology/
<MODULE>/<Submodule>/<Concept>` - no `<prefix>_<num>` convention.
We extract the last URI segment and prefix it with "FIBO:" so IDs
are e.g. `FIBO:FinancialInstrument` / `FIBO:Money`. Foreign-import
classes (BFO, CMNS that FIBO imports) keep their native unprefixed
last-segment IDs - same fallback as the other OWL modules.

Lifecycle uses a custom `import_fibo` orchestrator (because the
shared `import_ontology_data` assumes a single-file download) but delegates
to the shared `write_ontology_terms` family helpers for the Neo4j-write steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from knowledge_agent import _http_client
from knowledge_agent.kg.ontology_helpers import (
    OntologyTerm,
    _owl_id_extractor,
    ensure_cached,
    extract_terms_owl,
    get_cache_dir,
    delete_ontology_terms,
    is_ontology_imported,
    write_ontology_terms,
)
from knowledge_agent.kg.ontology_provenance import OntologyProvenance
from knowledge_agent.kg.schema import FIBO_IS_A_REL, FIBO_TERM_LABEL

logger = logging.getLogger(__name__)

# Re-exported for backward-compatible test patching.
_ = ensure_cached  # noqa: F841


# ---------------------------------------------------------------------------
# Module metadata + download configuration.
# ---------------------------------------------------------------------------

DOMAIN_TAGS: tuple[str, ...] = ("finance",)
"""Domain tags this ontology serves. FIBO is THE finance ontology -
the wizard suggests it for any finance corpus that references
financial instruments, transactions, parties, agreements, or
market infrastructure."""

# GitHub repo coordinates. Master branch tracks the latest release.
FIBO_REPO = "edmcouncil/fibo"
FIBO_BRANCH = "master"
FIBO_TREE_API_URL = (
    f"https://api.github.com/repos/{FIBO_REPO}/git/trees/{FIBO_BRANCH}"
    "?recursive=1"
)
FIBO_RAW_BASE_URL = (
    f"https://raw.githubusercontent.com/{FIBO_REPO}/{FIBO_BRANCH}"
)
FIBO_CACHE_SUBDIR = "fibo"
FIBO_ID_PREFIX = "FIBO"
# Approximate size of all ~70 modules combined, surfaced in the
# install dialog. Quarterly releases keep this roughly stable.
DOWNLOAD_SIZE_MB = 30

_ONTOLOGY_NAME = "FIBO"



# ---------------------------------------------------------------------------
# Provenance: structured per-ontology install-dialog metadata. See
# kg.ontology_provenance.OntologyProvenance for field semantics. Stored
# as a module-level constant so the registry can wire it through to the
# install dialog without an extra factory hop.
# ---------------------------------------------------------------------------

# L6a entity_type labels with strong coverage by this ontology. Joined
# against each extractor's `emitted_labels` by the cross-link helpers
# in kg.ontology_lifecycle to drive the install dialog's "pairs well
# with" surface.
_FIBO_COVERS_LABELS: tuple[str, ...] = ()

_FIBO_PROVENANCE = OntologyProvenance(
    ontology_name="fibo",
    full_name="Financial Industry Business Ontology",
    publisher="EDM Council",
    license="MIT",
    source_url="https://spec.edmcouncil.org/fibo/",
    download_url="https://github.com/edmcouncil/fibo",
    file_format="RDF modular (GitHub walker, ~70 .rdf files)",
    download_size_mb=DOWNLOAD_SIZE_MB,
    estimated_terms=15000,
    domain_tags=DOMAIN_TAGS,
    covers_labels=_FIBO_COVERS_LABELS,
    description=(
                "Financial instruments, transactions, parties, agreements, "
        "corporations, market infrastructure. Modular distribution; "
        "first install is 1-2 min of GitHub fetches. "
    ),
    heavy_warning=None,
)

# Paths in the repo that don't carry actual financial-class definitions:
# - `etc/` carries test fixtures and developer scripts.
# - `.github/` carries CI config.
# Filtering them out shaves a few network calls on first install.
_SKIP_PATH_PREFIXES: tuple[str, ...] = ("etc/", ".github/")


# ---------------------------------------------------------------------------
# Public API: thin wrappers around the shared write-helpers.
# ---------------------------------------------------------------------------


async def is_imported(client) -> bool:
    """True when at least one `:FIBOTerm` node exists in Neo4j."""
    return await is_ontology_imported(
        client, term_label=FIBO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def import_fibo(client,
    *,
    force: bool = False,
    xrefs_mode: str = "none",
) -> bool:
    """Download (multi-file) + parse + write the FIBO ontology to Neo4j.

    Custom orchestrator because the shared `import_ontology_data`
    assumes a single-file download. The shape is identical otherwise:
    returns `True` when the import actually ran, `False` when FIBO was
    already imported (no-op). Download / parse / write failures
    propagate to the caller (typed-errors contract).

    On a fresh install this walks the GitHub tree, downloads ~70
    module files, then loads them into one rdflib graph for term
    extraction. Subsequent imports re-use the local cache.
    """
    if not force and await is_imported(client):
        logger.info(
            "%s: already imported; use force=True to re-import",
            _ONTOLOGY_NAME,
        )
        return False

    if force:
        logger.info(
            "%s: force=True - dropping existing data before re-import",
            _ONTOLOGY_NAME,
        )
        await delete_imported(client)

    cache_dir = await _walk_and_cache_fibo()
    terms = _read_and_extract(cache_dir)

    if not terms:
        raise RuntimeError(
            f"{_ONTOLOGY_NAME}: extracted 0 terms - unexpected, aborting write"
        )

    await write_terms(client, terms)
    return True


async def delete_imported(client) -> None:
    """DETACH DELETE every :FIBOTerm node + its :FIBO_IS_A edges."""
    await delete_ontology_terms(
        client, term_label=FIBO_TERM_LABEL, ontology_name=_ONTOLOGY_NAME,
    )


async def write_terms(client,
    terms: list[OntologyTerm],
    *,
    xrefs_mode: str = "none",
) -> None:
    """Write `:OntologyTerm:FIBOTerm` nodes + `:FIBO_IS_A` edges."""
    await write_ontology_terms(
        client, terms,
        term_label=FIBO_TERM_LABEL,
        hierarchy_rel=FIBO_IS_A_REL,
        ontology_name=_ONTOLOGY_NAME,
        xrefs_mode=xrefs_mode,
    )


# ---------------------------------------------------------------------------
# Walker: list FIBO .rdf files via GitHub API + cache each.
# ---------------------------------------------------------------------------


async def _list_fibo_rdf_paths() -> list[str]:
    """Fetch the FIBO master tree from GitHub, return all `.rdf` file paths.

    Filters out paths under `etc/` (test fixtures) and `.github/` (CI
    config) — only the actual ontology modules remain. Returns paths
    relative to the repo root, sorted for deterministic walking.

    Raises `httpx.HTTPError` on network failure; caller catches and
    surfaces as "FIBO download failed".
    """
    logger.info("%s: listing module files via %s", _ONTOLOGY_NAME, FIBO_TREE_API_URL)
    response = await _http_client.request(FIBO_TREE_API_URL, timeout=60)
    response.raise_for_status()
    tree = response.json().get("tree", [])
    paths = sorted(
        entry["path"]
        for entry in tree
        if entry.get("type") == "blob"
        and entry.get("path", "").endswith(".rdf")
        and not any(
            entry["path"].startswith(prefix)
            for prefix in _SKIP_PATH_PREFIXES
        )
    )
    logger.info(
        "%s: found %d module files to cache", _ONTOLOGY_NAME, len(paths),
    )
    return paths


async def _walk_and_cache_fibo(*, paths: list[str] | None = None) -> Path:
    """Ensure every FIBO module file is downloaded into the local cache.

    Returns the FIBO subdirectory under the ontology cache - the
    caller (`_read_and_extract`) walks it to find all `.rdf` files
    for parsing.

    `paths` override is used by tests to avoid hitting the GitHub
    API. Production callers pass nothing.
    """
    if paths is None:
        paths = await _list_fibo_rdf_paths()

    for repo_path in paths:
        url = f"{FIBO_RAW_BASE_URL}/{repo_path}"
        # ensure_cached creates parent dirs idempotently (patched for
        # this exact multi-file ontology case).
        cache_filename = f"{FIBO_CACHE_SUBDIR}/{repo_path}"
        await ensure_cached(url, cache_filename)

    return get_cache_dir() / FIBO_CACHE_SUBDIR


# ---------------------------------------------------------------------------
# Multi-file parse + extract.
# ---------------------------------------------------------------------------


def _read_and_extract(cache_dir: Path) -> list[OntologyTerm]:
    """Load every cached FIBO `.rdf` file into one rdflib graph, extract.

    `cache_dir` is the FIBO subdirectory (`<cache>/fibo/`). Walks
    recursively for `*.rdf` files. Uses the shared
    `extract_terms_owl` with a FIBO-specific id_extractor so class
    URIs get the "FIBO:" prefix our convention expects.
    """
    import rdflib

    rdf_files = sorted(cache_dir.rglob("*.rdf"))
    logger.info(
        "%s: parsing %d module files into a single graph",
        _ONTOLOGY_NAME, len(rdf_files),
    )

    graph = rdflib.Graph()
    for rdf_path in rdf_files:
        try:
            graph.parse(str(rdf_path), format="xml")
        except Exception as exc:
            # One module's parse failure shouldn't kill the whole
            # import - log + skip. Most FIBO modules are independent
            # enough that the rest still yield useful term coverage.
            logger.warning(
                "%s: parse failed for %s: %r - skipping",
                _ONTOLOGY_NAME, rdf_path, exc,
            )

    return extract_terms_owl(
        graph,
        id_prefix=FIBO_ID_PREFIX,
        id_extractor=_fibo_id_extractor,
    )


def _fibo_id_extractor(uri: Any) -> str:
    """Custom id extractor for FIBO URIs.

    FIBO URIs follow `https://spec.edmcouncil.org/fibo/ontology/.../
    <ClassName>` - no `<prefix>_<num>` pattern, just a bare class
    name in the last URI segment. We add the `FIBO:` prefix so IDs
    follow the rest of the project's prefixed convention
    (e.g. `FIBO:FinancialInstrument`).

    Foreign-import classes (BFO, CMNS, OWL built-ins) keep their
    unprefixed last-segment IDs via the default `_owl_id_extractor`
    fallback - same behaviour as the other OWL modules.
    """
    text = str(uri)
    if "edmcouncil.org/fibo" in text:
        if "#" in text:
            last = text.rsplit("#", 1)[1]
        else:
            last = text.rsplit("/", 1)[1]
        if last:
            return f"FIBO:{last}"
    return _owl_id_extractor(uri)
