"""DOI extraction + OpenAlex resolution.

Two-stage: collect DOI candidates for a document, then resolve the first
via OpenAlex's `works/doi:{doi}` endpoint. Candidates are a structured
JATS `<article-id>` DOI (read straight from the XML file — docling's JATS
backend drops it from the chunk text) ahead of the chunk-text regex hits.
OpenAlex then supplies the rich metadata (title, authors, journal, licence)
for free, so the DOI is the only thing we extract locally.

Error policy (typed-errors contract):
  - `resolve_doi` distinguishes "DOI not found" (legitimate `None` — 404)
    from "API call failed" (raises — network error, non-2xx other than
    404, malformed JSON).
  - `resolve_metadata` is the per-candidate orchestrator: it catches
    `resolve_doi` raises so one transient API hit doesn't kill the walk
    of the other candidates. Returns `None` when no candidate succeeds
    (mirrors the prior fail-soft `None` return at the higher level).

For research papers, the DOI is almost always on page 1 (header, footer,
or first paragraph). We search only the first few chunks - keeps
extraction fast and avoids false positives from in-text citations of
OTHER papers that appear later in the document.
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from knowledge_agent import _http_client
from knowledge_agent.config import get_settings
from knowledge_agent.ingestion.parse import ParsedChunk

logger = logging.getLogger(__name__)

# CrossRef-recommended DOI pattern, case-insensitive. Captures the "10."
# prefix + registrant + suffix. Post-processing strips trailing punctuation
# that running text often appends.
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b", re.IGNORECASE)

# Trailing characters that can appear after a DOI in running text but
# aren't part of the DOI itself.
_DOI_TRAILING_NOISE = ".,;)]>"

OPENALEX_BASE_URL = "https://api.openalex.org"


def extract_doi_candidates(chunks: list[ParsedChunk], max_chunks_to_search: int = 3) -> list[str]:
    """Find DOI candidates in the first few chunks.

    Returns matches deduplicated, in order of first appearance, lowercased.
    Empty list = no DOI detected in the searched range.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for chunk in chunks[:max_chunks_to_search]:
        for match in _DOI_RE.finditer(chunk.text):
            raw = match.group(1)
            cleaned = raw.rstrip(_DOI_TRAILING_NOISE).lower()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                candidates.append(cleaned)
    return candidates


def doi_from_jats(path: Path) -> str | None:
    """Read the article DOI straight from a JATS XML file's
    ``<article-id pub-id-type="doi">`` element.

    JATS keeps the DOI in structured front-matter metadata that docling's
    JATS backend does NOT emit into the chunk text — so the text-regex path
    (`extract_doi_candidates`) misses it for XML articles. This reads it
    directly. Returns the lowercased DOI, or None when the file isn't
    parseable as XML or carries no article-level DOI id.

    Matches ``article-id`` specifically (the article's own id), NOT the
    ``pub-id`` elements inside reference citations, so a paper's
    bibliography can't hijack its DOI. Namespace-agnostic (matches on the
    local tag name) so it works whether or not the file declares a default
    JATS namespace.

    Uses stdlib ElementTree — fine for the user's own corpus files. An
    adversarial XML could abuse entity expansion; swap in `defusedxml` if
    untrusted sources ever get ingested.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        logger.info("doi_from_jats: could not parse %s: %r", path, exc)
        return None
    for elem in root.iter():
        local = elem.tag.rsplit("}", 1)[-1]  # drop "{namespace}" prefix if present
        if local == "article-id" and (elem.get("pub-id-type") or "").lower() == "doi":
            doi = (elem.text or "").strip()
            if doi:
                return doi.lower()
    return None


def collect_doi_candidates(
    chunks: list[ParsedChunk], *, source_path: Path | None = None
) -> list[str]:
    """DOI candidates for one document, most-authoritative first.

    A JATS ``<article-id>`` DOI (read straight from ``source_path`` when it
    is an ``.xml`` / ``.nxml`` file) is the reliable source and goes FIRST;
    the text-regex candidates from the parsed chunks follow. Deduplicated,
    order-preserving, lowercased. This is the single list both
    `resolve_metadata` and the pipeline's metadata-status check consume, so
    the two never disagree on whether a DOI exists.
    """
    ordered: list[str] = []
    if source_path is not None and source_path.suffix.lower() in {".xml", ".nxml"}:
        structured = doi_from_jats(source_path)
        if structured:
            ordered.append(structured)
    ordered.extend(extract_doi_candidates(chunks))
    seen: set[str] = set()
    deduped: list[str] = []
    for doi in ordered:
        if doi not in seen:
            seen.add(doi)
            deduped.append(doi)
    return deduped


async def resolve_doi(doi: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """Resolve a DOI to its OpenAlex work record (async).

    Returns the work JSON dict on a 200 response, or `None` on 404
    (legitimate "DOI not in OpenAlex"). Raises on:
      - network error / timeout (`httpx.HTTPError`)
      - non-200, non-404 status (`RuntimeError` wrapping the status)
      - malformed JSON in a 200 body (`ValueError`)

    The caller (`resolve_metadata`) catches these so one transient
    failure doesn't abort the walk of the remaining candidates.

    Polite-pool email (`mailto` query param) is added if `openalex_mailto`
    is set in Settings. Retry policy (429 / 5xx / network errors with
    exponential backoff) comes from the central HTTP client.
    """
    settings = get_settings()
    params: dict[str, str] = {}
    if settings.openalex_mailto:
        params["mailto"] = settings.openalex_mailto

    url = f"{OPENALEX_BASE_URL}/works/doi:{doi}"
    response = await _http_client.request(url, params=params, timeout=timeout)

    if response.status_code == 404:
        logger.info("OpenAlex: DOI not found: %r", doi)
        return None
    if response.status_code != 200:
        raise RuntimeError(f"OpenAlex: unexpected status {response.status_code} for doi={doi!r}")
    return response.json()


async def resolve_metadata(
    chunks: list[ParsedChunk], *, source_path: Path | None = None
) -> dict[str, Any] | None:
    """Collect DOI candidates and resolve the first that succeeds.

    Candidates come from `collect_doi_candidates`: a structured JATS DOI
    (when `source_path` is an `.xml` / `.nxml` article) first, then the
    chunk-text regex hits. Returns the OpenAlex work dict on success, or
    None if no DOI was found OR every candidate either missed (404) or
    failed (API exception).

    Per-candidate boundary: `resolve_doi` raises on real API failure, so
    one transient error doesn't kill the walk - we log + continue to the
    next candidate.
    """
    candidates = collect_doi_candidates(chunks, source_path=source_path)
    if not candidates:
        logger.info("metadata: no DOI candidates found")
        return None
    for doi in candidates:
        try:
            work = await resolve_doi(doi)
        except Exception as exc:
            logger.warning(
                "metadata: resolve_doi failed for %r: %r; trying next candidate",
                doi,
                exc,
            )
            continue
        if work is not None:
            logger.info("metadata: resolved DOI %r", doi)
            return work
    logger.info("metadata: %d candidate DOI(s) but none resolved", len(candidates))
    return None
