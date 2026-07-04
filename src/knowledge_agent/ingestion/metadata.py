"""DOI extraction + OpenAlex resolution.

Two-stage: scan chunk text for DOI candidates (regex), then resolve the
first candidate via OpenAlex's `works/doi:{doi}` endpoint.

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


async def resolve_metadata(chunks: list[ParsedChunk]) -> dict[str, Any] | None:
    """Extract DOI candidates from chunks and resolve the first that succeeds.

    Returns the OpenAlex work dict on success, or None if no DOI was found
    OR every candidate either missed (404) or failed (API exception).

    Per-candidate boundary: `resolve_doi` raises on real API failure, so
    one transient error doesn't kill the walk - we log + continue to the
    next candidate.
    """
    candidates = extract_doi_candidates(chunks)
    if not candidates:
        logger.info("metadata: no DOI candidates found in chunks")
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
