"""Deterministic metric compute functions.

Pure functions over primitives (lists of doc_ids / chunk texts / the
answer string) — no LLM, no KA objects, no framework. The adapter/engine
pulls these primitives out of the typed graph state and passes them in,
so the metrics stay provider- and framework-agnostic and trivially
testable.

Two relevance notions:
  - source-level: a retrieved chunk is relevant if its doc_id is in the
    case's gold `expected_sources`.
  - chunk-level: a retrieved chunk is relevant if any gold snippet from
    `expected_chunks` is a substring of its normalized text.

A metric family returns None for every metric when the case carries no
gold for it, so `safe_mean` drops it from run-level averages rather than
scoring a vacuous 0 that drags the mean down.
"""

from __future__ import annotations

import math
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_SOURCE_KEYS = ("hit_at_k", "mrr", "precision_at_k", "recall_at_k", "ndcg_at_k")
_CHUNK_KEYS = (
    "chunk_hit_at_k",
    "chunk_mrr",
    "chunk_precision_at_k",
    "chunk_recall_at_k",
    "chunk_ndcg_at_k",
)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, drop punctuation, and collapse whitespace — so
    substring matches ignore case/accent/punctuation/spacing noise.

    Punctuation handling mirrors the reference harness (among symbols it keeps
    only ``.`` ``%`` ``-``), so keyword and snippet matching are comparably
    forgiving. UNLIKE the reference — which drops every non-ASCII character —
    non-ASCII *letters and digits* are kept (Greek, CJK, accented bases): a
    general, multi-domain corpus shouldn't lose ``β``/``µ``/``北`` just because
    they aren't ASCII, and keeping them makes matching more precise (``α`` stays
    distinct from ``β``). Accents (combining marks) are still stripped, so
    ``café`` == ``cafe``.
    """
    nfkd = unicodedata.normalize("NFKD", text or "")
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    # Keep letters/digits of ANY script (`isalnum`) plus the few symbols the
    # reference keeps; every other character (comma, slash, underscore, quotes,
    # …) becomes a space. This strips punctuation like the reference but without
    # its ASCII-only restriction.
    kept = "".join(c if (c.isalnum() or c in ".%-" or c.isspace()) else " " for c in no_accents)
    return " ".join(kept.split())


def safe_mean(values: Iterable[float | None]) -> float | None:
    """Mean of the non-None values, or None when there are none."""
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


# ---------------------------------------------------------------------------
# Ranking primitives (binary relevance list, in rank order).
# ---------------------------------------------------------------------------


def hit_at_k(relevance: Sequence[bool]) -> float:
    """1.0 if any retrieved item is relevant, else 0.0."""
    return 1.0 if any(relevance) else 0.0


def mrr(relevance: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none)."""
    for i, rel in enumerate(relevance):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(relevance: Sequence[bool]) -> float:
    """Fraction of retrieved slots that are relevant."""
    return (sum(relevance) / len(relevance)) if relevance else 0.0


def ndcg_at_k(relevance: Sequence[bool]) -> float:
    """Binary-relevance NDCG, matching the reference harness's
    `_ndcg_from_relevance`: IDCG is the DCG of the ideal ordering of the SAME
    relevance labels (`sorted(..., reverse=True)`), so a real ranking can never
    beat the ideal and the score stays in [0, 1] — even when a gold item is
    retrieved in several slots (which previously inflated it above 1)."""
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevance) if rel)
    idcg = sum(
        1.0 / math.log2(i + 2) for i, rel in enumerate(sorted(relevance, reverse=True)) if rel
    )
    return (dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Family compute functions.
# ---------------------------------------------------------------------------


def compute_source_metrics(
    retrieved_doc_ids: Sequence[str], expected_sources: Sequence[str]
) -> dict[str, float | None]:
    """Source-level retrieval metrics (relevance by doc_id membership).

    Recall dedups documents (a doc retrieved 3× counts once toward finding
    the gold doc); precision/ndcg/mrr/hit use the per-slot relevance list.
    All None when the case has no gold `expected_sources`.
    """
    gold = set(expected_sources)
    if not gold:
        return dict.fromkeys(_SOURCE_KEYS, None)
    rel = [doc_id in gold for doc_id in retrieved_doc_ids]
    found_docs = {doc_id for doc_id in retrieved_doc_ids if doc_id in gold}
    return {
        "hit_at_k": hit_at_k(rel),
        "mrr": mrr(rel),
        "precision_at_k": precision_at_k(rel),
        "recall_at_k": len(found_docs) / len(gold),
        "ndcg_at_k": ndcg_at_k(rel),
    }


def compute_chunk_metrics(
    retrieved_texts: Sequence[str], expected_chunks: Sequence[str]
) -> dict[str, float | None]:
    """Chunk-level retrieval metrics (relevance by gold-snippet substring).

    Recall is snippet-centric (fraction of gold snippets found in at least
    one retrieved chunk); the rest use per-chunk relevance. All None when
    the case has no gold `expected_chunks`.
    """
    # Filter on the NORMALIZED value, not the raw: a punctuation-only snippet
    # passes `.strip()` but normalizes to "" — and "" is a substring of every
    # text, which would falsely mark every retrieved chunk as a hit.
    snippets = [ns for s in expected_chunks if (ns := normalize_text(s))]
    if not snippets:
        return dict.fromkeys(_CHUNK_KEYS, None)
    texts = [normalize_text(t) for t in retrieved_texts]
    rel = [any(s in t for s in snippets) for t in texts]
    found_snippets = sum(1 for s in snippets if any(s in t for t in texts))
    return {
        "chunk_hit_at_k": hit_at_k(rel),
        "chunk_mrr": mrr(rel),
        "chunk_precision_at_k": precision_at_k(rel),
        "chunk_recall_at_k": found_snippets / len(snippets),
        "chunk_ndcg_at_k": ndcg_at_k(rel),
    }


def required_keyword_hit_rate(answer: str, required: Sequence[str]) -> float:
    """Fraction of required keywords present in the answer (substring after
    normalization). Vacuously 1.0 when the case lists no required keywords."""
    # Filter on the NORMALIZED value: a punctuation-only keyword passes
    # `.strip()` but normalizes to "", which is a substring of every answer and
    # would falsely count as always-present (inflating the hit rate).
    reqs = [nk for k in required if (nk := normalize_text(k))]
    if not reqs:
        return 1.0
    na = normalize_text(answer)
    return sum(1 for k in reqs if k in na) / len(reqs)


def disallowed_keyword_hits(answer: str, disallowed: Sequence[str]) -> int:
    """Count of disallowed keywords present in the answer (0 = clean)."""
    na = normalize_text(answer)
    # Filter on the NORMALIZED value: a punctuation-only disallowed keyword
    # normalizes to "", which matches every answer and would falsely flag a
    # clean answer as containing it (forcing a spurious REVIEW).
    dis = [nk for k in disallowed if (nk := normalize_text(k))]
    return sum(1 for k in dis if k in na)


def chunk_source_grounding(
    cited_chunk_ids: Sequence[str], retrieved_chunk_ids: Sequence[str]
) -> float:
    """Fraction of the answer's chunk citations that point at an
    actually-retrieved chunk — a structural anti-hallucination check.
    1.0 when the answer cited nothing (nothing to ground)."""
    if not cited_chunk_ids:
        return 1.0
    retrieved = set(retrieved_chunk_ids)
    return sum(1 for cid in cited_chunk_ids if cid in retrieved) / len(cited_chunk_ids)


# ---------------------------------------------------------------------------
# KG metrics (Phase 2). All deterministic — read cypher_query / kg_hits /
# kg_sources straight out of the typed state. An entity is "found" when its
# normalized string appears in the normalized text of any returned KG row.
# ---------------------------------------------------------------------------

_KG_ENTITY_KEYS = ("kg_hit_at_k", "kg_entity_recall")


def _hit_contains_entity(hit: dict, entity_norm: str) -> bool:
    blob = normalize_text(" ".join(str(v) for v in hit.values()))
    return entity_norm in blob


def compute_kg_entity_metrics(
    kg_hits: Sequence[dict], expected_entities: Sequence[str]
) -> dict[str, float | None]:
    """`kg_hit_at_k` (any gold entity present in the rows) + `kg_entity_recall`
    (fraction of gold entities present). Both None when the case has no gold
    `expected_entities`."""
    ents = [normalize_text(e) for e in expected_entities if e.strip()]
    if not ents:
        return dict.fromkeys(_KG_ENTITY_KEYS, None)
    found = sum(1 for e in ents if any(_hit_contains_entity(h, e) for h in kg_hits))
    return {"kg_hit_at_k": 1.0 if found else 0.0, "kg_entity_recall": found / len(ents)}


def kg_source_grounding(cited_kg_indices: Sequence[int], n_kg_hits: int) -> float:
    """Fraction of the answer's KG citations (hit_index) that point at a real
    returned row — the graph twin of `chunk_source_grounding`. 1.0 when the
    answer cited no KG rows."""
    if not cited_kg_indices:
        return 1.0
    valid = sum(1 for i in cited_kg_indices if 0 <= i < n_kg_hits)
    return valid / len(cited_kg_indices)


def cypher_validity(cypher_read_only: bool | None, kg_retrieval_error: str | None) -> float:
    """1.0 iff the Cypher passed the read-only rail AND executed without a
    retrieval error (catches write attempts + malformed/failed Cypher)."""
    return 1.0 if (cypher_read_only and not kg_retrieval_error) else 0.0


def mode_routing_correct(routed_mode: str | None, expected_mode: str) -> float:
    """1.0 iff the mode-classifier routed to the case's expected leg."""
    return 1.0 if routed_mode == expected_mode else 0.0
