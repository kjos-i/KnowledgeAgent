"""Pronto-based OBO ontology reading + term extraction.

One of three format families split out of the historical
`helpers.py` (2026-06-29):

  - `pronto.py` (THIS FILE) — OBO Foundry via `pronto` library
  - `rdf.py` — RDF/OWL/SKOS via `rdflib` library
  - `writes.py` — Neo4j Cypher writes (format-agnostic)

Pronto is lazy-imported inside `read_obo` so corpora that only use
RDF ontologies don't pay the pronto import cost.

OBO ontologies that route through this module: GO, ChEBI, MONDO, HPO,
UBERON, ECO, SO, PR, CL, PO, FOODON, ENVO, NCBITaxon (14 of the 18
shipped ontologies). The remaining four (MeSH RDF, OBI/EFO/DRON OWL,
FIBO RDF) use the `rdf.py` family instead — pronto silently
drops oboInOwl synonyms on OWL inputs, which OBI / EFO need.
"""

from __future__ import annotations

import logging

from knowledge_agent.kg.ontologies.helpers import OntologyTerm

logger = logging.getLogger(__name__)


def read_obo(path):  # type: ignore[no-untyped-def]
    """Parse an OBO / OBO-JSON / OWL file via pronto.

    Pronto auto-detects the format from the file extension. Returns a
    `pronto.Ontology` object whose `.terms()` iterator the caller walks
    to extract terms. Lazy-imports pronto so corpora that only use RDF
    ontologies don't pay the (small) pronto import cost.

    Heavy: parses the entire file at once. A 200 MB GO file takes
    several seconds to load; an 800 MB ontology is on the order of
    tens of seconds. Called once per ontology import.
    """
    import pronto

    logger.info("ontology: parsing OBO file %s", path)
    return pronto.Ontology(str(path))


def extract_terms_obo(
    ontology,  # type: ignore[no-untyped-def]
    id_prefix: str,
) -> list[OntologyTerm]:
    """Walk a pronto Ontology and yield `OntologyTerm` records.

    Covers the standard OBO Foundry layout: each term has an ID,
    label, optional synonyms, optional definition, and zero-or-more
    parent terms via `is_a`. Obsolete terms are skipped.

    Only `is_a` parents land in `parents` for now - the OBO Foundry
    `is_a` is what corresponds to the `:GO_IS_A` / `:CHEBI_IS_A`
    edges we write. Other relations (`part_of`, `regulates`) can be
    added later as separate output fields if a downstream module needs
    them.

    Args:
      ontology: result of `read_obo(path)`.
      id_prefix: the ontology's canonical ID prefix (e.g. "GO" for
        GO:0008150). Pronto's `term.id` already includes this prefix
        for OBO Foundry ontologies, so this parameter is documentary
        rather than transformational.

    Returns a list (not a generator) because per-ontology write
    modules need to UNWIND it as a batch. List sizes are bounded by
    ontology term counts (~50K for GO, ~30K for MeSH, etc.) -
    comfortable in memory.
    """
    terms: list[OntologyTerm] = []
    for term in ontology.terms():
        if term.obsolete:
            continue
        if not term.id or not term.name:
            continue
        # Pronto exposes synonyms as Synonym objects; .description is the
        # string. Lowercase for downstream case-insensitive matching.
        synonyms = tuple(sorted({s.description.lower() for s in term.synonyms if s.description}))
        parents = tuple(
            sorted(p.id for p in term.superclasses(distance=1, with_self=False) if p.id)
        )
        # L7 cross-ontology xrefs. Pronto parses OBO `xref:` lines into
        # `Xref` objects with an `.id` attribute carrying the prefixed
        # canonical ID (e.g. "MESH:D003920"). Stored verbatim - no
        # normalisation. Deduped via set, sorted for determinism.
        xrefs = tuple(sorted({x.id for x in (term.xrefs or ()) if getattr(x, "id", None)}))
        terms.append(
            OntologyTerm(
                id=term.id,
                label=term.name,
                synonyms=synonyms,
                parents=parents,
                definition=str(term.definition) if term.definition else None,
                xrefs=xrefs,
            )
        )
    logger.info(
        "ontology: extracted %d terms from OBO ontology (prefix=%s)",
        len(terms),
        id_prefix,
    )
    return terms
