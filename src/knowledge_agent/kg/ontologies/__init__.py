"""Ontology layer (L7) — per-ontology term importers + shared helpers.

Grouped here (moved out of a flat `kg/` on 2026-07-23) to keep the KG
package's core — `client`, `schema`, and the L1-L10 layer writes —
uncluttered, mirroring how `parsers/` sits under `ingestion/`.

- `<name>_writes.py` — one thin adapter per ontology (mesh, go, hpo,
  chebi, …): source constants + delegating calls.
- `writes.py` / `helpers.py` / `pronto.py` / `rdf.py` — the shared,
  format-agnostic readers (OBO via pronto, RDF/OWL/SKOS via rdflib) and
  the Cypher write helpers the per-ontology adapters delegate to.
- `linking.py` — the L7 canonical-linking pass (`:Entity` ->
  `:CANONICAL_TO` -> `:OntologyTerm`) + the `ONTOLOGY_REGISTRY`.
- `xrefs.py` — cross-ontology `:<X>_XREF` backfill/clear primitives.
- `provenance.py` — `OntologyProvenance` (model origin / trust).
- `lifecycle.py` — GUI-facing install / link / delete plan-execute ops.
"""
