"""Structured per-ontology install-dialog metadata.

Leaf module — depends only on `dataclasses`, no other `kg/` modules.
That isolation matters because each per-ontology
`kg/ontologies/*_writes.py` module builds a `_<NAME>_PROVENANCE =
OntologyProvenance(...)` constant at module load time, and any import
back through `lifecycle.py` (which imports the registry which
imports those very modules) would cycle.

`lifecycle.py` re-exports `OntologyProvenance` from this
module so callers still write
`from knowledge_agent.kg.ontologies.lifecycle import OntologyProvenance`
— the leaf location is an implementation detail.

Mirrors `entity_extractors.extractor_lifecycle.ModelProvenance` on the
extractor side: gives the install dialog the same level of pre-install
detail (publisher, license, source URL, file format, size) for ontology
imports that users already get for downloadable models. Adds two
ontology-specific surfaces beyond ModelProvenance:

  - `domain_tags`: 20-tag taxonomy slots this ontology covers. Drives
    the wizard's "suggested ontologies for your corpus" rendering.
  - `covers_labels`: L6 entity_type labels (DISEASE, CHEMICAL, GENE,
    ...) this ontology has strong term coverage for. The cross-link
    helpers in `lifecycle.py` join this against each
    extractor's `emitted_labels` to produce the "pairs well with"
    surface in BOTH the ontology install dialog and the extractor
    install dialog. The mapping lives ON the ontology because the
    ontology knows what labels its terms answer to.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OntologyProvenance:
    """Structured per-ontology metadata for the install dialog.

    Fields:
      - `ontology_name`: registry key (e.g. "mesh"). Matches the
        `ONTOLOGY_REGISTRY` dict key in `linking.py`.
      - `full_name`: human-readable expansion (e.g. "Medical Subject
        Headings"). Used in dialog headers.
      - `publisher`: organisation maintaining the ontology, with
        affiliation in parens when relevant.
      - `license`: one-line license summary. Faithful to the source's
        terms; brevity over legal precision (the source_url has the
        full text).
      - `source_url`: canonical homepage / PURL the user can click
        through for full license + documentation.
      - `download_url`: the actual file URL we fetch. Often a
        sub-page of `source_url`; surfaced separately so the dialog
        can show both.
      - `file_format`: short label, one of "N-Triples (gzipped)",
        "OBO", "OWL/RDF-XML", "RDF modular (GitHub walker)" — drives
        the dialog's "how this gets parsed" line.
      - `download_size_mb`: the bytes that will be fetched. Single
        source of truth — the per-module `DOWNLOAD_SIZE_MB` constants
        are kept as back-compat re-exports.
      - `estimated_terms`: approximate term count (~30K for MeSH,
        ~50K for GO, ~2.74M for NCBITaxon). For the dialog's
        "what you'll get" line.
      - `domain_tags`: 20-tag taxonomy entries this ontology serves.
      - `covers_labels`: extractor output labels this ontology has
        strong term coverage for. The cross-link helpers route
        extractor → ontology candidates via this field.
      - `description`: one-paragraph user-facing summary. Compact
        (1-3 sentences); the install dialog renders it inline.
      - `heavy_warning`: optional one-line caveat for ontologies
        with notable resource costs (NCBITaxon ~440 MB / several GB
        RAM at import; DRON ~220 MB). None for normal-sized ontologies.
    """

    ontology_name: str
    full_name: str
    publisher: str
    license: str
    source_url: str
    download_url: str
    file_format: str
    download_size_mb: int
    estimated_terms: int
    domain_tags: tuple[str, ...]
    covers_labels: tuple[str, ...]
    description: str
    heavy_warning: str | None = None
