"""KG schema rendered as prompt text for the cypher_builder node.

The cypher_builder LLM gets this text in its system prompt so it can write
Cypher that targets real labels, relationships, and properties - never
fabricating ones that don't exist.

The renderer is config-aware: only the sections for layers + sub-labels
the corpus actually uses are emitted. Two channels filter the prompt:
  - `config.layers.openalex_papers` and `config.layers.chunks` gate the
    L1-L4 OpenAlex blocks and the L5 chunk blocks respectively.
  - `config.allowed_types` controls which Document/Artifact sub-labels
    appear (a corpus that doesn't allow `Patent` won't tell the LLM
    about `:Patent`). Loose `:Document` / `:Artifact` (no sub-label) is
    always valid so both top-level blocks always appear.

DRIFT WARNING: this module is only PARTIALLY auto-synced with the schema.
    - Auto-synced: label names and relationship-type names come from
      f-strings against the constants in `kg/schema.py`. Rename a
      constant and the prompt updates automatically.
    - NOT auto-synced: property names, types, and descriptions are inline
      strings here, because the same properties are inline strings in
      `kg/openalex_writes.py` / `kg/chunk_writes.py` Cypher writes. There
      is no constant declaring them.

When you change the schema:
    - Add or rename a label / relationship in `schema.py` -> the prompt
      auto-updates BUT you still have to add the new label's property
      block to the body below.
    - Add or remove a property -> manually update the relevant block below
      AND update `tests/unit/test_schema_as_prompt.py` (which acts as a
      checklist of expected properties).
    - Add a new sub-label to `DOCUMENT_SUB_LABELS` / `ARTIFACT_SUB_LABELS`
      -> the sub-label list rendering picks it up automatically once the
      corpus.toml's `allowed_types` includes it. Add a per-subtype
      property block here only if that subtype has properties beyond what
      the parent `:Document` / `:Artifact` already declares.
    - Land L6 entities -> add a new block + gate it on the new layer flag
      in `corpus_config.LayerFlags`, in lockstep with the schema + writes
      changes.

Future hardening (NOT done): refactor property names + types into a single
declarative manifest in `schema.py`, then iterate it here so adding a
property in one place propagates everywhere. Bigger change; revisit when
the schema grows past a handful of properties.
"""

from knowledge_agent.kg.corpus_config import CorpusConfig
from knowledge_agent.kg.schema import (
    ABOUT_TOPIC_REL,
    ARTIFACT_LABEL,
    ARTIFACT_SUB_LABELS,
    AUTHOR_LABEL,
    AUTHORED_REL,
    CANONICAL_TO_REL,
    CHEBI_IS_A_REL,
    CHEBI_TERM_LABEL,
    CHUNK_LABEL,
    CITES_REL,
    CL_IS_A_REL,
    CL_TERM_LABEL,
    DOCUMENT_LABEL,
    DOCUMENT_SUB_LABELS,
    DRON_IS_A_REL,
    DRON_TERM_LABEL,
    ECO_IS_A_REL,
    ECO_TERM_LABEL,
    EFO_IS_A_REL,
    EFO_TERM_LABEL,
    ENTITY_LABEL,
    ENVO_IS_A_REL,
    ENVO_TERM_LABEL,
    FIBO_IS_A_REL,
    FIBO_TERM_LABEL,
    FOODON_IS_A_REL,
    FOODON_TERM_LABEL,
    GO_IS_A_REL,
    GO_TERM_LABEL,
    HPO_IS_A_REL,
    HPO_TERM_LABEL,
    MONDO_IS_A_REL,
    MONDO_TERM_LABEL,
    NCBITAXON_IS_A_REL,
    NCBITAXON_TERM_LABEL,
    OBI_IS_A_REL,
    OBI_TERM_LABEL,
    PO_IS_A_REL,
    PO_TERM_LABEL,
    PR_IS_A_REL,
    PR_TERM_LABEL,
    SO_IS_A_REL,
    SO_TERM_LABEL,
    UBERON_IS_A_REL,
    UBERON_TERM_LABEL,
    MENTIONS_REL,
    MESH_BROADER_REL,
    MESH_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
    ONTOLOGY_XREF_RELS,
    PAPER_LABEL,
    PART_OF_REL,
    PUBLISHED_IN_REL,
    RELATED_BY_XREF_REL,
    RELATED_TO_REL,
    TOPIC_LABEL,
    TRIPLE_PREDICATE_RELS,
    VENUE_LABEL,
)


def format_schema_for_prompt(config: CorpusConfig) -> str:
    """Return a textual description of the KG schema for the LLM prompt.

    Sections are filtered against `config`:
      - L1-L4 blocks (`:Author`, `:Venue`, `:Topic`, citation / authorship
        / venue / topic edges, and the `:Paper` subtype line) appear only
        when `openalex_papers` is on AND `Paper` is in `allowed_types`.
      - `:Chunk` and `:PART_OF` appear only when `chunks` is on.
      - Sub-label lists (`:Document:Paper`, `:Artifact:Dataset`, ...)
        reflect the corpus's `allowed_types`.
      - `:Document` and `:Artifact` top-level blocks always appear,
        since loose-file ingest (no sub-label) is always valid.

    Not cached - re-rendering the small text per call is cheap, and
    `CorpusConfig` isn't hashable by default in Pydantic v2 (would need
    `frozen=True`), so `lru_cache` doesn't apply cleanly.
    """
    sections: list[str] = [_HEADER, "", "Node labels:"]

    # Node-label blocks - blank line between blocks (matches the original
    # rendering style).
    node_blocks: list[str] = [_document_block(config), _artifact_block(config)]
    if _openalex_active(config):
        node_blocks.extend([_AUTHOR_BLOCK, _VENUE_BLOCK, _TOPIC_BLOCK])
    if config.layers.chunks:
        node_blocks.append(_CHUNK_BLOCK)
    if config.layers.entities:
        node_blocks.append(_entity_block(config))
    # L7 ontology blocks: one per enabled ontology layer. Renders the
    # ontology-specific sub-label (`:MeSHTerm`, `:GOTerm`, ...) under the
    # shared `:OntologyTerm` super-label so the LLM can scope either way.
    for ontology_name in _enabled_ontologies(config):
        node_blocks.append(_ontology_term_block(ontology_name))
    sections.append("\n\n".join(node_blocks))

    # Relationship blocks - consecutive within their section, no inter-block
    # blanks. Header + section only emitted if at least one edge type is
    # active (otherwise the "Relationships:" header would dangle).
    rel_blocks: list[str] = []
    if _openalex_active(config):
        rel_blocks.extend(
            [
                _CITES_BLOCK,
                _AUTHORED_BLOCK,
                _PUBLISHED_IN_BLOCK,
                _ABOUT_TOPIC_BLOCK,
            ]
        )
    if config.layers.chunks:
        rel_blocks.append(_PART_OF_BLOCK)
    if config.layers.entities:
        rel_blocks.append(_MENTIONS_BLOCK)
    # L7 hierarchy + linking edges, gated per enabled ontology.
    for ontology_name in _enabled_ontologies(config):
        rel_blocks.append(_ontology_hierarchy_block(ontology_name))
    if _enabled_ontologies(config):
        rel_blocks.append(_CANONICAL_TO_BLOCK)
    if config.layers.triples:
        rel_blocks.append(_TRIPLES_BLOCK)
    # L7 xref edges: only rendered in "use" mode. In "collect_only"
    # the strings are stored as `dangling_xrefs` properties but no
    # `:<X>_XREF` edges exist yet — telling the LLM about a predicate
    # that doesn't materialise would invite empty result sets.
    if config.layers.xrefs == "use":
        rel_blocks.append(_XREFS_BLOCK)
    if config.layers.cross_doc:
        rel_blocks.append(_RELATED_TO_BLOCK)
    # L10 concept-level cross-doc. Independent of L9 — both edges can
    # coexist; semantics distinct (L9 = shared raw entities,
    # L10 = shared concepts via xref equivalence).
    if config.layers.cross_doc_xrefs:
        rel_blocks.append(_RELATED_BY_XREF_BLOCK)
    if rel_blocks:
        sections.extend(["", "Relationships:"])
        sections.append("\n".join(rel_blocks))

    sections.extend(["", _identifier_conventions(config)])
    return "\n".join(sections) + "\n"


_HEADER = (
    "Knowledge graph schema (Neo4j). Use ONLY these labels, relationships, and\n"
    "properties when writing Cypher - never invent ones that aren't listed."
)


def _openalex_active(config: CorpusConfig) -> bool:
    """True when the L1-L4 OpenAlex entities are reachable for this corpus:
    the layer is on AND the corpus allows the `Paper` sub-label (the only
    sub-label that triggers OpenAlex writes today)."""
    return (
        config.layers.openalex_papers
        and PAPER_LABEL in config.allowed_types
    )


def _allowed_in(
    config: CorpusConfig, candidates: tuple[str, ...]
) -> list[str]:
    """Return the candidates that appear in `config.allowed_types`,
    preserving the canonical order from schema.py rather than the user's
    write order."""
    allowed = set(config.allowed_types)
    return [c for c in candidates if c in allowed]


def _document_block(config: CorpusConfig) -> str:
    """:Document block. Always present (loose-file ingest is valid). The
    sub-label list reflects `config.allowed_types`. OpenAlex-derived
    properties (openalex_id, doi, in_corpus) appear only when the L1-L4
    layer is active."""
    sub_labels = _allowed_in(config, DOCUMENT_SUB_LABELS)
    lines = [
        f"- :{DOCUMENT_LABEL}        - primary written work parsed as text "
        "(papers, books, notes, etc.) or referenced as a citation.",
    ]
    if sub_labels:
        rendered = ", ".join(f":{s}" for s in sub_labels)
        lines.append(
            f"  Co-applied subtype labels for this corpus: {rendered}."
        )
        lines.append(
            f"  Use a subtype to scope queries (e.g. `MATCH (d:{sub_labels[0]})`)."
        )
    lines.append("  Properties:")
    lines.append(
        "    doc_id       (str)  : SHA-256 of file bytes. Present on every CORPUS document."
    )
    lines.append(
        "                          Universal join key with the LanceDB store."
    )
    if _openalex_active(config):
        lines.append(
            '    openalex_id  (str)  : Bare OpenAlex work ID (e.g. "W1234567890"). Present'
        )
        lines.append(
            "                          for papers resolved against OpenAlex AND for shadow"
        )
        lines.append("                          nodes (cited works not yet ingested).")
        lines.append(
            "    doi          (str)  : Normalized DOI (no URL prefix, lowercased). Optional."
        )
        lines.append(
            "    in_corpus    (bool) : true = ingested document. false = shadow (cited but"
        )
        lines.append(
            "                          not ingested). Filter on this to exclude shadows."
        )
    return "\n".join(lines)


def _artifact_block(config: CorpusConfig) -> str:
    """:Artifact block. Always present (loose-file ingest is valid).
    Holds supporting / derived content (datasets, code, figures, audio,
    video, diagrams). No OpenAlex-derived properties - artifacts don't
    resolve against OpenAlex."""
    sub_labels = _allowed_in(config, ARTIFACT_SUB_LABELS)
    lines = [
        f"- :{ARTIFACT_LABEL}        - supporting / derived content "
        "(datasets, code, figures, audio, video, diagrams) - not a primary written work.",
    ]
    if sub_labels:
        rendered = ", ".join(f":{s}" for s in sub_labels)
        lines.append(
            f"  Co-applied subtype labels for this corpus: {rendered}."
        )
        lines.append(
            f"  Use a subtype to scope queries (e.g. `MATCH (a:{sub_labels[0]})`)."
        )
    lines.append("  Properties:")
    lines.append(
        "    doc_id       (str)  : SHA-256 of file bytes. Universal join key with LanceDB."
    )
    return "\n".join(lines)


_AUTHOR_BLOCK = f"""\
- :{AUTHOR_LABEL}          - paper authors.
  Properties:
    openalex_id   (str) : Bare OpenAlex author ID (e.g. "A1234567890").
    display_name  (str) : Author display name from OpenAlex."""


_VENUE_BLOCK = f"""\
- :{VENUE_LABEL}           - publication venue (journal, repository, conference).
  Properties:
    openalex_id  (str)  : Bare OpenAlex source ID (e.g. "S137773608").
    name         (str)  : Venue display name (e.g. "Nature", "JAMA").
    type         (str)  : "journal", "repository", "conference", or "book series".
    issn         (str)  : Canonical ISSN (issn_l from OpenAlex). Optional."""


_TOPIC_BLOCK = f"""\
- :{TOPIC_LABEL}           - OpenAlex topic label assigned to a paper.
  Properties:
    openalex_id   (str) : Bare OpenAlex topic ID (e.g. "T11537").
    display_name  (str) : Topic label (e.g. "Vitamin D and Bone Health").
                          Topics are broad categories - hundreds of papers per topic."""


_CHUNK_BLOCK = f"""\
- :{CHUNK_LABEL}           - one chunk of text from a document or artifact. Identity +
  structural metadata only; the actual text + embedding live in LanceDB (join via chunk_id).
  Properties:
    chunk_id      (str) : "{{doc_id}}#{{chunk_index}}" form - join key with LanceDB.
    doc_id        (str) : parent's doc_id (denormalised for direct queries).
    chunk_index   (int) : 0-based position within the parent.
    section       (str) : section heading the chunk belongs to (e.g. "Methods"). Optional.
    page          (int) : page number in the source. Optional.
    content_type  (str) : "text", "figure", or "table"."""


_CITES_BLOCK = f"""\
- (:{DOCUMENT_LABEL})-[:{CITES_REL}]->(:{DOCUMENT_LABEL})
    Citation edge. Source = citing document, target = cited document.
    No edge properties."""


_AUTHORED_BLOCK = f"""\
- (:{AUTHOR_LABEL})-[:{AUTHORED_REL}]->(:{DOCUMENT_LABEL})
    Authorship edge. Edge properties:
      position         (str)  : "first", "middle", or "last".
      is_corresponding (bool) : true if author is the corresponding author."""


_PUBLISHED_IN_BLOCK = f"""\
- (:{DOCUMENT_LABEL})-[:{PUBLISHED_IN_REL}]->(:{VENUE_LABEL})
    Publication-venue edge. No edge properties."""


_ABOUT_TOPIC_BLOCK = f"""\
- (:{DOCUMENT_LABEL})-[:{ABOUT_TOPIC_REL}]->(:{TOPIC_LABEL})
    Topic-assignment edge. Edge properties:
      score (float) : OpenAlex relevance score 0-1 (per-paper-per-topic).
                      Higher = stronger topical match."""


_PART_OF_BLOCK = f"""\
- (:{CHUNK_LABEL})-[:{PART_OF_REL}]->(:{DOCUMENT_LABEL} | :{ARTIFACT_LABEL})
    Chunk-to-parent edge. Every :{CHUNK_LABEL} has exactly one :{PART_OF_REL}
    edge to either a :{DOCUMENT_LABEL} or an :{ARTIFACT_LABEL}. No edge properties."""


def _entity_block(config: CorpusConfig) -> str:
    """:Entity block (L6). Lists the corpus's configured `entity_types`
    so the LLM scopes Cypher to the labels actually present. Empty list
    in config = open vocabulary (entity_type values are LLM-chosen
    SCREAMING_SNAKE_CASE); the block flags that explicitly."""
    # config.entities is guaranteed non-None when layers.entities=true
    # (model validator), so this helper is only called in that case.
    assert config.entities is not None  # noqa: S101
    types = config.entities.entity_types
    if types:
        types_clause = (
            "Configured entity_type values for this corpus: "
            f"{', '.join(types)}."
        )
    else:
        types_clause = (
            "entity_type values are open-vocabulary (LLM-chosen "
            "SCREAMING_SNAKE_CASE labels - inspect the data to discover "
            "what's been written)."
        )
    return (
        f"- :{ENTITY_LABEL}          - extracted entity mention. Lowercased text "
        "+ category co-identify the node.\n"
        "  Properties:\n"
        "    key           (str)  : lowercased mention text - the MERGE key. Use this for\n"
        "                           string matching (already case-insensitive).\n"
        "    entity_type   (str)  : category label (e.g. GENE, DISEASE, CHEMICAL).\n"
        "    canonicalised (bool) : true = linked to a canonical ontology (L6b+);\n"
        "                           false = raw extracted mention only (L6 default).\n"
        f"  {types_clause}"
    )


_MENTIONS_BLOCK = f"""\
- (:{CHUNK_LABEL})-[:{MENTIONS_REL}]->(:{ENTITY_LABEL})
    Mention edge - source chunk's text contained this entity. Edge properties:
      offset     (int)   : char position of the mention in the chunk text. NER-only;
                           null for LLM extractions.
      confidence (float) : extractor-reported score in [0, 1]. NER-only; null for LLM
                           extractions AND for NER backends that don't expose per-entity scores."""


# ---- L7 ontology blocks ----


def _enabled_ontologies(config: CorpusConfig) -> list[str]:
    """Names of L7 ontology layers currently enabled in `config.layers`.
    Used to decide which ontology-specific blocks to render. Reads the
    same helper the pipeline uses."""
    return config._enabled_ontology_layers()


# Per-ontology block configuration. Adding a new ontology adapter means
# adding one entry here (sub-label, hierarchy edge type, description).
_ONTOLOGY_BLOCK_SPECS: dict[str, dict[str, str]] = {
    "mesh": {
        "sub_label": MESH_TERM_LABEL,
        "hierarchy_rel": MESH_BROADER_REL,
        "description": (
            "Medical Subject Headings - NLM's biomedical thesaurus. Covers "
            "diseases, chemicals, anatomy, organisms, methods."
        ),
        "id_example": "MESH:D003920",
    },
    "go": {
        "sub_label": GO_TERM_LABEL,
        "hierarchy_rel": GO_IS_A_REL,
        "description": (
            "Gene Ontology - biological processes, molecular functions, "
            "and cellular components."
        ),
        "id_example": "GO:0008150",
    },
    "hpo": {
        "sub_label": HPO_TERM_LABEL,
        "hierarchy_rel": HPO_IS_A_REL,
        "description": (
            "Human Phenotype Ontology - clinical features, signs, "
            "symptoms, lab abnormalities, behavioural traits."
        ),
        "id_example": "HP:0001249",
    },
    "uberon": {
        "sub_label": UBERON_TERM_LABEL,
        "hierarchy_rel": UBERON_IS_A_REL,
        "description": (
            "Uber Anatomy Ontology - multi-species anatomical "
            "structures: tissues, organs, body parts."
        ),
        "id_example": "UBERON:0002107",
    },
    "mondo": {
        "sub_label": MONDO_TERM_LABEL,
        "hierarchy_rel": MONDO_IS_A_REL,
        "description": (
            "Mondo Disease Ontology - diseases integrated across DOID, "
            "OMIM, Orphanet, EFO, NCIT, ICD-11."
        ),
        "id_example": "MONDO:0005148",
    },
    "chebi": {
        "sub_label": CHEBI_TERM_LABEL,
        "hierarchy_rel": CHEBI_IS_A_REL,
        "description": (
            "Chemical Entities of Biological Interest - small molecules "
            "(drugs, metabolites, signaling molecules, nutrients)."
        ),
        "id_example": "CHEBI:17234",
    },
    "eco": {
        "sub_label": ECO_TERM_LABEL,
        "hierarchy_rel": ECO_IS_A_REL,
        "description": (
            "Evidence & Conclusion Ontology - evidence codes qualifying "
            "annotations (experimental evidence, sequence orthology, "
            "author statement, etc.)."
        ),
        "id_example": "ECO:0000006",
    },
    "so": {
        "sub_label": SO_TERM_LABEL,
        "hierarchy_rel": SO_IS_A_REL,
        "description": (
            "Sequence Ontology - sequence features and variants: gene, "
            "promoter, intron, CDS, regulatory region, SNV, indel, "
            "structural variant."
        ),
        "id_example": "SO:0000704",
    },
    "pr": {
        "sub_label": PR_TERM_LABEL,
        "hierarchy_rel": PR_IS_A_REL,
        "description": (
            "Protein Ontology - protein classes, isoforms, complexes "
            "and ortholog relationships beyond UniProt's flat model."
        ),
        "id_example": "PR:000000001",
    },
    "cl": {
        "sub_label": CL_TERM_LABEL,
        "hierarchy_rel": CL_IS_A_REL,
        "description": (
            "Cell Ontology - cell types across organisms (neurons, "
            "lymphocytes, epithelial cells, etc.)."
        ),
        "id_example": "CL:0000540",
    },
    "po": {
        "sub_label": PO_TERM_LABEL,
        "hierarchy_rel": PO_IS_A_REL,
        "description": (
            "Plant Ontology - plant anatomy and developmental stages "
            "(root, leaf, flower, seedling, anthesis, etc.)."
        ),
        "id_example": "PO:0009001",
    },
    "foodon": {
        "sub_label": FOODON_TERM_LABEL,
        "hierarchy_rel": FOODON_IS_A_REL,
        "description": (
            "Food Ontology - foods, food products, dietary components, "
            "and processing methods."
        ),
        "id_example": "FOODON:03301720",
    },
    "envo": {
        "sub_label": ENVO_TERM_LABEL,
        "hierarchy_rel": ENVO_IS_A_REL,
        "description": (
            "Environment Ontology - biomes, environmental features, "
            "environmental materials, and exposures."
        ),
        "id_example": "ENVO:00000111",
    },
    "ncbitaxon": {
        "sub_label": NCBITAXON_TERM_LABEL,
        "hierarchy_rel": NCBITAXON_IS_A_REL,
        "description": (
            "NCBI Taxonomy - biological organisms (species, genus, "
            "family, ... up through life). The canonical taxonomy "
            "mirror of NCBI's data."
        ),
        "id_example": "NCBITaxon:9606",
    },
    "obi": {
        "sub_label": OBI_TERM_LABEL,
        "hierarchy_rel": OBI_IS_A_REL,
        "description": (
            "Ontology for Biomedical Investigations - experimental "
            "methods, assay protocols, study designs, laboratory "
            "equipment, planned biomedical processes."
        ),
        "id_example": "OBI:0000489",
    },
    "efo": {
        "sub_label": EFO_TERM_LABEL,
        "hierarchy_rel": EFO_IS_A_REL,
        "description": (
            "Experimental Factor Ontology - experimental factors "
            "across EBI studies (diseases, cell types, cell lines, "
            "traits, measurement methods); integrates UBERON, CL, "
            "MONDO, ChEBI, ORDO."
        ),
        "id_example": "EFO:0000001",
    },
    "dron": {
        "sub_label": DRON_TERM_LABEL,
        "hierarchy_rel": DRON_IS_A_REL,
        "description": (
            "Drug Ontology - drug products, ingredients, dose forms, "
            "manufacturers, packaging. Built on RxNorm with a realist "
            "upper-level structure."
        ),
        "id_example": "DRON:00000005",
    },
    "fibo": {
        "sub_label": FIBO_TERM_LABEL,
        "hierarchy_rel": FIBO_IS_A_REL,
        "description": (
            "Financial Industry Business Ontology - financial "
            "instruments, transactions, parties, agreements, "
            "corporations, market infrastructure."
        ),
        "id_example": "FIBO:FinancialInstrument",
    },
}


def _ontology_term_block(ontology_name: str) -> str:
    """Per-ontology :OntologyTerm:<sub_label> node block."""
    spec = _ONTOLOGY_BLOCK_SPECS[ontology_name]
    sub_label = spec["sub_label"]
    return (
        f"- :{ONTOLOGY_TERM_LABEL}:{sub_label}  - {spec['description']}\n"
        f"  Properties:\n"
        f"    id            (str)  : canonical ontology ID, e.g. {spec['id_example']!r}.\n"
        f"    label         (str)  : primary label / preferred name.\n"
        f"    synonyms      (list) : alternative names + abbreviations, lowercased.\n"
        f"    definition    (str)  : textual definition. Optional - may be null."
    )


def _ontology_hierarchy_block(ontology_name: str) -> str:
    """Per-ontology hierarchy edge block."""
    spec = _ONTOLOGY_BLOCK_SPECS[ontology_name]
    sub_label = spec["sub_label"]
    rel = spec["hierarchy_rel"]
    return (
        f"- (:{sub_label})-[:{rel}]->(:{sub_label})\n"
        f"    Hierarchy edge within {ontology_name.upper()}: child term -> "
        f"broader/parent term. Use\n"
        f"    `MATCH (a)-[:{rel}*]->(b)` for variable-depth ancestor traversal."
    )


_CANONICAL_TO_BLOCK = f"""\
- (:{ENTITY_LABEL})-[:{CANONICAL_TO_REL}]->(:{ONTOLOGY_TERM_LABEL})
    Canonical-linking edge - written by the L7 linking pass after an
    ontology is imported. Source :Entity, target one specific ontology term.
    Edge properties:
      strategy   (str)   : 'exact' or 'fuzzy' - how this link was found.
      confidence (float) : heuristic score in [0, 1] for fuzzy matches;
                           null for exact matches."""


# L8: 15 typed entity-to-entity relation edges. The predicate IS the edge
# label (not a property on a generic :RELATION edge). One block describes
# the 15 edge types together since they share the same node pattern and
# property schema; per-edge differences are only in the relation name.
_TRIPLES_PREDICATE_LIST_RENDERED = "\n      ".join(
    f":{p}" for p in TRIPLE_PREDICATE_RELS
)
_TRIPLES_BLOCK = f"""\
- (:{ENTITY_LABEL})-[<predicate>]->(:{ENTITY_LABEL})
    Typed entity-to-entity relation, extracted by an LLM from chunk text
    (L8). The predicate IS the edge label (no `predicate` property on a
    generic relation - the 15 types below are direct Neo4j edge labels).
    Use a typed match for predicate-specific queries:
      MATCH (a:{ENTITY_LABEL})-[:INHIBITS]->(b:{ENTITY_LABEL}) ...
    Use the pipe syntax to span multiple predicates:
      MATCH (a)-[:INHIBITS|REGULATES]->(b) ...
    Available predicate edge types (one of):
      {_TRIPLES_PREDICATE_LIST_RENDERED}
    Edge properties (same on every predicate):
      chunk_id      (str) : the :{CHUNK_LABEL}.chunk_id this triple was extracted from.
      doc_id        (str) : the parent document's doc_id. Same as :{CHUNK_LABEL}.doc_id.
      evidence_span (str) : verbatim chunk snippet supporting the assertion."""


# L9: materialised cross-document edges. Undirected; the same edge is
# matched in either direction. Endpoints span both :Document and
# :Artifact since the layer covers cross-type linking ("this dataset
# is related to that paper").
_RELATED_TO_BLOCK = f"""\
- (:{DOCUMENT_LABEL} | :{ARTIFACT_LABEL})-[:{RELATED_TO_REL}]-(:{DOCUMENT_LABEL} | :{ARTIFACT_LABEL})
    Cross-document synthesis edge (L9). Materialised between any two
    document or artifact nodes that share at least 2 distinct :{ENTITY_LABEL}
    nodes (default threshold). UNDIRECTED - match in either direction:
      MATCH (d)-[:{RELATED_TO_REL}]-(other) ...
    Rebuilt per-doc on every ingest / backfill; reflects the current
    L6 entity overlap.
    Edge properties:
      shared_entities (list) : entity keys (lowercased) shared by both endpoints.
      shared_count    (int)  : size of shared_entities; >= 2. Higher = stronger
                               topical overlap; sort by this to rank related docs.
      computed_at     (datetime) : when this edge was last (re)written."""


# L7 cross-ontology xrefs. The predicate IS the edge label, named
# after the SOURCE ontology (`:MESH_XREF` edges originate at :MeSHTerm
# nodes). 18 predicates total — one per shipped ontology — rendered
# inline from the canonical `ONTOLOGY_XREF_RELS` tuple so the prompt
# stays in sync as new ontologies ship.
_XREF_PREDICATE_LIST_RENDERED = "\n      ".join(
    f":{p}" for p in ONTOLOGY_XREF_RELS
)
_XREFS_BLOCK = f"""\
- (:{ONTOLOGY_TERM_LABEL})-[<predicate>]->(:{ONTOLOGY_TERM_LABEL})
    Cross-ontology xref edge (L7). Source ontology declares its concept
    is equivalent / closely-related to a concept in another ontology
    (OBO `xref:` lines, oboInOwl:hasDbXref, SKOS closeMatch/exactMatch).
    The predicate IS the edge label, named after the SOURCE ontology
    (e.g. `:MESH_XREF` edges originate at :{MESH_TERM_LABEL} nodes).
    Use a typed match for source-scoped queries:
      MATCH (s:{MESH_TERM_LABEL})-[:MESH_XREF]->(t:{ONTOLOGY_TERM_LABEL}) ...
    Use the pipe syntax to span source ontologies:
      MATCH (a:{ONTOLOGY_TERM_LABEL})-[:MESH_XREF|GO_XREF|CHEBI_XREF]-(b:{ONTOLOGY_TERM_LABEL}) ...
    Available xref predicate edge types (one per source ontology):
      {_XREF_PREDICATE_LIST_RENDERED}
    No edge properties."""


# L10 concept-level cross-document edge. Parallel shape to L9's
# RELATED_TO but joins via the xref graph (same concept OR connected
# by an xref edge), so the edge captures cross-ontology equivalence
# (same drug under MeSH and ChEBI, same disease under MONDO and DOID).
_RELATED_BY_XREF_BLOCK = f"""\
- (:{DOCUMENT_LABEL} | :{ARTIFACT_LABEL})-[:{RELATED_BY_XREF_REL}]-(:{DOCUMENT_LABEL} | :{ARTIFACT_LABEL})
    Concept-level cross-document edge (L10). Materialised between any
    two document or artifact nodes that share at least 2 canonical
    concepts via xref equivalence (default threshold). Equivalence
    means either the SAME :{ONTOLOGY_TERM_LABEL} node OR connected by
    a `:<X>_XREF` edge. UNDIRECTED - match in either direction:
      MATCH (d)-[:{RELATED_BY_XREF_REL}]-(other) ...
    Parallel to L9 (:{RELATED_TO_REL}) but uses CANONICAL CONCEPTS
    rather than raw :{ENTITY_LABEL} keys — surfaces same-concept
    overlap that varies in surface form (MeSH vs ChEBI for the same
    drug, MONDO vs DOID for the same disease).
    Rebuilt per-doc on every ingest / backfill.
    Edge properties:
      shared_concepts (list) : :{ONTOLOGY_TERM_LABEL}.id values that link the pair.
      shared_count    (int)  : size of shared_concepts; >= 2. Higher = stronger
                               topical overlap.
      computed_at     (datetime) : when this edge was last (re)written."""


def _identifier_conventions(config: CorpusConfig) -> str:
    """Trailing 'Identifier conventions' section. Bullets included based
    on which identifiers actually appear in the rendered schema, so the
    LLM doesn't see advice about properties it can't use."""
    lines = ["Identifier conventions:"]
    if config.layers.chunks:
        lines.append(
            "- Use `chunk_id` for chunk-level joins with LanceDB; use `doc_id` for"
        )
        lines.append("  document-level joins.")
    else:
        lines.append("- Use `doc_id` for document-level joins with LanceDB.")
    if _openalex_active(config):
        lines.append(
            "- Prefer `openalex_id` for traversing cite/author/venue/topic graphs."
        )
        lines.append("- `doi` is optional; use it only when the user asks by DOI.")
        lines.append(
            "- Use `in_corpus = true` to exclude shadow citations when the user wants"
        )
        lines.append("  only documents we've ingested.")
    return "\n".join(lines)
