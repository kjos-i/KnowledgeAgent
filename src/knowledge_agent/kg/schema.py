"""Single source of truth for KG schema strings.

Labels, relationship types, and the constraint statements applied at startup
(via `Neo4jClient.ensure_constraints`) all live here — the strings for every
layer L1-L10 (OpenAlex entities, chunks, domain entities, ontology terms,
typed triples, cross-document links). Any new layer extends the constants
below and appends its constraints - never define labels inline.

Neo4j unique constraints treat missing properties as fine - only nodes that
HAVE the property must have unique values - so shadow documents without a
`doi` or `doc_id` (cited papers we haven't ingested) don't violate those
constraints.

Identifier model:
- `doc_id`     - locally-generated SHA-256 of file bytes. The universal join
                 key with LanceDB. Present on every ingested document.
- `chunk_id`   - `{doc_id}#{chunk_index}`. The per-chunk join key with
                 LanceDB. Present on every `:Chunk` node.
- `openalex_id`- present for scholarly papers that resolved against OpenAlex
                 AND for shadow nodes (cited works we haven't ingested).
                 Also the canonical identifier for `:Venue` (`S<id>`),
                 `:Topic` (`T<id>`), and `:Author` (`A<id>`).
- `doi`        - present when extractable from the source PDF.

Two-tier label model (see also: `corpus_config.LayerFlags` and
`ingest_document(main_label, sub_label)`):

- `:Document` = primary written work parsed as text (PDF, Markdown, HTML,
                DOCX, ...). Things that are themselves the artefact of
                authorship.
- `:Artifact` = supporting / derived content (datasets, code, figures,
                diagrams, audio, video). Things that go alongside written
                works rather than being one.

Every ingested node carries one of these as its top-level label. A
sub-label is co-applied when the user picks one at ingest (e.g.
`:Document:Paper`, `:Artifact:Dataset`). The sub-label is optional - a
file ingested with just a top-level label (`:Document` alone) is valid
and stays queryable via the universal label.
"""

# ---- Top-level (main) labels.

DOCUMENT_LABEL = "Document"
ARTIFACT_LABEL = "Artifact"

MAIN_LABELS: tuple[str, ...] = (DOCUMENT_LABEL, ARTIFACT_LABEL)

# ---- :Document sub-labels.
#
# Each represents a kind of parseable written work. Multi-label co-applied
# alongside :Document (e.g. (:Document:Paper)). Singular naming convention
# matches the existing :Paper / :Author / :Chunk style - one instance is
# one paper, one note, etc.

PAPER_LABEL = "Paper"
BOOK_LABEL = "Book"
NOTE_LABEL = "Note"
REPORT_LABEL = "Report"
PRESENTATION_LABEL = "Presentation"
ARTICLE_LABEL = "Article"
CORRESPONDENCE_LABEL = "Correspondence"
TRANSCRIPT_LABEL = "Transcript"

DOCUMENT_SUB_LABELS: tuple[str, ...] = (
    PAPER_LABEL,
    BOOK_LABEL,
    NOTE_LABEL,
    REPORT_LABEL,
    PRESENTATION_LABEL,
    ARTICLE_LABEL,
    CORRESPONDENCE_LABEL,
    TRANSCRIPT_LABEL,
)

# ---- :Artifact sub-labels.
#
# Supporting / derived content. Multi-label co-applied alongside :Artifact
# (e.g. (:Artifact:Dataset)). Several of these need parser strategies that
# don't exist yet (Audio/Video transcription, structured-data parsing for
# Dataset, image parsing for Figure/Diagram) - Phase 4 work. The labels
# exist now so the schema is stable when parsers arrive.

DATASET_LABEL = "Dataset"
CODE_LABEL = "Code"
FIGURE_LABEL = "Figure"
DIAGRAM_LABEL = "Diagram"
AUDIO_LABEL = "Audio"
VIDEO_LABEL = "Video"

ARTIFACT_SUB_LABELS: tuple[str, ...] = (
    DATASET_LABEL,
    CODE_LABEL,
    FIGURE_LABEL,
    DIAGRAM_LABEL,
    AUDIO_LABEL,
    VIDEO_LABEL,
)

# All sub-labels (union) - used for top-level validation that a string
# refers to a known sub-label without yet checking which main-label group
# it belongs to.
ALL_SUB_LABELS: tuple[str, ...] = DOCUMENT_SUB_LABELS + ARTIFACT_SUB_LABELS

# Maps each sub-label back to its required main-label. Used at ingest
# time to validate `(main_label, sub_label)` consistency - rejects
# combinations like (Document, Dataset) before any write happens.
SUB_LABEL_TO_MAIN: dict[str, str] = {
    **{lbl: DOCUMENT_LABEL for lbl in DOCUMENT_SUB_LABELS},
    **{lbl: ARTIFACT_LABEL for lbl in ARTIFACT_SUB_LABELS},
}

# ---- Other node labels.

AUTHOR_LABEL = "Author"
VENUE_LABEL = "Venue"
TOPIC_LABEL = "Topic"
CHUNK_LABEL = "Chunk"

# L6: chunk-level entity mentions. Each :Entity is the deduped node
# for a (lowercased text, entity_type) pair. The original spelling
# from the source text is NOT stored on the node - for NER extractions
# it stays recoverable from the chunk text via the :MENTIONS.offset
# property; for LLM extractions it's lost (the LLM rewriting answers
# can re-capitalise from context). `canonicalised` flips to true at
# L7 when an ontology-linking layer matches this entity to a canonical
# ID (MeSH, GO, ...).
ENTITY_LABEL = "Entity"

# L7: ontology canonical concept nodes. Multi-label: every imported
# concept carries both :OntologyTerm (the shared super-label) and an
# ontology-specific sub-label (:MeSHTerm, :GOTerm, ...). This mirrors
# the :Document:Paper pattern at the top of the file - lets queries
# either scope to one ontology (`MATCH (t:MeSHTerm)`) or span all
# ontologies (`MATCH (t:OntologyTerm)`).
ONTOLOGY_TERM_LABEL = "OntologyTerm"
MESH_TERM_LABEL = "MeSHTerm"
GO_TERM_LABEL = "GOTerm"
HPO_TERM_LABEL = "HPOTerm"
UBERON_TERM_LABEL = "UBERONTerm"
MONDO_TERM_LABEL = "MONDOTerm"
CHEBI_TERM_LABEL = "ChEBITerm"
ECO_TERM_LABEL = "ECOTerm"
SO_TERM_LABEL = "SOTerm"
PR_TERM_LABEL = "PRTerm"
CL_TERM_LABEL = "CLTerm"
PO_TERM_LABEL = "POTerm"
FOODON_TERM_LABEL = "FOODONTerm"
ENVO_TERM_LABEL = "ENVOTerm"
NCBITAXON_TERM_LABEL = "NCBITaxonTerm"
OBI_TERM_LABEL = "OBITerm"
EFO_TERM_LABEL = "EFOTerm"
DRON_TERM_LABEL = "DRONTerm"
FIBO_TERM_LABEL = "FIBOTerm"

# All ontology sub-labels (union). Each new ontology adds its sub-label
# constant here so dispatcher / schema-as-prompt code can iterate.
ONTOLOGY_SUB_LABELS: tuple[str, ...] = (
    MESH_TERM_LABEL,
    GO_TERM_LABEL,
    HPO_TERM_LABEL,
    UBERON_TERM_LABEL,
    MONDO_TERM_LABEL,
    CHEBI_TERM_LABEL,
    ECO_TERM_LABEL,
    SO_TERM_LABEL,
    PR_TERM_LABEL,
    CL_TERM_LABEL,
    PO_TERM_LABEL,
    FOODON_TERM_LABEL,
    ENVO_TERM_LABEL,
    NCBITAXON_TERM_LABEL,
    OBI_TERM_LABEL,
    EFO_TERM_LABEL,
    DRON_TERM_LABEL,
    FIBO_TERM_LABEL,
)

# ---- Relationship types.

CITES_REL = "CITES"
AUTHORED_REL = "AUTHORED"
PUBLISHED_IN_REL = "PUBLISHED_IN"
ABOUT_TOPIC_REL = "ABOUT_TOPIC"
PART_OF_REL = "PART_OF"

# L6: (:Chunk)-[:MENTIONS {offset, confidence}]->(:Entity). offset +
# confidence are nullable on the edge - NER adapters populate them,
# LLM adapter leaves them null.
MENTIONS_REL = "MENTIONS"

# L7: hierarchy edges WITHIN one ontology. Per-ontology edge type (not
# a shared :IS_A) because each ontology's semantics differ: OBO's
# `is_a` is strict subsumption; SKOS's `skos:broader` is a looser
# "broader concept" relation. Per-ontology edges preserve those
# semantics, and cross-ontology traversal stays possible via Cypher's
# pipe syntax: `MATCH (a)-[:MESH_BROADER|GO_IS_A*]->(b)`.
MESH_BROADER_REL = "MESH_BROADER"
GO_IS_A_REL = "GO_IS_A"
HPO_IS_A_REL = "HPO_IS_A"
UBERON_IS_A_REL = "UBERON_IS_A"
MONDO_IS_A_REL = "MONDO_IS_A"
CHEBI_IS_A_REL = "CHEBI_IS_A"
ECO_IS_A_REL = "ECO_IS_A"
SO_IS_A_REL = "SO_IS_A"
PR_IS_A_REL = "PR_IS_A"
CL_IS_A_REL = "CL_IS_A"
PO_IS_A_REL = "PO_IS_A"
FOODON_IS_A_REL = "FOODON_IS_A"
ENVO_IS_A_REL = "ENVO_IS_A"
NCBITAXON_IS_A_REL = "NCBITAXON_IS_A"
OBI_IS_A_REL = "OBI_IS_A"
EFO_IS_A_REL = "EFO_IS_A"
DRON_IS_A_REL = "DRON_IS_A"
FIBO_IS_A_REL = "FIBO_IS_A"

# L7 cross-ontology xref edges. Each ontology's source data declares
# that certain of its concepts are equivalent / closely related to
# concepts in OTHER ontologies (OBO `xref:` lines, OWL `oboInOwl:
# hasDbXref` triples, SKOS `closeMatch` / `exactMatch`). When the
# user enables the `xrefs` layer, those cross-ontology pointers are
# materialised as Neo4j edges between :OntologyTerm sub-label nodes.
# Per-source-ontology edge types (not a shared :XREF) - faster typed
# queries, distinct sub-label-aware semantics, schema-as-prompt
# disambiguates each ontology's xrefs cleanly. Cross-source traversal
# stays possible via Cypher's type-predicate filtering:
#   MATCH (a:OntologyTerm)-[r]-(b:OntologyTerm)
#   WHERE type(r) ENDS WITH '_XREF'
# Edge naming is derivable from `<TERM_LABEL minus "Term">_XREF` -
# the `write_ontology_terms` helper computes it at write time, so
# per-ontology modules don't pass these constants explicitly.
# Constants here are for declarative documentation +
# schema-as-prompt rendering + bulk_op dispatch.
MESH_XREF_REL = "MESH_XREF"
GO_XREF_REL = "GO_XREF"
HPO_XREF_REL = "HPO_XREF"
UBERON_XREF_REL = "UBERON_XREF"
MONDO_XREF_REL = "MONDO_XREF"
CHEBI_XREF_REL = "CHEBI_XREF"
ECO_XREF_REL = "ECO_XREF"
SO_XREF_REL = "SO_XREF"
PR_XREF_REL = "PR_XREF"
CL_XREF_REL = "CL_XREF"
PO_XREF_REL = "PO_XREF"
FOODON_XREF_REL = "FOODON_XREF"
ENVO_XREF_REL = "ENVO_XREF"
NCBITAXON_XREF_REL = "NCBITAXON_XREF"
OBI_XREF_REL = "OBI_XREF"
EFO_XREF_REL = "EFO_XREF"
DRON_XREF_REL = "DRON_XREF"
FIBO_XREF_REL = "FIBO_XREF"

# Canonical ordered tuple of all 18 xref edge types. Drives:
#   - schema_as_prompt (block listing predicates for the agent)
#   - bulk_op dispatch (e.g. `clear_xref_edges`)
#   - test assertions
ONTOLOGY_XREF_RELS: tuple[str, ...] = (
    MESH_XREF_REL,
    GO_XREF_REL,
    HPO_XREF_REL,
    UBERON_XREF_REL,
    MONDO_XREF_REL,
    CHEBI_XREF_REL,
    ECO_XREF_REL,
    SO_XREF_REL,
    PR_XREF_REL,
    CL_XREF_REL,
    PO_XREF_REL,
    FOODON_XREF_REL,
    ENVO_XREF_REL,
    NCBITAXON_XREF_REL,
    OBI_XREF_REL,
    EFO_XREF_REL,
    DRON_XREF_REL,
    FIBO_XREF_REL,
)

# L7: the entity -> canonical-term link. Carries `strategy` ("exact"
# or "fuzzy") and `confidence` (float | null - exact matches leave it
# null, fuzzy matches set the similarity score). One :Entity can have
# multiple :CANONICAL_TO edges - one per ontology it matched.
CANONICAL_TO_REL = "CANONICAL_TO"

# L8: typed entity-to-entity relations extracted by an LLM from chunk
# text. The predicate IS the edge type (NOT a `predicate` property on a
# generic `:RELATION` edge) - 15 fixed types, idiomatic Neo4j, fast
# typed queries ("what inhibits X?" = MATCH (a)-[:INHIBITS]->(:Entity
# {key:"x"})). Each edge carries `chunk_id`, `doc_id`, and
# `evidence_span` (verbatim snippet) as properties for provenance.
# ONE edge per chunk assertion - if the same triple shows up in 5
# chunks, 5 edges exist (one per chunk_id), each with its own
# evidence; query-time `count()` / `collect()` dedupes if needed.
# Vocabulary is CONSTRAINED - LLM picks from these 15 or emits
# nothing (better to lose a relation than invent a new predicate that
# breaks query consistency).
INHIBITS_REL = "INHIBITS"
ACTIVATES_REL = "ACTIVATES"
REGULATES_REL = "REGULATES"
INTERACTS_WITH_REL = "INTERACTS_WITH"
BINDS_TO_REL = "BINDS_TO"
IS_PART_OF_REL = "IS_PART_OF"
IS_A_REL = "IS_A"
CAUSES_REL = "CAUSES"
ASSOCIATED_WITH_REL = "ASSOCIATED_WITH"
TREATS_REL = "TREATS"
MEASURED_BY_REL = "MEASURED_BY"
EXPRESSED_IN_REL = "EXPRESSED_IN"
LOCATED_IN_REL = "LOCATED_IN"
USES_REL = "USES"
COMPARED_WITH_REL = "COMPARED_WITH"

# L9: cross-document synthesis - materialised edge between two document
# (or artifact) nodes that share at least N L6 entities. Undirected
# (one edge per pair, written via MERGE on the undirected pattern).
# Properties: `shared_entities` (list of entity keys), `shared_count`
# (size), `computed_at` (timestamp). Recomputed per-doc on ingest /
# backfill; the focal node's :Document DETACH DELETE in `delete_doc`
# wipes the edges automatically, so there's no separate L9 cleanup
# primitive. Threshold lives in `cross_doc_writes.py`.
RELATED_TO_REL = "RELATED_TO"

# L10: concept-level cross-document synthesis - materialised edge
# between two document (or artifact) nodes that share at least N
# canonical concepts via xref equivalence. Parallel to L9 but uses
# the xref graph: two docs share a concept when their entities
# canonicalise (via :CANONICAL_TO) to :OntologyTerm nodes that are
# connected by an xref edge (or are the same node). Undirected.
# Properties: `shared_concepts` (list of OntologyTerm IDs),
# `shared_count` (size), `computed_at` (timestamp). Requires the
# `xrefs` layer in `"use"` mode (xref edges must exist). Threshold
# configurable per-corpus; default 2. Recomputed per-doc on ingest /
# backfill; the focal node's :Document DETACH DELETE in `delete_doc`
# wipes the edges automatically (cascade).
RELATED_BY_XREF_REL = "RELATED_BY_XREF"

# Canonical ordered tuple of the 15 L8 predicate edge types. Drives:
#   - LLM prompt (vocabulary the model is constrained to)
#   - Cypher generators (one MERGE per predicate type)
#   - schema_as_prompt (block listing predicates for the agent)
#   - delete_triples_by_doc_id (one MATCH per type)
# Order is "common-first": regulatory/causal verbs before structural
# before methodological. Re-ordering is cosmetic; the LLM treats them
# as a set.
TRIPLE_PREDICATE_RELS: tuple[str, ...] = (
    INHIBITS_REL,
    ACTIVATES_REL,
    REGULATES_REL,
    INTERACTS_WITH_REL,
    BINDS_TO_REL,
    IS_PART_OF_REL,
    IS_A_REL,
    CAUSES_REL,
    ASSOCIATED_WITH_REL,
    TREATS_REL,
    MEASURED_BY_REL,
    EXPRESSED_IN_REL,
    LOCATED_IN_REL,
    USES_REL,
    COMPARED_WITH_REL,
)

# ---- Constraint statements.
#
# Idempotent (`IF NOT EXISTS`), safe to apply on every startup. Each enforces
# uniqueness on an identifier that, when present on a node, must be globally
# unique. Adding new layers means appending new constraints here.
#
# doc_id is enforced unique on BOTH top-level labels so each store stays
# consistent. The same SHA-256 bytes-hash could in principle apply to a
# Document and an Artifact node, but in practice SHA-256 collision
# probability is astronomical, so per-label uniqueness is sufficient.

CONSTRAINT_STATEMENTS: tuple[str, ...] = (
    f"CREATE CONSTRAINT document_doc_id IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) REQUIRE d.doc_id IS UNIQUE",
    f"CREATE CONSTRAINT document_openalex_id IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) REQUIRE d.openalex_id IS UNIQUE",
    f"CREATE CONSTRAINT document_doi IF NOT EXISTS "
    f"FOR (d:{DOCUMENT_LABEL}) REQUIRE d.doi IS UNIQUE",
    f"CREATE CONSTRAINT artifact_doc_id IF NOT EXISTS "
    f"FOR (a:{ARTIFACT_LABEL}) REQUIRE a.doc_id IS UNIQUE",
    f"CREATE CONSTRAINT author_openalex_id IF NOT EXISTS "
    f"FOR (a:{AUTHOR_LABEL}) REQUIRE a.openalex_id IS UNIQUE",
    f"CREATE CONSTRAINT venue_openalex_id IF NOT EXISTS "
    f"FOR (v:{VENUE_LABEL}) REQUIRE v.openalex_id IS UNIQUE",
    f"CREATE CONSTRAINT topic_openalex_id IF NOT EXISTS "
    f"FOR (t:{TOPIC_LABEL}) REQUIRE t.openalex_id IS UNIQUE",
    f"CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
    f"FOR (c:{CHUNK_LABEL}) REQUIRE c.chunk_id IS UNIQUE",
    # L6: composite NODE KEY enforces (key, entity_type) globally unique
    # AND non-null on both fields. Same word with different types
    # (e.g. "apple" as GENE vs as COMPANY) stays correctly distinct.
    f"CREATE CONSTRAINT entity_key_node_key IF NOT EXISTS "
    f"FOR (e:{ENTITY_LABEL}) REQUIRE (e.key, e.entity_type) IS NODE KEY",
    # L7: ontology canonical IDs are unique per sub-label. Constraints
    # live on the sub-label (`:MeSHTerm`, `:GOTerm`), NOT on
    # `:OntologyTerm`, because each ontology has its own ID namespace
    # (MeSH uses "D003920", GO uses "GO:0008150") - they can't collide
    # within their own ontology, but the cross-ontology constraint
    # would falsely conflate them. Constraints fire idempotently at
    # startup; harmless when the layer's data doesn't exist yet.
    f"CREATE CONSTRAINT mesh_term_id IF NOT EXISTS "
    f"FOR (t:{MESH_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT go_term_id IF NOT EXISTS FOR (t:{GO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT hpo_term_id IF NOT EXISTS FOR (t:{HPO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT uberon_term_id IF NOT EXISTS "
    f"FOR (t:{UBERON_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT mondo_term_id IF NOT EXISTS "
    f"FOR (t:{MONDO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT chebi_term_id IF NOT EXISTS "
    f"FOR (t:{CHEBI_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT eco_term_id IF NOT EXISTS FOR (t:{ECO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT so_term_id IF NOT EXISTS FOR (t:{SO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT pr_term_id IF NOT EXISTS FOR (t:{PR_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT cl_term_id IF NOT EXISTS FOR (t:{CL_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT po_term_id IF NOT EXISTS FOR (t:{PO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT foodon_term_id IF NOT EXISTS "
    f"FOR (t:{FOODON_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT envo_term_id IF NOT EXISTS "
    f"FOR (t:{ENVO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT ncbitaxon_term_id IF NOT EXISTS "
    f"FOR (t:{NCBITAXON_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT obi_term_id IF NOT EXISTS FOR (t:{OBI_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT efo_term_id IF NOT EXISTS FOR (t:{EFO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT dron_term_id IF NOT EXISTS "
    f"FOR (t:{DRON_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
    f"CREATE CONSTRAINT fibo_term_id IF NOT EXISTS "
    f"FOR (t:{FIBO_TERM_LABEL}) REQUIRE t.id IS UNIQUE",
)
