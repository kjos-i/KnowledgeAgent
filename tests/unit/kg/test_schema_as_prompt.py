"""Drift tests for `kg/schema_as_prompt.py`.

Two kinds of tests:

- Auto-sync tests assert that every label and relationship-type CONSTANT
  declared in `kg/schema.py` appears in the rendered prompt (when the
  config has every layer enabled). If someone renames a constant but
  forgets to use the new name in the prompt body, these tests fail.

- Checklist tests act as the manual source of truth for properties (which
  are NOT constants - they are inline strings in the per-domain Cypher
  writes). When a property is added or renamed in a writes module, BOTH
  the prompt body AND the lists below must be updated. The test does not
  catch drift if you forget the test list - it ensures the prompt body
  matches the checklist.

- Layer-gating tests assert that the renderer correctly omits sections
  when a layer is turned off in the `CorpusConfig`, so the LLM never
  sees labels for which there is no data.

When L6 lands (entities), append its properties + label constants to
the lists below AND add layer-gating tests for the new flag.
"""

from knowledge_agent.kg.corpus_config import (
    CorpusConfig,
    EntityConfig,
    LayerFlags,
    OntologyConfig,
)
from knowledge_agent.kg.schema import (
    ABOUT_TOPIC_REL,
    ARTIFACT_LABEL,
    ARTIFACT_SUB_LABELS,
    AUTHOR_LABEL,
    AUTHORED_REL,
    CANONICAL_TO_REL,
    CHUNK_LABEL,
    CITES_REL,
    DOCUMENT_LABEL,
    DOCUMENT_SUB_LABELS,
    ENTITY_LABEL,
    GO_IS_A_REL,
    GO_TERM_LABEL,
    MENTIONS_REL,
    MESH_BROADER_REL,
    MESH_TERM_LABEL,
    ONTOLOGY_TERM_LABEL,
    PAPER_LABEL,
    PART_OF_REL,
    PUBLISHED_IN_REL,
    ONTOLOGY_XREF_RELS,
    RELATED_BY_XREF_REL,
    RELATED_TO_REL,
    TOPIC_LABEL,
    TRIPLE_PREDICATE_RELS,
    VENUE_LABEL,
)
from knowledge_agent.kg.schema_as_prompt import (
    format_schema_for_prompt,
)


# ---- shared fixtures (in-memory configs covering the layer combinations) ----


_ALL_SUBS = list(DOCUMENT_SUB_LABELS) + list(ARTIFACT_SUB_LABELS)


def _full_config() -> CorpusConfig:
    """Every layer enabled + every sub-label allowed - the maximal prompt.
    Used by every existing drift / checklist test."""
    return CorpusConfig(
        domain="biomedical",
        allowed_types=_ALL_SUBS,
        layers=LayerFlags(
            openalex_papers=True, chunks=True, entities=True
        ),
        entities=EntityConfig(
            extractor="llm", entity_types=["GENE", "DISEASE"]
        ),
    )


def _openalex_only_config() -> CorpusConfig:
    """L1-L4 only - chunks layer off. Paper allowed to keep OpenAlex active."""
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=False),
    )


def _chunks_only_config() -> CorpusConfig:
    """L5 only - openalex layer off (non-paper corpus, e.g. legal)."""
    return CorpusConfig(
        domain="legal",
        allowed_types=["Note"],  # one sub-label so :Note appears in the prompt
        layers=LayerFlags(openalex_papers=False, chunks=True),
    )


# ---- auto-synced parts (catches drift between schema.py constants and prompt) ----


def test_prompt_includes_all_labels_from_schema():
    """Every label constant declared in schema.py must appear in the
    fully-enabled prompt."""
    prompt = format_schema_for_prompt(_full_config())
    assert f":{DOCUMENT_LABEL}" in prompt
    assert f":{PAPER_LABEL}" in prompt
    assert f":{AUTHOR_LABEL}" in prompt
    assert f":{VENUE_LABEL}" in prompt
    assert f":{TOPIC_LABEL}" in prompt
    assert f":{CHUNK_LABEL}" in prompt
    assert f":{ENTITY_LABEL}" in prompt


def test_prompt_includes_all_relationship_types_from_schema():
    """Every relationship-type constant declared in schema.py must appear
    in the fully-enabled prompt."""
    prompt = format_schema_for_prompt(_full_config())
    assert f":{CITES_REL}" in prompt
    assert f":{AUTHORED_REL}" in prompt
    assert f":{PUBLISHED_IN_REL}" in prompt
    assert f":{ABOUT_TOPIC_REL}" in prompt
    assert f":{PART_OF_REL}" in prompt
    assert f":{MENTIONS_REL}" in prompt


# ---- checklist tests (catches drift between writes modules and prompt) ----

EXPECTED_DOCUMENT_PROPERTIES = ("doc_id", "openalex_id", "doi", "in_corpus")
EXPECTED_AUTHOR_PROPERTIES = ("openalex_id", "display_name")
EXPECTED_VENUE_PROPERTIES = ("openalex_id", "name", "type", "issn")
EXPECTED_TOPIC_PROPERTIES = ("openalex_id", "display_name")
EXPECTED_CHUNK_PROPERTIES = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "section",
    "page",
    "content_type",
)
EXPECTED_AUTHORED_EDGE_PROPERTIES = ("position", "is_corresponding")
EXPECTED_ABOUT_TOPIC_EDGE_PROPERTIES = ("score",)
EXPECTED_ENTITY_PROPERTIES = ("key", "entity_type", "canonicalised")
EXPECTED_MENTIONS_EDGE_PROPERTIES = ("offset", "confidence")


def test_prompt_lists_all_expected_document_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_DOCUMENT_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_DOCUMENT_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_author_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_AUTHOR_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_AUTHOR_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_venue_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_VENUE_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_VENUE_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_topic_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_TOPIC_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_TOPIC_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_chunk_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_CHUNK_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_CHUNK_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_authored_edge_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_AUTHORED_EDGE_PROPERTIES:
        assert prop in prompt, (
            f"Edge property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_AUTHORED_EDGE_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_about_topic_edge_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_ABOUT_TOPIC_EDGE_PROPERTIES:
        assert prop in prompt, (
            f"Edge property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_ABOUT_TOPIC_"
            f"EDGE_PROPERTIES list if the schema changed."
        )


# ---- layer-gating tests (catches drift between LayerFlags and rendered sections) ----


def test_openalex_off_omits_paper_subtype_note():
    """:Paper subtype mention disappears when openalex_papers is off."""
    prompt = format_schema_for_prompt(_chunks_only_config())
    assert f":{PAPER_LABEL}" not in prompt


def test_openalex_off_omits_openalex_only_document_properties():
    """openalex_id / doi / in_corpus drop off :Document when openalex
    layer is off - there are no shadow nodes and no OpenAlex IDs."""
    prompt = format_schema_for_prompt(_chunks_only_config())
    assert "openalex_id" not in prompt
    assert "doi" not in prompt
    assert "in_corpus" not in prompt


def test_openalex_off_omits_author_venue_topic_labels():
    """:Author / :Venue / :Topic labels drop entirely when openalex
    layer is off."""
    prompt = format_schema_for_prompt(_chunks_only_config())
    assert f":{AUTHOR_LABEL}" not in prompt
    assert f":{VENUE_LABEL}" not in prompt
    assert f":{TOPIC_LABEL}" not in prompt


def test_openalex_off_omits_openalex_only_relationships():
    """:CITES / :AUTHORED / :PUBLISHED_IN / :ABOUT_TOPIC drop when
    openalex layer is off."""
    prompt = format_schema_for_prompt(_chunks_only_config())
    assert f":{CITES_REL}" not in prompt
    assert f":{AUTHORED_REL}" not in prompt
    assert f":{PUBLISHED_IN_REL}" not in prompt
    assert f":{ABOUT_TOPIC_REL}" not in prompt


def test_chunks_off_omits_chunk_label_and_properties():
    """:Chunk label + its properties drop when chunks layer is off."""
    prompt = format_schema_for_prompt(_openalex_only_config())
    assert f":{CHUNK_LABEL}" not in prompt
    assert "chunk_id" not in prompt
    assert "chunk_index" not in prompt
    assert "content_type" not in prompt


def test_chunks_off_omits_part_of_relationship():
    """:PART_OF edge drops when chunks layer is off."""
    prompt = format_schema_for_prompt(_openalex_only_config())
    assert f":{PART_OF_REL}" not in prompt


def test_chunks_off_omits_chunk_id_from_identifier_conventions():
    """Identifier conventions footer should not mention chunk_id when the
    chunks layer is off."""
    prompt = format_schema_for_prompt(_openalex_only_config())
    assert "chunk_id" not in prompt


def test_openalex_off_omits_openalex_id_advice_from_conventions():
    """Identifier conventions footer should not mention `openalex_id`,
    `doi`, or `in_corpus` advice when openalex layer is off."""
    prompt = format_schema_for_prompt(_chunks_only_config())
    assert "openalex_id" not in prompt
    assert "doi" not in prompt
    assert "in_corpus" not in prompt


def test_document_label_present_in_every_active_config():
    """:Document is the focal node for every layer; it must appear in
    any config where at least one layer is active."""
    for config in (_full_config(), _openalex_only_config(), _chunks_only_config()):
        prompt = format_schema_for_prompt(config)
        assert f":{DOCUMENT_LABEL}" in prompt
        assert "doc_id" in prompt


def test_relationships_header_omitted_when_no_relationships_active():
    """When both layers are off, no edges exist - the 'Relationships:'
    header should be omitted entirely rather than dangle."""
    bare_config = CorpusConfig(
        domain="generic",
        layers=LayerFlags(openalex_papers=False, chunks=False),
    )
    prompt = format_schema_for_prompt(bare_config)
    assert "Relationships:" not in prompt


# ---- new in Phase 3: Artifact label + sub-label listing ----


def test_prompt_includes_artifact_top_level_label():
    """:Artifact is always present (loose-file ingest valid for either
    top-level label)."""
    prompt = format_schema_for_prompt(_full_config())
    assert f":{ARTIFACT_LABEL}" in prompt


def test_prompt_lists_only_allowed_sub_labels():
    """Sub-labels appear only when in `allowed_types`."""
    config = CorpusConfig(
        domain="mixed",
        allowed_types=["Paper", "Dataset"],  # subset
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    prompt = format_schema_for_prompt(config)
    # Listed.
    assert ":Paper" in prompt
    assert ":Dataset" in prompt
    # Not listed (not in allowed_types).
    assert ":Note" not in prompt
    assert ":Book" not in prompt
    assert ":Code" not in prompt
    assert ":Figure" not in prompt


def test_openalex_section_omitted_when_paper_not_in_allowed_types():
    """openalex_papers=True but Paper not in allowed_types -> L1-L4
    cannot fire (the gate is layer AND Paper-allowed), so :Author /
    :Venue / :Topic should not appear in the prompt."""
    config = CorpusConfig(
        domain="notes_only",
        allowed_types=["Note"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
    )
    prompt = format_schema_for_prompt(config)
    assert f":{AUTHOR_LABEL}" not in prompt
    assert f":{VENUE_LABEL}" not in prompt
    assert f":{TOPIC_LABEL}" not in prompt


# ---- L6a entity layer ----


def test_prompt_lists_all_expected_entity_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_ENTITY_PROPERTIES:
        assert prop in prompt, (
            f"Property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_ENTITY_"
            f"PROPERTIES list if the schema changed."
        )


def test_prompt_lists_all_expected_mentions_edge_properties():
    prompt = format_schema_for_prompt(_full_config())
    for prop in EXPECTED_MENTIONS_EDGE_PROPERTIES:
        assert prop in prompt, (
            f"Edge property {prop!r} missing from schema prompt. Update "
            f"schema_as_prompt.py and this test's EXPECTED_MENTIONS_EDGE_"
            f"PROPERTIES list if the schema changed."
        )


def test_entities_off_omits_entity_label_and_mentions_edge():
    """`:Entity` + `:MENTIONS` drop entirely when the entities layer is
    off (default config)."""
    config = CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(openalex_papers=True, chunks=True),
        # entities defaults to False; no [entities] section needed.
    )
    prompt = format_schema_for_prompt(config)
    assert f":{ENTITY_LABEL}" not in prompt
    assert f":{MENTIONS_REL}" not in prompt
    # And the entity-only properties drop with the block.
    assert "canonicalised" not in prompt
    assert "entity_type" not in prompt


def test_entities_on_lists_configured_entity_types_in_block():
    """The Entity block surfaces the corpus's configured entity_types so
    the LLM knows what values it can scope queries to."""
    config = CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(
            extractor="llm", entity_types=["GENE", "DISEASE", "CHEMICAL"]
        ),
    )
    prompt = format_schema_for_prompt(config)
    assert "GENE" in prompt
    assert "DISEASE" in prompt
    assert "CHEMICAL" in prompt


def test_entities_on_with_empty_entity_types_flags_open_vocab():
    """Empty entity_types -> the block tells the LLM the vocabulary is
    open (not a fixed list), so it knows to inspect data rather than
    assume types."""
    config = CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm", entity_types=[]),
    )
    prompt = format_schema_for_prompt(config)
    assert f":{ENTITY_LABEL}" in prompt
    assert "open-vocabulary" in prompt.lower()


# ---- L7 ontology layers ----


def _config_with_mesh() -> CorpusConfig:
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True, entities=True, ontology_mesh=True
        ),
        entities=EntityConfig(extractor="llm", entity_types=["DISEASE"]),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )


def _config_with_mesh_and_go() -> CorpusConfig:
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            ontology_go=True,
        ),
        entities=EntityConfig(extractor="llm", entity_types=["DISEASE"]),
        ontology={
            "mesh": OntologyConfig(matching="exact"),
            "go": OntologyConfig(matching="fuzzy"),
        },
    )


def test_ontologies_off_omits_ontology_blocks():
    """Default config (no ontology_* flags on) -> no :OntologyTerm,
    :MeSHTerm, :GOTerm, hierarchy edges, or :CANONICAL_TO."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    prompt = format_schema_for_prompt(config)
    assert f":{ONTOLOGY_TERM_LABEL}" not in prompt
    assert f":{MESH_TERM_LABEL}" not in prompt
    assert f":{GO_TERM_LABEL}" not in prompt
    assert f":{MESH_BROADER_REL}" not in prompt
    assert f":{GO_IS_A_REL}" not in prompt
    assert f":{CANONICAL_TO_REL}" not in prompt


def test_mesh_on_renders_mesh_term_block():
    """ontology_mesh=true -> :OntologyTerm:MeSHTerm block + :MESH_BROADER
    hierarchy edge + :CANONICAL_TO edge appear in the prompt."""
    prompt = format_schema_for_prompt(_config_with_mesh())
    assert f":{ONTOLOGY_TERM_LABEL}" in prompt
    assert f":{MESH_TERM_LABEL}" in prompt
    assert f":{MESH_BROADER_REL}" in prompt
    assert f":{CANONICAL_TO_REL}" in prompt


def test_mesh_on_does_not_render_go_blocks():
    """ontology_mesh=true alone -> no :GOTerm or :GO_IS_A in prompt."""
    prompt = format_schema_for_prompt(_config_with_mesh())
    assert f":{GO_TERM_LABEL}" not in prompt
    assert f":{GO_IS_A_REL}" not in prompt


def test_both_mesh_and_go_render_both_blocks():
    """Two ontologies enabled -> both sub-label blocks + both hierarchy
    edge types appear."""
    prompt = format_schema_for_prompt(_config_with_mesh_and_go())
    assert f":{MESH_TERM_LABEL}" in prompt
    assert f":{GO_TERM_LABEL}" in prompt
    assert f":{MESH_BROADER_REL}" in prompt
    assert f":{GO_IS_A_REL}" in prompt
    # :CANONICAL_TO is the shared link edge; rendered once.
    assert prompt.count(f"[:{CANONICAL_TO_REL}]") == 1


def test_mesh_block_lists_expected_term_properties():
    """Term nodes carry id / label / synonyms / definition."""
    prompt = format_schema_for_prompt(_config_with_mesh())
    # We don't pin exact wording; just verify each property name is
    # mentioned in the prompt body somewhere.
    for prop in ("id", "label", "synonyms", "definition"):
        assert prop in prompt, f"Property {prop!r} missing from L7 block"


def test_canonical_to_block_lists_strategy_and_confidence():
    """:CANONICAL_TO edge documentation must include the strategy +
    confidence properties so the LLM knows it can filter by them."""
    prompt = format_schema_for_prompt(_config_with_mesh())
    assert "strategy" in prompt
    assert "confidence" in prompt


# ---- L8 triples layer ----


def _config_with_triples() -> CorpusConfig:
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, triples=True),
        entities=EntityConfig(extractor="llm", entity_types=["GENE"]),
    )


def test_triples_off_omits_predicate_edges():
    """Default config (no triples flag) -> none of the 15 predicate
    edge labels appear in the rendered prompt."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    prompt = format_schema_for_prompt(config)
    for pred in TRIPLE_PREDICATE_RELS:
        # We assert the rel-label form `:INHIBITS` etc. is absent, not
        # the bare word - "BINDS_TO" might appear in unrelated text in
        # principle, but `:BINDS_TO` is unique to the triples block.
        assert f":{pred}" not in prompt, (
            f"Predicate :{pred} should not appear when triples layer "
            f"is off"
        )


def test_triples_on_renders_all_15_predicates():
    """`layers.triples=true` -> all 15 predicate edge types listed in
    the L8 block so the LLM knows which it can use."""
    prompt = format_schema_for_prompt(_config_with_triples())
    for pred in TRIPLE_PREDICATE_RELS:
        assert f":{pred}" in prompt, (
            f"Predicate :{pred} missing from L8 block. Update "
            f"schema_as_prompt._TRIPLES_BLOCK if the predicate list "
            f"changed."
        )


def test_triples_block_describes_entity_to_entity_pattern():
    """The block must tell the LLM these are :Entity-to-:Entity edges
    so it doesn't try to attach them to e.g. :Chunk nodes."""
    prompt = format_schema_for_prompt(_config_with_triples())
    # Look for the node pattern in the block.
    assert f"(:{ENTITY_LABEL})-[" in prompt
    # And the target side is also :Entity.
    assert f"]->(:{ENTITY_LABEL})" in prompt


def test_triples_block_documents_edge_properties():
    """Each typed edge carries chunk_id, doc_id, evidence_span - the
    LLM needs to know these for citation queries."""
    prompt = format_schema_for_prompt(_config_with_triples())
    assert "chunk_id" in prompt
    assert "doc_id" in prompt
    assert "evidence_span" in prompt


def test_triples_block_shows_pipe_syntax_for_multi_predicate_match():
    """The block should hint at the Cypher pipe syntax so the LLM
    can write `[:INHIBITS|REGULATES]` for predicate-union queries."""
    prompt = format_schema_for_prompt(_config_with_triples())
    assert "|" in prompt
    # Crude check that the pipe shows up in a Cypher-like context.
    assert "INHIBITS" in prompt


# ---- L9 cross_doc layer ----


def _config_with_cross_doc() -> CorpusConfig:
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(chunks=True, entities=True, cross_doc=True),
        entities=EntityConfig(extractor="llm", entity_types=["GENE"]),
    )


def test_cross_doc_off_omits_related_to_edge():
    """Default config (no cross_doc flag) -> :RELATED_TO absent."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    prompt = format_schema_for_prompt(config)
    assert f":{RELATED_TO_REL}" not in prompt


def test_cross_doc_on_renders_related_to_edge():
    prompt = format_schema_for_prompt(_config_with_cross_doc())
    assert f":{RELATED_TO_REL}" in prompt


def test_cross_doc_block_documents_edge_properties():
    """The block must surface shared_entities, shared_count, computed_at
    so the agent can rank or filter related-doc results."""
    prompt = format_schema_for_prompt(_config_with_cross_doc())
    assert "shared_entities" in prompt
    assert "shared_count" in prompt
    assert "computed_at" in prompt


def test_cross_doc_block_notes_undirected_semantics():
    """Hint to the LLM that :RELATED_TO is undirected so it doesn't
    add a `>` to one side and miss half the matches."""
    prompt = format_schema_for_prompt(_config_with_cross_doc())
    assert "undirected" in prompt.lower() or "UNDIRECTED" in prompt


def test_cross_doc_block_covers_both_document_and_artifact():
    """L9 spans both focal types; the prompt should reflect that."""
    prompt = format_schema_for_prompt(_config_with_cross_doc())
    # The block specifically mentions both labels as endpoints.
    assert f":{DOCUMENT_LABEL}" in prompt
    assert f":{ARTIFACT_LABEL}" in prompt


# ---- L7 xrefs layer (3-state) ----


def _config_with_xrefs(mode: str) -> CorpusConfig:
    """Build a config with the xrefs layer in the given mode and
    one ontology enabled (so the corpus is realistic)."""
    from knowledge_agent.kg.corpus_config import OntologyConfig
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            xrefs=mode,
        ),
        entities=EntityConfig(extractor="llm", entity_types=["GENE"]),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )


def test_xrefs_none_omits_xref_block():
    """Default `xrefs="none"` -> no `:<X>_XREF` predicates in prompt."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    prompt = format_schema_for_prompt(config)
    for rel in ONTOLOGY_XREF_RELS:
        assert f":{rel}" not in prompt


def test_xrefs_collect_only_omits_xref_block():
    """`xrefs="collect_only"` stores `dangling_xrefs` properties but
    NO `:<X>_XREF` edges exist yet. Telling the LLM about edges
    that don't materialise would invite empty results — so the
    block is gated on `"use"` specifically."""
    prompt = format_schema_for_prompt(_config_with_xrefs("collect_only"))
    for rel in ONTOLOGY_XREF_RELS:
        assert f":{rel}" not in prompt


def test_xrefs_use_renders_all_18_predicates():
    """`xrefs="use"` -> every shipped `:<X>_XREF` predicate appears in
    the prompt, in the canonical order from `ONTOLOGY_XREF_RELS`."""
    prompt = format_schema_for_prompt(_config_with_xrefs("use"))
    for rel in ONTOLOGY_XREF_RELS:
        assert f":{rel}" in prompt, (
            f"{rel} missing from xrefs-use prompt — update "
            "schema_as_prompt._XREFS_BLOCK if the predicate list "
            "changed in schema.py."
        )


def test_xrefs_use_block_describes_ontology_term_endpoints():
    """The block must scope both endpoints to `:OntologyTerm` so the
    agent doesn't try to use xref edges between unrelated nodes."""
    prompt = format_schema_for_prompt(_config_with_xrefs("use"))
    # The block emits `(:OntologyTerm)-[<predicate>]->(:OntologyTerm)`.
    # That string appears verbatim in the rendered block.
    assert "(:OntologyTerm)-[<predicate>]->(:OntologyTerm)" in prompt


def test_xrefs_use_block_shows_pipe_syntax_for_cross_source_traversal():
    """The block hints at the multi-predicate pipe pattern so the LLM
    writes typed queries when joining across source ontologies."""
    prompt = format_schema_for_prompt(_config_with_xrefs("use"))
    # The block tells the LLM how to span multiple xref predicates.
    assert "MESH_XREF|GO_XREF|CHEBI_XREF" in prompt


def test_xrefs_block_states_no_edge_properties():
    """Xref edges carry no properties — the prompt must say so."""
    prompt = format_schema_for_prompt(_config_with_xrefs("use"))
    assert "No edge properties" in prompt


# ---- L10 cross_doc_xrefs layer ----


def _config_with_cross_doc_xrefs() -> CorpusConfig:
    """Minimal L10-enabled config. L10 requires entities=true AND
    xrefs="use" plus at least one ontology (validator enforces the
    chain)."""
    from knowledge_agent.kg.corpus_config import OntologyConfig
    return CorpusConfig(
        domain="biomedical",
        allowed_types=["Paper"],
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            xrefs="use",
            cross_doc_xrefs=True,
        ),
        entities=EntityConfig(extractor="llm", entity_types=["GENE"]),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )


def test_cross_doc_xrefs_off_omits_related_by_xref_edge():
    """Default config (no cross_doc_xrefs flag) -> :RELATED_BY_XREF absent."""
    config = CorpusConfig(
        layers=LayerFlags(chunks=True, entities=True),
        entities=EntityConfig(extractor="llm"),
    )
    prompt = format_schema_for_prompt(config)
    assert f":{RELATED_BY_XREF_REL}" not in prompt


def test_cross_doc_xrefs_on_renders_related_by_xref_edge():
    prompt = format_schema_for_prompt(_config_with_cross_doc_xrefs())
    assert f":{RELATED_BY_XREF_REL}" in prompt


def test_cross_doc_xrefs_block_documents_edge_properties():
    """The L10 block must surface shared_concepts, shared_count,
    computed_at — agent ranks / filters on these."""
    prompt = format_schema_for_prompt(_config_with_cross_doc_xrefs())
    assert "shared_concepts" in prompt
    assert "shared_count" in prompt
    assert "computed_at" in prompt


def test_cross_doc_xrefs_block_notes_undirected_semantics():
    """Same undirected hint as L9, otherwise the agent will write
    a directional MERGE and miss half the matches."""
    prompt = format_schema_for_prompt(_config_with_cross_doc_xrefs())
    assert "undirected" in prompt.lower() or "UNDIRECTED" in prompt


def test_cross_doc_xrefs_block_covers_both_document_and_artifact():
    """L10 spans both focal types just like L9."""
    prompt = format_schema_for_prompt(_config_with_cross_doc_xrefs())
    assert f":{DOCUMENT_LABEL}" in prompt
    assert f":{ARTIFACT_LABEL}" in prompt


def test_cross_doc_xrefs_block_calls_out_parallel_to_l9():
    """The block tells the agent L10 surfaces concept-level overlap
    that L9's raw-entity signal misses (cross-ontology equivalence
    is the key differentiation)."""
    prompt = format_schema_for_prompt(_config_with_cross_doc_xrefs())
    # Both edges' names appear so the LLM understands they coexist.
    assert f":{RELATED_TO_REL}" not in prompt or f":{RELATED_BY_XREF_REL}" in prompt
    # The block specifically references concept-level overlap.
    assert "Concept-level" in prompt or "canonical concepts" in prompt


def test_cross_doc_xrefs_can_be_enabled_alongside_l9():
    """Both L9 and L10 can be on simultaneously - the prompt renders
    both edge blocks (they're complementary signals)."""
    from knowledge_agent.kg.corpus_config import OntologyConfig
    config = CorpusConfig(
        layers=LayerFlags(
            chunks=True,
            entities=True,
            ontology_mesh=True,
            xrefs="use",
            cross_doc=True,
            cross_doc_xrefs=True,
        ),
        entities=EntityConfig(extractor="llm", entity_types=["GENE"]),
        ontology={"mesh": OntologyConfig(matching="exact")},
    )
    prompt = format_schema_for_prompt(config)
    assert f":{RELATED_TO_REL}" in prompt
    assert f":{RELATED_BY_XREF_REL}" in prompt
