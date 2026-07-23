"""Knowledge-graph layer (Neo4j).

Owns the schema (labels, relationship types, constraints) in `schema.py` and
the async driver wrapper in `client.py`, used by the ingestion pipeline and the
agent's query path. Built layer-by-layer, each independently on/off per corpus
(`corpus_config.LayerFlags`; token-costly L6/L8 default off):

- L1: :Paper + :CITES (from OpenAlex referenced_works)
- L2: :Author + :AUTHORED (from OpenAlex authorships)
- L3: :Venue + :PUBLISHED_IN (from OpenAlex primary_location)
- L4: :Topic + :ABOUT_TOPIC (from OpenAlex topics)
- L5: :Chunk + :PART_OF (from the parse layer's chunker; pivots the KG from
      structural overlay to full-content store)
- L6: :Entity + :MENTIONS (extracted concepts; NER + LLM entity extractors)
- L7: :OntologyTerm canonicals + :CANONICAL_TO from :Entity, plus *_XREF
      cross-ontology equivalence and *_IS_A hierarchy (the imported formal
      ontologies)
- L8: typed entity-to-entity triples — the predicate IS the edge (:INHIBITS,
      :TREATS, …; 15 in schema.TRIPLE_PREDICATE_RELS), from LLM extraction
- L9: :RELATED_TO cross-document edges (docs sharing ≥ N L6 entities)
- L10: :RELATED_BY_XREF cross-document edges (docs sharing ≥ N canonical
      concepts via L7 xref equivalence)
"""
