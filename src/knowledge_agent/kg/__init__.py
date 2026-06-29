"""Knowledge-graph layer (Neo4j).

Owns the schema (labels, relationship types, constraints) and the driver
wrapper used by the ingestion pipeline and the agent's query path. Built
layer-by-layer:

- L1: :Paper + :CITES (from OpenAlex referenced_works)
- L2: :Author + :AUTHORED (from OpenAlex authorships)
- L3: :Venue + :PUBLISHED_IN (from OpenAlex primary_location)
- L4: :Topic + :ABOUT_TOPIC (from OpenAlex topics)
- L5: :Chunk + :PART_OF (from the docling chunker; pivots the KG from
      structural overlay to full-content store)
- L6: domain entities (:Gene, :Protein, ...) linked via NER + entity linkers
GROBID, NER, FORMAL ONTOLOGIES, DOCLING-GRAPH
"""
