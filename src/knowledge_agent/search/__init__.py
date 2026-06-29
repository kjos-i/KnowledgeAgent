"""Hybrid retrieval layer (LanceDB).

Owns the chunk-table schema and the embedded LanceDB connection used by the
ingestion pipeline (writer side) and the agent (reader side, later).
Self-contained: LanceDB is embedded (no server), so this package owns the
on-disk dataset location via `settings.lancedb_path`.

Built in layers, same staging discipline as `kg/`:

- Writer side (now):
  - `schema.py` - pyarrow schema for the chunks table (single source of truth)
  - `client.py` - `LanceClient` wrapper: ensure_schema, write_chunks
- Reader side (later, when the agent is being built):
  - `hybrid_search(query, top_k, filters)` - BM25 + vector with RRF fusion,
    optional rerankers (Cohere, ColBERT, cross-encoder)

The schema mirrors the chunk-level fields of the old ES schema in
ResearchArticlesAgent (chunk_id, doc_id, text, embedding, section, ...) plus
a small denormalised cache of the most-displayed metadata fields (title,
year, authors_display, doi, ...). Authors-as-structured-objects, citations,
and venues live in Neo4j as the normalised source of truth; LanceDB joins
to them by `doc_id` when those fields are needed.
"""
