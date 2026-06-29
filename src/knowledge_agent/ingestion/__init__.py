"""Ingestion pipeline.

Orchestrates the path from one input file to populated KG + LanceDB rows:

  ids.py       - compute_doc_id (SHA-256 of file bytes), make_chunk_id
  parse.py     - docling parser wrapper (file -> text + structure)
  metadata.py  - OpenAlex resolver (DOI -> rich work dict)
  embed.py     - Voyage embedder (text chunks -> vectors)
  pipeline.py  - the orchestrator that ties everything together

Each helper is single-purpose and independently testable. Only `pipeline.py`
calls the others; the others have no upstream dependencies on each other.
"""
