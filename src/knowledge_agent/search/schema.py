"""Single source of truth for the LanceDB chunks-table schema.

The schema is a function that reads the embedding dimension from Settings,
so changing the model + dim is a config change - never a schema edit.
Other table-shape constants (table name, content-type values,
metadata-status values) live here too, so writers and readers agree.

LanceDB tables are columnar (Lance format). The vector column uses
`fixed_size_list(float32, dim)` so the per-row size is known up front
(needed by the HNSW / IVF vector indexes built later).
"""

import pyarrow as pa

from knowledge_agent.config import get_settings

# ---- Table name.

CHUNKS_TABLE = "chunks"

# ---- Enumerated value sets (kept here so writers + readers agree).

CONTENT_TYPES: tuple[str, ...] = ("text", "figure", "table")

METADATA_STATUS_VALUES: tuple[str, ...] = (
    "enriched",  # DOI resolved against OpenAlex, full metadata available.
    "pending",  # DOI resolution attempted, failed - eligible for retry.
    "baseline",  # No DOI resolution attempted (non-PDF, or resolve_doi off).
    "manual",  # User-edited via the Metadata view.
)


def chunks_schema() -> pa.Schema:
    """Return the chunks table schema.

    Reads `embedding_dims` from Settings so the vector column matches the
    embedding model's output dimension. Changing the model + dim requires
    a corpus re-embed and a new table.
    """
    dims = get_settings().embedding_dims
    return pa.schema(
        [
            # ---- Identity & structure.
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("doc_id", pa.string(), nullable=False),
            pa.field("chunk_index", pa.int32(), nullable=False),
            pa.field("section", pa.string(), nullable=True),
            pa.field("page", pa.int32(), nullable=True),
            pa.field("char_start", pa.int32(), nullable=True),
            pa.field("char_end", pa.int32(), nullable=True),
            pa.field("content_type", pa.string(), nullable=False),
            pa.field("image_ref", pa.string(), nullable=True),
            # ---- Search signals.
            pa.field("text", pa.string(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.float32(), dims),
                nullable=False,
            ),
            # ---- KG label mirrors. main_label is always set; sub_label
            #      is nullable so files ingested without picking a subtype
            #      still produce valid rows.
            pa.field("main_label", pa.string(), nullable=False),
            pa.field("sub_label", pa.string(), nullable=True),
            # ---- Denormalised metadata cache (for hit display without a
            #      Neo4j round-trip).
            pa.field("doi", pa.string(), nullable=True),
            pa.field("openalex_id", pa.string(), nullable=True),
            pa.field("title", pa.string(), nullable=True),
            pa.field("year", pa.int32(), nullable=True),
            pa.field("authors_display", pa.string(), nullable=True),
            pa.field("venue", pa.string(), nullable=True),
            pa.field("source_url", pa.string(), nullable=True),
            pa.field("metadata_status", pa.string(), nullable=False),
            pa.field("language", pa.string(), nullable=True),
            # ---- Source-path hint. Set when ingest knows the file location;
            #      consumed by `bulk_ops.sync` to detect MOVED (same doc_id,
            #      different path) and EDITED (same path, different doc_id).
            #      Nullable so partial-pipeline ops that don't touch the
            #      original file (re-embed, backfill) don't have to invent
            #      a value.
            pa.field("source_path", pa.string(), nullable=True),
            pa.field("ingested_at", pa.timestamp("us"), nullable=False),
        ]
    )
