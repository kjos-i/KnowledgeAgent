"""Pydantic schemas used across the agent graph.

Two families live here:

- LLM-output schemas (`ModeChoice`, `SearchQueryRewrite`,
  `CypherQueryRewrite`, `AgentAnswer`, `ChunkSource`, `KGSource`)
  - LLM nodes use LangChain's `with_structured_output(Model)` so the LLM
    is forced to emit JSON matching these shapes.
- Retrieval schemas (`RetrievedChunk`, `KGHit`)
  - The return shape of search methods on `LanceClient` and of the Neo4j
    retriever. Carried through graph state so downstream nodes have typed
    access to hit fields.

Keeping all schemas in one module - rather than scattered across nodes -
means callers and tests reference one place.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchQueryRewrite(BaseModel):
    """Output of the `query_builder` node.

    The LLM reads the user's raw question and produces a search-optimised
    rewrite: keyword-dense, no conversational filler, no "and so on" tails.
    For document search this typically means extracting the
    domain terms and dropping question phrasing.
    """

    search_query: str = Field(
        ...,
        description=(
            "Polished query string optimised for hybrid (BM25 + vector) "
            "search. Keyword-dense, no filler."
        ),
    )


class CypherQueryRewrite(BaseModel):
    """Output of the `cypher_builder` node.

    The LLM reads the user's question plus the KG schema and produces a
    Cypher query targeting the Neo4j graph store. Sibling of
    SearchQueryRewrite for the LanceDB side.
    """

    cypher_query: str = Field(
        ...,
        description=(
            "Cypher query string targeting the KG (labels and relationships "
            "from `kg/schema.py`). Read-only - the runtime rejects DELETE / "
            "DROP / CREATE / MERGE / SET / REMOVE."
        ),
    )


class ModeChoice(BaseModel):
    """Output of the `mode_classifier` node (auto mode only).

    The LLM reads the user's question and picks ONE of the five concrete
    retrieval modes. `auto` itself is NOT a valid choice - the classifier
    exists precisely to resolve it to a concrete mode.
    """

    mode: Literal[
        "lancedb_only",
        "neo4j_only",
        "lancedb_then_neo4j",
        "neo4j_then_lancedb",
        "parallel_fused",
    ] = Field(
        ...,
        description=(
            "Chosen retrieval mode. Written into the graph state as "
            "`routed_mode`; the static router then dispatches accordingly."
        ),
    )


class RetrievedChunk(BaseModel):
    """One hit returned by a search method on `LanceClient`.

    Carries the chunk's content + a relevance score + the denormalised
    display fields cached on the LanceDB row (so the agent can render hits
    without a Neo4j round-trip). Required fields are the identity + text;
    everything else is nullable because metadata may not have resolved.
    """

    # ---- identity & content (required) ----
    chunk_id: str = Field(..., description="`{doc_id}#{chunk_index}` form.")
    doc_id: str = Field(..., description="SHA-256 file-bytes hash.")
    text: str = Field(..., description="The chunk's body text.")

    # ---- relevance signal ----
    score: float | None = Field(
        default=None,
        description=(
            "Relevance score from the search engine. RRF fused score in "
            "hybrid mode, cosine similarity in vector mode, BM25 in fts mode. "
            "Scales differ across modes - useful for ranking within a single "
            "result set, not for comparing across modes."
        ),
    )

    # ---- denormalised display fields (nullable; mirror the LanceDB row) ----
    title: str | None = Field(default=None)
    year: int | None = Field(default=None)
    authors_display: str | None = Field(default=None)
    doi: str | None = Field(default=None)
    openalex_id: str | None = Field(default=None)
    venue: str | None = Field(default=None)
    source_url: str | None = Field(default=None)
    section: str | None = Field(default=None)
    page: int | None = Field(default=None)
    # ---- multimodal fields (nullable; only set for figure chunks) ----
    content_type: str | None = Field(
        default=None,
        description=(
            "LanceDB row's content_type: 'text', 'figure', 'row', "
            "'json_object', 'code', 'transcript', etc. Populated by "
            "the retriever from the LanceDB row so the synthesizer / "
            "GUI can distinguish figure chunks from text chunks."
        ),
    )
    image_ref: str | None = Field(
        default=None,
        description=(
            "LanceDB row's image_ref — path to the picture file for "
            "figure chunks. Absolute for standalone image inputs; "
            "corpus-folder path for extracted PDF/Word/PPT pictures. "
            "None on text chunks."
        ),
    )


class KGHit(BaseModel):
    """One row returned by the Neo4j retriever.

    The shape of a Cypher result row depends on what the LLM-generated
    query asked for, so the contents are stored as a flexible dict rather
    than typed fields. The retriever wraps each row in a KGHit; the
    synthesizer renders them in its prompt; the LLM cites them by index
    (see `KGSource`).
    """

    data: dict[str, Any] = Field(
        ...,
        description=(
            "The row's column data, as returned by Neo4j. Keys depend on "
            "the Cypher query that produced the row."
        ),
    )


class ChunkSource(BaseModel):
    """One source reference inside an `AgentAnswer`.

    Anchors a claim in the answer to a specific chunk that supported it -
    "this sentence in the answer came from this chunk". Distinct from a
    paper-to-paper citation in the KG (`:CITES` edges); those are
    bibliographic, these are answer-attribution.
    """

    chunk_id: str = Field(
        ...,
        description="The `chunk_id` of the supporting chunk.",
    )
    doc_id: str = Field(
        ...,
        description="The `doc_id` of the document the chunk belongs to.",
    )
    quote: str | None = Field(
        default=None,
        description=(
            "Optional short verbatim quote from the chunk that anchors the "
            "citation. Useful for the GUI to show the supporting text."
        ),
    )
    content_type: str | None = Field(
        default=None,
        description=(
            "Optional chunk content type: 'text' (default), 'figure' "
            "(multimodal picture citation), 'row', 'json_object', "
            "'code', 'transcript', etc. None means the caller (usually "
            "the synthesizer) didn't populate it — the GUI falls back "
            "to text-chunk render. Populated by the synthesizer from "
            "the retrieval hit's row metadata."
        ),
    )
    image_ref: str | None = Field(
        default=None,
        description=(
            "For figure chunks: path to the picture file. Absolute "
            "path for standalone image inputs; corpus-folder path "
            "(`<corpus folder>/figures/<doc_id>/<i>.png`) for pictures "
            "extracted from PDF / Word / PPT. None for text chunks."
        ),
    )
    page: int | None = Field(
        default=None,
        description=(
            "Optional page number for figure chunks extracted from "
            "PDFs / Word / PPT — used by the citation format "
            "'[Figure N, page P of DocTitle]'. None for text chunks "
            "or when the source didn't expose a page number."
        ),
    )


class KGSource(BaseModel):
    """One source reference inside an `AgentAnswer` for a graph-row citation.

    Cited by index into the `kg_hits` list rather than by a natural ID:
    LLM-generated Cypher can return composite rows (e.g.,
    `{paper, author, year_count}`) that have no single canonical identifier.
    """

    hit_index: int = Field(
        ...,
        description=(
            "Zero-based index into the `kg_hits` list this answer was "
            "synthesised from. The full row data lives there."
        ),
    )
    quote: str | None = Field(
        default=None,
        description=(
            "Optional short verbatim or paraphrased snippet from the row "
            "that anchors the citation. Useful for the GUI to show what "
            "was used."
        ),
    )
    row: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The full cited row data (`kg_hits[hit_index].data`), copied in "
            "deterministically after synthesis (not LLM-produced). Carries the "
            "graph finding's provenance (doc_id / chunk_id / evidence_span when "
            "the query returned them), which joins back to the LanceDB source "
            "chunk the finding came from. None until enriched."
        ),
    )


class AgentAnswer(BaseModel):
    """Output of the `synthesizer` node.

    The user-facing answer plus the evidence that supports it. The answer
    text is required; both source lists may be empty (e.g., when no
    relevant evidence was retrieved and the synthesizer says so).

    Two evidence lists, one per retrieval store:
    - `chunk_sources`: passages from the LanceDB vector + BM25 store.
    - `kg_sources`: rows from the Neo4j graph.
    """

    answer: str = Field(
        ...,
        description=(
            "Natural-language answer to the user's question, grounded in "
            "the retrieved evidence. May include source markers ([1], [2], "
            "etc.) that correspond to entries in `chunk_sources` and/or "
            "`kg_sources`."
        ),
    )
    chunk_sources: list[ChunkSource] = Field(
        default_factory=list,
        description=(
            "Chunks that support the answer, in the order they appear in "
            "the answer text. Empty list = no supporting chunks (e.g., the "
            "answer is 'I don't have enough information', or the agent ran "
            "in `neo4j_only` mode where `kg_sources` carries the evidence)."
        ),
    )
    kg_sources: list[KGSource] = Field(
        default_factory=list,
        description=(
            "Graph rows that support the answer, in the order they appear "
            "in the answer text. Empty list = no supporting kg hits (e.g., "
            "the agent ran in `lancedb_only` mode where `chunk_sources` "
            "carries the evidence)."
        ),
    )
