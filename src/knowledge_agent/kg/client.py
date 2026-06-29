"""Neo4j driver wrapper for the knowledge-graph store.

Owns the connection lifecycle (lazy open, explicit close), applies schema
constraints, exposes the agent's read path (`read_query`), and binds the
per-layer write functions as async methods.

Per-layer write modules:
  - L1 + L2 + L3 + L4 -> `kg/openalex_writes.py` (OpenAlex-derived writes)
  - L5 (chunks)       -> `kg/chunk_writes.py`
  - L6a (entities)    -> `kg/entity_writes.py`   (raw entity extraction)
  - L7 (ontologies)   -> `kg/ontology_<name>_writes.py` (one module per
                                                         ontology: MeSH, GO,
                                                         ChEBI, ...). Each
                                                         imports its source
                                                         file once + writes
                                                         canonical-term nodes.

`Neo4jClient` exposes each write function as a 1-line async wrapper
method, so callers do `await client.write_citations(...)` and the
implementation lives in the layer's own (sync) module. The wrapper
runs the sync write in a worker thread via `asyncio.to_thread`,
keeping the event loop free for concurrent work in the orchestrator.
The underlying writes modules stay sync until a future sprint
migrates them to the Neo4j AsyncDriver natively.

Error policy (typed-errors contract): KG write/read failures propagate
to the caller. The ingestion pipeline + bulk_ops orchestrators are the
boundary that catches and records on `IngestResult._error` fields.
When the agent reads from the KG, `read_query` goes through
`session.execute_read` for the read-only safety belt.

LanceDB<->KG consistency is **best-effort, not transactional**: each
write is independently idempotent (re-running the same write is a no-op),
and per-step failures are caught at the orchestrator boundary so one
failed step doesn't abort the others. A future cross-store transaction
would need a saga/outbox pattern; not warranted at this scale.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from neo4j import Driver, GraphDatabase

from knowledge_agent.config import Settings, get_settings
from knowledge_agent.entity_extractors.base import Mention
from knowledge_agent.kg import (
    chunk_writes,
    cross_doc_writes,
    cross_doc_xrefs_writes,
    entity_writes,
    ontology_chebi_writes,
    ontology_cl_writes,
    ontology_dron_writes,
    ontology_eco_writes,
    ontology_efo_writes,
    ontology_envo_writes,
    ontology_fibo_writes,
    ontology_foodon_writes,
    ontology_go_writes,
    ontology_hpo_writes,
    ontology_linking,
    ontology_mesh_writes,
    ontology_mondo_writes,
    ontology_ncbitaxon_writes,
    ontology_obi_writes,
    ontology_po_writes,
    ontology_pr_writes,
    ontology_so_writes,
    ontology_uberon_writes,
    openalex_writes,
    triples_writes,
)
from knowledge_agent.kg.schema import CONSTRAINT_STATEMENTS

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thin sync wrapper around the neo4j `Driver`.

    One instance per app — the driver IS the connection pool (see neo4j
    docs). Lazy-opens on first use so import-time has no DB dependency.
    `NEO4J_PASSWORD` is required (validated at `Settings` load), so any
    Neo4jClient that gets constructed already has working credentials.
    """

    # ---- lifecycle ----

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        """Lazy driver (connection pool). `close()` it at shutdown."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(
                    self._settings.neo4j_user,
                    self._settings.neo4j_password,
                ),
            )
        return self._driver

    async def close(self) -> None:
        """Close the driver + reset the lazy handle. Async-compatible
        wrapper around the sync `driver.close()` (the underlying Neo4j
        sync driver's close is fast + blocking; `asyncio.to_thread`
        keeps the event loop responsive for ordered shutdown)."""
        if self._driver is not None:
            await asyncio.to_thread(self._driver.close)
            self._driver = None

    # ---- constraints ----

    async def ensure_constraints(self) -> None:
        """Apply schema constraints idempotently. Safe to call every startup.

        Cypher failures propagate to the caller under the typed-errors
        contract; the orchestrator boundary (pipeline / bulk_ops) catches.
        """
        def _run_all():
            with self.driver.session() as session:
                for stmt in CONSTRAINT_STATEMENTS:
                    session.run(stmt)
        await asyncio.to_thread(_run_all)

    # ---- OpenAlex writes (L1-L4) - implementations in `openalex_writes.py` ----

    async def write_citations(self, doc_id: str, work: dict[str, Any]) -> None:
        """L1: focal :Document + cited shadows + :CITES edges.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.write_citations`.
        """
        return await asyncio.to_thread(
            openalex_writes.write_citations,
            self, doc_id, work,
        )
    async def write_authorships(self, doc_id: str, work: dict[str, Any]) -> None:
        """L2: :Author nodes + :AUTHORED edges (with position, is_corresponding).

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.write_authorships`.
        """
        return await asyncio.to_thread(
            openalex_writes.write_authorships,
            self, doc_id, work,
        )
    async def write_venue(self, doc_id: str, work: dict[str, Any]) -> None:
        """L3: :Venue node + :PUBLISHED_IN edge.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.write_venue`.
        """
        return await asyncio.to_thread(
            openalex_writes.write_venue,
            self, doc_id, work,
        )
    async def write_topics(self, doc_id: str, work: dict[str, Any]) -> None:
        """L4: :Topic nodes + :ABOUT_TOPIC edges (with score).

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.write_topics`.
        """
        return await asyncio.to_thread(
            openalex_writes.write_topics,
            self, doc_id, work,
        )
    async def delete_doc(self, doc_id: str) -> None:
        """Wipe a paper's L1-L4 KG data: focal :Document + edges + GC orphans.

        Called by the ingestion pipeline BEFORE the L1-L4 writes, so re-ingest
        produces a clean state. Fires unconditionally (not gated by
        metadata resolution).

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.delete_doc`.
        """
        return await asyncio.to_thread(
            openalex_writes.delete_doc,
            self, doc_id,
        )
    async def delete_doc_l1_l4_edges(self, doc_id: str) -> None:
        """Wipe L1-L4 edges only; preserve focal :Document + :PART_OF.

        Surgical variant of `delete_doc` for partial ops (`resolve_openalex`)
        that need to re-write L1-L4 without orphaning `:Chunk` nodes.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `openalex_writes.delete_doc_l1_l4_edges`.
        """
        return await asyncio.to_thread(
            openalex_writes.delete_doc_l1_l4_edges,
            self, doc_id,
        )
    async def delete_chunks_by_doc_id(self, doc_id: str) -> None:
        """L5: wipe every :Chunk node for `doc_id`. Idempotent.

        Mirrors `LanceClient.delete_chunks_by_doc_id` - the ingestion
        pipeline calls this before `write_chunks` so a re-chunk leaves
        no orphans.

        Raises on failure (empty `doc_id` → ValueError; Cypher failures
        propagate). Delegates to `chunk_writes.delete_chunks_by_doc_id`.
        """
        return await asyncio.to_thread(
            chunk_writes.delete_chunks_by_doc_id,
            self, doc_id,
        )
    async def write_chunks(
        self,
        doc_id: str,
        chunks: list[Any],
        main_label: str,
        sub_label: str | None = None,
    ) -> None:
        """L5: :Chunk nodes + :PART_OF edges for one document.

        `chunks` is a list of objects with `chunk_index`, `section`, `page`,
        and `content_type` attributes (typically `ParsedChunk`). Chunk
        text + embedding are NEVER stored in the KG; those stay in LanceDB.

        `main_label` (`"Document"` or `"Artifact"`) and the optional
        `sub_label` (`"Paper"`, `"Note"`, ...) are applied to the focal
        node when this call creates it.

        Raises on failure (empty doc_id / unknown label → ValueError;
        Cypher failures propagate). Delegates to `chunk_writes.write_chunks`.
        """
        return await asyncio.to_thread(
            chunk_writes.write_chunks,
            self, doc_id, chunks, main_label, sub_label,
        )

    # ---- entity writes (L6a) - implementations in `entity_writes.py` ----

    async def delete_entities_by_doc_id(self, doc_id: str) -> None:
        """L6a: drop :MENTIONS edges from this doc's chunks + GC orphan
        :Entity nodes. Idempotent.

        Mirrors the orphan-GC step in `openalex_writes.delete_doc`. The
        ingestion pipeline calls this before `write_entities` so a re-
        extract leaves no stale edges or orphan entities.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `entity_writes.delete_entities_by_doc_id`.
        """
        return await asyncio.to_thread(
            entity_writes.delete_entities_by_doc_id,
            self, doc_id,
        )
    async def write_entities(
        self,
        doc_id: str,
        chunk_mentions: list[tuple[str, list[Mention]]],
    ) -> None:
        """L6a: :Entity nodes + :MENTIONS edges for one document.

        `chunk_mentions` is `[(chunk_id, [Mention, ...]), ...]` produced
        by the pipeline's per-chunk extraction loop. Each Mention's
        `raw_text` is lowercased to compute the :Entity merge key; the
        original spelling is not stored on the node (recoverable from
        chunk text via offset for NER, lost for LLM).

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `entity_writes.write_entities`.
        """
        return await asyncio.to_thread(
            entity_writes.write_entities, self, doc_id, chunk_mentions,
        )

    # ---- L8 triples writes - implementations in `triples_writes.py` ----

    async def get_entities_by_chunk(
        self, doc_id: str
    ) -> dict[str, list[tuple[str, str]]]:
        """L8 backfill helper: read this doc's L6a entity vocabulary
        grouped by chunk_id. Returns `{chunk_id: [(key, type), ...]}`.

        Used by `pipeline.backfill_triples` so backfill can run without
        re-running L6a - reuses the entities already in Neo4j.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `triples_writes.get_entities_by_chunk`.
        """
        return await asyncio.to_thread(
            triples_writes.get_entities_by_chunk, self, doc_id,
        )

    async def delete_triples_by_doc_id(self, doc_id: str) -> None:
        """L8: drop typed-relation edges this doc was the source of.

        One Cypher unions all 15 predicate edge types via the
        `[:R1|R2|...]` pipe syntax and deletes any edge whose
        `doc_id` property matches. Idempotent.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `triples_writes.delete_triples_by_doc_id`.
        """
        return await asyncio.to_thread(
            triples_writes.delete_triples_by_doc_id,
            self, doc_id,
        )
    async def write_triples(
        self,
        doc_id: str,
        chunk_triples: list[
            tuple[str, list[triples_writes.ExtractedTriple]]
        ],
    ) -> None:
        """L8: typed entity-to-entity edges from per-chunk LLM extraction.

        `chunk_triples` is `[(chunk_id, [ExtractedTriple, ...]), ...]`.
        Each ExtractedTriple's `predicate` must be one of the 15
        constants in `schema.TRIPLE_PREDICATE_RELS`; out-of-vocabulary
        predicates are logged + skipped per-row, not raised.

        Raises on failure (empty doc_id → ValueError; Cypher failures
        propagate). Delegates to `triples_writes.write_triples`.
        """
        return await asyncio.to_thread(
            triples_writes.write_triples, self, doc_id, chunk_triples,
        )

    # ---- L9 cross-doc writes - implementation in `cross_doc_writes.py` ----

    async def recompute_cross_doc_edges(
        self,
        doc_id: str,
        threshold: int = cross_doc_writes.DEFAULT_SHARED_COUNT_THRESHOLD,
    ) -> int:
        """L9: wipe + recompute `:RELATED_TO` edges incident to this doc.

        Walks `(focal)<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(:Entity)
        <-[:MENTIONS]-(:Chunk)-[:PART_OF]->(other)` and writes one
        undirected `:RELATED_TO` edge per (this, other) pair whose
        shared distinct entity count meets `threshold`. Edge
        properties carry the shared keys + count + timestamp.

        Returns the count of edges (re)written. 0 = no other doc met
        the threshold (a clean outcome).

        Raises on failure (empty doc_id / threshold < 1 → ValueError;
        Cypher failures propagate). Delegates to
        `cross_doc_writes.recompute_cross_doc_edges`.
        """
        return await asyncio.to_thread(
            cross_doc_writes.recompute_cross_doc_edges,
            self, doc_id, threshold,
        )

    # ---- L10 cross-doc xrefs - implementation in `cross_doc_xrefs_writes.py` ----

    async def recompute_cross_doc_xrefs_edges(
        self,
        doc_id: str,
        threshold: int = (
            cross_doc_xrefs_writes.DEFAULT_SHARED_COUNT_THRESHOLD
        ),
    ) -> int:
        """L10: wipe + recompute `:RELATED_BY_XREF` edges incident to
        this doc.

        Walks `(focal)<-[:PART_OF]-(:Chunk)-[:MENTIONS]->(:Entity)
        -[:CANONICAL_TO]->(:OntologyTerm)` on both sides; joins where
        the two terms are identical OR connected by a `:<X>_XREF`
        edge. Writes one undirected `:RELATED_BY_XREF` edge per
        (this, other) pair whose distinct shared-concept count meets
        `threshold`. Edge properties carry the shared concept IDs +
        count + timestamp.

        Returns the count of edges (re)written. 0 = no other doc met
        the threshold (a clean outcome).

        Raises on failure (empty doc_id / threshold < 1 → ValueError;
        Cypher failures propagate). Delegates to
        `cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges`.
        """
        return await asyncio.to_thread(
            cross_doc_xrefs_writes.recompute_cross_doc_xrefs_edges,
            self, doc_id, threshold,
        )

    # ---- L7 ontology imports - implementations in `kg/ontology_*_writes.py` ----

    async def is_mesh_imported(self) -> bool:
        """True when at least one `:MeSHTerm` node exists in Neo4j.

        Cheap label-index lookup. Used by the pipeline's auto-on-first-
        ingest path to decide whether to trigger a fresh MeSH import.

        Delegates to `ontology_mesh_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_mesh_writes.is_imported,
            self,
        )
    async def import_mesh(self, *, force: bool = False) -> bool:
        """Download + parse + write the MeSH ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a yearly MeSH
        release update). Download / parse / write failures
        propagate to the caller (typed-errors contract).

        Delegates to `ontology_mesh_writes.import_mesh`.
        """
        return await asyncio.to_thread(
            ontology_mesh_writes.import_mesh,
            self, force=force,
        )
    async def delete_mesh(self) -> None:
        """Drop every `:MeSHTerm` node + its edges. Idempotent.

        Used by maintenance flows that want a clean state before a
        version-bumped re-import. Normal per-doc re-ingestion does NOT
        delete MeSH - the ontology is corpus-scoped, not doc-scoped.

        Delegates to `ontology_mesh_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_mesh_writes.delete_imported,
            self,
        )
    async def is_go_imported(self) -> bool:
        """True when at least one `:GOTerm` node exists in Neo4j.

        Delegates to `ontology_go_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_go_writes.is_imported,
            self,
        )
    async def import_go(self, *, force: bool = False) -> bool:
        """Download + parse + write the Gene Ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a GO release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_go_writes.import_go`.
        """
        return await asyncio.to_thread(
            ontology_go_writes.import_go,
            self, force=force,
        )
    async def delete_go(self) -> None:
        """Drop every `:GOTerm` node + its edges. Idempotent.

        Delegates to `ontology_go_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_go_writes.delete_imported,
            self,
        )
    async def is_hpo_imported(self) -> bool:
        """True when at least one `:HPOTerm` node exists in Neo4j.

        Delegates to `ontology_hpo_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_hpo_writes.is_imported,
            self,
        )
    async def import_hpo(self, *, force: bool = False) -> bool:
        """Download + parse + write the Human Phenotype Ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a HPO release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_hpo_writes.import_hpo`.
        """
        return await asyncio.to_thread(
            ontology_hpo_writes.import_hpo,
            self, force=force,
        )
    async def delete_hpo(self) -> None:
        """Drop every `:HPOTerm` node + its edges. Idempotent.

        Delegates to `ontology_hpo_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_hpo_writes.delete_imported,
            self,
        )
    async def is_uberon_imported(self) -> bool:
        """True when at least one `:UBERONTerm` node exists in Neo4j.

        Delegates to `ontology_uberon_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_uberon_writes.is_imported,
            self,
        )
    async def import_uberon(self, *, force: bool = False) -> bool:
        """Download + parse + write the Uber Anatomy Ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a UBERON release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_uberon_writes.import_uberon`.
        """
        return await asyncio.to_thread(
            ontology_uberon_writes.import_uberon,
            self, force=force,
        )
    async def delete_uberon(self) -> None:
        """Drop every `:UBERONTerm` node + its edges. Idempotent.

        Delegates to `ontology_uberon_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_uberon_writes.delete_imported,
            self,
        )
    async def is_mondo_imported(self) -> bool:
        """True when at least one `:MONDOTerm` node exists in Neo4j.

        Delegates to `ontology_mondo_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_mondo_writes.is_imported,
            self,
        )
    async def import_mondo(self, *, force: bool = False) -> bool:
        """Download + parse + write the Mondo Disease Ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a MONDO release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_mondo_writes.import_mondo`.
        """
        return await asyncio.to_thread(
            ontology_mondo_writes.import_mondo,
            self, force=force,
        )
    async def delete_mondo(self) -> None:
        """Drop every `:MONDOTerm` node + its edges. Idempotent.

        Delegates to `ontology_mondo_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_mondo_writes.delete_imported,
            self,
        )
    async def is_chebi_imported(self) -> bool:
        """True when at least one `:ChEBITerm` node exists in Neo4j.

        Delegates to `ontology_chebi_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_chebi_writes.is_imported,
            self,
        )
    async def import_chebi(self, *, force: bool = False) -> bool:
        """Download + parse + write the ChEBI ontology (LITE variant) to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a ChEBI release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_chebi_writes.import_chebi`.
        """
        return await asyncio.to_thread(
            ontology_chebi_writes.import_chebi,
            self, force=force,
        )
    async def delete_chebi(self) -> None:
        """Drop every `:ChEBITerm` node + its edges. Idempotent.

        Delegates to `ontology_chebi_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_chebi_writes.delete_imported,
            self,
        )
    async def is_eco_imported(self) -> bool:
        """True when at least one `:ECOTerm` node exists in Neo4j.

        Delegates to `ontology_eco_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_eco_writes.is_imported,
            self,
        )
    async def import_eco(self, *, force: bool = False) -> bool:
        """Download + parse + write the Evidence & Conclusion Ontology to Neo4j.

        Returns True when the import actually ran; False when the
        ontology was already imported (no-op). Pass `force=True`
        to drop and re-import (useful after a ECO release
        update). Download / parse / write failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_eco_writes.import_eco`.
        """
        return await asyncio.to_thread(
            ontology_eco_writes.import_eco,
            self, force=force,
        )
    async def delete_eco(self) -> None:
        """Drop every `:ECOTerm` node + its edges. Idempotent.

        Delegates to `ontology_eco_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_eco_writes.delete_imported,
            self,
        )
    async def is_so_imported(self) -> bool:
        """True when at least one `:SOTerm` node exists in Neo4j.

        Delegates to `ontology_so_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_so_writes.is_imported,
            self,
        )
    async def import_so(self, *, force: bool = False) -> bool:
        """Download + parse + write the Sequence Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_so_writes.import_so`.
        """
        return await asyncio.to_thread(
            ontology_so_writes.import_so,
            self, force=force,
        )
    async def delete_so(self) -> None:
        """Drop every `:SOTerm` node + its edges. Idempotent.

        Delegates to `ontology_so_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_so_writes.delete_imported,
            self,
        )
    async def is_pr_imported(self) -> bool:
        """True when at least one `:PRTerm` node exists in Neo4j.

        Delegates to `ontology_pr_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_pr_writes.is_imported,
            self,
        )
    async def import_pr(self, *, force: bool = False) -> bool:
        """Download + parse + write the Protein Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_pr_writes.import_pr`.
        """
        return await asyncio.to_thread(
            ontology_pr_writes.import_pr,
            self, force=force,
        )
    async def delete_pr(self) -> None:
        """Drop every `:PRTerm` node + its edges. Idempotent.

        Delegates to `ontology_pr_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_pr_writes.delete_imported,
            self,
        )
    async def is_cl_imported(self) -> bool:
        """True when at least one `:CLTerm` node exists in Neo4j.

        Delegates to `ontology_cl_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_cl_writes.is_imported,
            self,
        )
    async def import_cl(self, *, force: bool = False) -> bool:
        """Download + parse + write the Cell Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_cl_writes.import_cl`.
        """
        return await asyncio.to_thread(
            ontology_cl_writes.import_cl,
            self, force=force,
        )
    async def delete_cl(self) -> None:
        """Drop every `:CLTerm` node + its edges. Idempotent.

        Delegates to `ontology_cl_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_cl_writes.delete_imported,
            self,
        )
    async def is_po_imported(self) -> bool:
        """True when at least one `:POTerm` node exists in Neo4j.

        Delegates to `ontology_po_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_po_writes.is_imported,
            self,
        )
    async def import_po(self, *, force: bool = False) -> bool:
        """Download + parse + write the Plant Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_po_writes.import_po`.
        """
        return await asyncio.to_thread(
            ontology_po_writes.import_po,
            self, force=force,
        )
    async def delete_po(self) -> None:
        """Drop every `:POTerm` node + its edges. Idempotent.

        Delegates to `ontology_po_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_po_writes.delete_imported,
            self,
        )
    async def is_foodon_imported(self) -> bool:
        """True when at least one `:FOODONTerm` node exists in Neo4j.

        Delegates to `ontology_foodon_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_foodon_writes.is_imported,
            self,
        )
    async def import_foodon(self, *, force: bool = False) -> bool:
        """Download + parse + write the Food Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_foodon_writes.import_foodon`.
        """
        return await asyncio.to_thread(
            ontology_foodon_writes.import_foodon,
            self, force=force,
        )
    async def delete_foodon(self) -> None:
        """Drop every `:FOODONTerm` node + its edges. Idempotent.

        Delegates to `ontology_foodon_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_foodon_writes.delete_imported,
            self,
        )
    async def is_envo_imported(self) -> bool:
        """True when at least one `:ENVOTerm` node exists in Neo4j.

        Delegates to `ontology_envo_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_envo_writes.is_imported,
            self,
        )
    async def import_envo(self, *, force: bool = False) -> bool:
        """Download + parse + write the Environment Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        Delegates to `ontology_envo_writes.import_envo`.
        """
        return await asyncio.to_thread(
            ontology_envo_writes.import_envo,
            self, force=force,
        )
    async def delete_envo(self) -> None:
        """Drop every `:ENVOTerm` node + its edges. Idempotent.

        Delegates to `ontology_envo_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_envo_writes.delete_imported,
            self,
        )
    async def is_ncbitaxon_imported(self) -> bool:
        """True when at least one `:NCBITaxonTerm` node exists in Neo4j.

        Delegates to `ontology_ncbitaxon_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_ncbitaxon_writes.is_imported,
            self,
        )
    async def import_ncbitaxon(self, *, force: bool = False) -> bool:
        """Download + parse + write NCBI Taxonomy to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import.

        NCBITaxon is the largest ontology on the menu (~2.74M classes,
        ~440 MB OBO source) - see `ontology_ncbitaxon_writes` module
        docstring for memory caveats at parse time.

        Delegates to `ontology_ncbitaxon_writes.import_ncbitaxon`.
        """
        return await asyncio.to_thread(
            ontology_ncbitaxon_writes.import_ncbitaxon,
            self, force=force,
        )
    async def delete_ncbitaxon(self) -> None:
        """Drop every `:NCBITaxonTerm` node + its edges. Idempotent.

        Delegates to `ontology_ncbitaxon_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_ncbitaxon_writes.delete_imported,
            self,
        )
    async def is_obi_imported(self) -> bool:
        """True when at least one `:OBITerm` node exists in Neo4j.

        Delegates to `ontology_obi_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_obi_writes.is_imported,
            self,
        )
    async def import_obi(self, *, force: bool = False) -> bool:
        """Download + parse + write the Ontology for Biomedical
        Investigations to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import. OBI is
        OWL/RDF-XML, parsed via rdflib (not pronto) so the synonym
        surface is preserved - see `ontology_obi_writes` module
        docstring for the OWL reader rationale.

        Delegates to `ontology_obi_writes.import_obi`.
        """
        return await asyncio.to_thread(
            ontology_obi_writes.import_obi,
            self, force=force,
        )
    async def delete_obi(self) -> None:
        """Drop every `:OBITerm` node + its edges. Idempotent.

        Delegates to `ontology_obi_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_obi_writes.delete_imported,
            self,
        )
    async def is_efo_imported(self) -> bool:
        """True when at least one `:EFOTerm` node exists in Neo4j.

        Delegates to `ontology_efo_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_efo_writes.is_imported,
            self,
        )
    async def import_efo(self, *, force: bool = False) -> bool:
        """Download + parse + write the Experimental Factor Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import. EFO is
        OWL/RDF-XML, parsed via rdflib (not pronto) so the synonym
        surface is preserved - see `ontology_efo_writes` module
        docstring for the OWL reader rationale.

        Delegates to `ontology_efo_writes.import_efo`.
        """
        return await asyncio.to_thread(
            ontology_efo_writes.import_efo,
            self, force=force,
        )
    async def delete_efo(self) -> None:
        """Drop every `:EFOTerm` node + its edges. Idempotent.

        Delegates to `ontology_efo_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_efo_writes.delete_imported,
            self,
        )
    async def is_dron_imported(self) -> bool:
        """True when at least one `:DRONTerm` node exists in Neo4j.

        Delegates to `ontology_dron_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_dron_writes.is_imported,
            self,
        )
    async def import_dron(self, *, force: bool = False) -> bool:
        """Download + parse + write the Drug Ontology (DRON) to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import. DRON is
        the second-largest ontology on the menu (~700K classes, ~220
        MB OWL) - see `ontology_dron_writes` module docstring for
        memory caveats at parse time.

        Delegates to `ontology_dron_writes.import_dron`.
        """
        return await asyncio.to_thread(
            ontology_dron_writes.import_dron,
            self, force=force,
        )
    async def delete_dron(self) -> None:
        """Drop every `:DRONTerm` node + its edges. Idempotent.

        Delegates to `ontology_dron_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_dron_writes.delete_imported,
            self,
        )
    async def is_fibo_imported(self) -> bool:
        """True when at least one `:FIBOTerm` node exists in Neo4j.

        Delegates to `ontology_fibo_writes.is_imported`.
        """
        return await asyncio.to_thread(
            ontology_fibo_writes.is_imported,
            self,
        )
    async def import_fibo(self, *, force: bool = False) -> bool:
        """Download (multi-file via GitHub walker) + parse + write the
        Financial Industry Business Ontology to Neo4j.

        Idempotent. Pass `force=True` to drop and re-import. FIBO
        ships modularly (~70 .rdf files); the FIBO module's walker
        fetches each from raw.githubusercontent.com on first install,
        caches them locally, then loads them into one rdflib graph
        for term extraction. Subsequent imports re-use the local
        cache.

        Delegates to `ontology_fibo_writes.import_fibo`.
        """
        return await asyncio.to_thread(
            ontology_fibo_writes.import_fibo,
            self, force=force,
        )
    async def delete_fibo(self) -> None:
        """Drop every `:FIBOTerm` node + its edges. Idempotent.

        Delegates to `ontology_fibo_writes.delete_imported`.
        """
        return await asyncio.to_thread(
            ontology_fibo_writes.delete_imported,
            self,
        )
    async def count_ontology_terms(self, ontology_name: str) -> int:
        """Count the term nodes of a given ontology.

        Used by `ontology_lifecycle.delete_ontology_plan` so the
        confirmation dialog can show "Delete MeSH (30,142 terms)". Also
        useful as a post-import sanity check. Cypher failures propagate
        to the caller (typed-errors contract).

        Delegates to `ontology_linking.count_ontology_terms`.
        """
        return await asyncio.to_thread(
            ontology_linking.count_ontology_terms,
            self, ontology_name,
        )
    async def count_canonical_links(self, ontology_name: str) -> int:
        """Count `:CANONICAL_TO` edges pointing at a given ontology's terms.

        Used by `ontology_lifecycle.delete_ontology_plan` so the dialog
        can show how many entity->ontology links die with the term wipe.
        Cypher failures propagate (typed-errors contract).

        Delegates to `ontology_linking.count_canonical_links`.
        """
        return await asyncio.to_thread(
            ontology_linking.count_canonical_links,
            self, ontology_name,
        )
    async def ensure_ontology_imported(
        self,
        ontology_name: str,
        *,
        xrefs_mode: str = "none",
    ) -> bool:
        """Make sure the named ontology is imported. Returns
        `was_already_imported` (True = no-op, False = import just ran).

        Idempotent. The pipeline uses this to decide whether to run a
        global linking pass (first import) or just link the current
        doc's entities (subsequent ingests).

        `xrefs_mode` is forwarded to the underlying import. Default
        `"none"`. See `kg.ontology_helpers.write_ontology_terms` for
        the accepted values and their semantics.

        Raises on failure (unknown ontology → ValueError; download /
        parse / write failures propagate). Delegates to
        `ontology_linking.ensure_ontology_imported`.
        """
        return await asyncio.to_thread(
            ontology_linking.ensure_ontology_imported,
            self, ontology_name, xrefs_mode=xrefs_mode,
        )

    async def link_entities_to_ontology(
        self,
        ontology_name: str,
        matching_strategy: str,
        *,
        doc_id: str | None = None,
    ) -> int:
        """Run the L7 linking pass for one ontology. Returns the count
        of `:CANONICAL_TO` edges written.

        With `doc_id` set, only entities mentioned by chunks of THAT
        doc are linked. With `doc_id=None`, runs globally across all
        entities not yet linked to this ontology - used on first-time
        ontology import.

        Delegates to `ontology_linking.link_entities`. The matching
        strategy ('exact' / 'fuzzy') comes from the per-corpus
        `OntologyConfig.matching` setting.
        """
        return await asyncio.to_thread(
            ontology_linking.link_entities,
            self,
            ontology_name,
            matching_strategy,  # type: ignore[arg-type]
            doc_id=doc_id,
        )

    # ---- reads (agent path) ----

    async def read_query(
        self, cypher: str, **params: Any
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query in a read-only transaction.

        Uses `session.execute_read(...)`, which Neo4j refuses to use for
        any query containing write clauses. Combined with the keyword
        validator in `cypher_safety.py`, this is the second of three
        safety layers around LLM-generated Cypher.

        Returns one dict per row, with keys = the RETURN aliases from the
        query (via Record.data()).

        Does NOT fail-soft: Cypher syntax errors, missing labels, and
        connection failures all raise. The caller (the neo4j_retriever
        node) wraps this in try/except and logs + returns empty kg_hits.
        Keeping the raise here means tests can assert error behaviour
        without it being swallowed in the client.
        """
        def _run():
            with self.driver.session() as session:
                records = session.execute_read(
                    lambda tx: list(tx.run(cypher, **params))
                )
            return [record.data() for record in records]
        return await asyncio.to_thread(_run)


@lru_cache(maxsize=1)
def get_kg_client() -> Neo4jClient:
    """Process-wide singleton `Neo4jClient`. Mirrors the cached config."""
    return Neo4jClient()
